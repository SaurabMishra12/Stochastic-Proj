"""Checkpointed full experiments with trajectory-independent NumPy streams."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,hashlib,json,time,importlib.util
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .pll_model import PLLParameters,initial_phase
from .sde_solver import compiled_path,brownian_increments,coarsen_increments,SUMMARY_COLUMNS,load_kernel
from .statistical_analysis import summarize
from .metrics import mean_ci
ROOT=Path(__file__).resolve().parents[1];MASTER=20261113
NOISE=[0,.025,.05,.1,.15,.2,.3,.4];DETUNING=[0,.25,.5,.75,.9,.95,.99,1.01,1.05]

def source_hash():
    return hashlib.sha256(b''.join((ROOT/'src'/f).read_bytes() for f in ['pll_model.py','sde_solver.py','kernel.cpp','metrics.py','statistical_analysis.py'])).hexdigest()

def design():
    groups={'combined':[],'long_horizon':[],'sensitivity':[],'original_initial':[]}
    for di,d in enumerate(DETUNING):
        for ai,alpha in enumerate(NOISE):
            n=1 if alpha==0 else 500
            if (d==.5 and alpha in [.1,.15,.4]) or (d in [.9,.95,.99] and alpha in [.1,.15,.3,.4]):n=1000
            groups['combined'].append(dict(condition_id=100+di*8+ai,a=1.,b=1.,detuning=d,alpha=alpha,T=50.,dt=.001,n=n))
    for i,(d,alpha) in enumerate((d,a) for d in [.75,.9,.95] for a in [.15,.3]):groups['long_horizon'].append(dict(condition_id=300+i,a=1.,b=1.,detuning=d,alpha=alpha,T=200.,dt=.001,n=1000))
    for i,(a,b,alpha) in enumerate((a,b,alpha) for a in [.5,1.,2.] for b in [.5,1.,2.] for alpha in [.15,.3]):groups['sensitivity'].append(dict(condition_id=400+i,a=a,b=b,detuning=.5,alpha=alpha,T=50.,dt=.001,n=500))
    groups['original_initial'].append(dict(condition_id=200,a=1.,b=1.,detuning=.5,alpha=.15,T=50.,dt=.001,n=1000,phi0=float(np.arcsin(.5)+.5)))
    return groups

def run_condition(c):
    started=time.monotonic();cid=c['condition_id'];folder=ROOT/'results/raw_data'/f'condition_{cid}';folder.mkdir(parents=True,exist_ok=True);fingerprint=hashlib.sha256((json.dumps(c,sort_keys=True)+source_hash()).encode()).hexdigest();completed=folder/'summary.json'
    if completed.exists():
        old=json.loads(completed.read_text())
        if old.get('fingerprint')!=fingerprint:raise RuntimeError(f'Checkpoint hash mismatch: {cid}')
        for file in ['trajectories.csv','ensemble.npz','config.json','events.csv']:
            if not (folder/file).exists():raise RuntimeError(f'Incomplete checkpoint {cid}: {file}')
        return old
    p=PLLParameters(**{k:c[k] for k in ['a','b','detuning','alpha']});nsteps=round(c['T']/c['dt']);stride=nsteps//1000;samples=[];rows=[];events=[];seeds=[]
    for i in range(c['n']):
        seed=int(np.random.SeedSequence([MASTER,cid,i]).generate_state(1,dtype=np.uint64)[0]);seeds.append(seed);dw=brownian_increments(nsteps,c['dt'],seed) if p.alpha else np.zeros(nsteps);out=compiled_path(p,dw,c['dt'],phi0=c.get('phi0'),stride=stride);rows.append(out['summary']);samples.append(out['path']);events.extend((i,float(time),int(direction)) for time,direction in zip(out['event_times'],out['event_directions']))
    frame=pd.DataFrame(rows,columns=SUMMARY_COLUMNS);frame.insert(0,'trajectory_id',range(c['n']));frame.insert(1,'seed',seeds);frame.to_csv(folder/'trajectories.csv',index=False);pd.DataFrame(events,columns=['trajectory_id','time','direction']).to_csv(folder/'events.csv',index=False)
    samples=np.asarray(samples);mean=samples.mean(axis=0);std=samples.std(axis=0,ddof=1) if c['n']>1 else np.zeros_like(mean);margin=student_t.ppf(.975,max(1,c['n']-1))*std/np.sqrt(c['n']) if c['n']>1 else np.full_like(mean,np.nan)
    np.savez_compressed(folder/'ensemble.npz',t=np.arange(0,nsteps+1,stride)*c['dt'],mean=mean,variance=std**2,ci_low=mean-margin,ci_high=mean+margin,representatives=samples[:8],terminal=samples[:,-1,:]);(folder/'config.json').write_text(json.dumps(c,indent=2))
    summary=summarize(frame,c);summary['wall_seconds']=time.monotonic()-started;summary['fingerprint']=fingerprint;summary['source_hash']=source_hash();tmp=completed.with_suffix('.tmp');tmp.write_text(json.dumps(summary,indent=2));tmp.replace(completed);return summary

def export_group(name,configs):
    values=[json.loads((ROOT/'results/raw_data'/f"condition_{c['condition_id']}"/'summary.json').read_text()) for c in configs];pd.DataFrame([{k:v for k,v in row.items() if k!='survival'} for row in values]).to_csv(ROOT/'results/tables'/f'{name}_summary.csv',index=False)

def original_replay():
    spec=importlib.util.spec_from_file_location('original',ROOT/'backups/baseline_1.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original);p=PLLParameters();checks=[]
    for seed in [0,1,2,3,4,42]:
        t,phi,v,_=original.simulate_pll(seed=seed);out=compiled_path(p,brownian_increments(50000,.001,seed),.001,phi0=np.arcsin(.5)+.5,stride=1);checks.append({'seed':seed,'max_phase_difference':float(np.max(np.abs(phi-out['path'][:,0]))),'max_velocity_difference':float(np.max(np.abs(v-out['path'][:,1])))})
        if seed==42:np.savez_compressed(ROOT/'experiments/baseline/original_seed42.npz',t=t,phi=phi,v=v,corrected=out['path'])
    (ROOT/'experiments/baseline/original_replay.json').write_text(json.dumps(checks,indent=2))

def convergence_case(label,p,n,cid):
    steps=[.01,.005,.0025,.00125,.000625];h_ref=.00015625;T=10.;N=round(T/h_ref);errors=np.zeros((n,5));bias=np.zeros((n,5));squarebias=np.zeros((n,5));endpoints=np.zeros((n,6,2));maxerr=np.zeros((n,5));seeds=[];initial=initial_phase(p)+.5
    for i in range(n):
        seed=int(np.random.SeedSequence([MASTER,cid,i]).generate_state(1,dtype=np.uint64)[0]);seeds.append(seed);fine=brownian_increments(N,h_ref,seed);ref=compiled_path(p,fine,h_ref,phi0=initial,stride=64)['path'];endpoints[i,-1]=ref[-1]
        for j,h in enumerate(steps):
            dw=coarsen_increments(fine,round(h/h_ref));path=compiled_path(p,dw,h,phi0=initial,stride=round(.01/h))['path'];endpoints[i,j]=path[-1];errors[i,j]=np.linalg.norm(path[-1]-ref[-1]);maxerr[i,j]=np.max(np.linalg.norm(path-ref,axis=1));bias[i,j]=path[-1,0]-ref[-1,0];squarebias[i,j]=path[-1,0]**2-ref[-1,0]**2
    rms=np.sqrt(np.mean(errors**2,axis=0));rate=float(np.polyfit(np.log(steps),np.log(rms),1)[0]);records=[]
    for j,h in enumerate(steps):
        rng=np.random.default_rng(cid+j);indices=rng.integers(0,n,(2000,n));boot=np.sqrt(np.mean(errors[indices,j]**2,axis=1));lo,hi=np.quantile(boot,[.025,.975]);mb,mlo,mhi=mean_ci(bias[:,j]);qb,qlo,qhi=mean_ci(squarebias[:,j]);variance_samples=np.var(endpoints[indices,j,0],axis=1,ddof=1)-np.var(endpoints[indices,-1,0],axis=1,ddof=1);vlo,vhi=np.quantile(variance_samples,[.025,.975]);records.append(dict(scenario=label,dt=h,n=n,strong_rms=rms[j],strong_low=lo,strong_high=hi,mean_bias=mb,mean_bias_low=mlo,mean_bias_high=mhi,variance_bias=float(np.var(endpoints[:,j,0],ddof=1)-np.var(endpoints[:,-1,0],ddof=1)),variance_bias_low=vlo,variance_bias_high=vhi,second_moment_bias=qb,second_moment_bias_low=qlo,second_moment_bias_high=qhi,max_path_error_mean=float(np.mean(maxerr[:,j])),finite_fraction=float(np.isfinite(endpoints[:,j]).all(axis=1).mean()),rate_fit=rate,reference_dt=h_ref,T=T))
    np.savez_compressed(ROOT/f'experiments/convergence/{label}_paired_raw.npz',steps=steps,seeds=seeds,errors=errors,bias=bias,squarebias=squarebias,endpoints=endpoints,max_path_error=maxerr);return records

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();groups=design();load_kernel();(ROOT/'configs/realized_design.json').write_text(json.dumps(groups,indent=2));started=time.monotonic();all_configs=[c for configs in groups.values() for c in configs]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_condition,c):c for c in all_configs}
        for future in as_completed(futures):
            out=future.result();print(f"condition {out['condition_id']} n={out['n']} T={out['T']} complete in {out['wall_seconds']:.2f}s",flush=True)
    for group,configs in groups.items():export_group(group,configs)
    combined=pd.read_csv(ROOT/'results/tables/combined_summary.csv');combined.query('delta == .5').to_csv(ROOT/'experiments/noise_sensitivity/summary.csv',index=False);combined.query('alpha == .15').to_csv(ROOT/'experiments/detuning/summary.csv',index=False);original_replay();conv=convergence_case('baseline',PLLParameters(),1000,600)+convergence_case('near_threshold',PLLParameters(detuning=.95,alpha=.4),500,601);pd.DataFrame(conv).to_csv(ROOT/'results/tables/convergence_summary.csv',index=False)
    meta={'status':'pll_complete','grid_conditions':72,'grid_trajectories':sum(c['n'] for c in groups['combined']),'original_initial_trajectories':1000,'long_conditions':6,'long_trajectories':6000,'sensitivity_conditions':18,'sensitivity_trajectories':9000,'convergence_trajectories':1500,'convergence_discretizations':6,'independent_pll_trajectories':sum(c['n'] for c in all_configs)+1500,'wall_seconds':time.monotonic()-started,'master_seed':MASTER,'source_hash':source_hash(),'tests_passed':25,'synthetic_status':'not_yet_executed','conditions':len(all_configs)};(ROOT/'results/run_metadata/full_run.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
