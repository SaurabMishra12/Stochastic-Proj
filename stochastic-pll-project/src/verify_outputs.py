"""Fail-closed consistency audit: raw rows, saved summaries and original hashes."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from .experiment_runner import design,source_hash
ROOT=Path(__file__).resolve().parents[1]

def verify():
    checks={};errors=[];all_seeds=[];n_total=0
    original=json.loads((ROOT/'results/run_metadata/original_manifest.json').read_text())
    for name,info in original.items():
        checks['original_'+name]=hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()==info['sha256']
        checks['backup_'+name]=hashlib.sha256((ROOT/'backups'/name).read_bytes()).hexdigest()==info['sha256']
    integrity=json.loads((ROOT/'results/run_metadata/pll_raw_integrity.json').read_text())
    for rel,info in integrity.items():
        path=ROOT/rel
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:errors.append('Raw integrity mismatch '+rel)
    for family,configs in design().items():
        summary=pd.read_csv(ROOT/f'results/tables/{family}_summary.csv');checks[family+'_condition_count']=len(summary)==len(configs)
        for c in configs:
            folder=ROOT/f"results/raw_data/condition_{c['condition_id']}";raw=pd.read_csv(folder/'trajectories.csv',dtype={'seed':'uint64'});events=pd.read_csv(folder/'events.csv');row=summary.loc[summary.condition_id==c['condition_id']].iloc[0];meta=json.loads((folder/'summary.json').read_text());n_total+=len(raw);all_seeds.extend(raw.seed.tolist())
            if len(raw)!=c['n']:errors.append(f"Wrong n {c['condition_id']}")
            for metric,actual in [('phi_mean',raw.phi_final.mean()),('phi_variance',raw.phi_final.var(ddof=1) if len(raw)>1 else 0.),('tail_velocity_mean',raw.tail_velocity.mean())]:
                if not np.isclose(row[metric],actual,rtol=2e-11,atol=1e-12):errors.append(f"Mismatch {c['condition_id']} {metric}")
            if abs(row.delta)<1:
                if int(row.slips_total)!=len(events) or int(row.slips_total)!=int((raw.slips_forward+raw.slips_backward).sum()):errors.append(f"Event denominator mismatch {c['condition_id']}")
                if int(row.first_slip_events)!=int((raw.first_slip>=0).sum()):errors.append(f"Censor mismatch {c['condition_id']}")
            elif not np.isnan(row.slip_rate) or not np.isnan(row.locking_probability):errors.append('Unlocked slip/lock assigned')
            if meta['source_hash']!=source_hash():errors.append(f"Source changed since experiment {c['condition_id']}")
            samples=np.load(folder/'ensemble.npz')
            if not np.isfinite(samples['mean']).all() or samples['mean'].shape!=(1001,2):errors.append('Nonfinite/misaligned samples')
    checks['all_trajectory_seeds_unique']=len(all_seeds)==len(set(all_seeds));checks['pll_raw_paths_excluding_convergence']=n_total==55009
    conv=pd.read_csv(ROOT/'results/tables/convergence_summary.csv');checks['convergence_grid']=len(conv)==10 and set(conv.scenario)=={'baseline','near_threshold'};checks['convergence_finite']=bool((conv.finite_fraction==1).all());checks['convergence_decreases']=all(np.all(np.diff(g.sort_values('dt').strong_rms)>0) for _,g in conv.groupby('scenario'))
    for k,v in checks.items():
        if not v:errors.append(k)
    result={'passed':not errors,'checks':checks,'errors':errors,'raw_files_hashed':len(integrity),'raw_trajectory_rows':n_total,'independent_coupled_convergence_paths':1500}
    (ROOT/'results/run_metadata/output_quality.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if errors:raise AssertionError(errors)
    return result
if __name__=='__main__':verify()
