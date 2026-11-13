"""Generate every scientific chart from saved raw results, without new inference."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .pll_model import PLLParameters,equilibria,drift,potential,barrier_height
from .sde_solver import deterministic,brownian_increments
from .phase_slips import saddle_manifold
ROOT=Path(__file__).resolve().parents[1];FIG=ROOT/'results/figures';CAP={}
COLORS=['#146b65','#b25332','#273c58','#666666']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.prop_cycle':plt.cycler(color=COLORS),'svg.fonttype':'none','figure.dpi':120,'axes.titlepad':10,'savefig.bbox':'tight'})

def save(fig,name,title,explanation,interpretation,limitations):
    fig.savefig(FIG/f'{name}.svg');fig.savefig(ROOT/'report/figures'/f'{name}.pdf');plt.close(fig);CAP[f'{name}.svg']={'title':title,'explanation':explanation,'interpretation':interpretation,'limitations':limitations}

def condition(cid):return np.load(ROOT/f'results/raw_data/condition_{cid}/ensemble.npz')

def baseline():
    data=condition(200);t=data['t'];mean=data['mean'];rep=data['representatives'];p=PLLParameters();s=equilibria(p)[0];det=deterministic(p,t,phi0=s+.5)
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for path in rep[:5]:ax[0,0].plot(t,path[:,0],alpha=.4,lw=.7)
    ax[0,0].plot(t,det[:,0],color='black',lw=1.4,label='Deterministic acquisition');ax[0,0].plot(t,mean[:,0],color=COLORS[0],label='1000-path mean');ax[0,0].axhline(s,ls=':',color='gray',label='Stable equilibrium');ax[0,0].legend(fontsize=8);ax[0,0].set(title='Baseline acquisition and representative paths',xlabel='Time (model units)',ylabel='Unwrapped phase (rad)')
    ax[0,1].plot(t,mean[:,0],label='Mean phase');ax[0,1].fill_between(t,data['ci_low'][:,0],data['ci_high'][:,0],alpha=.3,label='Pointwise 95% mean CI');ax[0,1].plot(t,det[:,0],ls='--',label='Deterministic');ax[0,1].legend(fontsize=8);ax[0,1].set(title='Sampling uncertainty of the ensemble mean',xlabel='Time (model units)',ylabel='Phase mean (rad)')
    ax[1,0].plot(t,data['variance'][:,0],label='Empirical phase variance');ax[1,0].axhline(p.alpha**2/(2*p.a*p.b*np.cos(s)),ls='--',color=COLORS[1],label='Local linear stationary approximation');ax[1,0].legend(fontsize=8);ax[1,0].set(title='Trajectory variability',xlabel='Time (model units)',ylabel='Phase variance (rad²)')
    for path in rep[:5]:ax[1,1].plot(path[:,0],path[:,1],lw=.7,alpha=.45)
    ax[1,1].plot(det[:,0],det[:,1],color='black',label='Deterministic');ax[1,1].scatter([s],[0],label='Stable equilibrium',s=35);ax[1,1].legend(fontsize=8);ax[1,1].set(title='Phase portrait during acquisition',xlabel='Phase (rad)',ylabel='Velocity (rad / model time)')
    save(fig,'baseline','E1: 1000-path baseline acquisition','a=b=1, Delta=.5, alpha=.15, T=50, dt=.001; original initial phase asin(.5)+.5. Representative paths are first five recorded streams.','The ensemble mean describes acquisition; the CI quantifies uncertainty in that mean. Individual paths fluctuate more widely.','Pointwise CIs are not simultaneous bands and do not enclose 95% of trajectories. The stationary formula is local and assumes no slips.')
    series={'t':t.tolist(),'deterministic_phi':det[:,0].tolist(),'stochastic_phi':rep[0,:,0].tolist(),'mean_phi':mean[:,0].tolist(),'variance_phi':data['variance'][:,0].tolist(),'mean_ci_low':data['ci_low'][:,0].tolist(),'mean_ci_high':data['ci_high'][:,0].tolist()};(ROOT/'results/tables/baseline_series.json').write_text(json.dumps(series))

def sensitivities(grid,long):
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained');d=grid[grid.delta==.5]
    ax[0,0].errorbar(d.alpha,d.phi_variance,yerr=[d.phi_variance-d.phi_variance_low,d.phi_variance_high-d.phi_variance],fmt='o-',label='Observed terminal variance');ax[0,0].plot(d.alpha,d.alpha**2/(2*np.cos(np.arcsin(.5))),ls='--',label='Local linear approximation');ax[0,0].set(title='E2: terminal variability at delta=.5',xlabel='Noise coefficient alpha',ylabel='Phase variance (rad²)');ax[0,0].legend(fontsize=8)
    ax[0,1].plot(d.alpha,d.mean_absolute_phase_deviation,'o-',label='Tail mean absolute circular error');ax[0,1].set(title='E2: mean phase deviation at delta=.5',xlabel='Noise coefficient alpha',ylabel='Absolute circular deviation (rad)');ax[0,1].legend(fontsize=8)
    d=grid[grid.alpha==.15];ax[1,0].errorbar(d.delta,d.tail_velocity_mean,yerr=[d.tail_velocity_mean-d.tail_velocity_mean_low,d.tail_velocity_mean_high-d.tail_velocity_mean],fmt='o-',label='Tail mean velocity');ax[1,0].axvline(1,ls=':',color='black',label='Equilibrium threshold');ax[1,0].set(title='E3: phase velocity across detuning',xlabel='Normalized detuning delta',ylabel='Mean velocity (rad / model time)');ax[1,0].legend(fontsize=8)
    locked=d[d.delta<1];ax[1,1].errorbar(locked.delta,locked.locking_probability,yerr=[locked.locking_probability-locked.locking_probability_low,locked.locking_probability_high-locked.locking_probability],fmt='o-',label='Terminal capture probability');ax[1,1].plot(locked.delta,locked.retained_lock_probability,'s--',label='No-slip retained capture');ax[1,1].set(title='E3: operational capture below threshold',xlabel='Normalized detuning delta',ylabel='Probability (unitless)',ylim=(-.05,1.05));ax[1,1].legend(fontsize=8)
    save(fig,'noise_detuning','E2-E3: noise and detuning sensitivity','Plots are slices of the full grid, not additional independent simulations. Error bars use trajectory bootstrap variance intervals or t/Wilson intervals as appropriate.','Increasing noise broadens the phase distribution. Detuning approaches the local saddle-node and changes capture and transport.','Terminal capture is an operational finite-time criterion. Above-threshold cells have no locking or phase-slip probability.')
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for a,col,title,label in zip(ax.ravel(),['tail_velocity_mean','phi_variance','retained_lock_probability','slip_rate'],['Mean tail velocity','Terminal unwrapped variance','Retained capture probability','Committed slip frequency'],['rad / model time','rad²','Probability','slips / model time']):
        table=grid.pivot(index='delta',columns='alpha',values=col);im=a.imshow(table.to_numpy(),origin='lower',aspect='auto',cmap='viridis');a.set_xticks(range(len(table.columns)),[f'{x:g}' for x in table.columns]);a.set_yticks(range(len(table.index)),[f'{x:g}' for x in table.index]);a.set(xlabel='Noise coefficient alpha',ylabel='Normalized detuning delta',title=title);fig.colorbar(im,ax=a,label=label)
    save(fig,'combined_grid','E4: complete 72-condition grid','All eight noise values and nine detunings were executed. Missing locking/slip cells above threshold are deliberately masked.','Variance, phase transport and committed transitions describe distinct outcomes; no single metric defines global lock.','T=50 is finite. Deterministic alpha=0 controls have one path; stochastic cells have 500 or 1000 independent paths.')
    fig,ax=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for _,r in long.iterrows():
        raw=json.loads((ROOT/f"results/raw_data/condition_{int(r.condition_id)}/summary.json").read_text());km=raw['survival'];ax[0].step(km['time'],km['survival'],where='post',label=f"delta={r.delta:g}, alpha={r.alpha:g}")
    ax[0].set(title='E6: time to first committed slip',xlabel='Time (model units)',ylabel='Estimated no-slip survival');ax[0].legend(fontsize=7)
    x=np.arange(len(long));ax[1].errorbar(x,long.slip_rate,yerr=[long.slip_rate-long.slip_rate_low,long.slip_rate_high-long.slip_rate],fmt='o',label='Primary detector, bootstrap 95% CI');ax[1].plot(x,long.loose_slip_rate,'s',label='Broad core, .25 residence');ax[1].plot(x,long.strict_slip_rate,'^',label='Narrow core, 1.0 residence');ax[1].set_xticks(x,[f'{r.delta:g}\n{r.alpha:g}' for _,r in long.iterrows()]);ax[1].set(title='E6: detection-rule sensitivity at T=200',xlabel='Condition: delta (top), alpha (bottom)',ylabel='Slips / model time');ax[1].legend(fontsize=7)
    save(fig,'slip_survival','E6: censoring and detector sensitivity','Six 1000-path ensembles run to T=200. Kaplan-Meier curves retain all right-censored trajectories; no event is represented by censoring, not a fictitious zero waiting time.','Longer runs resolve selected near-threshold escapes. Differences between detectors quantify sensitivity to commitment definitions.','Zero observed events do not establish a zero true rate. A bootstrap interval is degenerate at zero when every path has zero events; use the separately reported probability upper bounds.')

def slip_and_geometry():
    cid=155;p=PLLParameters(detuning=.99,alpha=.4);s,u=equilibria(p);d=condition(cid);events=pd.read_csv(ROOT/f'results/raw_data/condition_{cid}/events.csv');rep=d['representatives'];ids=events.trajectory_id.unique();i=int(next((i for i in ids if i<len(rep)),0));path=rep[i]
    fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');ax[0].plot(d['t'],path[:,0],label=f'Representative trajectory {i}')
    for k in range(int(np.floor(path[:,0].min()/(2*np.pi)))-1,int(np.ceil(path[:,0].max()/(2*np.pi)))+1):ax[0].axhline(s+2*np.pi*k,color='gray',ls=':',lw=.6)
    ev=events[events.trajectory_id==i]
    for _,row in ev.iterrows():ax[0].axvline(row.time,color=COLORS[1],alpha=.55,lw=.7)
    ax[0].plot([],[],color=COLORS[1],label='Commitment times');ax[0].plot([],[],color='gray',ls=':',label='Stable-well centers');ax[0].set(title='E6: completed transitions in an actual path',xlabel='Time (model units)',ylabel='Unwrapped phase (rad)');ax[0].legend(fontsize=7)
    q=PLLParameters();s0,u0=equilibria(q);x=np.linspace(-1,5,26);y=np.linspace(-2,2,20);X,Y=np.meshgrid(x,y);U=Y;V=q.detuning-Y-np.sin(X);scale=np.sqrt(U**2+V**2)+1e-10;ax[1].quiver(X,Y,U/scale,V/scale,alpha=.35,color='gray')
    for j,branch in enumerate(saddle_manifold(q,duration=14.)):ax[1].plot(branch[:,0],branch[:,1],color=COLORS[1],label='Saddle stable manifold (local continuation)' if j==0 else None)
    ax[1].scatter([s0,u0],[0,0],c=[COLORS[0],COLORS[1]],s=45);ax[1].annotate('stable',(s0,0),xytext=(s0,-.45));ax[1].annotate('saddle',(u0,0),xytext=(u0,.35));ax[1].set(xlim=(-1,5),ylim=(-2,2),title='Deterministic basin geometry: delta=.5',xlabel='Phase (rad)',ylabel='Velocity (rad / model time)');ax[1].legend(fontsize=7)
    save(fig,'slip_geometry','E6: well transitions and saddle geometry','Left: recorded condition delta=.99, alpha=.4 with primary committed-slip events. Right: backward integration along the stable eigenvector of the saddle at delta=.5, epsilon=1e-6.','The true deterministic basin separator is two-dimensional; a phase-only boundary cannot encode velocity-dependent capture.','The manifold is a local numerical approximation. The detector uses persistent well cores, not an exact stochastic separatrix. Commitment times lag barrier crossings.')
    fig,ax=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for delta in [.5,.95,1.05]:
        q=PLLParameters(detuning=delta);x=np.linspace(-1,7,500);ax[0].plot(x,potential(x,q),label=f'delta={delta:g}')
    ax[0].set(title='Tilted phase potential',xlabel='Phase (rad)',ylabel='Potential (model energy units)');ax[0].legend()
    ds=np.linspace(0,.9999,300);ax[1].plot(ds,[barrier_height(PLLParameters(detuning=float(v))) for v in ds],label='Downhill barrier');ax[1].set(title='Barrier collapses at the local threshold',xlabel='Normalized detuning delta',ylabel='Barrier height (model energy units)');ax[1].legend()
    save(fig,'potential_barrier','Mathematical mechanism: shrinking escape barrier','Derived from U(phi)=-Delta*phi-ab*cos(phi), a=b=1. This is an analytical diagram, not an estimated escape-rate curve.','The downhill barrier tends to zero as delta approaches one, providing a mechanistic explanation for increased escape susceptibility.','Barrier height alone does not give a finite-noise inertial escape rate, global capture range, or proof of asymptotic stochastic stability.')

def convergence_and_parameters(conv,sens):
    fig,ax=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for label,d in conv.groupby('scenario'):
        ax[0].errorbar(d.dt,d.strong_rms,yerr=[d.strong_rms-d.strong_low,d.strong_high-d.strong_rms],fmt='o-',label=f'{label}, slope {d.rate_fit.iloc[0]:.2f}');ax[1].errorbar(d.dt,d.mean_bias,yerr=[d.mean_bias-d.mean_bias_low,d.mean_bias_high-d.mean_bias],fmt='o-',label=label)
    ax[0].set(xscale='log',yscale='log',title='E5: coupled endpoint strong error',xlabel='Step size dt (model time)',ylabel='RMS state error (model norm)');ax[0].legend(fontsize=8);ax[1].set(xscale='log',title='E5: paired weak phase difference',xlabel='Step size dt (model time)',ylabel='Mean phase difference (rad)');ax[1].axhline(0,color='gray',ls=':');ax[1].legend(fontsize=8)
    save(fig,'convergence','E5: coupled numerical refinement','1000 baseline and 500 near-threshold Brownian paths, T=10. Reference dt=.00015625; all coarser increments are sums of the same fine increments. Strong intervals bootstrap paired path errors.','Observed fitted slopes describe finite-step error relative to this reference. Paired weak intervals expose sampling uncertainty in mean differences.','The reference is numerical; slopes are empirical and partly affected by reference error. No improved Milstein method is claimed. T=10 convergence does not prove T=200 rare-event count accuracy.')
    fig,ax=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for a,alpha in zip(ax,[.15,.3]):
        d=sens[sens.alpha==alpha];table=d.pivot(index='a',columns='b',values='tail_velocity_mean');im=a.imshow(table.to_numpy(),origin='lower',aspect='auto',cmap='viridis');a.set_xticks(range(3),[.5,1,2]);a.set_yticks(range(3),[.5,1,2]);a.set(title=f'E7: mean velocity, alpha={alpha}',xlabel='Parameter b',ylabel='Parameter a');fig.colorbar(im,ax=a,label='rad / model time')
        for (i,j),val in np.ndenumerate(table.to_numpy()):a.text(j,i,f'{val:.3f}',ha='center',va='center',color='white' if val<table.to_numpy().max()*.55 else 'black')
    save(fig,'parameter_sensitivity','E7: varying a, b and noise at fixed Delta=.5','18 conditions, 500 independent paths per condition, T=50, dt=.001. The raw table also reports variance, capture, events and first-passage outcomes.','Changing a and b changes normalized detuning, damping and noise response simultaneously; combinations below the equilibrium threshold exhibit phase transport.','This fixed-Delta design measures joint physical changes, not a standardized one-at-a-time causal derivative. Finite rotation is not numerical divergence.')

def brownian():
    dt=.001;dw=brownian_increments(2000,dt,20261113);t=np.arange(2001)*dt;fig,ax=plt.subplots(1,2,figsize=(12,3.7),layout='constrained');ax[0].plot(t,np.r_[0,np.cumsum(dw)],label='One generated Brownian path');ax[0].set(title='Numerical Brownian motion',xlabel='Time (model units)',ylabel='W(t) (sqrt(model time))');ax[0].legend(fontsize=8);ax[1].hist(dw,bins=45,density=True,label='2000 increments',alpha=.65);x=np.linspace(-.12,.12,300);ax[1].plot(x,np.exp(-x*x/(2*dt))/np.sqrt(2*np.pi*dt),label='Normal(0,dt) density');ax[1].set(title='Why sqrt(dt) scales the noise',xlabel='Increment dW (sqrt(model time))',ylabel='Probability density');ax[1].legend(fontsize=8)
    save(fig,'brownian','Brownian increments and variance scaling','A reproducible generated example (seed 20261113), separate from the experiment cohorts. Independent Gaussian increments have variance dt.','Multiplying standard-normal samples by sqrt(dt) produces the correct Wiener increment scale.','One path is an illustration; the 200,000-increment automated preflight supplies the variance check.')

def main():
    FIG.mkdir(parents=True,exist_ok=True);(ROOT/'report/figures').mkdir(exist_ok=True)
    grid=pd.read_csv(ROOT/'results/tables/combined_summary.csv');long=pd.read_csv(ROOT/'results/tables/long_horizon_summary.csv');conv=pd.read_csv(ROOT/'results/tables/convergence_summary.csv');sens=pd.read_csv(ROOT/'results/tables/sensitivity_summary.csv')
    baseline();sensitivities(grid,long);slip_and_geometry();convergence_and_parameters(conv,sens);brownian()
    existing=FIG/'captions.json'
    old=json.loads(existing.read_text()) if existing.exists() else {};old.update(CAP);existing.write_text(json.dumps(old,indent=2));print(f'Generated {len(CAP)} data-linked scientific figures')
if __name__=='__main__':main()
