"""Synthetic delayed Kuramoto networks; this is separate from the PLL model.

The delay applies to the entire relative phase error. Communication uses the
last relative-state observation, with constant initial history. Brownian noise
is equicorrelated, and Bernoulli messages are renewed each numerical step.
"""
from __future__ import annotations
import argparse
import csv
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class SyntheticConfig:
    n_agents: int = 3
    topology: str = 'complete'
    rho: float = 0.
    dropout: float = 0.
    delay: float = 0.
    n_trajectories: int = 500
    T: float = 50.
    dt: float = .01
    coupling: float = 2.
    sigma: float = .35
    sample_dt: float = .1
    master_seed: int = 20261113
    threads: int = 4

    def validate(self):
        if self.n_agents < 2 or self.n_trajectories < 1:
            raise ValueError('At least two nodes and one independent trajectory are required.')
        if self.topology not in ('complete', 'ring', 'star', 'sparse'):
            raise ValueError('Unknown topology.')
        if not 0 <= self.rho <= 1 or not 0 <= self.dropout <= 1:
            raise ValueError('Correlation and dropout must lie in [0,1].')
        if self.T <= 0 or self.dt <= 0 or self.delay < 0 or self.sigma < 0 or self.threads < 1:
            raise ValueError('Invalid time, noise, or thread parameter.')
        for value in (self.T, self.delay, self.sample_dt):
            if abs(value/self.dt-round(value/self.dt)) > 1e-8:
                raise ValueError('Horizon, delay, and sampling interval must be multiples of dt.')
        if self.sample_dt < self.dt:
            raise ValueError('Sampling cannot be faster than integration.')
        return self


def topology_matrix(n: int, name: str, seed: int = 20261113) -> np.ndarray:
    """Return a fixed undirected graph, with row i identifying senders to i."""
    if n < 2:
        raise ValueError('At least two nodes are required.')
    a = np.zeros((n, n), dtype=np.float64)
    if name == 'complete':
        a[:] = 1; np.fill_diagonal(a, 0)
    elif name == 'ring':
        for i in range(n):
            a[i, (i+1) % n] = a[(i+1) % n, i] = 1
    elif name == 'star':
        a[0, 1:] = a[1:, 0] = 1
    elif name == 'sparse':
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, 710]))
        order = rng.permutation(n)
        for i in range(1, n):
            parent = int(rng.integers(i))
            a[order[i], order[parent]] = a[order[parent], order[i]] = 1
        # A connected random tree plus fixed 0.05-density extra undirected edges.
        for i in range(n):
            for j in range(i+1, n):
                if not a[i, j] and rng.random() < .05:
                    a[i, j] = a[j, i] = 1
    else:
        raise ValueError('Unknown topology.')
    return a


def noise_covariance(n: int, rho: float) -> np.ndarray:
    if not 0 <= rho <= 1:
        raise ValueError('rho must lie in [0,1].')
    return (1-rho)*np.eye(n) + rho*np.ones((n, n))


def coupled_drift(relative_phases, frequencies, adjacency, coupling, message_mask=None):
    """Evaluate original-degree-normalized coupling at one error-observation time."""
    phase = np.asarray(relative_phases, dtype=float)
    adjacency = np.asarray(adjacency, dtype=float)
    mask = 1 if message_mask is None else np.asarray(message_mask)
    degree = adjacency.sum(axis=1)
    if np.any(degree == 0):
        raise ValueError('Zero-degree nodes are outside this protocol.')
    error = phase[None, :] - phase[:, None]
    return np.asarray(frequencies) + coupling*np.sum(adjacency*mask*np.sin(error), axis=1)/degree


def delayed_index(step: int, delay_steps: int) -> int:
    return max(0, step-delay_steps)


def _stream_seeds(c):
    # Cell labels are deliberately absent: comparisons reuse identical network inputs.
    seeds = np.empty((c.n_trajectories, 3), dtype=np.uint64)
    for r in range(c.n_trajectories):
        for stream in range(3):
            seeds[r, stream] = np.random.SeedSequence(
                [c.master_seed, 820, c.n_agents, r, stream]).generate_state(1, dtype=np.uint64)[0]
    return seeds


_KERNEL = r'''
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <omp.h>
static inline uint64_t mix64(uint64_t z) {
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return z^(z>>31);
}
extern "C" void simulate(int n,int nr,int steps,int delay,int stride,int nsave,
 double dt,double K,double sigma,double rho,double dropout,int threads,
 const double* adjacency,const double* omega,const double* initial,
 const uint64_t* seeds,double* metrics,double* orders,double* samples) {
    int ns=steps/stride+1, histlen=delay+1;
    double sigind=sigma*std::sqrt((1-rho)*dt), sigcom=sigma*std::sqrt(rho*dt);
    int tailstart=(4*steps)/5;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int r=0;r<nr;r++) {
        std::mt19937_64 rng(seeds[r*3+1]); std::normal_distribution<double> normal(0.,1.);
        std::vector<double> x(n),history(histlen*n),drift(n),sins(n),coss(n),z(n),tailx(n),degree(n);
        double global=0.;
        for(int i=0;i<n;i++) global+=initial[r*n+i]/n;
        for(int i=0;i<n;i++) {
            x[i]=initial[r*n+i]-global;
            for(int h=0;h<histlen;h++) history[h*n+i]=x[i];
            degree[i]=0.;for(int j=0;j<n;j++) degree[i]+=adjacency[i*n+j];
        }
        double sumR=0.,sumR2=0.;int ntail=0;
        double finalR=0.;
        for(int s=0;s<=steps;s++) {
            if(s>=tailstart || s%stride==0) {
                double real=0.,imag=0.;
                for(int i=0;i<n;i++){real+=std::cos(x[i]);imag+=std::sin(x[i]);}
                finalR=std::sqrt(real*real+imag*imag)/n;
                if(s>=tailstart){sumR+=finalR;sumR2+=finalR*finalR;ntail++;}
                if(s%stride==0){
                    int k=s/stride; orders[r*ns+k]=finalR;
                    if(r<nsave) for(int i=0;i<n;i++) samples[(r*ns+k)*n+i]=x[i]+global;
                }
            }
            if(s==tailstart) tailx=x;
            if(s==steps)break;
            int oldslot=(s>=delay ? (s-delay)%histlen : 0);
            for(int j=0;j<n;j++){
                sins[j]=std::sin(history[oldslot*n+j]);
                coss[j]=std::cos(history[oldslot*n+j]);
            }
            double driftmean=0.;
            for(int i=0;i<n;i++) {
                double coupling=0.;
                for(int j=0;j<n;j++) if(adjacency[i*n+j]) {
                    bool received=true;
                    if(dropout>0.) {
                        uint64_t key=seeds[r*3+2]^((uint64_t)(s+1)*UINT64_C(0x9e3779b97f4a7c15))
                                     ^((uint64_t)(i*n+j+1)*UINT64_C(0xd1b54a32d192ed03));
                        double u=(mix64(key)>>11)*0x1.0p-53;
                        received=(u>=dropout);
                    }
                    if(received) coupling+=sins[j]*coss[i]-coss[j]*sins[i];
                }
                drift[i]=omega[i]+K*coupling/degree[i]; driftmean+=drift[i]/n;
            }
            double zmean=0.;
            for(int i=0;i<n;i++){z[i]=normal(rng);zmean+=z[i]/n;}
            double shared=normal(rng);
            global+=driftmean*dt+sigind*zmean+sigcom*shared;
            for(int i=0;i<n;i++) x[i]+=(drift[i]-driftmean)*dt+sigind*(z[i]-zmean);
            int slot=(s+1)%histlen;
            for(int i=0;i<n;i++)history[slot*n+i]=x[i];
        }
        double fm=0.,f2=0.,tailduration=(steps-tailstart)*dt;
        for(int i=0;i<n;i++){
            double f=(x[i]-tailx[i])/tailduration;fm+=f/n;f2+=f*f/n;
        }
        double meanR=sumR/ntail,meanR2=sumR2/ntail;
        metrics[r*8]=meanR;
        metrics[r*8+1]=std::max(0.,meanR2-meanR*meanR);
        metrics[r*8+2]=1.-meanR;
        metrics[r*8+3]=(1.-meanR2)*n/(n-1.);
        metrics[r*8+4]=std::sqrt(std::max(0.,f2-fm*fm));
        metrics[r*8+5]=finalR;metrics[r*8+6]=global;metrics[r*8+7]=ntail;
    }
}
'''


def _kernel(output_root):
    directory = Path(output_root); directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_KERNEL.encode()).hexdigest()
    cpp = directory/'synthetic_kernel.cpp'
    library = directory/f'synthetic_kernel_{digest[:16]}.so'
    if not library.exists():
        cpp.write_text(_KERNEL)
        proc = subprocess.run(['g++', '-O3', '-std=c++17', '-fPIC', '-shared',
                               '-fopenmp', str(cpp), '-o', str(library)],
                              capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError('Synthetic kernel build failed: '+proc.stderr)
    loaded = ctypes.CDLL(str(library.resolve()))
    function = loaded.simulate
    function.argtypes = [ctypes.c_int]*6+[ctypes.c_double]*5+[ctypes.c_int]+[ctypes.c_void_p]*7
    function.restype = None
    return function, digest


_METRICS = ('tail_R_mean', 'tail_R_time_variance', 'tail_circular_variance',
            'tail_pairwise_disagreement', 'tail_frequency_dispersion', 'terminal_R',
            'terminal_mean_phase', 'tail_samples')


def simulate_networks(config: SyntheticConfig, output_root, initial_phases=None, frequencies=None):
    """Return independent network rows, sampled order parameters and five paths.

    C++ evolves the phase mean and its centered deviations separately. This is
    algebraically identical to Euler--Maruyama, while shared noise cancels from
    relative phases exactly rather than by floating-point subtraction.
    """
    c = config.validate(); n=c.n_agents; nr=c.n_trajectories
    seeds = _stream_seeds(c)
    if initial_phases is None:
        initial_phases = np.stack([np.random.default_rng(int(seed)).uniform(-np.pi, np.pi, n)
                                   for seed in seeds[:, 0]])
    initial = np.ascontiguousarray(initial_phases, dtype=np.float64)
    if initial.shape != (nr, n):
        raise ValueError('Initial phases must have shape (networks, agents).')
    if frequencies is None:
        frequencies = np.random.default_rng(np.random.SeedSequence(
            [c.master_seed, n, 711])).permutation(np.linspace(-.5, .5, n))
    frequencies = np.ascontiguousarray(frequencies, dtype=np.float64)
    if frequencies.shape != (n,) or not np.isfinite(initial).all() or not np.isfinite(frequencies).all():
        raise ValueError('Initial phases and frequencies must be finite and correctly shaped.')
    adjacency = topology_matrix(n, c.topology, c.master_seed)
    steps=round(c.T/c.dt); stride=round(c.sample_dt/c.dt); ns=steps//stride+1
    nsave=min(5, nr)
    metrics=np.empty((nr, 8), dtype=np.float64)
    order=np.empty((nr, ns), dtype=np.float64)
    samples=np.empty((nsave, ns, n), dtype=np.float64)
    function,digest = _kernel(output_root)
    arguments = [adjacency, frequencies, initial, seeds, metrics, order, samples]
    function(n,nr,steps,round(c.delay/c.dt),stride,nsave,
             c.dt,c.coupling,c.sigma,c.rho,c.dropout,c.threads,
             *[a.ctypes.data for a in arguments])
    dtype=[('replicate', 'i4'), ('initial_seed', 'u8'), ('brownian_seed', 'u8'),
           ('dropout_seed', 'u8')]+[(name, 'f8') for name in _METRICS]
    rows=np.empty(nr, dtype=dtype);rows['replicate']=np.arange(nr)
    for i,name in enumerate(('initial_seed','brownian_seed','dropout_seed')):rows[name]=seeds[:, i]
    for i,name in enumerate(_METRICS):rows[name]=metrics[:, i]
    return dict(rows=rows, order_parameter=order, phase_samples=samples,
                time=np.arange(ns)*c.sample_dt, adjacency=adjacency,
                frequencies=frequencies, initial_phases=initial, kernel_sha256=digest)


def summarize_networks(rows, config):
    """Student t intervals use one aggregate value per independent network."""
    result=asdict(config);result['n']=len(rows)
    for name in _METRICS[:-2]:
        values=np.asarray(rows[name]);n=len(values)
        mean=float(np.mean(values));sd=float(np.std(values,ddof=1)) if n>1 else 0.
        half=float(student_t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else float('nan')
        result.update({name+'_mean':mean,name+'_sd':sd,name+'_median':float(np.median(values)),
                       name+'_ci_low':mean-half,name+'_ci_high':mean+half})
    return result


def full_matrix():
    for n,topology,rho,dropout,delay in itertools.product(
            (3,8,20),('complete','ring','star','sparse'),(0.,.5,1.),(0.,.3),(0.,.2)):
        yield SyntheticConfig(n_agents=n,topology=topology,rho=rho,dropout=dropout,delay=delay)


def _cell_name(c):
    return f'N{c.n_agents}_{c.topology}_rho{c.rho:g}_drop{c.dropout:g}_delay{c.delay:g}'


def _write_rows(path, rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(rows.dtype.names)
        writer.writerows(rows.tolist())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_dicts(path, rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def _figures(project, summaries, cached):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False})
    figure_dir=project/'results/figures';figure_dir.mkdir(parents=True,exist_ok=True)
    colors=['#145b8f','#c16b22','#2d8064']
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
    topologies=['complete','ring','star','sparse']
    for ax,n in zip(axes,(3,8,20)):
        for k,rho in enumerate((0.,.5,1.)):
            cells=[next(s for s in summaries if s['n_agents']==n and s['topology']==topology
                        and s['rho']==rho and s['dropout']==0 and s['delay']==0) for topology in topologies]
            y=np.array([s['tail_R_mean_mean'] for s in cells]);lo=np.array([s['tail_R_mean_ci_low'] for s in cells]);hi=np.array([s['tail_R_mean_ci_high'] for s in cells])
            ax.errorbar(np.arange(4)+(k-1)*.13,y,yerr=[y-lo,hi-y],fmt='o-',capsize=3,color=colors[k],label=f'noise rho={rho:g}')
        ax.set_xticks(range(4),topologies,rotation=25);ax.set_title(f'{n} agents');ax.set_ylim(0,1.04)
        ax.set_xlabel('Communication topology')
    axes[0].set_ylabel('Tail mean order parameter R (unitless)');axes[-1].legend(fontsize=8)
    fig.suptitle('Synthetic networks: synchronization without delay or dropout; 95% network CIs')
    fig.tight_layout();fig.savefig(figure_dir/'synthetic_synchronization.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
    states=[(0.,0.),(.3,0.),(0.,.2),(.3,.2)]
    for ax,n in zip(axes,(3,8,20)):
        for k,rho in enumerate((0.,.5,1.)):
            vals=[];errs=[]
            for dropout,delay in states:
                s=next(s for s in summaries if s['n_agents']==n and s['topology']=='sparse'
                       and s['rho']==rho and s['dropout']==dropout and s['delay']==delay)
                vals.append(s['tail_pairwise_disagreement_mean']);errs.append((s['tail_pairwise_disagreement_ci_high']-s['tail_pairwise_disagreement_ci_low'])/2)
            ax.errorbar(np.arange(4)+(k-1)*.1,vals,yerr=errs,fmt='o-',capsize=3,color=colors[k],label=f'rho={rho:g}')
        ax.set_title(f'{n} agents, sparse');ax.set_xticks(range(4),['neither','dropout','delay','both'],rotation=25)
        ax.set_xlabel('Communication impairment')
    axes[0].set_ylabel('Tail pairwise mean (1-cos phase error)');axes[-1].legend(fontsize=8)
    fig.suptitle('Synthetic networks: disagreement with delayed relative-state messages')
    fig.tight_layout();fig.savefig(figure_dir/'synthetic_impairments.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
    for ax,n in zip(axes,(3,8,20)):
        for k,rho in enumerate((0.,.5,1.)):
            c=SyntheticConfig(n_agents=n,topology='ring',rho=rho,dropout=.3,delay=.2)
            data=np.load(cached[_cell_name(c)]/'sampled_paths.npz')
            r=data['order_parameter'];mean=r.mean(0);half=student_t.ppf(.975,r.shape[0]-1)*r.std(0,ddof=1)/np.sqrt(r.shape[0])
            ax.plot(data['time'],mean,color=colors[k],label=f'rho={rho:g}')
            ax.fill_between(data['time'],mean-half,mean+half,color=colors[k],alpha=.15)
        ax.set_title(f'{n} agents, ring');ax.set_xlabel('Time (model units)');ax.set_ylim(0,1.04)
    axes[0].set_ylabel('Ensemble mean R with pointwise 95% mean CI');axes[-1].legend(fontsize=8)
    fig.suptitle('Synchronization history: dropout=0.3, relative-error delay=0.2')
    fig.tight_layout();fig.savefig(figure_dir/'synthetic_timeseries.svg');plt.close(fig)


def _paired_effects(project, cached):
    effects=[]
    for n,topology,rho in itertools.product((3,8,20),('complete','ring','star','sparse'),(0.,.5,1.)):
        base=SyntheticConfig(n_agents=n,topology=topology,rho=rho)
        left=np.load(cached[_cell_name(base)]/'trajectory_rows.npy')
        for dropout,delay in ((.3,0.),(0.,.2),(.3,.2)):
            config=SyntheticConfig(n_agents=n,topology=topology,rho=rho,dropout=dropout,delay=delay)
            right=np.load(cached[_cell_name(config)]/'trajectory_rows.npy')
            for metric in ('tail_R_mean','tail_pairwise_disagreement','tail_frequency_dispersion'):
                difference=right[metric]-left[metric];sd=difference.std(ddof=1)
                half=student_t.ppf(.975,len(difference)-1)*sd/np.sqrt(len(difference))
                effects.append(dict(n_agents=n,topology=topology,rho=rho,dropout=dropout,delay=delay,
                                    contrast='impairment minus no impairment',metric=metric,n=len(difference),
                                    paired_difference_mean=difference.mean(),ci_low=difference.mean()-half,
                                    ci_high=difference.mean()+half))
    _write_dicts(project/'experiments/synthetic_agents/paired_effects.csv',effects)


def run_full(project):
    project=Path(project);root=project/'experiments/synthetic_agents';root.mkdir(parents=True,exist_ok=True)
    meta_path=project/'results/run_metadata/synthetic_run.json';meta_path.parent.mkdir(parents=True,exist_ok=True)
    code_sha=_sha(Path(__file__));started=time.perf_counter();summaries=[];cached={};manifest=[]
    matrix=list(full_matrix())
    state=dict(complete=False,started_utc=datetime.now(timezone.utc).isoformat(),
               requested_cells=len(matrix),requested_networks=sum(c.n_trajectories for c in matrix),
               completed_cells=0,completed_networks=0,source_sha256=code_sha,
               model='delayed-relative-error additive-noise Kuramoto',schema_version=1,
               compiler=subprocess.run(['g++','--version'],capture_output=True,text=True).stdout.splitlines()[0],
               compile_flags=['-O3','-std=c++17','-fPIC','-shared','-fopenmp'],
               numerical_method='Euler-Maruyama, centered relative state plus global mean',
               brownian_rng='std::mt19937_64 with std::normal_distribution per trajectory',
               preflight_tests_file='experiments/synthetic_agents/preflight_tests.txt')
    meta_path.write_text(json.dumps(state,indent=2))
    for index,c in enumerate(matrix):
        directory=root/_cell_name(c);directory.mkdir(exist_ok=True);cached[_cell_name(c)]=directory
        config=asdict(c);config_sha=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        checkpoint=directory/'checkpoint.json';valid=False
        if checkpoint.exists():
            prior=json.loads(checkpoint.read_text())
            valid=prior.get('complete') and prior.get('source_sha256')==code_sha and prior.get('config_sha256')==config_sha
            if valid:
                valid=all((directory/name).exists() and _sha(directory/name)==digest for name,digest in prior['artifacts'].items())
        if valid:
            rows=np.load(directory/'trajectory_rows.npy')
        else:
            before=time.perf_counter();data=simulate_networks(c,root);rows=data['rows']
            if not all(np.isfinite(rows[name]).all() for name in _METRICS):raise RuntimeError('Nonfinite network metrics.')
            np.save(directory/'trajectory_rows.npy',rows);_write_rows(directory/'trajectory_rows.csv',rows)
            np.savez_compressed(directory/'sampled_paths.npz',time=data['time'],order_parameter=data['order_parameter'],
                                phase_samples=data['phase_samples'],initial_phases=data['initial_phases'])
            (directory/'configuration.json').write_text(json.dumps(dict(config,adjacency=data['adjacency'].tolist(),
                frequencies=data['frequencies'].tolist(),normalization='original row degree; no dropout renormalization',
                delay_definition='theta_j(t-tau) minus theta_i(t-tau); constant prehistory',
                brownian_generator='std::mt19937_64 + std::normal_distribution, seeds recorded per replicate',
                dropout_generator='counter splitmix64 keyed by seed, timestep and directed node pair',
                kernel_sha256=data['kernel_sha256']),indent=2))
            artifacts={name:_sha(directory/name) for name in ['trajectory_rows.npy','trajectory_rows.csv','sampled_paths.npz','configuration.json']}
            prior=dict(complete=True,source_sha256=code_sha,config_sha256=config_sha,artifacts=artifacts,
                       wall_seconds=time.perf_counter()-before,n=len(rows))
            checkpoint.write_text(json.dumps(prior,indent=2))
        summaries.append(summarize_networks(rows,c));manifest.append(dict(cell=_cell_name(c),**prior))
        state.update(completed_cells=index+1,completed_networks=sum(s['n'] for s in summaries),wall_seconds=time.perf_counter()-started)
        meta_path.write_text(json.dumps(state,indent=2))
        print(f"synthetic {index+1}/{len(matrix)} {_cell_name(c)} networks={len(rows)} elapsed={state['wall_seconds']:.1f}s",flush=True)
    table=project/'results/tables/synthetic_summary.csv';table.parent.mkdir(parents=True,exist_ok=True)
    _write_dicts(table,summaries);_paired_effects(project,cached);_figures(project,summaries,cached)
    state.update(complete=True,finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started,
                 rows_per_cell=500,source_sha256=code_sha,cells=manifest,
                 total_euler_steps=sum(round(c.T/c.dt)*c.n_trajectories for c in matrix),
                 unique_matched_network_inputs=1500,summary_sha256=_sha(table),
                 uncertainty_unit='independent network trajectory; timesteps are never replicates',
                 caveats=['N=3 ring equals complete; not independent topology evidence',
                          'rho=1 additive shared noise cancels from all relative-state dynamics',
                          'dropout is a stepwise Bernoulli communication model tied to dt=.01',
                          'delay applies to a stale relative-state error, not sender-only phase',
                          'frequency dispersion uses tail net phase velocity, not instantaneous white-noise derivative',
                          'paired 95% intervals are descriptive and not multiplicity-adjusted significance tests'])
    meta_path.write_text(json.dumps(state,indent=2));return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    arguments=parser.parse_args()
    if arguments.full:run_full(arguments.project)
    elif arguments.benchmark:
        c=SyntheticConfig(n_agents=20,topology='complete',rho=.5,dropout=.3,delay=.2,
                          n_trajectories=100,T=5.)
        before=time.perf_counter();simulate_networks(c,Path(arguments.project)/'experiments/synthetic_agents')
        print(json.dumps(dict(config=asdict(c),wall_seconds=time.perf_counter()-before),indent=2))
    else:parser.error('Choose --full or --benchmark.')


if __name__=='__main__':main()
