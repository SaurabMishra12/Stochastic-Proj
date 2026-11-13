"""Replay two executed rho=1 ring rows to diagnose phase configurations."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project))
from src.synthetic_agents import SyntheticConfig,simulate_networks
root=project/'experiments/synthetic_agents'
cell=root/'N8_ring_rho1_drop0_delay0'
rows=np.load(cell/'trajectory_rows.npy')
paths=np.load(cell/'sampled_paths.npz')
config=json.loads((cell/'configuration.json').read_text())
low=int(np.flatnonzero(rows['tail_R_mean']<.5)[0])
high=int(np.flatnonzero(rows['tail_R_mean']>.9)[0])
records=[]
plt.rcParams.update({'svg.fonttype':'none','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,replicate in zip(axes,(high,low)):
    # At rho=1, changing sigma alters only the common angle, so sigma=0 is an
    # exact replay of the original relative dynamics, not a new replication.
    result=simulate_networks(SyntheticConfig(n_agents=8,topology='ring',rho=1.,sigma=0.,n_trajectories=1),
                             root,initial_phases=paths['initial_phases'][[replicate]],
                             frequencies=np.asarray(config['frequencies']))
    phase=result['phase_samples'][0,-1]
    edge_error=np.angle(np.exp(1j*(np.roll(phase,-1)-phase)))
    cumulative=np.r_[0,np.cumsum(edge_error)]
    winding=float(cumulative[-1]/(2*np.pi))
    error=float(abs(result['rows']['tail_R_mean'][0]-rows['tail_R_mean'][replicate]))
    if error>1e-12:raise RuntimeError('Diagnostic replay failed relative-state identity.')
    ax.plot(np.arange(9),cumulative,'o-',color='#145b8f')
    ax.axhline(0,color='#999999',linewidth=.8)
    ax.set_title(f'Original replicate {replicate}: tail R={rows["tail_R_mean"][replicate]:.4f}, winding={winding:.0f}')
    ax.set_xlabel('Successive node along the ring (8 closes the loop)')
    ax.set_ylabel('Accumulated wrapped neighbor phase gap (radians)')
    ax.set_xticks(range(9))
    records.append(dict(original_replicate=replicate,tail_R=float(rows['tail_R_mean'][replicate]),
                        frequency_dispersion=float(rows['tail_frequency_dispersion'][replicate]),
                        terminal_winding=winding,relative_metric_replay_absolute_error=error,
                        original_initial_phases=paths['initial_phases'][replicate].tolist(),
                        replay_final_phases=phase.tolist(),edge_phase_gaps=edge_error.tolist()))
fig.suptitle('Low global coherence can coexist with a frequency-locked twisted ring state\nN=8, rho=1, no dropout or delay; selected deterministic relative-state replays')
fig.tight_layout();fig.savefig(project/'results/figures/synthetic_ring_states.svg');plt.close(fig)
(root/'ring_state_diagnostic.json').write_text(json.dumps(dict(
    source_cell=str(cell.relative_to(project)),selection='first executed row with tail R>.9, first with tail R<.5',
    independent_additional_networks=0,diagnostic_replays=2,
    note='rho=1 common noise cancels; sigma=0 replay recovers executed relative dynamics. Two selected states do not classify all500 rows.',
    records=records),indent=2))
print(json.dumps(records,indent=2))
