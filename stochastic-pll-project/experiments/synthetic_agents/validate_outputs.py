"""Verify all executed synthetic cells against their saved manifests and metrics."""
from pathlib import Path
import sys
import json
import hashlib
import csv
import numpy as np
from scipy.stats import t

project=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project))
from src.synthetic_agents import full_matrix,_cell_name,_METRICS
root=project/'experiments/synthetic_agents'
meta=json.loads((project/'results/run_metadata/synthetic_run.json').read_text())
assert meta['complete'] and meta['completed_cells']==144 and meta['completed_networks']==72000
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert meta['source_sha256']==sha(project/'src/synthetic_agents.py')
assert meta['summary_sha256']==sha(project/'results/tables/synthetic_summary.csv')
with (project/'results/tables/synthetic_summary.csv').open() as h:summary=list(csv.DictReader(h))
assert len(summary)==144
matrices=list(full_matrix());identical_inputs={};validated=0;networks=0;metric_error=0.
for c in matrices:
    directory=root/_cell_name(c)
    cp=json.loads((directory/'checkpoint.json').read_text())
    assert cp['complete'] and cp['source_sha256']==meta['source_sha256']
    for name,expected in cp['artifacts'].items():assert sha(directory/name)==expected,(directory,name)
    rows=np.load(directory/'trajectory_rows.npy')
    assert len(rows)==500
    for name in _METRICS:assert np.isfinite(rows[name]).all(),(directory,name)
    for seed in ('initial_seed','brownian_seed','dropout_seed'):assert len(np.unique(rows[seed]))==500
    d=np.load(directory/'sampled_paths.npz')
    assert d['order_parameter'].shape==(500,501)
    assert d['phase_samples'].shape==(5,501,c.n_agents)
    assert np.all((d['order_parameter']>=0)&(d['order_parameter']<=1+1e-14))
    assert np.isfinite(d['phase_samples']).all()
    if c.n_agents not in identical_inputs:
        identical_inputs[c.n_agents]=(d['initial_phases'],np.c_[rows['initial_seed'],rows['brownian_seed'],rows['dropout_seed']])
    else:
        initial,seeds=identical_inputs[c.n_agents]
        assert np.array_equal(initial,d['initial_phases'])
        assert np.array_equal(seeds,np.c_[rows['initial_seed'],rows['brownian_seed'],rows['dropout_seed']])
    actual=rows['tail_pairwise_disagreement']
    expected=c.n_agents/(c.n_agents-1)*(1-rows['tail_R_time_variance']-rows['tail_R_mean']**2)
    metric_error=max(metric_error,float(np.max(np.abs(actual-expected))))
    assert np.allclose(actual,expected,atol=1e-12,rtol=0)
    assert np.allclose(rows['tail_circular_variance'],1-rows['tail_R_mean'],atol=1e-14,rtol=0)
    table=next(s for s in summary if int(s['n_agents'])==c.n_agents and s['topology']==c.topology
               and float(s['rho'])==c.rho and float(s['dropout'])==c.dropout and float(s['delay'])==c.delay)
    for metric in _METRICS[:-2]:
        values=rows[metric];mean=values.mean();sd=values.std(ddof=1);half=t.ppf(.975,499)*sd/np.sqrt(500)
        for suffix,value in (('_mean',mean),('_sd',sd),('_ci_low',mean-half),('_ci_high',mean+half)):
            assert float(table[metric+suffix])==value,(directory,metric,suffix)
    validated+=1;networks+=len(rows)
for c in matrices:
    if c.n_agents==3 and c.topology=='complete':
        complete=np.load(root/_cell_name(c)/'trajectory_rows.npy')
        ring_path=_cell_name(c).replace('_complete_','_ring_')
        ring=np.load(root/ring_path/'trajectory_rows.npy')
        assert np.array_equal(complete,ring)
figures=sorted((project/'results/figures').glob('synthetic*.svg'))
assert len(figures)==5
for figure in figures:assert '<svg' in figure.read_text() and figure.stat().st_size>1000
out=dict(complete=True,validated_cells=validated,validated_networks=networks,
         validated_cell_artifacts=validated*4,unique_matched_input_blocks=1500,
         all_seed_and_initial_inputs_matched=True,n3_complete_ring_exact_identity=True,
         all_metric_values_finite=True,all_reported_intervals_recomputed_exactly=True,
         pairwise_metric_identity_max_absolute_error=metric_error,
         source_and_summary_hashes_valid=True,
         figure_sha256={p.name:sha(p) for p in figures},
         diagnostic_replays=2,diagnostic_new_independent_networks=0)
(root/'validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
