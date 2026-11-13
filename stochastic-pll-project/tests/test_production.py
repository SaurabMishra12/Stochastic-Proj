import numpy as np
from src.pll_model import PLLParameters
from src.sde_solver import compiled_path,brownian_increments
from src.phase_slips import detect_slips
from src.statistical_analysis import summarize
import pandas as pd

def test_compiled_commitment_events_match_reference():
    p=PLLParameters(detuning=.99,alpha=.4);dw=brownian_increments(50000,.001,761)
    out=compiled_path(p,dw,.001,stride=1);events=detect_slips(out['path'][:,0],.001,np.arcsin(.99))
    assert len(events)>0
    np.testing.assert_allclose([x[0] for x in events],out['event_times'],atol=1e-10)
    np.testing.assert_array_equal([x[1] for x in events],out['event_directions'])

def test_unlocked_production_metrics_are_not_slips():
    p=PLLParameters(detuning=1.05);out=compiled_path(p,np.zeros(10000),.001,stride=10)
    assert np.isnan(out['summary'][5:7]).all()
    assert np.isnan(out['summary'][10:12]).all()
    assert len(out['event_times'])==0

def test_summarized_variance_and_censoring_use_all_rows():
    frame=pd.DataFrame({'phi_final':[1.,2.,3.],'v_final':[0.,0.,0.],'tail_velocity':[0.,0.,0.],'tail_phase_variance':[0.,0.,0.],'tail_absolute_circular_error':[0.,0.,0.],'tail_rotation_rate':[0.,0.,0.],'captured':[1.,1.,1.],'retained_lock':[1.,1.,1.],'slips_forward':[0.,0.,0.],'slips_backward':[0.,0.,0.],'first_slip':[-1.,-1.,-1.],'loose_slips':[0.,0.,0.],'strict_slips':[0.,0.,0.]})
    c=dict(condition_id=8,a=1.,b=1.,detuning=.5,alpha=.15,T=50.,dt=.001,n=3)
    out=summarize(frame,c)
    assert out['phi_variance']==1.
    assert out['rmst_first_slip']==50.
    assert out['median_first_slip'] is None
    assert out['slip_probability_high']>0
    assert out['zero_event_rate_upper_poisson']>0
