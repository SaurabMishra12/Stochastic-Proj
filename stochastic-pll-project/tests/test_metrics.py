from pathlib import Path
import importlib
import numpy as np
import pytest

def modules():
    assert Path('src/metrics.py').exists(), 'Statistical metrics are required'
    return importlib.import_module('src.metrics'),importlib.import_module('src.phase_slips')

def test_wilson_nonzero_uncertainty_with_zero_events():
    m,_=modules();lo,hi=m.wilson_interval(0,1000)
    assert lo==pytest.approx(0,abs=1e-15) and .003<hi<.0045

def test_mean_ci_uses_independent_rows():
    m,_=modules();mean,lo,hi=m.mean_ci(np.array([1.,2.,3.,4.]))
    assert mean==2.5 and lo<mean<hi

def test_bootstrap_and_censoring():
    m,_=modules();ci=m.bootstrap_ci(np.array([1.,2.,3.]),np.mean,seed=8,n_boot=1000)
    assert ci[0]<=2<=ci[1]
    km=m.kaplan_meier(np.array([1.,3.,3.]),np.array([1,0,0]),3.)
    assert km['restricted_mean']==pytest.approx(7/3)
    assert km['median'] is None

def test_completed_slip_not_temporary_crossing():
    _,s=modules();dt=.01;stable=0.;phi=np.zeros(800)
    phi[100:120]=np.pi+.1
    assert len(s.detect_slips(phi,dt,stable))==0
    phi[200:]=2*np.pi
    ev=s.detect_slips(phi,dt,stable)
    assert len(ev)==1 and ev[0][1]==1

def test_negative_completed_slip():
    _,s=modules();phi=np.r_[np.zeros(100),np.full(100,-2*np.pi)]
    assert s.detect_slips(phi,.01,0.)[0][1]==-1

def test_detector_core_and_persistence_sensitivity():
    _,s=modules();phi=np.r_[np.zeros(50),np.full(30,2*np.pi),np.zeros(50)]
    assert len(s.detect_slips(phi,.01,0.,dwell=.5))==0
    assert len(s.detect_slips(phi,.01,0.,dwell=.25))==2
