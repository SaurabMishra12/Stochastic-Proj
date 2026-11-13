import importlib
from pathlib import Path
import numpy as np
import pytest

def modules():
    assert Path('src/sde_solver.py').exists(), 'Numerical solver is required'
    return importlib.import_module('src.sde_solver'),importlib.import_module('src.pll_model')

def test_reproducible_and_finite():
    s,m=modules();p=m.PLLParameters()
    dw=s.brownian_increments(2000,.001,42)
    x=s.euler_maruyama(p,dw,.001)
    np.testing.assert_array_equal(x,s.euler_maruyama(p,s.brownian_increments(2000,.001,42),.001))
    assert np.isfinite(x).all()

def test_brownian_increment_variance():
    s,m=modules();dw=s.brownian_increments(200000,.002,123)
    assert abs(dw.mean())<.0004
    assert abs(dw.var()/.002-1)<.015

def test_milstein_equals_euler_for_additive_noise():
    s,m=modules();p=m.PLLParameters();dw=s.brownian_increments(1000,.001,81)
    np.testing.assert_array_equal(s.euler_maruyama(p,dw,.001),s.milstein(p,dw,.001))

def test_zero_noise_ode_and_first_order_refinement():
    s,m=modules();p=m.PLLParameters(alpha=0); errors=[]
    for h in [.01,.005,.0025]:
        n=round(4/h);x=s.euler_maruyama(p,np.zeros(n),h,phi0=1.)
        ref=s.deterministic(p,np.array([4.]),phi0=1.)
        errors.append(np.linalg.norm(x[-1]-ref[-1]))
    assert errors[-1]<.002
    assert errors[0]/errors[1]>1.8 and errors[1]/errors[2]>1.8

def test_compiled_kernel_matches_python_and_samples():
    s,m=modules();p=m.PLLParameters(alpha=.3,detuning=.95)
    dw=s.brownian_increments(5000,.001,718)
    x=s.euler_maruyama(p,dw,.001,phi0=1.4)
    out=s.compiled_path(p,dw,.001,phi0=1.4,stride=10)
    np.testing.assert_allclose(out['path'],x[::10],atol=2e-13,rtol=2e-13)

def test_coupled_increments_preserve_path():
    s,m=modules();dw=s.brownian_increments(1600,.000625,38)
    c=s.coarsen_increments(dw,16)
    assert abs(dw.sum()-c.sum())<1e-14

@pytest.mark.parametrize('dt',[0,-.01,float('nan')])
def test_invalid_step(dt):
    s,m=modules()
    with pytest.raises(ValueError):s.euler_maruyama(m.PLLParameters(),np.zeros(4),dt)
