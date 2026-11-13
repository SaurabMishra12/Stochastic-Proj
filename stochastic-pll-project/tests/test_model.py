from pathlib import Path
import importlib
import numpy as np
import pytest

def load():
    assert Path('src/pll_model.py').exists(), 'PLL model implementation is required'
    return importlib.import_module('src.pll_model')

def test_equilibria_and_classification():
    m=load(); p=m.PLLParameters()
    s,u=m.equilibria(p)
    assert abs(m.drift(np.array([s,0.]),p)[1])<1e-14
    assert np.max(np.real(m.eigenvalues(s,p)))<0
    assert np.max(np.real(m.eigenvalues(u,p)))>0

@pytest.mark.parametrize('delta',[1.,1.01,-1.,-1.1])
def test_no_asymptotically_stable_equilibrium_at_threshold(delta):
    m=load(); assert m.equilibria(m.PLLParameters(detuning=delta)) is None

@pytest.mark.parametrize('kwargs',[{'a':0},{'b':-1},{'alpha':-1},{'detuning':float('nan')}])
def test_invalid_parameters(kwargs):
    m=load()
    with pytest.raises(ValueError):m.PLLParameters(**kwargs)

def test_negative_detuning_symmetry():
    m=load(); s,u=m.equilibria(m.PLLParameters(detuning=-.5))
    assert np.isclose(s,-np.pi/6)
    assert m.barrier_height(m.PLLParameters(detuning=-.5))==pytest.approx(m.barrier_height(m.PLLParameters(detuning=.5)))
