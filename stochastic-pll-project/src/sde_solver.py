"""Euler-Maruyama, deterministic reference and validated compiled path execution."""
from pathlib import Path
import ctypes
import subprocess
import numpy as np
from scipy.integrate import solve_ivp
from .pll_model import drift,initial_phase

_LIB=None
SUMMARY_COLUMNS=['phi_final','v_final','tail_velocity','tail_phase_variance','tail_absolute_circular_error','slips_forward','slips_backward','first_slip','loose_slips','strict_slips','captured','retained_lock','tail_rotation_rate']

def brownian_increments(n,dt,seed):
    if n<=0 or not np.isfinite(dt) or dt<=0:
        raise ValueError('n and dt must be positive')
    return np.random.default_rng(seed).normal(0.,np.sqrt(dt),n)

def coarsen_increments(dw,factor):
    if factor<1 or len(dw)%factor:
        raise ValueError('Factor must divide the path length')
    return np.asarray(dw).reshape(-1,factor).sum(axis=-1)

def euler_maruyama(p,dw,dt,phi0=None,v0=0.):
    if not np.isfinite(dt) or dt<=0:
        raise ValueError('dt must be positive finite')
    x=np.empty((len(dw)+1,2));x[0]=[initial_phase(p) if phi0 is None else phi0,v0]
    for i,w in enumerate(dw):
        phi,v=x[i]
        x[i+1,0]=phi+v*dt
        x[i+1,1]=v+(p.detuning-v-p.a*p.b*np.sin(phi))/p.a*dt-p.alpha/p.a*w
    return x

def milstein(p,dw,dt,phi0=None,v0=0.):
    # The diffusion vector is constant, so its Jacobian and Milstein correction vanish.
    return euler_maruyama(p,dw,dt,phi0,v0)

def deterministic(p,t,phi0=None,v0=0.):
    t=np.asarray(t,float)
    sol=solve_ivp(lambda time,x:drift(x,p),(0,float(t[-1])),[initial_phase(p) if phi0 is None else phi0,v0],t_eval=t,method='DOP853',rtol=1e-11,atol=1e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T

def load_kernel():
    global _LIB
    if _LIB is None:
        source=Path(__file__).with_name('kernel.cpp');binary=source.with_suffix('.so')
        if not binary.exists() or binary.stat().st_mtime<source.stat().st_mtime:
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(binary)],check=True)
        _LIB=ctypes.CDLL(str(binary.resolve()))
        dptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        iptr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        _LIB.pll_path.argtypes=[dptr,ctypes.c_int]+[ctypes.c_double]*7+[ctypes.c_int,dptr,dptr,dptr,iptr,ctypes.POINTER(ctypes.c_int)]
        _LIB.pll_path.restype=None
    return _LIB

def compiled_path(p,dw,dt,phi0=None,v0=0.,stride=50):
    if not np.isfinite(dt) or dt<=0 or len(dw)%stride or len(dw)<5:
        raise ValueError('Positive dt and at least five steps divisible by stride required')
    dw=np.ascontiguousarray(dw,dtype=np.float64)
    path=np.empty((len(dw)//stride+1,2));summary=np.empty(13)
    times=np.empty(len(dw));directions=np.empty(len(dw),dtype=np.int32);count=ctypes.c_int()
    load_kernel().pll_path(dw,len(dw),dt,p.a,p.b,p.detuning,p.alpha,initial_phase(p) if phi0 is None else phi0,v0,stride,path.ravel(),summary,times,directions,ctypes.byref(count))
    if not np.isfinite(path).all():
        raise FloatingPointError('Nonfinite numerical path')
    return {'path':path,'summary':summary,'event_times':times[:count.value].copy(),'event_directions':directions[:count.value].copy()}
