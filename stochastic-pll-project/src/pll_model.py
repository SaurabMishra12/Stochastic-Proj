"""State-space model and local stability for textbook equation (10.4)."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PLLParameters:
    a: float=1.
    b: float=1.
    detuning: float=.5
    alpha: float=.15
    def __post_init__(self):
        if not all(np.isfinite([self.a,self.b,self.detuning,self.alpha])) or self.a<=0 or self.b<=0 or self.alpha<0:
            raise ValueError('Finite a,b>0, alpha>=0 and detuning are required')
    @property
    def delta(self):
        return self.detuning/(self.a*self.b)

def drift(state, p):
    phi,v=np.asarray(state)
    return np.array([v,(p.detuning-v-p.a*p.b*np.sin(phi))/p.a])

def diffusion(p):
    return np.array([0.,-p.alpha/p.a])

def equilibria(p):
    if abs(p.delta)>=1:
        return None
    s=float(np.arcsin(p.delta))
    return s,float(np.pi-s)

def eigenvalues(phi,p):
    return np.linalg.eigvals([[0.,1.],[-p.b*np.cos(phi),-1/p.a]])

def potential(phi,p):
    return -p.detuning*np.asarray(phi)-p.a*p.b*np.cos(phi)

def barrier_height(p):
    if abs(p.delta)>=1:
        return 0.
    d=abs(p.delta)
    return float(2*p.a*p.b*(np.sqrt(1-d*d)-d*np.arccos(d)))

def initial_phase(p):
    e=equilibria(p)
    return 0. if e is None else e[0]
