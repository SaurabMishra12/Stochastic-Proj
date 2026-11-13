"""Completed well-to-well transitions with a core and residence-time rule."""
import numpy as np
from scipy.integrate import solve_ivp
from .pll_model import drift,equilibria,eigenvalues

def detect_slips(phi,dt,stable,radius=np.pi/4,dwell=.5):
    if dt<=0 or dwell<=0 or not 0<radius<np.pi:
        raise ValueError('Positive dt/dwell and radius in (0,pi) required')
    occupied=int(np.floor((phi[0]-stable)/(2*np.pi)+.5)); candidate=occupied; age=0.
    events=[]
    for i,x in enumerate(phi[1:],1):
        k=int(np.floor((x-stable)/(2*np.pi)+.5))
        if k!=occupied and abs(x-stable-2*np.pi*k)<=radius:
            target=occupied+(1 if k>occupied else -1)
            if target!=k:
                age=0.;continue
            if candidate!=target:
                candidate=target;age=0.
            age+=dt
            if age+1e-12>=dwell:
                events.append((i*dt,target-occupied));occupied=target;candidate=occupied;age=0.
        else:
            candidate=occupied;age=0.
    return events

def saddle_manifold(p,duration=8.,epsilon=1e-6):
    eq=equilibria(p)
    if eq is None:
        raise ValueError('A saddle requires |delta|<1')
    u=eq[1]; A=np.array([[0,1],[-p.b*np.cos(u),-1/p.a]])
    vals,vecs=np.linalg.eig(A);direction=np.real(vecs[:,np.argmin(np.real(vals))])
    branches=[]
    for sign in [-1,1]:
        sol=solve_ivp(lambda t,x:drift(x,p),(0,-duration),np.array([u,0])+sign*epsilon*direction,rtol=1e-10,atol=1e-12,max_step=.01)
        keep=(abs(sol.y[0]-u)<2*np.pi)&(abs(sol.y[1])<5)
        branches.append(sol.y[:,keep].T)
    return branches
