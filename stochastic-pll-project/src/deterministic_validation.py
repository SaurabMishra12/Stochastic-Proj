"""Quantify zero-noise Euler error against DOP853 on a common T=50 grid."""
from pathlib import Path
import numpy as np
import pandas as pd
from .pll_model import PLLParameters,initial_phase
from .sde_solver import compiled_path,deterministic
ROOT=Path(__file__).resolve().parents[1]

def main():
    rows=[];times=np.linspace(0,50,5001)
    for delta in [.5,.95]:
        p=PLLParameters(detuning=delta,alpha=0);phi0=initial_phase(p)+.5;ref=deterministic(p,times,phi0=phi0)
        for h in [.01,.005,.0025,.00125,.000625]:
            out=compiled_path(p,np.zeros(round(50/h)),h,phi0=phi0,stride=round(.01/h))['path'];err=np.linalg.norm(out-ref,axis=1);rows.append(dict(delta=delta,dt=h,T=50,phi0=phi0,max_state_error=float(err.max()),time_of_max_error=float(times[np.argmax(err)]),endpoint_state_error=float(err[-1]),reference='SciPy DOP853 rtol1e-11 atol1e-13'))
    table=pd.DataFrame(rows);table.to_csv(ROOT/'results/tables/deterministic_validation.csv',index=False);print(table.to_string(index=False))
if __name__=='__main__':main()
