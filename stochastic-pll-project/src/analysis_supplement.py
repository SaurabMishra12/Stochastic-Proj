"""Derived descriptive tables and independent-cohort effect sizes from raw rows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .metrics import bootstrap_ci
ROOT=Path(__file__).resolve().parents[1]

def enrich_tables():
    for family in ['combined','long_horizon','sensitivity','original_initial']:
        path=ROOT/f'results/tables/{family}_summary.csv';table=pd.read_csv(path)
        for i,r in table.iterrows():
            raw=pd.read_csv(ROOT/f'results/raw_data/condition_{int(r.condition_id)}/trajectories.csv')
            for col,prefix in [('phi_final','phi'),('v_final','velocity'),('tail_velocity','tail_velocity')]:
                table.loc[i,prefix+'_median']=float(raw[col].median());table.loc[i,prefix+'_std']=float(raw[col].std(ddof=1)) if len(raw)>1 else 0.
            table.loc[i,'mean_absolute_phase_deviation_std']=float(raw.tail_absolute_circular_error.std(ddof=1)) if abs(r.delta)<1 and len(raw)>1 else np.nan
        table.to_csv(path,index=False)

def effect_sizes():
    pairs=[(119,120,'baseline_noise_0.15_minus_0.10'),(120,123,'baseline_noise_0.40_minus_0.15'),(120,152,'detuning_0.99_minus_0.50_at_noise_0.15'),(144,152,'detuning_0.99_minus_0.95_at_noise_0.15')]
    results=[]
    for left,right,name in pairs:
        x=pd.read_csv(ROOT/f'results/raw_data/condition_{left}/trajectories.csv');y=pd.read_csv(ROOT/f'results/raw_data/condition_{right}/trajectories.csv');rng=np.random.default_rng(20261113+left+right)
        for col in ['tail_velocity','captured','tail_absolute_circular_error']:
            boot=np.empty(2000)
            for start in range(0,2000,100):
                ix=rng.integers(0,len(x),(100,len(x)));iy=rng.integers(0,len(y),(100,len(y)));boot[start:start+100]=np.mean(y[col].to_numpy()[iy],axis=1)-np.mean(x[col].to_numpy()[ix],axis=1)
            lo,hi=np.quantile(boot,[.025,.975]);pooled=np.sqrt(((len(x)-1)*x[col].var(ddof=1)+(len(y)-1)*y[col].var(ddof=1))/(len(x)+len(y)-2));diff=y[col].mean()-x[col].mean()
            results.append(dict(comparison=name,left_condition=left,right_condition=right,metric=col,n_left=len(x),n_right=len(y),difference=diff,ci_low=lo,ci_high=hi,standardized_difference=diff/pooled if pooled>0 else np.nan,interpretation='Independent-cohort descriptive comparison; no familywise significance declaration'))
    pd.DataFrame(results).to_csv(ROOT/'results/tables/effect_sizes.csv',index=False)

if __name__=='__main__':enrich_tables();effect_sizes()
