"""Confidence intervals treat trajectories as independent sampling units."""
import numpy as np
from scipy.stats import t,norm

def mean_ci(x,confidence=.95):
    x=np.asarray(x,float);n=len(x);mean=float(np.mean(x))
    if n<2:return mean,float('nan'),float('nan')
    margin=float(t.ppf((1+confidence)/2,n-1)*np.std(x,ddof=1)/np.sqrt(n))
    return mean,mean-margin,mean+margin

def wilson_interval(successes,n,confidence=.95):
    if n<=0:raise ValueError('Positive n required')
    z=norm.ppf((1+confidence)/2);p=successes/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(max(0,mid-half)),float(min(1,mid+half))

def bootstrap_ci(x,statistic=np.mean,seed=0,n_boot=2000):
    x=np.asarray(x);rng=np.random.default_rng(seed)
    values=np.empty(n_boot)
    for start in range(0,n_boot,100):
        size=min(100,n_boot-start);batch=x[rng.integers(0,len(x),(size,len(x)))];values[start:start+size]=statistic(batch,axis=1)
    return tuple(float(v) for v in np.quantile(values,[.025,.975]))

def kaplan_meier(times,observed,horizon):
    times=np.asarray(times);observed=np.asarray(observed,bool)
    event_times=np.unique(times[observed]);survival=1.;last=0.;area=0.;median=None;out_t=[0.];out_s=[1.]
    for time in event_times:
        area+=survival*(time-last);risk=np.sum(times>=time);events=np.sum((times==time)&observed)
        survival*=1-events/risk;out_t.append(float(time));out_s.append(float(survival))
        if median is None and survival<=.5:median=float(time)
        last=time
    area+=survival*(horizon-last);out_t.append(float(horizon));out_s.append(float(survival))
    return {'time':out_t,'survival':out_s,'restricted_mean':float(area),'median':median,'events':int(observed.sum()),'censored':int((~observed).sum())}
