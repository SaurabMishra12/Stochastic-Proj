"""Condition-level inference uses independent trajectory rows."""
import numpy as np
from .metrics import mean_ci,wilson_interval,bootstrap_ci,kaplan_meier

def summarize(rows,c):
    out={k:c[k] for k in ['condition_id','a','b','detuning','alpha','T','dt','n']};out['delta']=c['detuning']/(c['a']*c['b']);n=len(rows);stochastic=c['alpha']>0
    for source,target in [('phi_final','phi'),('v_final','velocity'),('tail_velocity','tail_velocity')]:
        out[target+'_mean'],out[target+'_mean_low'],out[target+'_mean_high']=mean_ci(rows[source])
    values=np.asarray(rows['phi_final']);out['phi_variance']=float(np.var(values,ddof=1)) if n>1 else 0.
    out['phi_variance_low'],out['phi_variance_high']=bootstrap_ci(values,np.var,seed=c['condition_id']) if n>1 else (float('nan'),)*2
    out['tail_phase_variance_mean']=float(np.mean(rows['tail_phase_variance']));out['mean_absolute_phase_deviation']=float(np.mean(rows['tail_absolute_circular_error'])) if abs(out['delta'])<1 else float('nan');out['rotation_rate']=float(np.mean(rows['tail_rotation_rate']))
    if abs(out['delta'])<1:
        for source,target in [('captured','locking_probability'),('retained_lock','retained_lock_probability')]:
            out[target]=float(np.mean(rows[source]));out[target+'_low'],out[target+'_high']=wilson_interval(int(np.sum(rows[source])),n) if stochastic else (float('nan'),)*2
        counts=np.asarray(rows['slips_forward']+rows['slips_backward']);out['slips_total']=int(np.sum(counts));out['slip_rate']=float(np.mean(counts)/c['T'])
        out['slip_rate_low'],out['slip_rate_high']=np.asarray(bootstrap_ci(counts,np.mean,seed=c['condition_id']+8))/c['T'] if stochastic else (float('nan'),)*2
        out['slip_probability']=float(np.mean(counts>0));out['slip_probability_low'],out['slip_probability_high']=wilson_interval(int(np.sum(counts>0)),n) if stochastic else (float('nan'),)*2
        out['zero_event_rate_upper_poisson']=float(-np.log(.05)/(n*c['T'])) if stochastic and not np.any(counts) else float('nan')
        out['loose_slip_rate']=float(np.mean(rows['loose_slips'])/c['T']);out['strict_slip_rate']=float(np.mean(rows['strict_slips'])/c['T'])
        observed=np.asarray(rows['first_slip'])>=0;times=np.where(observed,rows['first_slip'],c['T']);km=kaplan_meier(times,observed,c['T']);out['rmst_first_slip']=km['restricted_mean'];out['median_first_slip']=km['median'];out['first_slip_events']=km['events'];out['first_slip_censored']=km['censored'];out['conditional_first_slip_mean']=float(np.mean(times[observed])) if np.any(observed) else float('nan');out['survival']=km
    else:
        for key in ['locking_probability','locking_probability_low','locking_probability_high','retained_lock_probability','retained_lock_probability_low','retained_lock_probability_high','slips_total','slip_rate','slip_rate_low','slip_rate_high','slip_probability','slip_probability_low','slip_probability_high','zero_event_rate_upper_poisson','loose_slip_rate','strict_slip_rate','rmst_first_slip','median_first_slip','first_slip_events','first_slip_censored','conditional_first_slip_mean']:out[key]=float('nan')
    return out
