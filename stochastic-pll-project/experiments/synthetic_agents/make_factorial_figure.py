"""Regenerate the all-cell synthetic matrix figure from executed summary rows."""
import csv
import itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project=Path(__file__).resolve().parents[2]
with (project/'results/tables/synthetic_summary.csv').open() as handle:
    rows=list(csv.DictReader(handle))
states=list(itertools.product((0.,.5,1.),(0.,.3),(0.,.2)))
labels=[f'rho {rho:g}\np {drop:g}, tau {delay:g}' for rho,drop,delay in states]
topologies=['complete','ring','star','sparse']
plt.rcParams.update({'svg.fonttype':'none','font.size':9})
fig,axes=plt.subplots(3,1,figsize=(14,8.5),constrained_layout=True)
for ax,n in zip(axes,(3,8,20)):
    values=np.empty((4,12))
    for i,topology in enumerate(topologies):
        for j,(rho,drop,delay) in enumerate(states):
            row=next(row for row in rows if int(row['n_agents'])==n and row['topology']==topology
                     and float(row['rho'])==rho and float(row['dropout'])==drop and float(row['delay'])==delay)
            values[i,j]=float(row['tail_R_mean_mean'])
    image=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    for (i,j),value in np.ndenumerate(values):
        ax.text(j,i,f'{value:.3f}',ha='center',va='center',fontsize=8,color='white' if value<.65 else '#17222c')
    ax.set_yticks(range(4),topologies);ax.set_ylabel(f'N={n}\nTopology')
    ax.set_xticks(range(12),labels,fontsize=8)
    ax.set_xlabel('Noise correlation rho; directed dropout probability p; relative-error delay tau (model units)')
    for x in (3.5,7.5):ax.axvline(x,color='white',linewidth=2)
fig.colorbar(image,ax=axes.tolist(),label='Mean tail R (unitless)',fraction=.025,pad=.02)
fig.suptitle('All 144 synthetic conditions: 500 independent network trajectories per cell; T=50, dt=.01',fontsize=13)
fig.savefig(project/'results/figures/synthetic_factorial.svg')
plt.close(fig)
