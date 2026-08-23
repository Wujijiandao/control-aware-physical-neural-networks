#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/e060_confirmatory/e060_confirmatory_summary.csv'
OUT=ROOT.parent/'submission/figures'
df=pd.read_csv(SRC)
conds=['nominal','moderate','strong']; labels=['Nominal','Moderate','Strong']
methods=['noise_aware_mse','control_aware_rank']; names=['Noise-aware MSE','Control-aware rank']
fig,axs=plt.subplots(1,2,figsize=(10.5,4.1))
x=np.arange(3);offsets=[-.16,.16]
for method,name,off in zip(methods,names,offsets):
 g=df[df.method==method].set_index('condition').loc[conds]
 axs[0].errorbar(x+off,g.stabilization_loss,yerr=g.se_stabilization_loss,marker='o',capsize=3,label=name)
 axs[1].errorbar(x+off,g.survival_steps,yerr=g.se_survival_steps,marker='o',capsize=3,label=name)
axs[0].set_xticks(x,labels);axs[1].set_xticks(x,labels)
axs[0].set_ylabel('Failure-padded stabilization loss')
axs[1].set_ylabel('Mean survival steps (max 500)')
axs[0].set_title('Closed-loop stabilization')
axs[1].set_title('Episode survival')
for ax in axs:
 ax.grid(alpha=.2);ax.set_xlabel('Deployment condition')
axs[0].legend(frameon=False,loc='upper left')
fig.suptitle('E-060C1 canonical Cart-Pole task generalization',fontsize=14)
fig.tight_layout(rect=[0,0,1,.94])
OUT.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT/'e060c1_cartpole_generalization.png',dpi=220,bbox_inches='tight')
fig.savefig(OUT/'e060c1_cartpole_generalization.pdf',bbox_inches='tight')
