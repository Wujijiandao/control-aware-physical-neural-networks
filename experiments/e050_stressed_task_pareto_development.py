#!/usr/bin/env python3
"""E-050D2 development: stressed task/viability Pareto ranges for frozen E-020/E-031 models."""
from __future__ import annotations
import csv,json,copy
from pathlib import Path
import numpy as np, torch
from piha.substrates import InterferometricOracle,NonlinearOscillatorOracle
from e020_robustness_development import episode as e020_episode
from e031_confirmatory_shard import episode as e031_episode
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/e050_development'
LAMBDAS=[0.010,0.014,0.018,0.020,0.022,0.026,0.030,0.036]

def i_model(folder,name):
 m=InterferometricOracle(paths=64,detectors=16);m.load_state_dict(torch.load(folder/name,weights_only=True));m.eval();return m

def o_model(folder,name):
 m=NonlinearOscillatorOracle(d=3,oscillators=24,integration_steps=14,dt=.055);m.load_state_dict(torch.load(folder/name,weights_only=True));m.eval();return m

def summarize(vals):
 return {k:float(np.mean([v[k] for v in vals])) for k in vals[0]}

def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[]
 # E020
 f=ROOT/'frozen/e020';conds={c['name']:c for c in json.loads((f/'conditions.json').read_text())}
 mods={'boundary_aware':i_model(f,'boundary_aware.pt'),'robust_control_aware':i_model(f,'robust_control_aware.pt')}
 seeds=list(range(71000,71012))
 for method,m in mods.items():
  for cname in ('moderate','strong'):
   for lam in LAMBDAS:
    vals=[e020_episode(m,s,conds[cname],steps=400,lam=lam) for s in seeds]
    r=summarize(vals);rows.append({'experiment':'E020','method':method,'condition':cname,'lambda_task':lam,**r});print('E020',method,cname,lam,r['mean_D'],r['cumulative_task'],flush=True)
 # E031
 f=ROOT/'frozen/e031';conds={c['name']:c for c in json.loads((f/'conditions.json').read_text())}
 mods={'boundary_aware':o_model(f,'boundary_aware.pt'),'robust_control_aware':o_model(f,'robust_control_aware.pt')}
 seeds=list(range(72000,72012))
 for method,m in mods.items():
  for cname in ('moderate','strong'):
   for lam in LAMBDAS:
    vals=[e031_episode(m,s,conds[cname],steps=400,lam=lam) for s in seeds]
    r=summarize(vals);rows.append({'experiment':'E031','method':method,'condition':cname,'lambda_task':lam,**r});print('E031',method,cname,lam,r['mean_D'],r['cumulative_task'],flush=True)
 with (OUT/'e050d2_stressed_pareto.csv').open('w',newline='',encoding='utf-8') as fh:
  w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/'e050d2_summary.json').write_text(json.dumps({'status':'DEVELOPMENT_ONLY','lambda_grid':LAMBDAS,'e020_seed_range':[71000,71011],'e031_seed_range':[72000,72011]},indent=2))
if __name__=='__main__':main()
