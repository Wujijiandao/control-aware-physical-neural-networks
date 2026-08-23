#!/usr/bin/env python3
"""E-050 development: task-utility/viability Pareto audit of frozen E-010 models.

Purpose: address the possibility that lower viability cost is obtained merely by doing
less task work. This development script chooses a fixed lambda grid and checks that the
two frozen controllers have overlapping task-utility ranges. No confirmatory seeds are
used here and no E-010 assets are modified.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
import torch
from piha.substrates import InterferometricOracle
from e010_matched_static_development import evaluate_closed_loop

ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'frozen/e010'
OUT=ROOT/'results/e050_development'
LAMBDAS=[0.010,0.014,0.018,0.020,0.022,0.026,0.030,0.036]

def load(name):
    m=InterferometricOracle(paths=64,detectors=16)
    m.load_state_dict(torch.load(F/name,weights_only=True));m.eval();return m

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    seeds=list(range(70000,70024))
    models={'mse':load('selected_mse.pt'),'control_aware':load('selected_control_aware.pt')}
    rows=[]
    for method,m in models.items():
        for lam in LAMBDAS:
            r=evaluate_closed_loop(m,seeds,steps=400,lam=lam)
            rows.append({'method':method,'lambda_task':lam,**r})
            print(method,lam,r['mean_D'],r['cumulative_task'],flush=True)
    with (OUT/'e050d1_pareto.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Aggregate task-range overlap, based only on the predeclared grid.
    by={k:[r for r in rows if r['method']==k] for k in models}
    lo=max(min(r['cumulative_task'] for r in by[k]) for k in models)
    hi=min(max(r['cumulative_task'] for r in by[k]) for k in models)
    summary={'status':'DEVELOPMENT_ONLY','lambda_grid':LAMBDAS,'seed_range':[seeds[0],seeds[-1]],
             'overlap_task_range':[lo,hi],'overlap_width':hi-lo}
    (OUT/'e050d1_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
