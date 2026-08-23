#!/usr/bin/env python3
"""E-020D2 development-only robustness comparison for equal-budget strong baselines.

This script is used only to choose perturbation magnitudes and verify that the planned
confirmatory experiment is informative. It must not be reported as confirmatory evidence.
"""
from __future__ import annotations
import copy, csv, json, math
from pathlib import Path
import numpy as np
import torch

from piha.dynamics import ACTIONS, TASK_GAIN, ambient_at, predicted_next, true_step
from piha.evaluation import model_oracle
from piha.substrates import InterferometricOracle
from piha.viability import viability_np
from e020_checkpoint_development import METHODS

CONDITIONS = (
    {"name":"nominal",  "measurement_noise":0.0,   "input_noise":0.0,   "phase_sigma":0.0,  "coupling_rel":0.0},
    {"name":"mild",     "measurement_noise":0.005, "input_noise":0.003, "phase_sigma":0.01, "coupling_rel":0.002},
    {"name":"moderate", "measurement_noise":0.010, "input_noise":0.006, "phase_sigma":0.03, "coupling_rel":0.005},
    {"name":"strong",   "measurement_noise":0.020, "input_noise":0.012, "phase_sigma":0.08, "coupling_rel":0.010},
)


def load_final(method: str, src: Path) -> InterferometricOracle:
    fam=torch.load(src/f'{method}_family.pt',weights_only=False)
    ck=fam[-1]
    if ck['step'] != 4000:
        raise RuntimeError(f'{method}: expected final step 4000')
    m=InterferometricOracle(paths=64,detectors=16); m.load_state_dict(ck['state']); m.eval()
    return m


def drifted_model(base: InterferometricOracle, seed: int, phase_sigma: float, coupling_rel: float):
    m=copy.deepcopy(base)
    if phase_sigma==0 and coupling_rel==0:
        return m
    rng=np.random.default_rng(seed)
    with torch.no_grad():
        if phase_sigma:
            m.b.add_(torch.tensor(rng.normal(0.0,phase_sigma,size=tuple(m.b.shape)),dtype=m.b.dtype))
        if coupling_rel:
            for p in (m.cre,m.cim,m.w):
                z=torch.tensor(rng.normal(0.0,coupling_rel,size=tuple(p.shape)),dtype=p.dtype)
                p.mul_(1.0+z)
    return m


def episode(model, seed:int, condition:dict, steps:int=400, lam:float=0.02):
    # Same environmental seed across methods; device-drift seed is a deterministic offset.
    rng=np.random.default_rng(seed)
    dm=drifted_model(model, 9000000+seed, condition['phase_sigma'], condition['coupling_rel'])
    h=np.array([0.60,0.45,0.15])+rng.normal(0.0,0.015,3)
    dvals=[]; viable=[]; severe=[]; agree=[]; task=[]
    for t in range(steps):
        amb=ambient_at(t)
        cand=np.array([predicted_next(h,a,amb) for a in range(len(ACTIONS))],dtype=np.float32)
        demand=1.0+0.35*np.sin(2*np.pi*t/80+0.4)
        bonus=lam*demand*TASK_GAIN
        exact=int(np.argmin(viability_np(cand)-bonus))
        costs=model_oracle(dm,cand,rng,condition['measurement_noise'],condition['input_noise'])
        action=int(np.argmin(costs-bonus))
        agree.append(action==exact)
        h=true_step(h,action,amb,rng,shocks=True)
        d=float(viability_np(h)); dvals.append(d); viable.append(d<0.05); severe.append(d>0.2)
        task.append(float(TASK_GAIN[action]*demand))
    return {
        'mean_D':float(np.mean(dvals)), 'p95_D':float(np.quantile(dvals,0.95)),
        'viability_occupancy':float(np.mean(viable)), 'severe_fraction':float(np.mean(severe)),
        'action_agreement':float(np.mean(agree)), 'cumulative_task':float(np.sum(task)),
    }


def main():
    src=Path('results/e020_equalbudget_development')
    out=Path('results/e020_robustness_development'); out.mkdir(parents=True,exist_ok=True)
    seeds=list(range(32000,32016))
    rows=[]; raw=[]
    for method in METHODS:
        m=load_final(method,src)
        for cond in CONDITIONS:
            vals=[]
            for s in seeds:
                r=episode(m,s,cond); vals.append(r); raw.append({'method':method,'condition':cond['name'],'seed':s,**r})
            row={'method':method,'condition':cond['name'],**{k:cond[k] for k in cond if k!='name'}}
            for metric in vals[0]:
                a=np.array([v[metric] for v in vals],dtype=float)
                row[metric]=float(a.mean()); row['se_'+metric]=float(a.std(ddof=1)/math.sqrt(len(a)))
            rows.append(row)
            print(method,cond['name'],'mean_D',row['mean_D'],'agree',row['action_agreement'],flush=True)
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    with (out/'raw_per_seed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0].keys()));w.writeheader();w.writerows(raw)
    (out/'conditions.json').write_text(json.dumps(CONDITIONS,indent=2),encoding='utf-8')
    print('DEVELOPMENT_ONLY_NOT_CONFIRMATORY')
if __name__=='__main__':main()
