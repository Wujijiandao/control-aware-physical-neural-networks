#!/usr/bin/env python3
"""E-030C1 single-use cross-substrate confirmatory runner."""
from __future__ import annotations
import csv, hashlib, json, multiprocessing as mp, os, time
from pathlib import Path
import numpy as np
import torch
from piha.substrates import NonlinearOscillatorOracle
from piha.dynamics import ACTIONS, TASK_GAIN, ambient_at, predicted_next, true_step
from piha.viability import viability_np

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'frozen/e030'
OUT=ROOT/'results/e030_confirmatory'
_G={}

def sha256(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()

def verify_freeze():
    for line in (FROZEN/'FREEZE_MANIFEST.sha256').read_text().splitlines():
        if not line.strip(): continue
        h,name=line.split('  ',1); got=sha256(FROZEN/name)
        if got!=h: raise RuntimeError(f'hash mismatch {name}: {got} != {h}')

def new_model(): return NonlinearOscillatorOracle(d=3,oscillators=24,integration_steps=14,dt=0.055)

def init_worker():
    torch.set_num_threads(1)
    models={}
    for label,file in [('mse','mse.pt'),('control_aware','control_aware.pt')]:
        m=new_model();m.load_state_dict(torch.load(FROZEN/file,weights_only=True));m.eval();models[label]=m
    _G['models']=models

@torch.no_grad()
def oracle_action(model,cand,bonus):
    x=torch.tensor(cand,dtype=torch.float32)
    return int(torch.argmin(model(x)-torch.tensor(bonus,dtype=torch.float32)).item())

def episode(model,seed:int,steps:int=400,lam:float=0.02):
    rng=np.random.default_rng(seed)
    h=np.array([0.60,0.45,0.15])+rng.normal(0.0,0.015,3)
    dvals=[];agree=[];viable=[];severe=[];task=[]
    for t in range(steps):
        amb=ambient_at(t)
        cand=np.array([predicted_next(h,a,amb) for a in range(len(ACTIONS))],dtype=np.float32)
        demand=1.0+0.35*np.sin(2*np.pi*t/80+0.4);bonus=lam*demand*TASK_GAIN
        exact=int(np.argmin(viability_np(cand)-bonus)); action=oracle_action(model,cand,bonus)
        agree.append(action==exact)
        h=true_step(h,action,amb,rng,shocks=True)
        d=float(viability_np(h));dvals.append(d);viable.append(d<0.05);severe.append(d>0.2);task.append(float(TASK_GAIN[action]*demand))
    return {'mean_D':float(np.mean(dvals)),'p95_D':float(np.quantile(dvals,0.95)),
            'viability_occupancy':float(np.mean(viable)),'severe_fraction':float(np.mean(severe)),
            'action_agreement':float(np.mean(agree)),'cumulative_task':float(np.sum(task))}

def work(seed:int):
    return {label:episode(m,seed) for label,m in _G['models'].items()}

def bootstrap_ci(delta,n=20000,seed=930301):
    rng=np.random.default_rng(seed);N=len(delta);vals=np.empty(n)
    for i in range(n): vals[i]=delta[rng.integers(0,N,N)].mean()
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def main():
    if OUT.exists(): raise SystemExit(f'refusing rerun/overwrite: {OUT}')
    verify_freeze();OUT.mkdir(parents=True)
    cfg=json.loads((FROZEN/'confirmatory_config.json').read_text());seeds=[int(x) for x in (FROZEN/'confirmatory_seeds.txt').read_text().split()]
    start=time.time();ctx=mp.get_context('fork')
    with ctx.Pool(processes=min(4,os.cpu_count() or 1),initializer=init_worker) as pool:
        vals=pool.map(work,seeds,chunksize=3)
    metrics=['mean_D','p95_D','viability_occupancy','severe_fraction','action_agreement','cumulative_task']
    rows=[];paired=[]
    for seed,r in zip(seeds,vals):
        for method in ('mse','control_aware'): rows.append({'seed':seed,'method':method,**r[method]})
        paired.append({'seed':seed,'mse_mean_D':r['mse']['mean_D'],'control_mean_D':r['control_aware']['mean_D'],
                       'delta_mean_D':r['control_aware']['mean_D']-r['mse']['mean_D']})
    with (OUT/'e030c1_raw_per_seed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['seed','method']+metrics);w.writeheader();w.writerows(rows)
    with (OUT/'e030c1_paired.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    delta=np.array([x['delta_mean_D'] for x in paired]);mse=np.array([x['mse_mean_D'] for x in paired]);ctl=np.array([x['control_mean_D'] for x in paired])
    ci=bootstrap_ci(delta,n=cfg['bootstrap_resamples'],seed=cfg['bootstrap_seed'])
    rel=float((mse.mean()-ctl.mean())/mse.mean());success=bool(ci[1]<0 and rel>=cfg['success_criteria']['minimum_relative_reduction'])
    descriptive={}
    for method in ('mse','control_aware'):
        rr=[r for r in rows if r['method']==method]
        descriptive[method]={k:float(np.mean([x[k] for x in rr])) for k in metrics}
    summary={'experiment':'E-030C1','status':'CONFIRMATORY_COMPLETED_ONCE','substrate':'NonlinearOscillatorOracle',
             'n_seeds':len(seeds),'horizon':cfg['horizon'],'mse_mean_D':float(mse.mean()),'control_mean_D':float(ctl.mean()),
             'paired_difference':float(delta.mean()),'bootstrap_ci95':ci,'relative_reduction':rel,
             'prespecified_minimum_relative_reduction':cfg['success_criteria']['minimum_relative_reduction'],
             'primary_success':success,'bootstrap_resamples':cfg['bootstrap_resamples'],'bootstrap_seed':cfg['bootstrap_seed'],
             'descriptive_secondary':descriptive,'runtime_seconds':time.time()-start,
             'seed_sha256':sha256(FROZEN/'confirmatory_seeds.txt'),'raw_sha256':sha256(OUT/'e030c1_raw_per_seed.csv'),
             'paired_sha256':sha256(OUT/'e030c1_paired.csv')}
    (OUT/'e030c1_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
