#!/usr/bin/env python3
"""Resume the frozen E-030C1 computation in deterministic seed shards.

This is a computational continuation of the already-frozen experiment after the
monolithic runner exceeded the execution wall-clock limit before writing any
outcome file. It does not change seeds, checkpoints, endpoints or analysis.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, multiprocessing as mp, os
from pathlib import Path
import numpy as np
import torch
from piha.substrates import NonlinearOscillatorOracle
from piha.dynamics import ACTIONS, TASK_GAIN, ambient_at, predicted_next, true_step
from piha.viability import viability_np

ROOT=Path(__file__).resolve().parents[1];FROZEN=ROOT/'frozen/e030';OUT=ROOT/'results/e030_confirmatory';_G={}

def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_freeze():
    for line in (FROZEN/'FREEZE_MANIFEST.sha256').read_text().splitlines():
        if not line.strip(): continue
        h,name=line.split('  ',1)
        if sha256(FROZEN/name)!=h: raise RuntimeError(f'hash mismatch: {name}')

def new_model(): return NonlinearOscillatorOracle(d=3,oscillators=24,integration_steps=14,dt=.055)

def init_worker():
    torch.set_num_threads(1);models={}
    for label,file in [('mse','mse.pt'),('control_aware','control_aware.pt')]:
        m=new_model();m.load_state_dict(torch.load(FROZEN/file,weights_only=True));m.eval();models[label]=m
    _G['models']=models

@torch.no_grad()
def run_model(model,seed,steps=400,lam=.02):
    rng=np.random.default_rng(seed);h=np.array([.60,.45,.15])+rng.normal(0,.015,3)
    ds=[];ag=[];vi=[];sev=[];tasks=[]
    for t in range(steps):
        amb=ambient_at(t);cand=np.array([predicted_next(h,a,amb) for a in range(len(ACTIONS))],dtype=np.float32)
        demand=1+.35*np.sin(2*np.pi*t/80+.4);bonus=lam*demand*TASK_GAIN
        exact=int(np.argmin(viability_np(cand)-bonus))
        score=_model_cost(model,cand)-bonus;act=int(np.argmin(score));ag.append(act==exact)
        h=true_step(h,act,amb,rng,shocks=True);d=float(viability_np(h));ds.append(d);vi.append(d<.05);sev.append(d>.2);tasks.append(float(TASK_GAIN[act]*demand))
    return {'mean_D':float(np.mean(ds)),'p95_D':float(np.quantile(ds,.95)),'viability_occupancy':float(np.mean(vi)),
            'severe_fraction':float(np.mean(sev)),'action_agreement':float(np.mean(ag)),'cumulative_task':float(np.sum(tasks))}

def _model_cost(model,cand):
    return model(torch.tensor(cand,dtype=torch.float32)).cpu().numpy()

def work(seed): return {'seed':seed,**{k:run_model(m,seed) for k,m in _G['models'].items()}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard-index',type=int,required=True);ap.add_argument('--num-shards',type=int,default=8);args=ap.parse_args()
    verify_freeze();seeds=[int(x) for x in (FROZEN/'confirmatory_seeds.txt').read_text().split()]
    if not 0<=args.shard_index<args.num_shards: raise SystemExit('bad shard')
    shard_seeds=seeds[args.shard_index::args.num_shards]
    sd=OUT/'shards';sd.mkdir(parents=True,exist_ok=True);path=sd/f'shard_{args.shard_index:02d}_of_{args.num_shards:02d}.csv'
    if path.exists(): raise SystemExit(f'refusing overwrite: {path}')
    ctx=mp.get_context('fork')
    with ctx.Pool(processes=min(4,os.cpu_count() or 1),initializer=init_worker) as pool: vals=pool.map(work,shard_seeds,chunksize=1)
    metrics=['mean_D','p95_D','viability_occupancy','severe_fraction','action_agreement','cumulative_task'];rows=[]
    for v in vals:
        for method in ('mse','control_aware'): rows.append({'seed':v['seed'],'method':method,**v[method]})
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['seed','method']+metrics);w.writeheader();w.writerows(rows)
    meta={'experiment':'E-030C1','continuation':'frozen computational shard','shard_index':args.shard_index,'num_shards':args.num_shards,
          'n_seeds':len(shard_seeds),'first_seed_position':args.shard_index,'seed_file_sha256':sha256(FROZEN/'confirmatory_seeds.txt'),'csv_sha256':sha256(path)}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
