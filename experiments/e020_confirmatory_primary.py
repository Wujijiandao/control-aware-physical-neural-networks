#!/usr/bin/env python3
"""E-020C1 frozen confirmatory primary analysis.

Evaluates only the prespecified primary comparison (robust-control-aware vs
boundary-aware, moderate+strong conditions). Secondary descriptive analyses are
run separately after this primary result is sealed.
"""
from __future__ import annotations
import csv, hashlib, json, math, multiprocessing as mp, os, time
from pathlib import Path
import numpy as np
import torch

from piha.substrates import InterferometricOracle
from e020_robustness_development import episode

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'frozen/e020'
OUT=ROOT/'results/e020_confirmatory'

_G={}

def sha256(p:Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def verify_freeze():
    for line in (FROZEN/'FREEZE_MANIFEST.sha256').read_text().splitlines():
        if not line.strip(): continue
        h,name=line.split('  ',1)
        got=sha256(FROZEN/name)
        if got!=h: raise RuntimeError(f'hash mismatch for {name}: {got} != {h}')

def init_worker(conditions):
    torch.set_num_threads(1)
    models={}
    for label,file in [('boundary_aware','boundary_aware.pt'),('robust_control_aware','robust_control_aware.pt')]:
        m=InterferometricOracle(paths=64,detectors=16)
        m.load_state_dict(torch.load(FROZEN/file,weights_only=True));m.eval();models[label]=m
    _G['models']=models;_G['conditions']={c['name']:c for c in conditions}

def work_seed(seed:int):
    rows=[]
    for method in ('boundary_aware','robust_control_aware'):
        for cname in ('moderate','strong'):
            r=episode(_G['models'][method],seed,_G['conditions'][cname])
            rows.append({'seed':seed,'method':method,'condition':cname,**r})
    return rows

def bootstrap_ci(delta, n=20000, seed=920201):
    rng=np.random.default_rng(seed);N=len(delta)
    vals=np.empty(n,dtype=float)
    for i in range(n): vals[i]=delta[rng.integers(0,N,N)].mean()
    return [float(np.quantile(vals,0.025)),float(np.quantile(vals,0.975))]

def main():
    if OUT.exists(): raise SystemExit(f'refusing rerun/overwrite: {OUT}')
    verify_freeze()
    OUT.mkdir(parents=True)
    seeds=[int(x) for x in (FROZEN/'confirmatory_seeds.txt').read_text().split()]
    conditions=json.loads((FROZEN/'conditions.json').read_text())
    start=time.time()
    ctx=mp.get_context('fork')
    with ctx.Pool(processes=min(4,os.cpu_count() or 1),initializer=init_worker,initargs=(conditions,)) as pool:
        nested=pool.map(work_seed,seeds,chunksize=4)
    rows=[r for rr in nested for r in rr]
    fields=['seed','method','condition','mean_D','p95_D','viability_occupancy','severe_fraction','action_agreement','cumulative_task']
    with (OUT/'e020c1_primary_raw.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])

    # Per-seed stressed endpoint: mean of moderate and strong condition means.
    by={}
    for r in rows: by[(r['seed'],r['method'],r['condition'])]=r
    paired=[]
    for s in seeds:
        b=0.5*(by[(s,'boundary_aware','moderate')]['mean_D']+by[(s,'boundary_aware','strong')]['mean_D'])
        c=0.5*(by[(s,'robust_control_aware','moderate')]['mean_D']+by[(s,'robust_control_aware','strong')]['mean_D'])
        paired.append((s,b,c,c-b))
    arr=np.asarray([[x[1],x[2],x[3]] for x in paired],dtype=float)
    bmean=float(arr[:,0].mean());cmean=float(arr[:,1].mean());delta=arr[:,2]
    ci=bootstrap_ci(delta)
    relative=float((bmean-cmean)/bmean)
    success=bool(ci[1]<0 and relative>=0.05)
    with (OUT/'e020c1_primary_paired.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['seed','boundary_stress_mean_D','robust_control_stress_mean_D','delta'])
        w.writerows(paired)
    summary={
        'experiment':'E-020C1','status':'CONFIRMATORY_COMPLETED_ONCE',
        'primary_comparator':'boundary_aware','primary_method':'robust_control_aware',
        'n_seeds':len(seeds),'conditions':['moderate','strong'],'horizon':400,
        'boundary_stress_mean_D':bmean,'robust_control_stress_mean_D':cmean,
        'paired_difference':float(delta.mean()),'bootstrap_ci95':ci,
        'relative_reduction':relative,'prespecified_minimum_relative_reduction':0.05,
        'primary_success':success,'bootstrap_resamples':20000,'bootstrap_seed':920201,
        'runtime_seconds':time.time()-start,
        'seed_sha256':sha256(FROZEN/'confirmatory_seeds.txt'),
        'raw_sha256':sha256(OUT/'e020c1_primary_raw.csv'),
        'paired_sha256':sha256(OUT/'e020c1_primary_paired.csv'),
    }
    (OUT/'e020c1_primary_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
