#!/usr/bin/env python3
"""Descriptive E-020C1 secondary evaluation using frozen models/seeds only."""
from __future__ import annotations
import argparse,csv,json,multiprocessing as mp,os
from pathlib import Path
import numpy as np, torch
from piha.substrates import InterferometricOracle
from e020_robustness_development import episode

ROOT=Path(__file__).resolve().parents[1];FROZEN=ROOT/'frozen/e020';OUT=ROOT/'results/e020_confirmatory_secondary'
_G={}
ALL_METHODS=('mse','noise_aware','boundary_aware','sharpness_aware','robust_control_aware')
FILES={m:f'{m}.pt' for m in ALL_METHODS}

def init_worker(methods,conditions):
    torch.set_num_threads(1);mods={}
    for m in methods:
        model=InterferometricOracle(paths=64,detectors=16);model.load_state_dict(torch.load(FROZEN/FILES[m],weights_only=True));model.eval();mods[m]=model
    _G['models']=mods;_G['conditions']={c['name']:c for c in conditions}

def work(args):
    seed,methods,cnames=args; rows=[]
    for m in methods:
        for c in cnames: rows.append({'seed':seed,'method':m,'condition':c,**episode(_G['models'][m],seed,_G['conditions'][c])})
    return rows

def run_part(part):
    conditions=json.loads((FROZEN/'conditions.json').read_text());seeds=[int(x) for x in (FROZEN/'confirmatory_seeds.txt').read_text().split()]
    if part=='nominal_all': methods=ALL_METHODS;cnames=('nominal',)
    elif part=='mild_all': methods=ALL_METHODS;cnames=('mild',)
    elif part=='moderate_other': methods=('mse','noise_aware','sharpness_aware');cnames=('moderate',)
    elif part=='strong_other': methods=('mse','noise_aware','sharpness_aware');cnames=('strong',)
    else: raise ValueError(part)
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/f'{part}_raw.csv'
    if path.exists(): raise SystemExit(f'refusing overwrite {path}')
    ctx=mp.get_context('fork')
    with ctx.Pool(processes=min(4,os.cpu_count() or 1),initializer=init_worker,initargs=(methods,conditions)) as pool:
        nested=pool.map(work,[(s,methods,cnames) for s in seeds],chunksize=4)
    rows=[r for rr in nested for r in rr];fields=list(rows[0].keys())
    with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(part,len(rows))

def aggregate():
    import pandas as pd
    primary=ROOT/'results/e020_confirmatory/e020c1_primary_raw.csv'
    parts=[OUT/'nominal_all_raw.csv',OUT/'mild_all_raw.csv',OUT/'moderate_other_raw.csv',OUT/'strong_other_raw.csv']
    if not all(p.exists() for p in [primary,*parts]): raise SystemExit('missing part for aggregate')
    d=pd.concat([pd.read_csv(primary),*(pd.read_csv(p) for p in parts)],ignore_index=True)
    # Remove any accidental duplicate keys; expected primary supplies boundary/control stressed only.
    assert not d.duplicated(['seed','method','condition']).any()
    assert len(d)==192*5*4
    d.to_csv(OUT/'e020c1_all_raw.csv',index=False)
    metrics=['mean_D','p95_D','viability_occupancy','severe_fraction','action_agreement','cumulative_task']
    rows=[]
    for (m,c),g in d.groupby(['method','condition'],sort=False):
        row={'method':m,'condition':c,'n_seeds':len(g)}
        for x in metrics:
            a=g[x].to_numpy(float);row[x]=a.mean();row['se_'+x]=a.std(ddof=1)/np.sqrt(len(a))
        rows.append(row)
    s=pd.DataFrame(rows);s.to_csv(OUT/'e020c1_all_summary.csv',index=False)
    # Stress aggregate for all methods.
    stress=d[d.condition.isin(['moderate','strong'])].groupby(['seed','method'],as_index=False)['mean_D'].mean()
    stress_summary=stress.groupby('method')['mean_D'].agg(['mean','std','count']).reset_index();stress_summary['se']=stress_summary['std']/np.sqrt(stress_summary['count'])
    stress_summary.to_csv(OUT/'e020c1_stress_summary.csv',index=False)
    print(s[['method','condition','mean_D','action_agreement','cumulative_task']].to_string(index=False))
    print('\nSTRESS\n',stress_summary.to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['nominal_all','mild_all','moderate_other','strong_other','aggregate'],required=True);a=ap.parse_args()
    aggregate() if a.part=='aggregate' else run_part(a.part)
