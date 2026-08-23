#!/usr/bin/env python3
"""Aggregate deterministic E-030C1 shards using the frozen analysis."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'frozen/e030';OUT=ROOT/'results/e030_confirmatory'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ci(delta,n,seed):
 rng=np.random.default_rng(seed);N=len(delta);x=np.empty(n)
 for i in range(n):x[i]=delta[rng.integers(0,N,N)].mean()
 return [float(np.quantile(x,.025)),float(np.quantile(x,.975))]
def main():
 cfg=json.loads((F/'confirmatory_config.json').read_text());seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()]
 files=sorted((OUT/'shards').glob('shard_*_of_08.csv'))
 if len(files)!=8:raise SystemExit(f'need 8 shards, found {len(files)}')
 rows=[]
 for p in files:
  with p.open() as f: rows.extend(list(csv.DictReader(f)))
 for r in rows:
  r['seed']=int(r['seed'])
  for k in ['mean_D','p95_D','viability_occupancy','severe_fraction','action_agreement','cumulative_task']:r[k]=float(r[k])
 keys=[(r['seed'],r['method']) for r in rows]
 expected={(s,m) for s in seeds for m in ('mse','control_aware')}
 if set(keys)!=expected or len(keys)!=len(expected):raise RuntimeError('shard coverage mismatch')
 order={s:i for i,s in enumerate(seeds)};rows.sort(key=lambda r:(order[r['seed']],r['method']))
 metrics=['mean_D','p95_D','viability_occupancy','severe_fraction','action_agreement','cumulative_task']
 raw=OUT/'e030c1_raw_per_seed.csv'
 with raw.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['seed','method']+metrics);w.writeheader();w.writerows(rows)
 by={(r['seed'],r['method']):r for r in rows};pairs=[]
 for s in seeds:
  a=by[(s,'mse')]['mean_D'];b=by[(s,'control_aware')]['mean_D'];pairs.append({'seed':s,'mse_mean_D':a,'control_mean_D':b,'delta_mean_D':b-a})
 paired=OUT/'e030c1_paired.csv'
 with paired.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
 mse=np.array([p['mse_mean_D'] for p in pairs]);ctl=np.array([p['control_mean_D'] for p in pairs]);delta=ctl-mse
 bci=ci(delta,cfg['bootstrap_resamples'],cfg['bootstrap_seed']);rel=float((mse.mean()-ctl.mean())/mse.mean());success=bool(bci[1]<0 and rel>=cfg['success_criteria']['minimum_relative_reduction'])
 desc={}
 for m in ('mse','control_aware'):
  rr=[r for r in rows if r['method']==m];desc[m]={k:float(np.mean([r[k] for r in rr])) for k in metrics}
 summary={'experiment':'E-030C1','status':'CONFIRMATORY_COMPLETED_ONCE_VIA_FROZEN_SHARDS','technical_note':'Monolithic execution exceeded wall-clock before writing outcomes; identical frozen task resumed in 8 deterministic seed shards.',
          'n_seeds':len(seeds),'horizon':cfg['horizon'],'mse_mean_D':float(mse.mean()),'control_mean_D':float(ctl.mean()),'paired_difference':float(delta.mean()),'bootstrap_ci95':bci,
          'relative_reduction':rel,'prespecified_minimum_relative_reduction':cfg['success_criteria']['minimum_relative_reduction'],'primary_success':success,
          'bootstrap_resamples':cfg['bootstrap_resamples'],'bootstrap_seed':cfg['bootstrap_seed'],'descriptive_secondary':desc,'seed_sha256':sha(F/'confirmatory_seeds.txt'),'raw_sha256':sha(raw),'paired_sha256':sha(paired),
          'shard_sha256':{p.name:sha(p) for p in files}}
 (OUT/'e030c1_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
