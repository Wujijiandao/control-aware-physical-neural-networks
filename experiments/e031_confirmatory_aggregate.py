#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'frozen/e031';OUT=ROOT/'results/e031_confirmatory'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def boot(d,n,seed):
 rng=np.random.default_rng(seed);N=len(d);x=np.empty(n)
 for i in range(n):x[i]=d[rng.integers(0,N,N)].mean()
 return [float(np.quantile(x,.025)),float(np.quantile(x,.975))]
def main():
 cfg=json.loads((F/'confirmatory_config.json').read_text());seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()];files=sorted((OUT/'shards').glob('shard_*_of_08.csv'))
 if len(files)!=8:raise SystemExit(f'need 8 shards, found {len(files)}')
 rows=[]
 for p in files:
  with p.open() as f:rows+=list(csv.DictReader(f))
 metrics=['mean_D','p95_D','viability_occupancy','severe_fraction','action_agreement','cumulative_task']
 for r in rows:
  r['seed']=int(r['seed'])
  for k in metrics:r[k]=float(r[k])
 keys=[(r['seed'],r['method'],r['condition']) for r in rows];expected={(s,m,c) for s in seeds for m in ('boundary_aware','robust_control_aware') for c in ('moderate','strong')}
 if set(keys)!=expected or len(keys)!=len(expected):raise RuntimeError('coverage mismatch')
 raw=OUT/'e031c1_primary_raw.csv'
 with raw.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['seed','method','condition']+metrics);w.writeheader();w.writerows(rows)
 by={(r['seed'],r['method'],r['condition']):r for r in rows};pairs=[]
 for s in seeds:
  b=.5*(by[(s,'boundary_aware','moderate')]['mean_D']+by[(s,'boundary_aware','strong')]['mean_D']);c=.5*(by[(s,'robust_control_aware','moderate')]['mean_D']+by[(s,'robust_control_aware','strong')]['mean_D']);pairs.append({'seed':s,'boundary_stress_mean_D':b,'robust_control_stress_mean_D':c,'delta':c-b})
 pp=OUT/'e031c1_primary_paired.csv'
 with pp.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
 b=np.array([p['boundary_stress_mean_D'] for p in pairs]);c=np.array([p['robust_control_stress_mean_D'] for p in pairs]);d=c-b;ci=boot(d,cfg['bootstrap_resamples'],cfg['bootstrap_seed']);rel=float((b.mean()-c.mean())/b.mean());success=bool(ci[1]<0 and rel>=cfg['success_criteria']['minimum_relative_reduction'])
 desc={}
 for m in ('boundary_aware','robust_control_aware'):
  desc[m]={}
  for cond in ('moderate','strong'):
   rr=[r for r in rows if r['method']==m and r['condition']==cond];desc[m][cond]={k:float(np.mean([x[k] for x in rr])) for k in metrics}
 summary={'experiment':'E-031C1','status':'CONFIRMATORY_COMPLETED_ONCE_VIA_FROZEN_SHARDS','n_seeds':len(seeds),'conditions':['moderate','strong'],'boundary_stress_mean_D':float(b.mean()),'robust_control_stress_mean_D':float(c.mean()),'paired_difference':float(d.mean()),'bootstrap_ci95':ci,'relative_reduction':rel,'prespecified_minimum_relative_reduction':cfg['success_criteria']['minimum_relative_reduction'],'primary_success':success,'bootstrap_resamples':cfg['bootstrap_resamples'],'bootstrap_seed':cfg['bootstrap_seed'],'descriptive_secondary':desc,'seed_sha256':sha(F/'confirmatory_seeds.txt'),'raw_sha256':sha(raw),'paired_sha256':sha(pp),'shard_sha256':{p.name:sha(p) for p in files}}
 (OUT/'e031c1_primary_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
