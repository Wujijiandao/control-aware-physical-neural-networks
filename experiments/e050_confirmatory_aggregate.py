#!/usr/bin/env python3
"""Aggregate frozen E-050C1 shards and compute task-matched Pareto primary endpoint."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'frozen/e050';OUT=ROOT/'results/e050_confirmatory'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bootstrap(x,n,seed):
 rng=np.random.default_rng(seed);N=len(x);v=np.empty(n)
 for i in range(n):v[i]=x[rng.integers(0,N,N)].mean()
 return [float(np.quantile(v,.025)),float(np.quantile(v,.975))]
def curve_endpoint(gb,gc):
 gb=gb.sort_values('cumulative_task');gc=gc.sort_values('cumulative_task');lo=max(gb.cumulative_task.min(),gc.cumulative_task.min());hi=min(gb.cumulative_task.max(),gc.cumulative_task.max())
 if not hi>lo:raise RuntimeError('no task overlap')
 w=hi-lo;lo2=lo+.1*w;hi2=hi-.1*w;grid=np.linspace(lo2,hi2,41);b=np.interp(grid,gb.cumulative_task,gb.mean_D);c=np.interp(grid,gc.cumulative_task,gc.mean_D)
 return float(b.mean()),float(c.mean()),float((c-b).mean()),float(lo),float(hi)
def main():
 if (OUT/'e050c1_summary.json').exists():raise SystemExit('refusing overwrite completed summary')
 cfg=json.loads((F/'confirmatory_config.json').read_text());seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()];files=sorted((OUT/'shards').glob('shard_*_of_04.csv'))
 if len(files)!=4:raise RuntimeError(f'expected 4 shards, got {len(files)}')
 df=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);expected=len(seeds)*2*2*len(cfg['lambda_grid'])
 keys=['seed','method','condition','lambda_task']
 if len(df)!=expected or df.duplicated(keys).any():raise RuntimeError(f'coverage failure rows={len(df)} expected={expected}')
 if set(df.seed)!=set(seeds):raise RuntimeError('seed coverage mismatch')
 per=[]
 for s in seeds:
  condrows=[]
  for cond in ('moderate','strong'):
   d=df[(df.seed==s)&(df.condition==cond)];b,c,delta,lo,hi=curve_endpoint(d[d.method=='boundary_aware'],d[d.method=='robust_control_aware']);condrows.append((b,c,delta));per.append({'seed':s,'condition':cond,'boundary_taskmatched_D':b,'robust_taskmatched_D':c,'delta':delta,'task_overlap_low':lo,'task_overlap_high':hi})
 # per seed primary average across conditions
 prim=[]
 for s in seeds:
  q=[r for r in per if r['seed']==s];b=np.mean([r['boundary_taskmatched_D'] for r in q]);c=np.mean([r['robust_taskmatched_D'] for r in q]);prim.append({'seed':s,'boundary_taskmatched_D':b,'robust_taskmatched_D':c,'delta':c-b})
 pdf=pd.DataFrame(prim);delta=pdf.delta.to_numpy();bmean=float(pdf.boundary_taskmatched_D.mean());cmean=float(pdf.robust_taskmatched_D.mean());rel=(bmean-cmean)/bmean;ci=bootstrap(delta,cfg['bootstrap_resamples'],cfg['bootstrap_seed']);success=bool(ci[1]<0 and rel>=cfg['success_criteria']['minimum_relative_reduction'])
 pd.DataFrame(per).to_csv(OUT/'e050c1_condition_paired.csv',index=False);pdf.to_csv(OUT/'e050c1_primary_paired.csv',index=False);df.to_csv(OUT/'e050c1_raw.csv',index=False)
 summary={'experiment':'E-050C1','status':'CONFIRMATORY_COMPLETED_ONCE','n_seeds':len(seeds),'boundary_taskmatched_mean_D':bmean,'robust_taskmatched_mean_D':cmean,'paired_difference':float(delta.mean()),'bootstrap_ci95':ci,'relative_reduction':float(rel),'minimum_relative_reduction':cfg['success_criteria']['minimum_relative_reduction'],'primary_success':success,'bootstrap_resamples':cfg['bootstrap_resamples'],'bootstrap_seed':cfg['bootstrap_seed'],'raw_sha256':sha(OUT/'e050c1_raw.csv'),'primary_paired_sha256':sha(OUT/'e050c1_primary_paired.csv')}
 (OUT/'e050c1_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
