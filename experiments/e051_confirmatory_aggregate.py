#!/usr/bin/env python3
"""Aggregate E-051C1 and compute co-primary one-step regret endpoints."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'frozen/e051';OUT=ROOT/'results/e051_confirmatory'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def boot(x,n,seed):
 rng=np.random.default_rng(seed);N=len(x);v=np.empty(n)
 for i in range(n):v[i]=x[rng.integers(0,N,N)].mean()
 return [float(np.quantile(v,.025)),float(np.quantile(v,.975))]
def main():
 cfg=json.loads((F/'confirmatory_config.json').read_text());seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()];files=sorted((OUT/'shards').glob('shard_*_of_04.csv'))
 if len(files)!=4:raise RuntimeError(f'expected 4 shards got {len(files)}')
 df=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);keys=['seed','experiment','method','condition'];expected=len(seeds)*2*2*2
 if len(df)!=expected or df.duplicated(keys).any() or set(df.seed)!=set(seeds):raise RuntimeError('coverage failure')
 paired=[];summ={};all_success=True
 for exp in ('E020','E031'):
  for s in seeds:
   d=df[(df.seed==s)&(df.experiment==exp)];b=d[d.method=='boundary_aware'].mean_regret.mean();c=d[d.method=='robust_control_aware'].mean_regret.mean();paired.append({'seed':s,'experiment':exp,'boundary_mean_regret':b,'robust_mean_regret':c,'delta':c-b})
  q=pd.DataFrame([r for r in paired if r['experiment']==exp]);delta=q.delta.to_numpy();bm=float(q.boundary_mean_regret.mean());cm=float(q.robust_mean_regret.mean());rel=(bm-cm)/bm;ci=boot(delta,cfg['bootstrap_resamples'],cfg['bootstrap_seed']+(20 if exp=='E020' else 31));success=bool(ci[1]<0 and rel>=cfg['success_criteria_each_substrate']['minimum_relative_regret_reduction']);all_success &= success;summ[exp]={'boundary_mean_regret':bm,'robust_mean_regret':cm,'paired_difference':float(delta.mean()),'bootstrap_ci95':ci,'relative_reduction':float(rel),'success':success}
 pd.DataFrame(paired).to_csv(OUT/'e051c1_primary_paired.csv',index=False);df.to_csv(OUT/'e051c1_raw.csv',index=False)
 # pooled descriptive condition/method means and quartiles
 metric_cols=['mean_regret','p95_regret','action_agreement','mean_gap']+[f'q{i}_{x}' for i in range(1,5) for x in ('regret','agreement')]
 desc=df.groupby(['experiment','method','condition'])[metric_cols].mean().reset_index();desc.to_csv(OUT/'e051c1_descriptive_summary.csv',index=False)
 summary={'experiment':'E-051C1','status':'CONFIRMATORY_COMPLETED_ONCE','n_seeds':len(seeds),'co_primary':summ,'full_mechanism_success':bool(all_success),'minimum_relative_regret_reduction':cfg['success_criteria_each_substrate']['minimum_relative_regret_reduction'],'raw_sha256':sha(OUT/'e051c1_raw.csv'),'paired_sha256':sha(OUT/'e051c1_primary_paired.csv')};(OUT/'e051c1_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
