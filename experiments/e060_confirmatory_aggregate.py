#!/usr/bin/env python3
"""Aggregate frozen E-060C1 shards and evaluate the pre-specified endpoint."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'frozen/e060';OUT=ROOT/'results/e060_confirmatory'

def bootstrap_ci(x,B=20000,seed=606060):
 x=np.asarray(x,float);r=np.random.default_rng(seed);n=len(x);chunk=1000;means=[]
 for st in range(0,B,chunk):
  b=min(chunk,B-st);idx=r.integers(0,n,size=(b,n));means.append(x[idx].mean(1))
 m=np.concatenate(means);return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]

def main():
 cfg=json.loads((F/'audit_config.json').read_text());seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()]
 files=sorted((OUT/'shards').glob('shard_*_of_08.csv'))
 if len(files)!=8:raise RuntimeError(f'expected 8 shards, found {len(files)}')
 rows=[]
 for p in files:
  with p.open() as f:rows.extend(csv.DictReader(f))
 expected=len(seeds)*2*3
 if len(rows)!=expected:raise RuntimeError(f'expected {expected} rows, got {len(rows)}')
 key={(int(r['seed']),r['method'],r['condition']) for r in rows}
 if len(key)!=expected:raise RuntimeError('duplicate/missing seed-method-condition rows')
 required={(s,m,c) for s in seeds for m in ('noise_aware_mse','control_aware_rank') for c in ('nominal','moderate','strong')}
 if key!=required:raise RuntimeError('coverage mismatch')
 # typed rows
 for r in rows:
  for k in ('stabilization_loss','survival_steps','success_500','action_agreement','mean_decision_regret','mean_action_gap','first_divergence_step','first_divergence_gap'):
   r[k]=float(r[k])
 # summary
 summary=[]
 for method in ('noise_aware_mse','control_aware_rank'):
  for cond in ('nominal','moderate','strong'):
   rr=[r for r in rows if r['method']==method and r['condition']==cond];d={'method':method,'condition':cond,'n':len(rr)}
   for metric in ('stabilization_loss','survival_steps','success_500','action_agreement','mean_decision_regret','mean_action_gap'):
    a=np.array([r[metric] for r in rr]);d[metric]=float(a.mean());d['se_'+metric]=float(a.std(ddof=1)/math.sqrt(len(a)))
   fg=np.array([r['first_divergence_gap'] for r in rr]);d['mean_first_divergence_gap']=float(np.nanmean(fg));summary.append(d)
 # primary paired seed endpoint
 by={(int(r['seed']),r['method'],r['condition']):r for r in rows};diff=[];base=[];ctrl=[]
 for s in seeds:
  b=np.mean([by[(s,'noise_aware_mse',c)]['stabilization_loss'] for c in ('moderate','strong')])
  q=np.mean([by[(s,'control_aware_rank',c)]['stabilization_loss'] for c in ('moderate','strong')])
  base.append(b);ctrl.append(q);diff.append(q-b)
 mb=float(np.mean(base));mc=float(np.mean(ctrl));md=float(np.mean(diff));ci=bootstrap_ci(diff,cfg['bootstrap_resamples'],cfg['bootstrap_seed']);rel=(mb-mc)/mb
 success=bool(ci[1]<0 and rel>=.10)
 primary={'experiment':'E-060C1','n_seeds':len(seeds),'noise_aware_mse_mean':mb,'control_aware_rank_mean':mc,'paired_delta_control_minus_mse':md,'ci95':ci,'relative_reduction':float(rel),'minimum_relative_reduction':.10,'primary_success':success,'endpoint':cfg['primary_endpoint']}
 OUT.mkdir(exist_ok=True)
 with (OUT/'e060_confirmatory_summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 with (OUT/'e060_confirmatory_raw.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/'e060_primary_result.json').write_text(json.dumps(primary,indent=2))
 (OUT/'run.log').write_text(json.dumps({'primary':primary,'summary':summary},indent=2))
 print(json.dumps(primary,indent=2))
if __name__=='__main__':main()
