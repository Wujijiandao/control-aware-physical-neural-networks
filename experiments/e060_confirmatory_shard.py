#!/usr/bin/env python3
"""Frozen E-060C1 cart-pole confirmatory shard runner."""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, multiprocessing as mp, os
from pathlib import Path
import numpy as np
import torch
from piha.cartpole import CartPoleParams, candidate_afterstates, exact_candidate_scores, failed, normalize_state, stage_cost, step
from piha.substrates import InterferometricOracle

ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'frozen/e060'; OUT=ROOT/'results/e060_confirmatory'
HORIZON=2; MAX_STEPS=500; FAIL_PAD=12.0
_G={}

def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()

def verify():
 for line in (F/'E060_FROZEN_MANIFEST.sha256').read_text().splitlines():
  expected,name=line.split(None,1);p=F/name.strip()
  if sha(p)!=expected:raise RuntimeError(f'frozen manifest mismatch: {name}')
 cfg=json.loads((F/'audit_config.json').read_text())
 if cfg['experiment']!='E-060C1' or cfg['confirmatory_n_seeds']!=128:raise RuntimeError('unexpected frozen config')

def model(path):
 m=InterferometricOracle(d=4,paths=64,detectors=16);m.load_state_dict(torch.load(path,weights_only=True));m.eval();return m

def drifted(base,seed,cond):
 m=copy.deepcopy(base);r=np.random.default_rng(7_060_000+seed)
 with torch.no_grad():
  if cond['phase_sigma']:m.b.add_(torch.tensor(r.normal(0,cond['phase_sigma'],size=tuple(m.b.shape)),dtype=m.b.dtype))
  if cond['coupling_rel']:
   for p in (m.cre,m.cim,m.w):p.mul_(1+torch.tensor(r.normal(0,cond['coupling_rel'],size=tuple(p.shape)),dtype=p.dtype))
 return m

def plant_params(seed,cond):
 r=np.random.default_rng(8_060_000+seed);b=CartPoleParams();pr=cond['plant_rel'];fr=cond['force_rel']
 return CartPoleParams(gravity=b.gravity,masscart=b.masscart,
  masspole=b.masspole*max(.55,1+r.normal(0,pr)),length=b.length*max(.55,1+r.normal(0,pr)),
  force_mag=b.force_mag*max(.65,1+r.normal(0,fr)),tau=b.tau)

def episode(base,seed,cond):
 r=np.random.default_rng(seed);p=plant_params(seed,cond);m=drifted(base,seed,cond)
 s=np.array([r.uniform(-.35,.35),r.uniform(-.40,.40),r.uniform(-.060,.060),r.uniform(-.40,.40)])
 costs=[];agree=[];reg=[];gaps=[];first=None;sensor=np.asarray(cond['sensor'])
 with torch.no_grad():
  for t in range(MAX_STEPS):
   obs=s+r.normal(0,sensor);cand=candidate_afterstates(obs);q=exact_candidate_scores(obs,horizon=HORIZON);order=np.argsort(q);best=int(order[0]);gap=float(q[order[1]]-q[order[0]])
   zin=normalize_state(cand)
   if cond['input_noise']:zin=np.clip(zin+r.normal(0,cond['input_noise'],size=zin.shape),0,1)
   pred=m(torch.tensor(zin,dtype=torch.float32)).cpu().numpy()
   if cond['measurement_noise']:pred=pred+r.normal(0,cond['measurement_noise'],size=pred.shape)
   act=int(np.argmin(pred));agree.append(act==best);reg.append(float(q[act]-q[best]));gaps.append(gap)
   if act!=best and first is None:first=(t,gap)
   s=step(s,act,p);costs.append(float(stage_cost(s)))
   if failed(s):break
 n=len(costs);loss=(sum(costs)+(MAX_STEPS-n)*FAIL_PAD)/MAX_STEPS
 return {'stabilization_loss':float(loss),'survival_steps':n,'success_500':float(n==MAX_STEPS),
  'action_agreement':float(np.mean(agree)),'mean_decision_regret':float(np.mean(reg)),
  'mean_action_gap':float(np.mean(gaps)),'first_divergence_step':(-1 if first is None else first[0]),
  'first_divergence_gap':(float('nan') if first is None else first[1])}

def init_worker():
 torch.set_num_threads(1);_G['models']={'noise_aware_mse':model(F/'noise_aware_mse.pt'),'control_aware_rank':model(F/'control_aware_rank.pt')};_G['conds']=json.loads((F/'conditions.json').read_text())

def work(seed):
 rows=[]
 for method,m in _G['models'].items():
  for cname in ('nominal','moderate','strong'):
   rows.append({'seed':seed,'method':method,'condition':cname,**episode(m,seed,_G['conds'][cname])})
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard-index',type=int,required=True);ap.add_argument('--num-shards',type=int,default=8);args=ap.parse_args();verify()
 seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()];ss=seeds[args.shard_index::args.num_shards]
 sd=OUT/'shards';sd.mkdir(parents=True,exist_ok=True);p=sd/f'shard_{args.shard_index:02d}_of_{args.num_shards:02d}.csv'
 if p.exists():raise SystemExit(f'refusing overwrite {p}')
 ctx=mp.get_context('fork')
 with ctx.Pool(processes=min(4,os.cpu_count() or 1),initializer=init_worker) as pool:nested=pool.map(work,ss,chunksize=1)
 rows=[r for rr in nested for r in rr];fields=list(rows[0])
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 meta={'experiment':'E-060C1','shard_index':args.shard_index,'num_shards':args.num_shards,'n_seeds':len(ss),'seed_sha256':sha(F/'confirmatory_seeds.txt'),'csv_sha256':sha(p)}
 p.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
