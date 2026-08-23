#!/usr/bin/env python3
"""E-031C1 frozen confirmatory primary, deterministic shard runner."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,multiprocessing as mp,os
from pathlib import Path
import numpy as np
import torch
from piha.substrates import NonlinearOscillatorOracle
from piha.dynamics import ACTIONS,TASK_GAIN,ambient_at,predicted_next,true_step
from piha.viability import viability_np
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'frozen/e031';OUT=ROOT/'results/e031_confirmatory';_G={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
 for line in (F/'FREEZE_MANIFEST.sha256').read_text().splitlines():
  if not line.strip():continue
  h,n=line.split('  ',1)
  if sha(F/n)!=h:raise RuntimeError(f'hash mismatch {n}')
def new():return NonlinearOscillatorOracle(d=3,oscillators=24,integration_steps=14,dt=.055)
def init_worker():
 torch.set_num_threads(1);mods={}
 for label,file in [('boundary_aware','boundary_aware.pt'),('robust_control_aware','robust_control_aware.pt')]:
  m=new();m.load_state_dict(torch.load(F/file,weights_only=True));m.eval();mods[label]=m
 _G['models']=mods;_G['conds']={c['name']:c for c in json.loads((F/'conditions.json').read_text())}
def drifted(base,seed,cond):
 m=copy.deepcopy(base);rng=np.random.default_rng(9_310_000+seed)
 with torch.no_grad():
  ds=cond['dynamic_sigma'];tr=cond['transduction_rel']
  if ds:
   for p in (m.raw_omega,m.raw_gamma,m.raw_alpha):p.add_(torch.tensor(rng.normal(0,ds,size=tuple(p.shape)),dtype=p.dtype))
   m.raw_coupling.add_(float(rng.normal(0,ds)))
  if tr:
   for p in (m.force,m.readout):p.mul_(1+torch.tensor(rng.normal(0,tr,size=tuple(p.shape)),dtype=p.dtype))
   m.force_bias.add_(torch.tensor(rng.normal(0,tr,size=tuple(m.force_bias.shape)),dtype=m.force_bias.dtype))
 return m
@torch.no_grad()
def episode(base,seed,cond,steps=400,lam=.02):
 rng=np.random.default_rng(seed);m=drifted(base,seed,cond);h=np.array([.60,.45,.15])+rng.normal(0,.015,3);ds=[];ag=[];vi=[];sev=[];task=[]
 for t in range(steps):
  amb=ambient_at(t);cand=np.array([predicted_next(h,a,amb) for a in range(len(ACTIONS))],dtype=np.float32);demand=1+.35*np.sin(2*np.pi*t/80+.4);bonus=lam*demand*TASK_GAIN
  exact=int(np.argmin(viability_np(cand)-bonus));xin=cand.copy()
  if cond['input_noise']:xin+=rng.normal(0,cond['input_noise'],size=xin.shape)
  pred=m(torch.tensor(np.clip(xin,0,1),dtype=torch.float32)).cpu().numpy()
  if cond['measurement_noise']:pred+=rng.normal(0,cond['measurement_noise'],size=pred.shape)
  act=int(np.argmin(np.maximum(0,pred)-bonus));ag.append(act==exact);h=true_step(h,act,amb,rng,shocks=True);d=float(viability_np(h));ds.append(d);vi.append(d<.05);sev.append(d>.2);task.append(float(TASK_GAIN[act]*demand))
 return {'mean_D':float(np.mean(ds)),'p95_D':float(np.quantile(ds,.95)),'viability_occupancy':float(np.mean(vi)),'severe_fraction':float(np.mean(sev)),'action_agreement':float(np.mean(ag)),'cumulative_task':float(np.sum(task))}
def work(seed):
 rows=[]
 for method,m in _G['models'].items():
  for cname in ('moderate','strong'):rows.append({'seed':seed,'method':method,'condition':cname,**episode(m,seed,_G['conds'][cname])})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard-index',type=int,required=True);ap.add_argument('--num-shards',type=int,default=8);args=ap.parse_args();verify();seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()];ss=seeds[args.shard_index::args.num_shards];sd=OUT/'shards';sd.mkdir(parents=True,exist_ok=True);p=sd/f'shard_{args.shard_index:02d}_of_{args.num_shards:02d}.csv'
 if p.exists():raise SystemExit(f'refusing overwrite {p}')
 ctx=mp.get_context('fork')
 with ctx.Pool(processes=min(4,os.cpu_count() or 1),initializer=init_worker) as pool:nested=pool.map(work,ss,chunksize=1)
 rows=[r for rr in nested for r in rr];fields=list(rows[0])
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 meta={'experiment':'E-031C1','shard_index':args.shard_index,'num_shards':args.num_shards,'n_seeds':len(ss),'seed_sha256':sha(F/'confirmatory_seeds.txt'),'csv_sha256':sha(p)};p.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
