#!/usr/bin/env python3
"""E-051C1 frozen confirmatory decision-regret shard runner."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,multiprocessing as mp,os
from pathlib import Path
import numpy as np,torch
from piha.substrates import InterferometricOracle,NonlinearOscillatorOracle
from piha.dynamics import ACTIONS,TASK_GAIN,ambient_at,predicted_next,true_step
from piha.viability import viability_np
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'frozen/e051';OUT=ROOT/'results/e051_confirmatory';_G={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
 for line in (F/'FREEZE_MANIFEST.sha256').read_text().splitlines():
  if not line.strip():continue
  h,n=line.split('  ',1)
  if sha(F/n)!=h:raise RuntimeError(f'hash mismatch {n}')
def new_i():return InterferometricOracle(paths=64,detectors=16)
def new_o():return NonlinearOscillatorOracle(d=3,oscillators=24,integration_steps=14,dt=.055)
def init_worker():
 torch.set_num_threads(1);mods={}
 for key,file,make in [('e020_boundary','e020_boundary.pt',new_i),('e020_robust','e020_robust.pt',new_i),('e031_boundary','e031_boundary.pt',new_o),('e031_robust','e031_robust.pt',new_o)]:
  m=make();m.load_state_dict(torch.load(F/file,weights_only=True));m.eval();mods[key]=m
 _G['models']=mods;_G['c20']={c['name']:c for c in json.loads((F/'e020_conditions.json').read_text())};_G['c31']={c['name']:c for c in json.loads((F/'e031_conditions.json').read_text())}
def exact_pool(seed,steps=220,lam=.02):
 rng=np.random.default_rng(seed);h=np.array([.60,.45,.15])+rng.normal(0,.015,3);C=[];B=[];S=[];best=[];gap=[]
 for t in range(steps):
  amb=ambient_at(t);cand=np.array([predicted_next(h,a,amb) for a in range(len(ACTIONS))],dtype=np.float32);demand=1+.35*np.sin(2*np.pi*t/80+.4);bonus=lam*demand*TASK_GAIN;score=viability_np(cand)-bonus;order=np.argsort(score);a=int(order[0]);C.append(cand);B.append(bonus);S.append(score);best.append(a);gap.append(float(score[order[1]]-score[order[0]]));h=true_step(h,a,amb,rng,shocks=True)
 return np.asarray(C),np.asarray(B),np.asarray(S),np.asarray(best),np.asarray(gap)
def drift_i(base,seed,c):
 m=copy.deepcopy(base);rng=np.random.default_rng(9_000_000+seed)
 with torch.no_grad():
  if c['phase_sigma']:m.b.add_(torch.tensor(rng.normal(0,c['phase_sigma'],size=tuple(m.b.shape)),dtype=m.b.dtype))
  if c['coupling_rel']:
   for p in (m.cre,m.cim,m.w):p.mul_(1+torch.tensor(rng.normal(0,c['coupling_rel'],size=tuple(p.shape)),dtype=p.dtype))
 return m
def drift_o(base,seed,c):
 m=copy.deepcopy(base);rng=np.random.default_rng(9_310_000+seed)
 with torch.no_grad():
  ds=c['dynamic_sigma'];tr=c['transduction_rel']
  if ds:
   for p in (m.raw_omega,m.raw_gamma,m.raw_alpha):p.add_(torch.tensor(rng.normal(0,ds,size=tuple(p.shape)),dtype=p.dtype))
   m.raw_coupling.add_(float(rng.normal(0,ds)))
  if tr:
   for p in (m.force,m.readout):p.mul_(1+torch.tensor(rng.normal(0,tr,size=tuple(p.shape)),dtype=p.dtype))
   m.force_bias.add_(torch.tensor(rng.normal(0,tr,size=tuple(m.force_bias.shape)),dtype=m.force_bias.dtype))
 return m
@torch.no_grad()
def metrics(base,seed,c,C,B,S,best,gap,substrate):
 if substrate=='E020':m=drift_i(base,seed,c);rng=np.random.default_rng(51_000_000+seed)
 else:m=drift_o(base,seed,c);rng=np.random.default_rng(52_000_000+seed)
 x=C.copy()
 if c['input_noise']:x+=rng.normal(0,c['input_noise'],size=x.shape)
 shp=x.shape;pred=m(torch.tensor(np.clip(x,0,1).reshape(-1,3),dtype=torch.float32)).cpu().numpy().reshape(shp[:2])
 if c['measurement_noise']:pred+=rng.normal(0,c['measurement_noise'],size=pred.shape)
 acts=np.argmin(np.maximum(0,pred)-B,axis=1);idx=np.arange(len(acts));reg=S[idx,acts]-S[idx,best];agree=(acts==best)
 qs=np.quantile(gap,[.25,.5,.75]);bins=np.digitize(gap,qs,right=True);out={'mean_regret':float(reg.mean()),'p95_regret':float(np.quantile(reg,.95)),'action_agreement':float(agree.mean()),'mean_gap':float(gap.mean())}
 for q in range(4):
  z=bins==q;out[f'q{q+1}_regret']=float(reg[z].mean());out[f'q{q+1}_agreement']=float(agree[z].mean())
 return out
def work(seed):
 C,B,S,best,gap=exact_pool(seed);rows=[]
 specs=[('E020','boundary_aware',_G['models']['e020_boundary'],_G['c20']),('E020','robust_control_aware',_G['models']['e020_robust'],_G['c20']),('E031','boundary_aware',_G['models']['e031_boundary'],_G['c31']),('E031','robust_control_aware',_G['models']['e031_robust'],_G['c31'])]
 for exp,method,m,conds in specs:
  for cname in ('moderate','strong'):rows.append({'seed':seed,'experiment':exp,'method':method,'condition':cname,**metrics(m,seed,conds[cname],C,B,S,best,gap,exp)})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard-index',type=int,required=True);ap.add_argument('--num-shards',type=int,default=4);args=ap.parse_args();verify();seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()];ss=seeds[args.shard_index::args.num_shards];sd=OUT/'shards';sd.mkdir(parents=True,exist_ok=True);p=sd/f'shard_{args.shard_index:02d}_of_{args.num_shards:02d}.csv'
 if p.exists():raise SystemExit(f'refusing overwrite {p}')
 ctx=mp.get_context('fork')
 with ctx.Pool(processes=min(4,os.cpu_count() or 1),initializer=init_worker) as pool:nested=pool.map(work,ss,chunksize=1)
 rows=[r for rr in nested for r in rr];fields=list(rows[0])
 with p.open('w',newline='',encoding='utf-8') as fh:w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(rows)
 meta={'experiment':'E-051C1','shard_index':args.shard_index,'num_shards':args.num_shards,'n_seeds':len(ss),'seed_sha256':sha(F/'confirmatory_seeds.txt'),'csv_sha256':sha(p)};p.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
