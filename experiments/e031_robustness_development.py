#!/usr/bin/env python3
"""E-031D3 development-only oscillator robustness comparison."""
from __future__ import annotations
import copy,csv,json,math,multiprocessing as mp,os
from pathlib import Path
import numpy as np
import torch
from piha.substrates import NonlinearOscillatorOracle
from piha.dynamics import ACTIONS,TASK_GAIN,ambient_at,predicted_next,true_step
from piha.viability import viability_np

CONDITIONS=(
 {'name':'nominal','measurement_noise':0.0,'input_noise':0.0,'dynamic_sigma':0.0,'transduction_rel':0.0},
 {'name':'mild','measurement_noise':0.005,'input_noise':0.003,'dynamic_sigma':0.01,'transduction_rel':0.002},
 {'name':'moderate','measurement_noise':0.010,'input_noise':0.006,'dynamic_sigma':0.03,'transduction_rel':0.005},
 {'name':'strong','measurement_noise':0.020,'input_noise':0.012,'dynamic_sigma':0.08,'transduction_rel':0.010},
)
METHODS=(
 ('boundary_aware','boundary_warm_f1000.pt'),
 ('robust_w001','robust_w0p001_warm.pt'),
 ('robust_w002','robust_w0p002_warm.pt'),
 ('robust_w004','robust_w0p004_warm.pt'),
)
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/e031_oscillator_robust_development';_G={}
def new_model():return NonlinearOscillatorOracle(d=3,oscillators=24,integration_steps=14,dt=.055)
def init_worker():
 torch.set_num_threads(1);mods={}
 for label,file in METHODS:
  m=new_model();m.load_state_dict(torch.load(SRC/file,weights_only=True));m.eval();mods[label]=m
 _G['models']=mods

def drifted(base,seed,cond):
 m=copy.deepcopy(base);rng=np.random.default_rng(9_310_000+seed)
 with torch.no_grad():
  ds=cond['dynamic_sigma'];tr=cond['transduction_rel']
  if ds:
   for p in (m.raw_omega,m.raw_gamma,m.raw_alpha):p.add_(torch.tensor(rng.normal(0,ds,size=tuple(p.shape)),dtype=p.dtype))
   m.raw_coupling.add_(float(rng.normal(0,ds)))
  if tr:
   for p in (m.force,m.readout):
    z=torch.tensor(rng.normal(0,tr,size=tuple(p.shape)),dtype=p.dtype);p.mul_(1+z)
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
def work(x):
 method,cname,seed=x;cond=next(c for c in CONDITIONS if c['name']==cname);return {'method':method,'condition':cname,'seed':seed,**episode(_G['models'][method],seed,cond)}
def main():
 out=SRC/'robustness_d3';out.mkdir(parents=True,exist_ok=True);seeds=list(range(64000,64016));tasks=[(m,c['name'],s) for m,_ in METHODS for c in CONDITIONS for s in seeds]
 ctx=mp.get_context('fork')
 with ctx.Pool(processes=min(4,os.cpu_count() or 1),initializer=init_worker) as pool:raw=pool.map(work,tasks,chunksize=2)
 fields=list(raw[0]);
 with (out/'raw.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(raw)
 rows=[]
 for m,_ in METHODS:
  for c in CONDITIONS:
   rr=[r for r in raw if r['method']==m and r['condition']==c['name']];row={'method':m,'condition':c['name']}
   for metric in ['mean_D','p95_D','viability_occupancy','severe_fraction','action_agreement','cumulative_task']:
    a=np.array([r[metric] for r in rr]);row[metric]=float(a.mean());row['se_'+metric]=float(a.std(ddof=1)/math.sqrt(len(a)))
   rows.append(row);print(m,c['name'],'meanD',row['mean_D'],'agree',row['action_agreement'],flush=True)
 with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'conditions.json').write_text(json.dumps(CONDITIONS,indent=2))
 # stressed aggregate for development selection
 agg={}
 for m,_ in METHODS:
  vals=[r['mean_D'] for r in rows if r['method']==m and r['condition'] in ('moderate','strong')];agg[m]=float(np.mean(vals))
 (out/'stressed_aggregate.json').write_text(json.dumps({'status':'DEVELOPMENT_ONLY','moderate_strong_mean_D':agg},indent=2));print('stress',json.dumps(agg,indent=2))
if __name__=='__main__':main()
