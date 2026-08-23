#!/usr/bin/env python3
"""E-051 development: action-gap stratified one-step decision-regret mechanism audit.

Uses common exact-teacher state trajectories and frozen E-020/E-031 checkpoints.
The aim is to distinguish unweighted action agreement from the consequence of a
misranking measured by exact one-step score regret.
"""
from __future__ import annotations
import copy,csv,json,math
from pathlib import Path
import numpy as np, torch
from piha.substrates import InterferometricOracle,NonlinearOscillatorOracle
from piha.dynamics import ACTIONS,TASK_GAIN,ambient_at,predicted_next,true_step
from piha.viability import viability_np
from e020_robustness_development import drifted_model as i_drift
from e031_confirmatory_shard import drifted as o_drift
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/e051_mechanism_development'

def load_i(f,name):
 m=InterferometricOracle(paths=64,detectors=16);m.load_state_dict(torch.load(f/name,weights_only=True));m.eval();return m

def load_o(f,name):
 m=NonlinearOscillatorOracle(d=3,oscillators=24,integration_steps=14,dt=.055);m.load_state_dict(torch.load(f/name,weights_only=True));m.eval();return m

def exact_states(seed:int,steps=220,lam=.02):
 rng=np.random.default_rng(seed);h=np.array([.60,.45,.15])+rng.normal(0,.015,3);rows=[]
 for t in range(steps):
  amb=ambient_at(t);cand=np.array([predicted_next(h,a,amb) for a in range(len(ACTIONS))],dtype=np.float32);demand=1+.35*np.sin(2*np.pi*t/80+.4);bonus=lam*demand*TASK_GAIN;score=viability_np(cand)-bonus;order=np.argsort(score);best=int(order[0]);gap=float(score[order[1]]-score[order[0]]);rows.append((cand,bonus,score,best,gap));h=true_step(h,best,amb,rng,shocks=True)
 return rows

def eval_interferometric(base,seed,cond,states):
 dm=i_drift(base,9_000_000+seed,cond['phase_sigma'],cond['coupling_rel']);rng=np.random.default_rng(51_000_000+seed);out=[]
 with torch.no_grad():
  for cand,bonus,true_score,best,gap in states:
   x=cand.copy()
   if cond['input_noise']: x += rng.normal(0,cond['input_noise'],size=x.shape)
   pred=dm(torch.tensor(np.clip(x,0,1),dtype=torch.float32)).cpu().numpy()
   if cond['measurement_noise']: pred += rng.normal(0,cond['measurement_noise'],size=pred.shape)
   act=int(np.argmin(np.maximum(0,pred)-bonus));reg=float(true_score[act]-true_score[best]);out.append((gap,act==best,reg))
 return out

def eval_oscillator(base,seed,cond,states):
 dm=o_drift(base,seed,cond);rng=np.random.default_rng(52_000_000+seed);out=[]
 with torch.no_grad():
  for cand,bonus,true_score,best,gap in states:
   x=cand.copy()
   if cond['input_noise']:x += rng.normal(0,cond['input_noise'],size=x.shape)
   pred=dm(torch.tensor(np.clip(x,0,1),dtype=torch.float32)).cpu().numpy()
   if cond['measurement_noise']:pred += rng.normal(0,cond['measurement_noise'],size=pred.shape)
   act=int(np.argmin(np.maximum(0,pred)-bonus));reg=float(true_score[act]-true_score[best]);out.append((gap,act==best,reg))
 return out

def summarize(rows):
 a=np.asarray(rows,dtype=float);gap=a[:,0];agree=a[:,1];reg=a[:,2];qs=np.quantile(gap,[.25,.5,.75]);bins=np.digitize(gap,qs,right=True);d={'n_decisions':len(a),'action_agreement':float(agree.mean()),'mean_regret':float(reg.mean()),'p95_regret':float(np.quantile(reg,.95)),'mean_gap':float(gap.mean())}
 for b in range(4):
  m=bins==b;d[f'q{b+1}_n']=int(m.sum());d[f'q{b+1}_agreement']=float(agree[m].mean());d[f'q{b+1}_regret']=float(reg[m].mean())
 return d

def main():
 OUT.mkdir(parents=True,exist_ok=True);allrows=[];summary=[]
 configs=[]
 f=ROOT/'frozen/e020';conds={c['name']:c for c in json.loads((f/'conditions.json').read_text())};configs.append(('E020','boundary_aware',load_i(f,'boundary_aware.pt'),conds,eval_interferometric));configs.append(('E020','robust_control_aware',load_i(f,'robust_control_aware.pt'),conds,eval_interferometric))
 f=ROOT/'frozen/e031';conds2={c['name']:c for c in json.loads((f/'conditions.json').read_text())};configs.append(('E031','boundary_aware',load_o(f,'boundary_aware.pt'),conds2,eval_oscillator));configs.append(('E031','robust_control_aware',load_o(f,'robust_control_aware.pt'),conds2,eval_oscillator))
 seeds=list(range(73000,73016));state_cache={s:exact_states(s) for s in seeds}
 for exp,method,model,conds,fn in configs:
  for cname in ('moderate','strong'):
   rows=[]
   for s in seeds:rows.extend(fn(model,s,conds[cname],state_cache[s]))
   sm=summarize(rows);summary.append({'experiment':exp,'method':method,'condition':cname,**sm});print(exp,method,cname,'agree',sm['action_agreement'],'regret',sm['mean_regret'],flush=True)
 with (OUT/'e051d1_summary.csv').open('w',newline='',encoding='utf-8') as fh:
  w=csv.DictWriter(fh,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (OUT/'e051d1_meta.json').write_text(json.dumps({'status':'DEVELOPMENT_ONLY','seed_range':[seeds[0],seeds[-1]],'steps_per_seed':220,'conditions':['moderate','strong'],'state_source':'exact_teacher_common_trajectories'},indent=2))
if __name__=='__main__':main()
