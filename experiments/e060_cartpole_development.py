#!/usr/bin/env python3
"""E-060 development-only canonical cart-pole generalization study.

This file may be used to choose a training weight and deployment envelope.  Its
seeds and outcomes are not confirmatory evidence.  Confirmatory evaluation must
use a separately frozen seed namespace and immutable checkpoints/configuration.
"""
from __future__ import annotations
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from piha.cartpole import (
    CartPoleParams, candidate_afterstates, exact_candidate_scores, failed,
    finite_horizon_value, normalize_state, stage_cost, step,
)
from piha.substrates import InterferometricOracle, NonlinearOscillatorOracle
from piha.training import set_seed

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/e060_cartpole_development'
HORIZON=2
MAX_STEPS=500
FAIL_PAD=12.0

CONDITIONS={
 'nominal':dict(plant_rel=0.0,force_rel=0.0,sensor=[0,0,0,0],input_noise=0.0,measurement_noise=0.0,
                phase_sigma=0.0,coupling_rel=0.0,dynamic_sigma=0.0,transduction_rel=0.0),
 'moderate':dict(plant_rel=0.08,force_rel=0.04,sensor=[0.010,0.050,0.003,0.050],input_noise=0.004,measurement_noise=0.010,
                 phase_sigma=0.025,coupling_rel=0.005,dynamic_sigma=0.012,transduction_rel=0.008),
 'strong':dict(plant_rel=0.15,force_rel=0.08,sensor=[0.020,0.080,0.006,0.080],input_noise=0.008,measurement_noise=0.020,
               phase_sigma=0.060,coupling_rel=0.010,dynamic_sigma=0.025,transduction_rel=0.016),
}


def sample_states(n:int, seed:int):
    r=np.random.default_rng(seed); n1=int(.7*n); n2=n-n1
    a=np.column_stack([r.uniform(-1.6,1.6,n1),r.uniform(-2.0,2.0,n1),r.uniform(-.15,.15,n1),r.uniform(-2.2,2.2,n1)])
    b=np.column_stack([r.uniform(-2.25,2.25,n2),r.uniform(-3.0,3.0,n2),r.uniform(-.23,.23,n2),r.uniform(-3.2,3.2,n2)])
    x=np.concatenate([a,b]);r.shuffle(x);return x


def collect_rank_states(n_seeds:int=32, steps_per_seed:int=160, seed0:int=61000):
    cand=[];scores=[];targets=[];gaps=[]
    for j in range(n_seeds):
        r=np.random.default_rng(seed0+j)
        s=np.array([r.uniform(-.45,.45),r.uniform(-.55,.55),r.uniform(-.075,.075),r.uniform(-.55,.55)])
        for _ in range(steps_per_seed):
            c=candidate_afterstates(s)
            q=exact_candidate_scores(s,horizon=HORIZON)
            order=np.argsort(q);cand.append(normalize_state(c));scores.append(q);targets.append(int(order[0]));gaps.append(float(q[order[1]]-q[order[0]]))
            s=step(s,int(order[0]))
            # Occasional bounded state perturbation keeps the candidate pool broad
            # without redefining the deterministic reference dynamics.
            if r.random()<.025:
                s=s+r.normal(0,[.025,.10,.008,.10])
            if failed(s):
                s=np.array([r.uniform(-.25,.25),r.uniform(-.3,.3),r.uniform(-.04,.04),r.uniform(-.3,.3)])
    return (torch.tensor(np.asarray(cand),dtype=torch.float32),
            torch.tensor(np.asarray(scores),dtype=torch.float32),
            torch.tensor(np.asarray(targets),dtype=torch.long),
            torch.tensor(np.asarray(gaps),dtype=torch.float32))


def regression_metrics(model,x,y):
    with torch.no_grad():
        p=model(x);e=p-y;mse=float(e.square().mean());rmse=math.sqrt(mse);mae=float(e.abs().mean());
        den=float((y-y.mean()).square().sum());r2=1-float(e.square().sum())/den
    return dict(mse=mse,rmse=rmse,mae=mae,r2=r2)


def candidate_metrics(model,cand,scores,target):
    with torch.no_grad():
        n,a,d=cand.shape;p=model(cand.reshape(n*a,d)).reshape(n,a);rmse=float((p-scores).square().mean().sqrt());agree=float((p.argmin(1)==target).float().mean())
    return dict(candidate_rmse=rmse,action_agreement=agree)


def make_model(substrate:str):
    if substrate=='interferometric':return InterferometricOracle(d=4,paths=64,detectors=16)
    if substrate=='oscillator':return NonlinearOscillatorOracle(d=4,oscillators=24,integration_steps=14,dt=.055)
    raise ValueError(substrate)


def train(substrate:str,method:str,rank_weight:float,steps:int=1800):
    torch.set_num_threads(1)
    # Immutable development datasets shared between methods within a substrate.
    field=sample_states(18000,60001);cal=sample_states(6000,60002)
    y=finite_horizon_value(field,horizon=HORIZON);yc=finite_horizon_value(cal,horizon=HORIZON)
    x=torch.tensor(normalize_state(field),dtype=torch.float32);yt=torch.tensor(y,dtype=torch.float32)
    xc=torch.tensor(normalize_state(cal),dtype=torch.float32);yct=torch.tensor(yc,dtype=torch.float32)
    cand,scores,target,gaps=collect_rank_states()
    set_seed(60601 if substrate=='interferometric' else 60602);init=make_model(substrate);state=copy.deepcopy(init.state_dict())
    set_seed(60611);m=make_model(substrate);m.load_state_dict(state)
    opt=torch.optim.Adam(m.parameters(),lr=2.5e-3,weight_decay=1e-5);gen=torch.Generator().manual_seed(60612)
    start=time.time()
    for st in range(steps):
        idx=torch.randint(0,len(x),(768,),generator=gen);pred=m(x[idx]);field_loss=F.mse_loss(pred,yt[idx])
        ir=torch.randint(0,len(cand),(192,),generator=gen);cb=cand[ir];sb=scores[ir];tb=target[ir]
        noise=.006*torch.randn(cb.shape,generator=gen);noisy=torch.clamp(cb+noise,0,1);n,a,d=noisy.shape;pp=m(noisy.reshape(n*a,d)).reshape(n,a)
        if method=='noise_mse':
            aux=F.mse_loss(pp,sb)
            loss=field_loss+.35*aux
        elif method=='control_rank':
            # Same noisy candidate states as the control baseline; only the
            # candidate objective changes from pointwise MSE to ordering.
            aux=F.cross_entropy(-pp/.035,tb)
            loss=field_loss+rank_weight*aux
        else:raise ValueError(method)
        loss.backward();opt.step();opt.zero_grad()
    return m,{**regression_metrics(m,xc,yct),**candidate_metrics(m,cand,scores,target),'runtime_seconds':time.time()-start}


def drifted(model,substrate,seed,cond):
    m=copy.deepcopy(model);r=np.random.default_rng(7_060_000+seed)
    with torch.no_grad():
        if substrate=='interferometric':
            if cond['phase_sigma']:m.b.add_(torch.tensor(r.normal(0,cond['phase_sigma'],size=tuple(m.b.shape)),dtype=m.b.dtype))
            if cond['coupling_rel']:
                for p in (m.cre,m.cim,m.w):p.mul_(1+torch.tensor(r.normal(0,cond['coupling_rel'],size=tuple(p.shape)),dtype=p.dtype))
        else:
            if cond['dynamic_sigma']:
                for p in (m.raw_omega,m.raw_gamma,m.raw_alpha):p.add_(torch.tensor(r.normal(0,cond['dynamic_sigma'],size=tuple(p.shape)),dtype=p.dtype))
                m.raw_coupling.add_(float(r.normal(0,cond['dynamic_sigma'])))
            if cond['transduction_rel']:
                for p in (m.force,m.readout):p.mul_(1+torch.tensor(r.normal(0,cond['transduction_rel'],size=tuple(p.shape)),dtype=p.dtype))
                m.force_bias.add_(torch.tensor(r.normal(0,cond['transduction_rel'],size=tuple(m.force_bias.shape)),dtype=m.force_bias.dtype))
    m.eval();return m


def plant_params(seed,cond):
    r=np.random.default_rng(8_060_000+seed);b=CartPoleParams();pr=cond['plant_rel'];fr=cond['force_rel']
    return CartPoleParams(gravity=b.gravity,masscart=b.masscart,
                          masspole=b.masspole*max(.55,1+r.normal(0,pr)),
                          length=b.length*max(.55,1+r.normal(0,pr)),
                          force_mag=b.force_mag*max(.65,1+r.normal(0,fr)),tau=b.tau)


def episode(model,substrate,seed,cond):
    r=np.random.default_rng(seed);p=plant_params(seed,cond);m=drifted(model,substrate,seed,cond)
    s=np.array([r.uniform(-.35,.35),r.uniform(-.40,.40),r.uniform(-.060,.060),r.uniform(-.40,.40)])
    costs=[];agreements=[];regrets=[];gaps=[];first_div=None
    sensor=np.asarray(cond['sensor'])
    with torch.no_grad():
        for t in range(MAX_STEPS):
            obs=s+r.normal(0,sensor);c=candidate_afterstates(obs);q=exact_candidate_scores(obs,horizon=HORIZON);order=np.argsort(q);best=int(order[0]);gap=float(q[order[1]]-q[order[0]])
            zin=normalize_state(c)
            if cond['input_noise']:zin=np.clip(zin+r.normal(0,cond['input_noise'],size=zin.shape),0,1)
            pred=m(torch.tensor(zin,dtype=torch.float32)).cpu().numpy()
            if cond['measurement_noise']:pred=pred+r.normal(0,cond['measurement_noise'],size=pred.shape)
            act=int(np.argmin(pred));agreements.append(act==best);regrets.append(float(q[act]-q[best]));gaps.append(gap)
            if act!=best and first_div is None:first_div=(t,gap)
            s=step(s,act,p);costs.append(float(stage_cost(s)))
            if failed(s):break
    n=len(costs);loss=(sum(costs)+(MAX_STEPS-n)*FAIL_PAD)/MAX_STEPS
    return dict(stabilization_loss=float(loss),survival_steps=n,success_500=float(n==MAX_STEPS),
                action_agreement=float(np.mean(agreements)),mean_decision_regret=float(np.mean(regrets)),
                mean_action_gap=float(np.mean(gaps)),first_divergence_step=(-1 if first_div is None else first_div[0]),
                first_divergence_gap=(float('nan') if first_div is None else first_div[1]))


def evaluate(models,substrate,seeds):
    rows=[]
    for method,m in models.items():
        for cname,cond in CONDITIONS.items():
            vals=[episode(m,substrate,s,cond) for s in seeds]
            row={'substrate':substrate,'method':method,'condition':cname,'n':len(vals)}
            for k in vals[0]:
                a=np.asarray([v[k] for v in vals],dtype=float);row[k]=float(np.nanmean(a));row['se_'+k]=float(np.nanstd(a,ddof=1)/math.sqrt(len(a)))
            rows.append(row);print(substrate,method,cname,'loss',row['stabilization_loss'],'survival',row['survival_steps'],'reg',row['mean_decision_regret'],flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--substrate',choices=['interferometric','oscillator','both'],default='both');ap.add_argument('--rank-weight',type=float,default=.010);ap.add_argument('--steps',type=int,default=1800);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);subs=['interferometric','oscillator'] if args.substrate=='both' else [args.substrate]
    allsum=[]
    for sub in subs:
        models={};metrics={}
        for method in ('noise_mse','control_rank'):
            m,met=train(sub,method,args.rank_weight,args.steps);models[method]=m;metrics[method]=met
            torch.save(m.state_dict(),OUT/f'{sub}_{method}_w{args.rank_weight:.4f}.pt')
        rows=evaluate(models,sub,list(range(76000,76024)));allsum.extend(rows)
        (OUT/f'{sub}_training_metrics_w{args.rank_weight:.4f}.json').write_text(json.dumps(metrics,indent=2))
    import csv
    with (OUT/f'development_summary_w{args.rank_weight:.4f}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(allsum[0]));w.writeheader();w.writerows(allsum)
    (OUT/f'development_meta_w{args.rank_weight:.4f}.json').write_text(json.dumps({'status':'DEVELOPMENT_ONLY_NOT_CONFIRMATORY','rank_weight':args.rank_weight,'training_steps':args.steps,'horizon':HORIZON,'evaluation_seeds':[76000,76023],'conditions':CONDITIONS},indent=2))

if __name__=='__main__':main()
