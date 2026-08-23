#!/usr/bin/env python3
"""E-020 development comparison of robustness-oriented training baselines.

Exploratory only. Same architecture, initialization, field pool, optimizer budget.
No external research source code is incorporated; the sharpness-aware baseline is
an independent implementation of a two-pass local parameter perturbation method.
"""
from __future__ import annotations
import copy, csv, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from piha.substrates import InterferometricOracle
from piha.training import set_seed
from piha.viability import viability_torch
from piha.evaluation import run_episode
from e010_matched_static_development import collect_candidate_dataset, make_field_pool, regression_metrics

METHODS = ("mse", "noise_aware", "boundary_aware", "sharpness_aware", "control_aware")


def train(method, init_state, x, y, rank_ds, steps=1200, seed=8111):
    set_seed(seed)
    m=InterferometricOracle(paths=64,detectors=16); m.load_state_dict(copy.deepcopy(init_state))
    opt=torch.optim.Adam(m.parameters(),lr=3e-3,weight_decay=1e-5)
    gen=torch.Generator().manual_seed(seed+313)
    for _ in range(steps):
        idx=torch.randint(0,len(x),(1024,),generator=gen)
        xb=x[idx]; yb=y[idx]
        if method=="noise_aware":
            xb=torch.clamp(xb+0.012*torch.randn(xb.shape,generator=gen),0,1)
            yb=viability_torch(xb)
        pred=m(xb)
        if method=="boundary_aware":
            w=0.5+0.05/(0.01+yb)
            field=((pred-yb).square()*w).mean()
        else:
            field=F.mse_loss(pred,yb)

        if method=="sharpness_aware":
            # First gradient determines a small local parameter perturbation.
            field.backward()
            grads=[p.grad for p in m.parameters() if p.grad is not None]
            gnorm=torch.sqrt(sum((g.detach().square().sum() for g in grads)))+1e-12
            rho=0.015
            perturb=[]
            with torch.no_grad():
                for p in m.parameters():
                    if p.grad is None: perturb.append(None); continue
                    e=rho*p.grad/gnorm; p.add_(e); perturb.append(e)
            opt.zero_grad()
            loss2=F.mse_loss(m(xb),yb)
            loss2.backward()
            with torch.no_grad():
                for p,e in zip(m.parameters(),perturb):
                    if e is not None: p.sub_(e)
            opt.step(); opt.zero_grad(); continue

        loss=field
        if method=="control_aware":
            ir=torch.randint(0,len(rank_ds.cand),(320,),generator=gen)
            cand=rank_ds.cand[ir]; bonus=rank_ds.bonus[ir]; target=rank_ds.target[ir]
            n,a,d=cand.shape
            score=m(cand.reshape(n*a,d)).reshape(n,a)-bonus
            rank=F.cross_entropy(-score/0.025,target)
            loss=field+2e-4*rank
        loss.backward(); opt.step(); opt.zero_grad()
    return m


def evaluate_noise(model, seeds, measurement_noise, input_noise):
    vals=[]
    dummy=np.zeros(4)
    for s in seeds:
        ep=run_episode(model,dummy,"physical",s,steps=400,measurement_noise=measurement_noise,input_noise=input_noise)
        D=ep[:,0]
        vals.append([D.mean(),(D<0.05).mean(),(D>0.2).mean(),ep[:,1].sum()])
    a=np.asarray(vals)
    return {"mean_D":float(a[:,0].mean()),"viability_occupancy":float(a[:,1].mean()),
            "severe_fraction":float(a[:,2].mean()),"cumulative_task":float(a[:,3].mean())}


def main():
    out=Path("results/e020_development"); out.mkdir(parents=True,exist_ok=True)
    rank=collect_candidate_dataset(1200,32,200)
    x,y=make_field_pool(8282,22000,rank,22000)
    g=torch.Generator().manual_seed(161803)
    xcal=torch.rand((8000,3),generator=g); ycal=viability_torch(xcal)
    set_seed(4242); init=InterferometricOracle(paths=64,detectors=16); init_state=copy.deepcopy(init.state_dict())
    seeds=list(range(7000,7040))
    noise_levels=[(0.0,0.0),(0.005,0.003),(0.01,0.006),(0.02,0.012)]
    rows=[]; static={}
    for method in METHODS:
        model=train(method,init_state,x,y,rank)
        static[method]=regression_metrics(model,xcal,ycal)
        torch.save(model.state_dict(),out/f"{method}.pt")
        for mn,inn in noise_levels:
            r=evaluate_noise(model,seeds,mn,inn)
            rows.append({"method":method,"measurement_noise":mn,"input_noise":inn,**static[method],**r})
            print(method,mn,inn,r["mean_D"])
    with (out/"e020_dev_noise_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out/"e020_dev_static.json").write_text(json.dumps(static,indent=2),encoding="utf-8")

if __name__=="__main__": main()
