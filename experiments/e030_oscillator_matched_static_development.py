#!/usr/bin/env python3
"""E-030D1: cross-substrate matched-static development on nonlinear oscillators.

Development-only.  The physical substrate is a project-original damped, coupled
Duffing oscillator network.  MSE and control-aware models share architecture,
initialization, field pool, optimizer settings and update budget.  Checkpoint
pairing uses static calibration metrics only.  Development closed-loop outcomes
are examined only after pair selection and may be used to choose the ranking
weight before the confirmatory protocol is frozen.
"""
from __future__ import annotations
import argparse, copy, csv, json, math, time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F

from piha.substrates import NonlinearOscillatorOracle
from piha.training import set_seed
from piha.viability import viability_torch
from e010_matched_static_development import (
    CandidateDataset, collect_candidate_dataset, make_field_pool,
    regression_metrics, action_metrics, select_static_matched_pair,
    evaluate_closed_loop,
)


def new_model():
    return NonlinearOscillatorOracle(d=3, oscillators=24, integration_steps=14, dt=0.055)


def train_family(strategy: str, initial_state: Dict[str, torch.Tensor], x_field, y_field,
                 rank_ds: CandidateDataset, x_cal, y_cal, cand_cal: CandidateDataset,
                 *, steps: int, checkpoint_every: int, seed: int,
                 rank_weight: float, rank_temperature: float = 0.025) -> List[Dict]:
    set_seed(seed)
    model = new_model(); model.load_state_dict(copy.deepcopy(initial_state))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed + 1701)
    out=[]
    for step in range(1, steps+1):
        idx=torch.randint(0,len(x_field),(1024,),generator=gen)
        field=F.mse_loss(model(x_field[idx]),y_field[idx])
        loss=field
        if strategy=='control_aware':
            ir=torch.randint(0,len(rank_ds.cand),(256,),generator=gen)
            cand=rank_ds.cand[ir]; bonus=rank_ds.bonus[ir]; target=rank_ds.target[ir]
            n,a,d=cand.shape
            scores=model(cand.reshape(n*a,d)).reshape(n,a)-bonus
            rank=F.cross_entropy(-scores/rank_temperature,target)
            loss=field+rank_weight*rank
        elif strategy!='mse':
            raise ValueError(strategy)
        loss.backward(); opt.step(); opt.zero_grad()
        if step % checkpoint_every == 0:
            gm=regression_metrics(model,x_cal,y_cal); am=action_metrics(model,cand_cal)
            out.append({'strategy':strategy,'step':step,'rank_weight':rank_weight,
                        **gm,**am,'state':copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})})
    return out


def save_metrics(path: Path, fam: List[Dict]):
    fields=['strategy','step','rank_weight','mse','rmse','mae','r2','candidate_rmse','candidate_mae','cal_action_agreement']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for ck in fam: w.writerow({k:ck[k] for k in fields})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=2400)
    ap.add_argument('--checkpoint-every',type=int,default=100)
    ap.add_argument('--weights',default='0.0002,0.0005,0.0010')
    ap.add_argument('--dev-seeds',type=int,default=48)
    ap.add_argument('--out',default='results/e030_oscillator_development')
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise SystemExit(f'refusing overwrite: {out}')
    out.mkdir(parents=True,exist_ok=True)
    start=time.time()

    # Disjoint development namespaces.  Confirmatory seeds do not appear here.
    rank=collect_candidate_dataset(41000,32,200)
    cand_cal=collect_candidate_dataset(42000,20,180)
    x_field,y_field=make_field_pool(43001,24000,rank,24000)
    g=torch.Generator().manual_seed(43002)
    x_cal=torch.rand((10000,3),generator=g); y_cal=viability_torch(x_cal)

    set_seed(43003); init=new_model(); init_state=copy.deepcopy(init.state_dict())
    base=train_family('mse',init_state,x_field,y_field,rank,x_cal,y_cal,cand_cal,
                      steps=args.steps,checkpoint_every=args.checkpoint_every,seed=43004,
                      rank_weight=0.0)
    torch.save(base,out/'mse_family.pt'); save_metrics(out/'mse_metrics.csv',base)

    weights=[float(x) for x in args.weights.split(',') if x.strip()]
    dev_seeds=list(range(44000,44000+args.dev_seeds))
    candidates=[]
    for i,wgt in enumerate(weights):
        fam=train_family('control_aware',init_state,x_field,y_field,rank,x_cal,y_cal,cand_cal,
                         steps=args.steps,checkpoint_every=args.checkpoint_every,seed=43004,
                         rank_weight=wgt)
        tag=f'w{wgt:.6g}'.replace('.','p')
        torch.save(fam,out/f'control_{tag}_family.pt'); save_metrics(out/f'control_{tag}_metrics.csv',fam)
        try:
            b,c,match=select_static_matched_pair(base,fam,r2_min=0.99)
        except RuntimeError as e:
            candidates.append({'rank_weight':wgt,'eligible':False,'error':str(e)}); continue
        mb=new_model(); mb.load_state_dict(b['state']); mb.eval()
        mc=new_model(); mc.load_state_dict(c['state']); mc.eval()
        br=evaluate_closed_loop(mb,dev_seeds); cr=evaluate_closed_loop(mc,dev_seeds)
        candidates.append({
            'rank_weight':wgt,'eligible':True,'matching':match,
            'mse_checkpoint':{k:v for k,v in b.items() if k!='state'},
            'control_checkpoint':{k:v for k,v in c.items() if k!='state'},
            'mse_closed_loop':br,'control_closed_loop':cr,
            'relative_mean_D_reduction':(br['mean_D']-cr['mean_D'])/br['mean_D'],
        })
        torch.save(b['state'],out/f'selected_mse_{tag}.pt')
        torch.save(c['state'],out/f'selected_control_{tag}.pt')
        print('weight',wgt,'R2',b['r2'],c['r2'],'cand',b['candidate_rmse'],c['candidate_rmse'],
              'meanD',br['mean_D'],cr['mean_D'],'relative',(br['mean_D']-cr['mean_D'])/br['mean_D'],flush=True)

    summary={'experiment':'E-030D1','status':'DEVELOPMENT_ONLY_NOT_CONFIRMATORY',
             'substrate':'project-original damped coupled Duffing oscillator network',
             'args':vars(args),'candidates':candidates,'runtime_seconds':time.time()-start}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
