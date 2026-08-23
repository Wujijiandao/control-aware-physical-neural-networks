#!/usr/bin/env python3
"""Development-only robust control-aware training variants for E-020."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import torch
import torch.nn.functional as F
from piha.substrates import InterferometricOracle
from piha.training import set_seed
from piha.viability import viability_torch
from e010_matched_static_development import collect_candidate_dataset, make_field_pool, regression_metrics, action_metrics


def train(weight:float, steps:int=4000, seed:int=8111):
    rank=collect_candidate_dataset(22000,32,200); cal=collect_candidate_dataset(26000,16,180)
    x,y=make_field_pool(8282,24000,rank,24000)
    gcal=torch.Generator().manual_seed(161803); xc=torch.rand((10000,3),generator=gcal); yc=viability_torch(xc)
    set_seed(4242); init=InterferometricOracle(paths=64,detectors=16); init_state=copy.deepcopy(init.state_dict())
    set_seed(seed); m=InterferometricOracle(paths=64,detectors=16);m.load_state_dict(init_state)
    opt=torch.optim.Adam(m.parameters(),lr=3e-3,weight_decay=1e-5); gen=torch.Generator().manual_seed(seed+313)
    for _ in range(steps):
        idx=torch.randint(0,len(x),(1024,),generator=gen); xb=x[idx]; yb=y[idx]
        field=F.mse_loss(m(xb),yb)
        ir=torch.randint(0,len(rank.cand),(320,),generator=gen)
        cand=rank.cand[ir]; bonus=rank.bonus[ir]; target=rank.target[ir]; n,a,d=cand.shape
        # Perturb candidate encoding during ranking training. Additive output-noise samples
        # force the ranking loss to prefer action margins that survive readout uncertainty.
        cnoise=0.006*torch.randn(cand.shape,generator=gen)
        noisy_cand=torch.clamp(cand+cnoise,0,1)
        score=m(noisy_cand.reshape(n*a,d)).reshape(n,a)-bonus
        score=score+0.010*torch.randn(score.shape,generator=gen)
        rank_loss=F.cross_entropy(-score/0.025,target)
        loss=field+weight*rank_loss
        loss.backward();opt.step();opt.zero_grad()
    return m,regression_metrics(m,xc,yc),action_metrics(m,cal)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--weight',type=float,required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    m,gm,am=train(args.weight)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),out)
    summary={'status':'DEVELOPMENT_ONLY','weight':args.weight,'global':gm,'candidate':am,'checkpoint':str(out)}
    out.with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
