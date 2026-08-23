#!/usr/bin/env python3
"""Recoverable E-030 development training for control-aware oscillator families."""
from __future__ import annotations
import argparse, copy, csv, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from piha.substrates import NonlinearOscillatorOracle
from piha.training import set_seed
from piha.viability import viability_torch
from e010_matched_static_development import collect_candidate_dataset, make_field_pool, regression_metrics, action_metrics

WEIGHTS=(2e-4,5e-4,1e-3)

def new_model(): return NonlinearOscillatorOracle(d=3,oscillators=24,integration_steps=14,dt=0.055)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weight',type=float,default=None); args=ap.parse_args()
    torch.set_num_threads(1)
    out=Path('results/e030_oscillator_development');out.mkdir(parents=True,exist_ok=True)
    rank=collect_candidate_dataset(41000,32,200); cal=collect_candidate_dataset(42000,20,180)
    x,y=make_field_pool(43001,24000,rank,24000)
    gcal=torch.Generator().manual_seed(43002); xc=torch.rand((10000,3),generator=gcal);yc=viability_torch(xc)
    set_seed(43003); init=new_model(); st=copy.deepcopy(init.state_dict())
    weights=(args.weight,) if args.weight is not None else WEIGHTS
    for wgt in weights:
        tag=f'w{wgt:.6g}'.replace('.','p')
        family_path=out/f'control_{tag}_family.pt'
        if family_path.exists():
            print('skip existing',family_path,flush=True);continue
        set_seed(43004);m=new_model();m.load_state_dict(st)
        opt=torch.optim.Adam(m.parameters(),lr=3e-3,weight_decay=1e-5)
        gen=torch.Generator().manual_seed(44705); fam=[]; start=time.time()
        for step in range(1,2201):
            idx=torch.randint(0,len(x),(1024,),generator=gen)
            field=F.mse_loss(m(x[idx]),y[idx])
            ir=torch.randint(0,len(rank.cand),(128,),generator=gen)
            cand=rank.cand[ir]; bonus=rank.bonus[ir]; target=rank.target[ir]; n,a,d=cand.shape
            score=m(cand.reshape(n*a,d)).reshape(n,a)-bonus
            rloss=F.cross_entropy(-score/0.025,target)
            loss=field+wgt*rloss
            loss.backward();opt.step();opt.zero_grad()
            if step%100==0:
                gm=regression_metrics(m,xc,yc);am=action_metrics(m,cal)
                fam.append({'strategy':'control_aware','rank_weight':wgt,'step':step,**gm,**am,
                            'state':copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})})
        torch.save(fam,family_path)
        fields=['strategy','rank_weight','step','mse','rmse','mae','r2','candidate_rmse','candidate_mae','cal_action_agreement']
        with (out/f'control_{tag}_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows([{k:c[k] for k in fields} for c in fam])
        (out/f'control_{tag}_summary.json').write_text(json.dumps({'status':'DEVELOPMENT_ONLY','weight':wgt,'runtime_seconds':time.time()-start,'final':{k:v for k,v in fam[-1].items() if k!='state'}},indent=2))
        print('done',wgt,'final r2',fam[-1]['r2'],'cand',fam[-1]['candidate_rmse'],'agree',fam[-1]['cal_action_agreement'],'sec',time.time()-start,flush=True)
if __name__=='__main__':main()
