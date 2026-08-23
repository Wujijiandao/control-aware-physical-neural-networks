#!/usr/bin/env python3
"""E-031D2 strong-baseline schedule: common MSE warm-up then method-specific fine-tune."""
from __future__ import annotations
import argparse, copy, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from piha.substrates import NonlinearOscillatorOracle
from piha.training import set_seed
from piha.viability import viability_torch
from e010_matched_static_development import collect_candidate_dataset, make_field_pool, regression_metrics, action_metrics

def model():return NonlinearOscillatorOracle(d=3,oscillators=24,integration_steps=14,dt=.055)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('method',choices=['boundary_aware','robust_control_aware']);ap.add_argument('--weight',type=float,default=.002);ap.add_argument('--warmup',type=int,default=2200);ap.add_argument('--finetune',type=int,default=1600);ap.add_argument('--rank-batch',type=int,default=128);args=ap.parse_args();torch.set_num_threads(1)
 out=Path('results/e031_oscillator_robust_development');out.mkdir(parents=True,exist_ok=True);tag=(f'boundary_warm_f{args.finetune}' if args.method=='boundary_aware' else f'robust_w{args.weight:.6g}_warm').replace('.','p');path=out/f'{tag}.pt'
 if path.exists():raise SystemExit(f'refusing overwrite: {path}')
 rank=collect_candidate_dataset(61000,32,200);cal=collect_candidate_dataset(62000,20,180);x,y=make_field_pool(63001,24000,rank,24000);gcal=torch.Generator().manual_seed(63002);xc=torch.rand((10000,3),generator=gcal);yc=viability_torch(xc)
 set_seed(63003);init=model();st=copy.deepcopy(init.state_dict());set_seed(63004);m=model();m.load_state_dict(st);opt=torch.optim.Adam(m.parameters(),lr=3e-3,weight_decay=1e-5);gen=torch.Generator().manual_seed(63005);start=time.time()
 for _ in range(args.warmup):
  idx=torch.randint(0,len(x),(1024,),generator=gen);loss=F.mse_loss(m(x[idx]),y[idx]);loss.backward();opt.step();opt.zero_grad()
 warm={'global':regression_metrics(m,xc,yc),'candidate':action_metrics(m,cal)}
 opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-6)
 for _ in range(args.finetune):
  idx=torch.randint(0,len(x),(1024,),generator=gen);xb=x[idx];yb=y[idx];sq=(m(xb)-yb).square()
  if args.method=='boundary_aware':
   w=.7+.03/(.015+yb);w=w/w.mean().detach();loss=(sq*w).mean()
  else:
   field=sq.mean();ir=torch.randint(0,len(rank.cand),(args.rank_batch,),generator=gen);cand=rank.cand[ir];bonus=rank.bonus[ir];target=rank.target[ir];n,a,d=cand.shape
   noisy=torch.clamp(cand+.006*torch.randn(cand.shape,generator=gen),0,1);score=m(noisy.reshape(n*a,d)).reshape(n,a)-bonus+.010*torch.randn((n,a),generator=gen);rankloss=F.cross_entropy(-score/.025,target);loss=field+args.weight*rankloss
  loss.backward();opt.step();opt.zero_grad()
 gm=regression_metrics(m,xc,yc);am=action_metrics(m,cal);torch.save(m.state_dict(),path);summary={'status':'DEVELOPMENT_ONLY','schedule':{'warmup':args.warmup,'finetune':args.finetune,'rank_batch':args.rank_batch},'method':args.method,'weight':args.weight if args.method!='boundary_aware' else None,'warmup_metrics':warm,'global':gm,'candidate':am,'runtime_seconds':time.time()-start,'checkpoint':str(path)};path.with_suffix('.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
