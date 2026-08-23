#!/usr/bin/env python3
"""Train one E-020 method and persist its full calibration checkpoint family.
Development only; no closed-loop evaluation is performed here.
"""
from __future__ import annotations
import argparse, copy, csv, json
from pathlib import Path
import torch
from piha.substrates import InterferometricOracle
from piha.training import set_seed
from piha.viability import viability_torch
from e010_matched_static_development import collect_candidate_dataset, make_field_pool
from e020_checkpoint_development import METHODS, train_family


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('method', choices=METHODS)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--checkpoint-every', type=int, default=100)
    ap.add_argument('--out', default='results/e020_checkpoint_families')
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    family_file=out/f'{args.method}_family.pt'
    if family_file.exists():
        raise SystemExit(f'refusing overwrite: {family_file}')

    train_rank=collect_candidate_dataset(22000,32,200)
    cand_cal=collect_candidate_dataset(26000,16,180)
    x_field,y_field=make_field_pool(8282,24000,train_rank,24000)
    g=torch.Generator().manual_seed(161803)
    x_cal=torch.rand((10000,3),generator=g); y_cal=viability_torch(x_cal)
    set_seed(4242); init=InterferometricOracle(paths=64,detectors=16)
    init_state=copy.deepcopy(init.state_dict())

    fam=train_family(args.method,init_state,x_field,y_field,train_rank,x_cal,y_cal,cand_cal,
                     steps=args.steps,checkpoint_every=args.checkpoint_every,seed=8111)
    torch.save(fam,family_file)
    fields=['method','step','mse','rmse','mae','r2','candidate_rmse','candidate_mae','cal_action_agreement']
    with (out/f'{args.method}_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for ck in fam: w.writerow({k:ck[k] for k in fields})
    summary={'status':'DEVELOPMENT_ONLY','method':args.method,'steps':args.steps,
             'checkpoint_every':args.checkpoint_every,'final':{k:v for k,v in fam[-1].items() if k!='state'}}
    (out/f'{args.method}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
