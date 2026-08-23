#!/usr/bin/env python3
"""Select a five-method E-020 checkpoint set using static calibration metrics only."""
from __future__ import annotations
import csv, json
from pathlib import Path
import torch
from e020_checkpoint_development import METHODS, select_matched_set


def main():
    src=Path('results/e020_checkpoint_families')
    out=Path('results/e020_static_selection'); out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise SystemExit('refusing overwrite of existing E-020 static selection')
    families={m:torch.load(src/f'{m}_family.pt',weights_only=False) for m in METHODS}
    selected,audit=select_matched_set(families,r2_min=0.99)
    fields=['method','step','mse','rmse','mae','r2','candidate_rmse','candidate_mae','cal_action_agreement']
    rows=[]
    for m in METHODS:
        ck=selected[m]; torch.save(ck['state'],out/f'selected_{m}.pt')
        rows.append({k:ck[k] for k in fields})
    with (out/'selected_static_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={'status':'DEVELOPMENT_ONLY_STATIC_SELECTION','audit':audit,
             'selection_uses':['global_r2','global_rmse','candidate_pointwise_rmse'],
             'selection_excludes':['cal_action_agreement','closed_loop_outcomes','perturbation_outcomes'],
             'selected':{m:{k:v for k,v in selected[m].items() if k!='state'} for m in METHODS}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
