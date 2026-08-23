#!/usr/bin/env python3
"""Select static-matched E-030 oscillator checkpoints, then development-only closed-loop compare."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import torch
from piha.substrates import NonlinearOscillatorOracle
from e010_matched_static_development import select_static_matched_pair, evaluate_closed_loop

WEIGHTS=(1.5e-3,2e-3)

def model(): return NonlinearOscillatorOracle(d=3,oscillators=24,integration_steps=14,dt=0.055)

def main():
    torch.set_num_threads(1)
    src=Path('results/e030_oscillator_development')
    base=torch.load(src/'mse_family.pt',weights_only=False)
    dev_seeds=list(range(44000,44016))
    rows=[]; start=time.time()
    for w in WEIGHTS:
        tag=f'w{w:.6g}'.replace('.','p')
        fam=torch.load(src/f'control_{tag}_family.pt',weights_only=False)
        b,c,match=select_static_matched_pair(base,fam,r2_min=0.99)
        mb=model();mb.load_state_dict(b['state']);mb.eval()
        mc=model();mc.load_state_dict(c['state']);mc.eval()
        rb=evaluate_closed_loop(mb,dev_seeds);rc=evaluate_closed_loop(mc,dev_seeds)
        rel=(rb['mean_D']-rc['mean_D'])/rb['mean_D']
        item={'rank_weight':w,'matching':match,
              'mse_checkpoint':{k:v for k,v in b.items() if k!='state'},
              'control_checkpoint':{k:v for k,v in c.items() if k!='state'},
              'mse_closed_loop':rb,'control_closed_loop':rc,'relative_mean_D_reduction':rel}
        rows.append(item)
        torch.save(b['state'],src/f'selected_mse_{tag}.pt');torch.save(c['state'],src/f'selected_control_{tag}.pt')
        print('weight',w,'steps',b['step'],c['step'],'r2',b['r2'],c['r2'],'ratios',match['rmse_ratio'],match['candidate_rmse_ratio'],'meanD',rb['mean_D'],rc['mean_D'],'rel',rel,flush=True)
    summary={'experiment':'E-030D2','status':'DEVELOPMENT_ONLY_NOT_CONFIRMATORY','development_seeds':dev_seeds,'results':rows,'runtime_seconds':time.time()-start}
    (src/'matched_development_summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
