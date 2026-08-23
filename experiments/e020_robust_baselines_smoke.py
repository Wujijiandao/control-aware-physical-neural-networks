#!/usr/bin/env python3
"""Engineering/development smoke for E-020 baseline feasibility; not evidence."""
import copy, csv, json
from pathlib import Path
import torch
from piha.substrates import InterferometricOracle
from piha.training import set_seed
from piha.viability import viability_torch
from e010_matched_static_development import collect_candidate_dataset, make_field_pool, regression_metrics
from e020_robust_baselines_development import METHODS, train, evaluate_noise

out=Path('results/e020_smoke'); out.mkdir(parents=True,exist_ok=True)
rank=collect_candidate_dataset(1300,10,120)
x,y=make_field_pool(9292,8000,rank,8000)
g=torch.Generator().manual_seed(271828)
xcal=torch.rand((3000,3),generator=g); ycal=viability_torch(xcal)
set_seed(4242); init=InterferometricOracle(paths=64,detectors=16); init_state=copy.deepcopy(init.state_dict())
seeds=list(range(7100,7108)); noise=[(0,0),(0.01,0.006),(0.02,0.012)]
rows=[]; static={}
for method in METHODS:
    model=train(method,init_state,x,y,rank,steps=500,seed=8111)
    static[method]=regression_metrics(model,xcal,ycal)
    for mn,inn in noise:
        r=evaluate_noise(model,seeds,mn,inn)
        rows.append({'method':method,'measurement_noise':mn,'input_noise':inn,**static[method],**r})
with (out/'e020_smoke.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
(out/'e020_smoke_static.json').write_text(json.dumps(static,indent=2),encoding='utf-8')
print(json.dumps({'status':'ENGINEERING_SMOKE_NOT_EVIDENCE','static':static,'rows':rows},indent=2))
