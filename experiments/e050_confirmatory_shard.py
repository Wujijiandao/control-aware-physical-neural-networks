#!/usr/bin/env python3
"""E-050C1 frozen confirmatory task-matched Pareto shard runner."""
from __future__ import annotations
import argparse,csv,hashlib,json,multiprocessing as mp,os
from pathlib import Path
import torch
from piha.substrates import InterferometricOracle
from e020_robustness_development import episode
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'frozen/e050';OUT=ROOT/'results/e050_confirmatory';_G={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
 for line in (F/'FREEZE_MANIFEST.sha256').read_text().splitlines():
  if not line.strip():continue
  h,n=line.split('  ',1)
  if sha(F/n)!=h:raise RuntimeError(f'hash mismatch {n}')
def init_worker():
 torch.set_num_threads(1);mods={}
 for label,file in [('boundary_aware','boundary_aware.pt'),('robust_control_aware','robust_control_aware.pt')]:
  m=InterferometricOracle(paths=64,detectors=16);m.load_state_dict(torch.load(F/file,weights_only=True));m.eval();mods[label]=m
 _G['models']=mods;_G['conds']={c['name']:c for c in json.loads((F/'conditions.json').read_text())};_G['grid']=json.loads((F/'confirmatory_config.json').read_text())['lambda_grid']
def work(seed):
 rows=[]
 for method,m in _G['models'].items():
  for cname in ('moderate','strong'):
   for lam in _G['grid']:
    r=episode(m,seed,_G['conds'][cname],steps=400,lam=float(lam));rows.append({'seed':seed,'method':method,'condition':cname,'lambda_task':lam,**r})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard-index',type=int,required=True);ap.add_argument('--num-shards',type=int,default=4);args=ap.parse_args();verify();seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()];ss=seeds[args.shard_index::args.num_shards];sd=OUT/'shards';sd.mkdir(parents=True,exist_ok=True);p=sd/f'shard_{args.shard_index:02d}_of_{args.num_shards:02d}.csv'
 if p.exists():raise SystemExit(f'refusing overwrite {p}')
 ctx=mp.get_context('fork')
 with ctx.Pool(processes=min(4,os.cpu_count() or 1),initializer=init_worker) as pool:nested=pool.map(work,ss,chunksize=1)
 rows=[r for rr in nested for r in rr];fields=list(rows[0])
 with p.open('w',newline='',encoding='utf-8') as fh:w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(rows)
 meta={'experiment':'E-050C1','shard_index':args.shard_index,'num_shards':args.num_shards,'n_seeds':len(ss),'seed_sha256':sha(F/'confirmatory_seeds.txt'),'csv_sha256':sha(p)};p.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
