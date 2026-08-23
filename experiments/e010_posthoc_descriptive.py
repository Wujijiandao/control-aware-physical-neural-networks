#!/usr/bin/env python3
"""Post-confirmatory descriptive analysis of already-frozen E-010C1 raw data.

This script does not rerun trajectories and cannot modify the frozen primary
endpoint. Secondary intervals are descriptive and were not part of the primary
success criterion.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/e010_confirmatory/e010c1_raw_per_seed.csv"
OUT = ROOT / "results/e010_confirmatory/e010c1_secondary_bootstrap.json"


def bootstrap_mean_ci(x, seed, n_boot=20000):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    vals=[]
    for _ in range(20):
        idx=rng.integers(0,len(x),size=(1000,len(x)))
        vals.append(x[idx].mean(axis=1))
    vals=np.concatenate(vals)[:n_boot]
    lo,hi=np.quantile(vals,[0.025,0.975])
    return {"mean":float(x.mean()),"ci95_low":float(lo),"ci95_high":float(hi)}


def main():
    df=pd.read_csv(RAW)
    metrics=["p95_D","viability_occupancy","severe_fraction","action_agreement","cumulative_task"]
    out={"status":"POSTHOC_DESCRIPTIVE_NOT_PRIMARY","n":len(df),"paired_differences":{}}
    for i,m in enumerate(metrics):
        d=df[f"control_{m}"]-df[f"mse_{m}"]
        out["paired_differences"][m]=bootstrap_mean_ci(d, 920000+i)
    OUT.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
