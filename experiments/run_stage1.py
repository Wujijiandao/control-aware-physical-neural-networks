#!/usr/bin/env python3
"""Exploratory Stage-1 benchmark. Results are not confirmatory."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from piha.training import train_oracle
from piha.evaluation import fit_linear, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--output", default="results/stage1.csv")
    args = ap.parse_args()
    model, metrics, x, y = train_oracle(quick=args.quick, control_relevant=True)
    linear = fit_linear(x, y)
    rows = evaluate(model, linear, seeds=args.seeds or (20 if args.quick else 100))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"fit": metrics, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
