#!/usr/bin/env python3
"""Exploratory test of static fit versus closed-loop behavior.

This script trains two models with the same architecture and random seed:
(A) uniform regression only and (B) control-relevant fine-tuning. It reports
both static fit and closed-loop metrics. It does NOT yet enforce preregistered
R^2 matching; that is reserved for the confirmatory protocol.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from piha.training import train_oracle
from piha.evaluation import fit_linear, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--output", default="results/static_vs_closed_loop.csv")
    args = ap.parse_args()
    rows = []
    for label, control in (("uniform", False), ("control_relevant", True)):
        model, fit, x, y = train_oracle(quick=args.quick, control_relevant=control, seed=0)
        linear = fit_linear(x, y)
        closed = evaluate(model, linear, seeds=args.seeds)
        physical = next(r for r in closed if r["oracle"] == "physical")
        rows.append({"training": label, **fit, **{f"closed_{k}": v for k, v in physical.items() if k != "oracle"}})
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
