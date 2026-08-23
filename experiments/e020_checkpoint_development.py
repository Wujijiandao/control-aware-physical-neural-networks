#!/usr/bin/env python3
"""E-020D1: development-only fair checkpoint construction for strong baselines.

The purpose of this script is *not* to establish a robustness result. It trains five
project-original implementations under the same physical architecture, initialization,
field pool, optimizer-update budget and calibration sets, then chooses checkpoints using
STATIC CALIBRATION METRICS ONLY. Closed-loop or perturbation outcomes are deliberately
absent from checkpoint selection.

Methods
-------
- mse: ordinary pointwise field MSE.
- noise_aware: pointwise MSE under calibrated input perturbations.
- boundary_aware: pointwise MSE reweighted toward low-viability-cost states, with batch
  weight normalization so the effective step scale remains comparable.
- sharpness_aware: independent two-pass local parameter-perturbation implementation.
- control_aware: pointwise MSE plus candidate-action ranking loss.

No third-party research source code is copied or imported.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from piha.substrates import InterferometricOracle
from piha.training import set_seed
from piha.viability import viability_torch
from e010_matched_static_development import (
    CandidateDataset,
    action_metrics,
    collect_candidate_dataset,
    make_field_pool,
    regression_metrics,
)

METHODS = ("mse", "noise_aware", "boundary_aware", "sharpness_aware", "control_aware")


def train_family(
    method: str,
    initial_state: Dict[str, torch.Tensor],
    x_field: torch.Tensor,
    y_field: torch.Tensor,
    rank_ds: CandidateDataset,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    cand_cal: CandidateDataset,
    *,
    steps: int = 3000,
    checkpoint_every: int = 100,
    seed: int = 8111,
) -> List[Dict]:
    set_seed(seed)
    model = InterferometricOracle(paths=64, detectors=16)
    model.load_state_dict(copy.deepcopy(initial_state))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed + 313)
    checkpoints: List[Dict] = []

    for step in range(1, steps + 1):
        idx = torch.randint(0, len(x_field), (1024,), generator=gen)
        xb = x_field[idx]
        yb = y_field[idx]

        if method == "noise_aware":
            xb = torch.clamp(xb + 0.012 * torch.randn(xb.shape, generator=gen), 0.0, 1.0)
            yb = viability_torch(xb)

        pred = model(xb)
        sq = (pred - yb).square()
        if method == "boundary_aware":
            # Low D states contain the viability surface and its immediate neighborhood.
            # Normalize weights per batch so this is a change in emphasis rather than an
            # uncontrolled change in effective learning-rate scale.
            w = 0.5 + 0.05 / (0.01 + yb)
            w = w / w.mean().detach()
            field = (sq * w).mean()
        else:
            field = sq.mean()

        if method == "sharpness_aware":
            field.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            gnorm = torch.sqrt(sum(g.detach().square().sum() for g in grads)) + 1e-12
            rho = 0.015
            perturb = []
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        perturb.append(None)
                    else:
                        e = rho * p.grad / gnorm
                        p.add_(e)
                        perturb.append(e)
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            with torch.no_grad():
                for p, e in zip(model.parameters(), perturb):
                    if e is not None:
                        p.sub_(e)
            opt.step()
            opt.zero_grad()
        else:
            loss = field
            if method == "control_aware":
                ir = torch.randint(0, len(rank_ds.cand), (320,), generator=gen)
                cand = rank_ds.cand[ir]
                bonus = rank_ds.bonus[ir]
                target = rank_ds.target[ir]
                n, a, d = cand.shape
                score = model(cand.reshape(n * a, d)).reshape(n, a) - bonus
                rank = F.cross_entropy(-score / 0.025, target)
                loss = field + 2e-4 * rank
            elif method not in ("mse", "noise_aware", "boundary_aware"):
                raise ValueError(method)
            loss.backward()
            opt.step()
            opt.zero_grad()

        if step % checkpoint_every == 0:
            gm = regression_metrics(model, x_cal, y_cal)
            cm = action_metrics(model, cand_cal)
            checkpoints.append({
                "method": method,
                "step": step,
                **gm,
                **cm,
                "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
            })
    return checkpoints


def static_distance(a: Dict, b: Dict) -> float:
    """Symmetric discrepancy using static calibration metrics only."""
    return max(
        abs(math.log(a["rmse"] / b["rmse"])),
        abs(math.log(a["candidate_rmse"] / b["candidate_rmse"])),
        abs(a["r2"] - b["r2"]) / 0.003,
    )


def select_matched_set(families: Dict[str, List[Dict]], r2_min: float = 0.99):
    """Anchor-search a five-method static-matched set without action/closed-loop use."""
    best = None
    for anchor in families["mse"]:
        if anchor["r2"] < r2_min:
            continue
        selected = {"mse": anchor}
        for method in METHODS[1:]:
            eligible = [x for x in families[method] if x["r2"] >= r2_min]
            if not eligible:
                selected = None
                break
            selected[method] = min(eligible, key=lambda x: static_distance(anchor, x))
        if selected is None:
            continue
        vals = list(selected.values())
        r2s = [x["r2"] for x in vals]
        rmses = [x["rmse"] for x in vals]
        crmses = [x["candidate_rmse"] for x in vals]
        score = max(
            max(r2s) - min(r2s),
            abs(math.log(max(rmses) / min(rmses))) * 0.003,
            abs(math.log(max(crmses) / min(crmses))) * 0.003,
        )
        audit = {
            "r2_range": max(r2s) - min(r2s),
            "global_rmse_ratio": max(rmses) / min(rmses),
            "candidate_rmse_ratio": max(crmses) / min(crmses),
            "selection_score": score,
        }
        item = (score, audit, selected)
        if best is None or item[0] < best[0]:
            best = item
    if best is None:
        raise RuntimeError("No five-method set satisfies R2 floor")
    return best[2], best[1]


def main():
    out = Path("results/e020_checkpoint_development")
    out.mkdir(parents=True, exist_ok=True)

    # Namespaces are disjoint from E-010 confirmatory seeds.
    train_rank = collect_candidate_dataset(22000, 32, 200)
    cand_cal = collect_candidate_dataset(26000, 16, 180)
    x_field, y_field = make_field_pool(8282, 24000, train_rank, 24000)
    g = torch.Generator().manual_seed(161803)
    x_cal = torch.rand((10000, 3), generator=g)
    y_cal = viability_torch(x_cal)

    set_seed(4242)
    init = InterferometricOracle(paths=64, detectors=16)
    init_state = copy.deepcopy(init.state_dict())

    families = {}
    for method in METHODS:
        print(f"training {method}", flush=True)
        families[method] = train_family(
            method, init_state, x_field, y_field, train_rank, x_cal, y_cal, cand_cal,
            steps=3000, checkpoint_every=100, seed=8111,
        )
        print(f"done {method}: final R2={families[method][-1]['r2']:.6f}", flush=True)

    selected, audit = select_matched_set(families, r2_min=0.99)

    rows = []
    for method in METHODS:
        ck = selected[method]
        torch.save(ck["state"], out / f"selected_{method}.pt")
        rows.append({k: v for k, v in ck.items() if k != "state"})

    with (out / "selected_static_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["method", "step", "mse", "rmse", "mae", "r2", "candidate_rmse", "candidate_mae", "cal_action_agreement"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r[k] for k in fields} for r in rows])

    with (out / "all_checkpoint_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["method", "step", "mse", "rmse", "mae", "r2", "candidate_rmse", "candidate_mae", "cal_action_agreement"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method in METHODS:
            for ck in families[method]:
                w.writerow({k: ck[k] for k in fields})

    summary = {
        "status": "DEVELOPMENT_ONLY_STATIC_SELECTION",
        "selection_uses": ["global_r2", "global_rmse", "candidate_pointwise_rmse"],
        "selection_excludes": ["cal_action_agreement", "closed_loop_outcomes", "perturbation_outcomes"],
        "audit": audit,
        "selected": {m: {k: v for k, v in selected[m].items() if k != "state"} for m in METHODS},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
