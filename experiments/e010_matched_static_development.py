#!/usr/bin/env python3
"""E-010 development experiment: static-matched PNNs with closed-loop evaluation.

This is a DEVELOPMENT-ONLY script. It may be used to tune the matching rule and
training recipe, but its closed-loop outcomes are not confirmatory evidence.

Design principles
-----------------
1. Same physical architecture and identical initialization for both models.
2. Same field-regression samples for both models.
3. Baseline optimizes only pointwise field MSE.
4. Control-aware model adds a candidate-action ranking loss.
5. Checkpoint pair is selected using static CALIBRATION metrics only.
6. Closed-loop DEVELOPMENT seeds are evaluated only after pair selection.

No third-party research source code is used.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from piha.dynamics import ACTIONS, TASK_GAIN, ambient_at, predicted_next, true_step
from piha.substrates import InterferometricOracle
from piha.training import set_seed
from piha.viability import viability_np, viability_torch


@dataclass
class CandidateDataset:
    cand: torch.Tensor      # [N, A, 3]
    bonus: torch.Tensor     # [N, A]
    target: torch.Tensor    # [N]


def collect_candidate_dataset(seed_start: int, n_seeds: int, steps: int, lam: float = 0.02) -> CandidateDataset:
    """Collect exact-teacher trajectories and all one-step candidate states."""
    cand_rows, bonus_rows, targets = [], [], []
    for s in range(seed_start, seed_start + n_seeds):
        rng = np.random.default_rng(s)
        h = np.array([0.60, 0.45, 0.15]) + rng.normal(0.0, 0.015, 3)
        for t in range(steps):
            amb = ambient_at(t)
            cand = np.array([predicted_next(h, a, amb) for a in range(len(ACTIONS))], dtype=np.float32)
            demand = 1.0 + 0.35 * np.sin(2 * np.pi * t / 80 + 0.4)
            bonus = (lam * demand * TASK_GAIN).astype(np.float32)
            exact_scores = viability_np(cand) - bonus
            best = int(np.argmin(exact_scores))
            cand_rows.append(cand)
            bonus_rows.append(bonus)
            targets.append(best)
            h = true_step(h, best, amb, rng, shocks=True)
    return CandidateDataset(
        cand=torch.tensor(np.asarray(cand_rows), dtype=torch.float32),
        bonus=torch.tensor(np.asarray(bonus_rows), dtype=torch.float32),
        target=torch.tensor(np.asarray(targets), dtype=torch.long),
    )


def regression_metrics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        p = model(x)
        e = p - y
        mse = float(e.square().mean())
        rmse = math.sqrt(mse)
        mae = float(e.abs().mean())
        denom = float((y - y.mean()).square().sum())
        r2 = 1.0 - float(e.square().sum()) / denom
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def action_metrics(model: torch.nn.Module, ds: CandidateDataset) -> Dict[str, float]:
    with torch.no_grad():
        n, a, d = ds.cand.shape
        pred_cost = model(ds.cand.reshape(n * a, d)).reshape(n, a)
        pred_scores = pred_cost - ds.bonus
        pred_action = pred_scores.argmin(dim=1)
        agreement = float((pred_action == ds.target).float().mean())
        # Pointwise RMSE over candidate viability values, still a static metric.
        exact_cost = viability_torch(ds.cand.reshape(n * a, d)).reshape(n, a)
        cand_rmse = float(torch.sqrt((pred_cost - exact_cost).square().mean()))
        cand_mae = float((pred_cost - exact_cost).abs().mean())
    return {"candidate_rmse": cand_rmse, "candidate_mae": cand_mae, "cal_action_agreement": agreement}


def make_field_pool(seed: int, n_uniform: int, cand_ds: CandidateDataset, n_candidate_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x_u = torch.rand((n_uniform, 3), generator=g)
    flat = cand_ds.cand.reshape(-1, 3)
    idx = torch.randint(0, len(flat), (n_candidate_points,), generator=g)
    x = torch.cat([x_u, flat[idx]], dim=0)
    y = viability_torch(x)
    return x, y


def train_family(
    strategy: str,
    initial_state: Dict[str, torch.Tensor],
    x_field: torch.Tensor,
    y_field: torch.Tensor,
    rank_ds: CandidateDataset,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    cand_cal: CandidateDataset,
    steps: int,
    checkpoint_every: int,
    seed: int,
    rank_weight: float,
    rank_temperature: float,
) -> List[Dict]:
    set_seed(seed)
    model = InterferometricOracle(paths=64, detectors=16)
    model.load_state_dict(copy.deepcopy(initial_state))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed + 991)
    checkpoints: List[Dict] = []

    for step in range(1, steps + 1):
        idx = torch.randint(0, len(x_field), (1024,), generator=gen)
        field_loss = F.mse_loss(model(x_field[idx]), y_field[idx])
        loss = field_loss

        if strategy == "control_aware":
            ir = torch.randint(0, len(rank_ds.cand), (320,), generator=gen)
            cand = rank_ds.cand[ir]
            bonus = rank_ds.bonus[ir]
            target = rank_ds.target[ir]
            n, a, d = cand.shape
            pred = model(cand.reshape(n * a, d)).reshape(n, a)
            score = pred - bonus
            rank_loss = F.cross_entropy(-score / rank_temperature, target)
            loss = field_loss + rank_weight * rank_loss
        elif strategy != "mse":
            raise ValueError(strategy)

        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % checkpoint_every == 0:
            m = regression_metrics(model, x_cal, y_cal)
            am = action_metrics(model, cand_cal)
            checkpoints.append({
                "strategy": strategy,
                "step": step,
                **m,
                **am,
                "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
            })
    return checkpoints


def select_static_matched_pair(base: List[Dict], control: List[Dict], r2_min: float = 0.985) -> Tuple[Dict, Dict, Dict]:
    """Choose pair using static metrics only; action agreement is NOT used in selection."""
    candidates = []
    for b in base:
        for c in control:
            if min(b["r2"], c["r2"]) < r2_min:
                continue
            r2_diff = abs(b["r2"] - c["r2"])
            rmse_ratio = max(b["rmse"], c["rmse"]) / min(b["rmse"], c["rmse"])
            cand_ratio = max(b["candidate_rmse"], c["candidate_rmse"]) / min(b["candidate_rmse"], c["candidate_rmse"])
            # Static-only lexicographic objective: first worst relative discrepancy, then R2.
            discrepancy = max(abs(math.log(rmse_ratio)), abs(math.log(cand_ratio)), r2_diff / 0.002)
            candidates.append((discrepancy, r2_diff, rmse_ratio, cand_ratio, b, c))
    if not candidates:
        raise RuntimeError("No eligible checkpoint pair")
    candidates.sort(key=lambda z: (z[0], z[1], abs(math.log(z[2])), abs(math.log(z[3]))))
    d, r2d, rr, cr, b, c = candidates[0]
    info = {"static_discrepancy": d, "r2_diff": r2d, "rmse_ratio": rr, "candidate_rmse_ratio": cr}
    return b, c, info


def oracle_action(model: torch.nn.Module, cand: np.ndarray, bonus: np.ndarray) -> int:
    with torch.no_grad():
        x = torch.tensor(cand, dtype=torch.float32)
        score = model(x).cpu().numpy() - bonus
    return int(np.argmin(score))


def evaluate_closed_loop(model: torch.nn.Module, seeds: List[int], steps: int = 400, lam: float = 0.02) -> Dict[str, float]:
    episode_metrics = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        h = np.array([0.60, 0.45, 0.15]) + rng.normal(0.0, 0.015, 3)
        dvals, agreements, severe, viable, taskvals = [], [], [], [], []
        for t in range(steps):
            amb = ambient_at(t)
            cand = np.array([predicted_next(h, a, amb) for a in range(len(ACTIONS))], dtype=np.float32)
            demand = 1.0 + 0.35 * np.sin(2 * np.pi * t / 80 + 0.4)
            bonus = lam * demand * TASK_GAIN
            exact_action = int(np.argmin(viability_np(cand) - bonus))
            action = oracle_action(model, cand, bonus)
            agreements.append(action == exact_action)
            h = true_step(h, action, amb, rng, shocks=True)
            dv = float(viability_np(h))
            dvals.append(dv)
            severe.append(dv > 0.2)
            viable.append(dv < 0.05)
            taskvals.append(float(TASK_GAIN[action] * demand))
        episode_metrics.append([
            np.mean(dvals), np.quantile(dvals, 0.95), np.mean(viable), np.mean(severe),
            np.mean(agreements), np.sum(taskvals)
        ])
    x = np.asarray(episode_metrics, dtype=float)
    names = ["mean_D", "p95_D", "viability_occupancy", "severe_fraction", "action_agreement", "cumulative_task"]
    out = {}
    for i, name in enumerate(names):
        out[name] = float(x[:, i].mean())
        out[f"se_{name}"] = float(x[:, i].std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return out


def hash_jsonable(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--rank-weight", type=float, default=0.0025)
    ap.add_argument("--rank-temperature", type=float, default=0.025)
    ap.add_argument("--development-seeds", type=int, default=80)
    ap.add_argument("--output-dir", default="results/e010_development")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Namespace separation: training < 5000, development closed-loop 5000-9999,
    # calibration 10000-19999. Confirmatory namespace is deliberately absent here.
    train_cand = collect_candidate_dataset(1000, 40, 220)
    cal_cand = collect_candidate_dataset(10000, 24, 180)
    x_field, y_field = make_field_pool(2718, 24000, train_cand, 24000)

    gcal = torch.Generator().manual_seed(314159)
    x_cal = torch.rand((12000, 3), generator=gcal)
    y_cal = viability_torch(x_cal)

    set_seed(4242)
    init = InterferometricOracle(paths=64, detectors=16)
    initial_state = copy.deepcopy(init.state_dict())

    base = train_family("mse", initial_state, x_field, y_field, train_cand,
                        x_cal, y_cal, cal_cand, args.steps, args.checkpoint_every,
                        seed=7001, rank_weight=args.rank_weight, rank_temperature=args.rank_temperature)
    control = train_family("control_aware", initial_state, x_field, y_field, train_cand,
                           x_cal, y_cal, cal_cand, args.steps, args.checkpoint_every,
                           seed=7001, rank_weight=args.rank_weight, rank_temperature=args.rank_temperature)

    b, c, match = select_static_matched_pair(base, control)

    dev_seeds = list(range(5000, 5000 + args.development_seeds))
    results = {}
    for label, ck in (("mse", b), ("control_aware", c)):
        m = InterferometricOracle(paths=64, detectors=16)
        m.load_state_dict(ck["state"])
        results[label] = {
            "checkpoint": {k: v for k, v in ck.items() if k != "state"},
            "closed_loop_development": evaluate_closed_loop(m, dev_seeds),
        }
        torch.save(ck["state"], out / f"selected_{label}.pt")

    summary = {
        "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "args": vars(args),
        "matching": match,
        "selected": results,
        "development_seed_range": [dev_seeds[0], dev_seeds[-1]],
    }
    summary["configuration_sha256"] = hash_jsonable({"args": vars(args), "matching_rule": "static-only-v1"})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Checkpoint audit table without state tensors.
    with (out / "checkpoint_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["strategy", "step", "mse", "rmse", "mae", "r2", "candidate_rmse", "candidate_mae", "cal_action_agreement"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ck in base + control:
            w.writerow({k: ck[k] for k in fieldnames})

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
