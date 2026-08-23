#!/usr/bin/env python3
"""Frozen confirmatory runner for E-010C1.

Do not modify this script after E-010C1 is run. Any changed analysis must receive
a new experiment identifier. Project-original implementation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from piha.dynamics import ACTIONS, TASK_GAIN, ambient_at, predicted_next, true_step
from piha.substrates import InterferometricOracle
from piha.viability import viability_np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "frozen" / "e010"
OUT = ROOT / "results" / "e010_confirmatory"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_frozen_inputs() -> Dict[str, str]:
    # Seed commitment
    seed_commit = (FROZEN / "SEED_COMMITMENT.txt").read_text(encoding="utf-8").splitlines()[0].split()[-1]
    actual_seed = sha256(FROZEN / "confirmatory_seeds.txt")
    if actual_seed != seed_commit:
        raise RuntimeError("confirmatory seed commitment mismatch")

    # Checkpoint manifest generated before outcome evaluation.
    expected = {}
    for line in (FROZEN / "CHECKPOINT_SHA256.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, path = line.split(maxsplit=1)
        expected[Path(path).name] = digest
    for name in ("selected_mse.pt", "selected_control_aware.pt", "development_selection_summary.json"):
        got = sha256(FROZEN / name)
        if got != expected[name]:
            raise RuntimeError(f"frozen input hash mismatch: {name}")
    return {"seed_sha256": actual_seed, **{f"sha256_{k}": v for k, v in expected.items()}}


def load_model(path: Path) -> InterferometricOracle:
    m = InterferometricOracle(paths=64, detectors=16)
    state = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def choose_actions(model: InterferometricOracle, candidates: np.ndarray, bonuses: np.ndarray) -> np.ndarray:
    """Candidates [N,A,3], bonuses [N,A]."""
    n, a, d = candidates.shape
    x = torch.tensor(candidates.reshape(n * a, d), dtype=torch.float32)
    pred = model(x).cpu().numpy().reshape(n, a)
    return np.argmin(pred - bonuses, axis=1)


def candidate_batch(states: np.ndarray, ambient: float) -> np.ndarray:
    n = len(states)
    out = np.empty((n, len(ACTIONS), 3), dtype=np.float32)
    # Keep a single source of dynamics truth by calling predicted_next.
    for i in range(n):
        for a in range(len(ACTIONS)):
            out[i, a] = predicted_next(states[i], a, ambient)
    return out


def evaluate_model(model: InterferometricOracle, seeds: List[int], steps: int, lam: float) -> List[Dict[str, float]]:
    n = len(seeds)
    rngs = [np.random.default_rng(s) for s in seeds]
    states = np.vstack([np.array([0.60, 0.45, 0.15]) + r.normal(0.0, 0.015, 3) for r in rngs])

    D_hist = np.empty((n, steps), dtype=np.float64)
    agree_hist = np.empty((n, steps), dtype=np.float64)
    task_hist = np.empty((n, steps), dtype=np.float64)

    for t in range(steps):
        amb = ambient_at(t)
        cand = candidate_batch(states, amb)
        demand = 1.0 + 0.35 * np.sin(2 * np.pi * t / 80 + 0.4)
        bonus_one = lam * demand * TASK_GAIN
        bonus = np.broadcast_to(bonus_one, (n, len(ACTIONS)))

        exact_actions = np.argmin(viability_np(cand) - bonus, axis=1)
        actions = choose_actions(model, cand, bonus)
        agree_hist[:, t] = (actions == exact_actions)
        task_hist[:, t] = TASK_GAIN[actions] * demand

        new_states = np.empty_like(states)
        for i in range(n):
            new_states[i] = true_step(states[i], int(actions[i]), amb, rngs[i], shocks=True)
        states = new_states
        D_hist[:, t] = viability_np(states)

    records = []
    for i, seed in enumerate(seeds):
        D = D_hist[i]
        records.append({
            "seed": int(seed),
            "mean_D": float(D.mean()),
            "p95_D": float(np.quantile(D, 0.95)),
            "viability_occupancy": float((D < 0.05).mean()),
            "severe_fraction": float((D > 0.2).mean()),
            "action_agreement": float(agree_hist[i].mean()),
            "cumulative_task": float(task_hist[i].sum()),
        })
    return records


def mean_ci_bootstrap(diff: np.ndarray, resamples: int, rng_seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    n = len(diff)
    # Chunk to bound memory; exact prespecified percentile bootstrap.
    vals = np.empty(resamples, dtype=np.float64)
    chunk = 1000
    pos = 0
    while pos < resamples:
        k = min(chunk, resamples - pos)
        idx = rng.integers(0, n, size=(k, n))
        vals[pos:pos+k] = diff[idx].mean(axis=1)
        pos += k
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"mean": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def summarize(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = ["mean_D", "p95_D", "viability_occupancy", "severe_fraction", "action_agreement", "cumulative_task"]
    out = {}
    for k in keys:
        x = np.array([r[k] for r in records], dtype=float)
        out[k] = {
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)),
            "se": float(x.std(ddof=1) / math.sqrt(len(x))),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise RuntimeError("E-010 confirmatory output directory is not empty; refusing to overwrite or rerun")

    hashes = verify_frozen_inputs()
    config = json.loads((FROZEN / "confirmatory_config.json").read_text(encoding="utf-8"))
    seeds = [int(x) for x in (FROZEN / "confirmatory_seeds.txt").read_text(encoding="utf-8").split()]
    if len(seeds) != config["n_seeds"] or len(set(seeds)) != len(seeds):
        raise RuntimeError("seed count/uniqueness mismatch")

    mse_model = load_model(FROZEN / "selected_mse.pt")
    ca_model = load_model(FROZEN / "selected_control_aware.pt")

    mse_records = evaluate_model(mse_model, seeds, config["steps"], config["lambda_task"])
    ca_records = evaluate_model(ca_model, seeds, config["steps"], config["lambda_task"])

    # Paired records by committed seed order.
    raw = []
    for m, c in zip(mse_records, ca_records):
        assert m["seed"] == c["seed"]
        row = {"seed": m["seed"]}
        for key in ("mean_D", "p95_D", "viability_occupancy", "severe_fraction", "action_agreement", "cumulative_task"):
            row[f"mse_{key}"] = m[key]
            row[f"control_{key}"] = c[key]
            row[f"delta_{key}"] = c[key] - m[key]
        raw.append(row)

    raw_path = OUT / "e010c1_raw_per_seed.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader(); w.writerows(raw)

    diff = np.array([r["delta_mean_D"] for r in raw], dtype=float)
    primary = mean_ci_bootstrap(diff, config["bootstrap_resamples"], config["bootstrap_seed"])
    mse_mean = float(np.mean([r["mse_mean_D"] for r in raw]))
    control_mean = float(np.mean([r["control_mean_D"] for r in raw]))
    relative_reduction = (mse_mean - control_mean) / mse_mean
    success_ci = primary["ci95_high"] < 0.0
    success_effect = relative_reduction >= config["primary_relative_reduction_threshold"]

    summary = {
        "experiment_id": config["experiment_id"],
        "status": "CONFIRMATORY_COMPLETED_ONCE",
        "frozen_input_hashes": hashes,
        "config": config,
        "primary": {
            "definition": "control_aware mean_D minus MSE mean_D, paired by seed",
            **primary,
            "mse_mean_D": mse_mean,
            "control_mean_D": control_mean,
            "relative_reduction": relative_reduction,
            "criterion_ci_upper_below_zero": success_ci,
            "criterion_relative_reduction_at_least_threshold": success_effect,
            "primary_success": bool(success_ci and success_effect),
        },
        "mse_summary": summarize(mse_records),
        "control_aware_summary": summarize(ca_records),
        "raw_csv_sha256": sha256(raw_path),
    }
    (OUT / "e010c1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "RUN_LOCK.txt").write_text(
        "E-010C1 has been evaluated. Do not overwrite, delete, reseed, or rerun under this experiment identifier.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
