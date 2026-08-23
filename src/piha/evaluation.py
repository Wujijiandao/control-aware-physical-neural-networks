"""Closed-loop evaluation. Project-original implementation."""
from __future__ import annotations
import math
import numpy as np
import torch
from .dynamics import ACTIONS, TASK_GAIN, ambient_at, predicted_next, true_step
from .viability import viability_np


@torch.no_grad()
def model_oracle(model, x, rng=None, measurement_noise: float = 0.0, input_noise: float = 0.0):
    x = np.asarray(x, dtype=np.float32)
    if input_noise > 0 and rng is not None:
        x = x + rng.normal(0.0, input_noise, size=x.shape)
    xt = torch.tensor(np.clip(x, 0.0, 1.0), dtype=torch.float32)
    y = model(xt).cpu().numpy()
    if measurement_noise > 0 and rng is not None:
        y = y + rng.normal(0.0, measurement_noise, size=np.shape(y))
    return np.maximum(0.0, y)


def fit_linear(x, y):
    A = np.c_[x, np.ones(len(x))]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def run_episode(model, linear_coef, oracle: str, seed: int, steps: int = 400, lam: float = 0.02,
                measurement_noise: float = 0.005, input_noise: float = 0.003):
    rng = np.random.default_rng(seed)
    h = np.array([0.60, 0.45, 0.15]) + rng.normal(0.0, 0.015, 3)
    rows = []
    for t in range(steps):
        amb = ambient_at(t)
        cand = np.array([predicted_next(h, a, amb) for a in range(len(ACTIONS))])
        if oracle == "exact":
            costs = viability_np(cand)
        elif oracle == "physical":
            costs = model_oracle(model, cand, rng, measurement_noise, input_noise)
        elif oracle == "linear":
            costs = np.maximum(0.0, cand @ linear_coef[:3] + linear_coef[3])
        elif oracle == "sham":
            costs = model_oracle(model, cand[:, [2, 0, 1]], rng, measurement_noise, input_noise)
        else:
            raise ValueError(oracle)
        demand = 1.0 + 0.35 * np.sin(2 * np.pi * t / 80 + 0.4)
        action = int(np.argmin(costs - lam * demand * TASK_GAIN))
        h = true_step(h, action, amb, rng, shocks=True)
        rows.append((float(viability_np(h)), float(TASK_GAIN[action] * demand)))
    return np.asarray(rows)


def evaluate(model, linear_coef, seeds: int = 100, steps: int = 400):
    records = []
    for oracle in ("exact", "physical", "linear", "sham"):
        values = []
        for seed in range(seeds):
            ep = run_episode(model, linear_coef, oracle, seed, steps=steps)
            D, task = ep[:, 0], ep[:, 1]
            values.append([D.mean(), np.quantile(D, 0.95), (D < 0.05).mean(), task.sum(), (D > 0.2).mean()])
        values = np.asarray(values)
        mean = values.mean(axis=0)
        se = values.std(axis=0, ddof=1) / math.sqrt(seeds) if seeds > 1 else np.zeros(5)
        records.append({
            "oracle": oracle,
            "mean_D": mean[0], "p95_D": mean[1], "viability_occupancy": mean[2],
            "cumulative_task": mean[3], "severe_fraction": mean[4],
            "se_mean_D": se[0], "se_p95_D": se[1], "se_viability": se[2],
            "se_task": se[3], "se_severe": se[4]
        })
    return records
