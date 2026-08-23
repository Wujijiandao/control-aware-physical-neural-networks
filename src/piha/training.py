"""Exploratory training routines. Project-original implementation."""
from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn.functional as F
from .dynamics import ACTIONS, TASK_GAIN, ambient_at, predicted_next, true_step
from .substrates import InterferometricOracle
from .viability import viability_np, viability_torch


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_exact_trajectory_states(n_seeds: int = 60, steps: int = 250, lam: float = 0.02):
    states = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        h = np.array([0.60, 0.45, 0.15]) + rng.normal(0.0, 0.015, 3)
        for t in range(steps):
            amb = ambient_at(t)
            cand = np.array([predicted_next(h, a, amb) for a in range(len(ACTIONS))])
            demand = 1.0 + 0.35 * np.sin(2 * np.pi * t / 80 + 0.4)
            score = viability_np(cand) - lam * demand * TASK_GAIN
            action = int(np.argmin(score))
            h = true_step(h, action, amb, rng, shocks=True)
            states.append(h.copy())
    return np.asarray(states)


def train_oracle(quick: bool = False, control_relevant: bool = True, seed: int = 0):
    set_seed(seed)
    n_uniform = 12000 if quick else 30000
    x_u = torch.rand(n_uniform, 3)
    y_u = viability_torch(x_u)
    x_test = torch.rand(4000 if quick else 10000, 3)
    y_test = viability_torch(x_test)

    model = InterferometricOracle(paths=48 if quick else 64, detectors=12 if quick else 16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    for _ in range(600 if quick else 1700):
        idx = torch.randint(0, n_uniform, (768 if quick else 1024,))
        loss = F.mse_loss(model(x_u[idx]), y_u[idx])
        loss.backward(); opt.step(); opt.zero_grad()

    if control_relevant:
        rel = collect_exact_trajectory_states(n_seeds=20 if quick else 80, steps=180 if quick else 300)
        rng = np.random.default_rng(123)
        rel = np.repeat(rel, 2, axis=0)
        rel = np.clip(rel + rng.normal(0.0, 0.035, size=rel.shape), 0.0, 1.0)
        x_r = torch.tensor(rel, dtype=torch.float32)
        y_r = viability_torch(x_r)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        for _ in range(250 if quick else 700):
            nrel = 512 if quick else 768
            ir = torch.randint(0, len(x_r), (nrel,))
            iu = torch.randint(0, n_uniform, (256,))
            xb = torch.cat([x_r[ir], x_u[iu]], dim=0)
            yb = torch.cat([y_r[ir], y_u[iu]], dim=0)
            pred = model(xb)
            weights = 0.5 + 0.06 / (0.01 + yb)
            loss = ((pred - yb).square() * weights).mean()
            loss.backward(); opt.step(); opt.zero_grad()

    with torch.no_grad():
        pred = model(x_test)
        mse = F.mse_loss(pred, y_test).item()
        mae = (pred - y_test).abs().mean().item()
        r2 = 1.0 - (pred - y_test).square().sum().item() / (y_test - y_test.mean()).square().sum().item()
    return model, {"mse": mse, "mae": mae, "r2": r2}, x_u.numpy(), y_u.numpy()
