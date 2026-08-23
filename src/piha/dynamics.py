"""Minimal auditable homeostatic environment. Project-original implementation."""
from __future__ import annotations
import numpy as np

ACTIONS = ("charge", "cool", "repair", "task", "fast_task")
TASK_GAIN = np.array([0.0, 0.0, 0.0, 1.0, 1.8])


def ambient_at(t: int) -> float:
    value = 0.43 + 0.09 * np.sin(2 * np.pi * t / 120) + 0.02 * np.sin(2 * np.pi * t / 37)
    if 150 <= t < 185 or 360 <= t < 390:
        value += 0.18
    return float(value)


def predicted_next(h, action: int, ambient: float):
    E, T, R = map(float, h)
    E -= 0.012
    T += 0.08 * (ambient - T)
    R = max(0.0, R - 0.004)
    if action == 0:
        E += 0.095; T += 0.025; R += 0.003
    elif action == 1:
        E -= 0.020; T -= 0.105; R += 0.001
    elif action == 2:
        E -= 0.018; T -= 0.015; R -= 0.080
    elif action == 3:
        E -= 0.038; T += 0.040; R += 0.022
    elif action == 4:
        E -= 0.070; T += 0.080; R += 0.050
    else:
        raise ValueError(f"unknown action {action}")
    return np.clip([E, T, R], 0.0, 1.0)


def true_step(h, action: int, ambient: float, rng: np.random.Generator, shocks: bool = True):
    x = predicted_next(h, action, ambient).astype(float)
    x += rng.normal(0.0, [0.004, 0.004, 0.003])
    if shocks:
        if rng.random() < 0.018:
            x[0] -= rng.uniform(0.06, 0.14)
        if rng.random() < 0.018:
            x[1] += rng.uniform(0.07, 0.15)
        if rng.random() < 0.018:
            x[2] += rng.uniform(0.05, 0.13)
    return np.clip(x, 0.0, 1.0)
