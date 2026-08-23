"""Homeostatic viability benchmark functions. Project-original implementation."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def softplus_np(x, beta: float = 20.0):
    return np.logaddexp(0.0, beta * np.asarray(x)) / beta


def viability_np(h):
    h = np.asarray(h)
    E, T, R = h[..., 0], h[..., 1], h[..., 2]
    return (
        softplus_np(0.35 - E) ** 2
        + softplus_np(E - 0.85) ** 2
        + 1.2 * softplus_np(0.25 - T) ** 2
        + 1.2 * softplus_np(T - 0.65) ** 2
        + 1.5 * softplus_np(R - 0.45) ** 2
        + softplus_np(T + 0.55 * R - 0.83) ** 2
        + softplus_np(0.50 - E + 0.45 * R) ** 2
    )


def viability_torch(h: torch.Tensor) -> torch.Tensor:
    E, T, R = h[..., 0], h[..., 1], h[..., 2]
    sp = lambda x: F.softplus(20.0 * x) / 20.0
    return (
        sp(0.35 - E) ** 2
        + sp(E - 0.85) ** 2
        + 1.2 * sp(0.25 - T) ** 2
        + 1.2 * sp(T - 0.65) ** 2
        + 1.5 * sp(R - 0.45) ** 2
        + sp(T + 0.55 * R - 0.83) ** 2
        + sp(0.50 - E + 0.45 * R) ** 2
    )
