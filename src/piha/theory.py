"""Small testable statements used by the manuscript theory layer."""
from __future__ import annotations
import numpy as np


def action_gap(scores) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    if s.size < 2:
        return float("inf")
    return float(s[1] - s[0])


def selected_action(scores) -> int:
    return int(np.argmin(np.asarray(scores, dtype=float)))


def two_epsilon_regret_bound(epsilon: float) -> float:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    return 2.0 * float(epsilon)


def action_preservation_sufficient(gap: float, epsilon: float) -> bool:
    """Sufficient condition: gap > 2 epsilon."""
    return float(gap) > two_epsilon_regret_bound(float(epsilon))


def subgaussian_action_mismatch_bound(gaps, pairwise_sigma: float) -> float:
    """Union bound on greedy action reversal under pairwise sub-Gaussian score error.

    For each competitor a, let gap[a] = J(a)-J(a*) > 0 and let the pairwise
    approximation-error difference Z_a = e_a-e_* be zero-mean sub-Gaussian with
    parameter ``pairwise_sigma`` in the tail convention
    P(Z_a <= -x) <= exp(-x^2 / (2 sigma^2)). Then

        P(argmin J+e != a*) <= sum_a exp(-gap[a]^2/(2 sigma^2)).

    Independence across competitors is not required for the union bound once the
    pairwise tail bounds are assumed.
    """
    g = np.asarray(gaps, dtype=float)
    if np.any(g <= 0):
        raise ValueError("all competitor gaps must be strictly positive")
    if pairwise_sigma < 0:
        raise ValueError("pairwise_sigma must be non-negative")
    if pairwise_sigma == 0:
        return 0.0
    p = np.exp(-(g ** 2) / (2.0 * float(pairwise_sigma) ** 2)).sum()
    return float(min(1.0, p))
