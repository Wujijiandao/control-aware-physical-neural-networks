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


def finite_horizon_first_divergence_bound(gap_sequences, pairwise_sigma):
    """Union bound for first action divergence along an exact-reference path.

    ``gap_sequences[t]`` contains the positive score gaps J_t(a)-J_t(a*_t)
    for every competitor at exact-reference state t. ``pairwise_sigma`` may be
    a scalar or one value per time step. If, conditional on having matched the
    exact controller up to time t, each pairwise score-error difference obeys
    the same sub-Gaussian tail convention used by
    :func:`subgaussian_action_mismatch_bound`, then the probability of any
    action divergence by horizon T is at most the sum of the per-step mismatch
    bounds. No independence across time is required for this union bound.
    """
    gaps=[np.asarray(g,dtype=float) for g in gap_sequences]
    if any(np.any(g<=0) for g in gaps):
        raise ValueError("all competitor gaps must be strictly positive")
    sig=np.asarray(pairwise_sigma,dtype=float)
    if sig.ndim==0:
        sig=np.full(len(gaps),float(sig))
    if len(sig)!=len(gaps):
        raise ValueError("pairwise_sigma must be scalar or have one value per time step")
    if np.any(sig<0):
        raise ValueError("pairwise_sigma must be non-negative")
    total=0.0
    for g,s in zip(gaps,sig):
        total += subgaussian_action_mismatch_bound(g,float(s))
    return float(min(1.0,total))


def finite_horizon_match_probability_lower_bound(gap_sequences, pairwise_sigma):
    """Lower bound on preserving the exact-reference action sequence."""
    return float(max(0.0, 1.0-finite_horizon_first_divergence_bound(gap_sequences,pairwise_sigma)))
