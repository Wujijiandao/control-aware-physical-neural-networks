"""Canonical cart-pole stabilization benchmark used for E-060.

The equations follow the standard frictionless cart-pole formulation used by the
classic Barto--Sutton--Anderson pole-balancing benchmark and modern CartPole
implementations. This module is project-maintained and has no Gym/Gymnasium
runtime dependency.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np


@dataclass(frozen=True)
class CartPoleParams:
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5  # half pole length in the standard equations
    force_mag: float = 10.0
    tau: float = 0.02


X_THRESHOLD = 2.4
THETA_THRESHOLD = 12.0 * np.pi / 180.0
# Input normalization bounds deliberately exceed termination thresholds for
# velocity and angle so near-failure states remain representable.
STATE_LOW = np.array([-2.4, -3.5, -0.32, -4.0], dtype=float)
STATE_HIGH = np.array([2.4, 3.5, 0.32, 4.0], dtype=float)
VALUE_SCALE = 10.0


def normalize_state(state):
    x = np.asarray(state, dtype=float)
    z = (x - STATE_LOW) / (STATE_HIGH - STATE_LOW)
    return np.clip(z, 0.0, 1.0)


def denormalize_state(z):
    z = np.asarray(z, dtype=float)
    return STATE_LOW + np.clip(z, 0.0, 1.0) * (STATE_HIGH - STATE_LOW)


def failed(state):
    x = np.asarray(state, dtype=float)
    return (np.abs(x[..., 0]) > X_THRESHOLD) | (np.abs(x[..., 2]) > THETA_THRESHOLD)


def stage_cost(state):
    """Smooth stabilization cost with a strong failure barrier.

    The primary benchmark endpoint additionally pads post-failure steps with a
    fixed penalty, so this smooth cost does not redefine the canonical failure
    condition.
    """
    s = np.asarray(state, dtype=float)
    xn = s[..., 0] / X_THRESHOLD
    xdn = s[..., 1] / 3.5
    thn = s[..., 2] / THETA_THRESHOLD
    thdn = s[..., 3] / 4.0
    base = 0.12 * xn**2 + 0.025 * xdn**2 + 1.75 * thn**2 + 0.06 * thdn**2
    # Smoothly penalize approaching/exceeding the canonical failure boundary.
    edge = np.maximum(0.0, np.abs(xn) - 0.85)**2 + 4.0 * np.maximum(0.0, np.abs(thn) - 0.85)**2
    hard = failed(s).astype(float) * 12.0
    return base + 2.5 * edge + hard


def step_force(state, force, params: CartPoleParams = CartPoleParams()):
    """One explicit-Euler cart-pole step under an arbitrary horizontal force."""
    s = np.asarray(state, dtype=float)
    x, x_dot, theta, theta_dot = [s[..., i] for i in range(4)]
    force = np.asarray(force, dtype=float)
    total_mass = params.masspole + params.masscart
    polemass_length = params.masspole * params.length
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    thetaacc = (params.gravity * sintheta - costheta * temp) / (
        params.length * (4.0 / 3.0 - params.masspole * costheta**2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    out = np.stack([
        x + params.tau * x_dot,
        x_dot + params.tau * xacc,
        theta + params.tau * theta_dot,
        theta_dot + params.tau * thetaacc,
    ], axis=-1)
    return out


def step(state, action, params: CartPoleParams = CartPoleParams()):
    """One cart-pole step for the canonical binary left/right action set."""
    force = np.where(np.asarray(action) > 0, params.force_mag, -params.force_mag)
    return step_force(state, force, params)


def linearize_origin(params: CartPoleParams = CartPoleParams(), eps: float = 1e-6):
    """Numerically linearize the discrete dynamics at the upright origin."""
    z=np.zeros(4,dtype=float)
    A=np.empty((4,4),dtype=float)
    for j in range(4):
        d=np.zeros(4);d[j]=eps
        A[:,j]=(step_force(z+d,0.0,params)-step_force(z-d,0.0,params))/(2*eps)
    B=((step_force(z,eps,params)-step_force(z,-eps,params))/(2*eps)).reshape(4,1)
    return A,B


def lqr_terminal_matrix(params: CartPoleParams = CartPoleParams()):
    """Discrete Riccati solution used only as a terminal stabilizing heuristic."""
    A,B=linearize_origin(params)
    Q=np.diag([0.8,0.08,18.0,0.18])
    R=np.array([[0.02]])
    P=Q.copy()
    for _ in range(10000):
        S=R+B.T@P@B
        K=np.linalg.solve(S,B.T@P@A)
        Pn=Q+A.T@P@A-A.T@P@B@K
        if np.max(np.abs(Pn-P))<1e-11:
            P=Pn;break
        P=Pn
    return P


_LQR_P=lqr_terminal_matrix()

def lqr_terminal_cost(state):
    s=np.asarray(state,dtype=float)
    return np.einsum("...i,ij,...j->...",s,_LQR_P,s)


def action_sequences(horizon: int):
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon == 0:
        return np.zeros((1, 0), dtype=np.int8)
    return np.asarray(list(product((0, 1), repeat=horizon)), dtype=np.int8)


def finite_horizon_value(states, horizon: int = 4, discount: float = 0.97,
                         params: CartPoleParams = CartPoleParams(), chunk: int = 1024,
                         terminal_weight: float = 0.035):
    """Exact short-horizon nonlinear MPC value with an LQR terminal heuristic.

    The binary action sequences are enumerated exactly.  The terminal quadratic
    comes from a local discrete Riccati solution and extends the effective
    planning horizon without introducing an external optimization package.
    """
    x = np.asarray(states, dtype=float)
    orig = x.shape[:-1]
    flat = x.reshape(-1, 4)
    seqs = action_sequences(horizon)
    out = np.empty(len(flat), dtype=float)
    for start in range(0, len(flat), chunk):
        b = flat[start:start+chunk]
        n = len(b); q = len(seqs)
        st = np.repeat(b[:, None, :], q, axis=1)
        total = np.broadcast_to(stage_cost(b)[:, None], (n, q)).copy()
        alive = ~failed(st)
        for k in range(horizon):
            acts = np.broadcast_to(seqs[None, :, k], (n, q))
            nxt = step(st, acts, params)
            total += (discount ** (k + 1)) * stage_cost(nxt)
            st = np.where(alive[..., None], nxt, st)
            alive &= ~failed(nxt)
        total += (discount ** (horizon + 1)) * terminal_weight * lqr_terminal_cost(st)
        out[start:start+n] = total.min(axis=1)
    return out.reshape(orig) / VALUE_SCALE


def candidate_afterstates(states, params: CartPoleParams = CartPoleParams()):
    s = np.asarray(states, dtype=float)
    left = step(s, 0, params)
    right = step(s, 1, params)
    return np.stack([left, right], axis=-2)


def exact_candidate_scores(states, horizon: int = 4, discount: float = 0.97,
                           params: CartPoleParams = CartPoleParams()):
    cand = candidate_afterstates(states, params)
    shape = cand.shape[:-1]
    vals = finite_horizon_value(cand.reshape(-1, 4), horizon=horizon,
                                discount=discount, params=params)
    return vals.reshape(shape)


def exact_action(states, horizon: int = 4, discount: float = 0.97,
                 params: CartPoleParams = CartPoleParams()):
    return np.argmin(exact_candidate_scores(states, horizon, discount, params), axis=-1)
