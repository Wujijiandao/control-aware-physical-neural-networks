import numpy as np
from piha.cartpole import (
    CartPoleParams, THETA_THRESHOLD, X_THRESHOLD, action_sequences,
    exact_candidate_scores, failed, finite_horizon_value, normalize_state, step,
)


def test_cartpole_equilibrium_shape_and_finiteness():
    s=np.zeros(4); n=step(s,1)
    assert n.shape==(4,)
    assert np.isfinite(n).all()


def test_cartpole_failure_boundaries():
    assert not bool(failed(np.zeros(4)))
    assert bool(failed(np.array([X_THRESHOLD+1e-3,0,0,0])))
    assert bool(failed(np.array([0,0,THETA_THRESHOLD+1e-3,0])))


def test_normalization_center():
    assert np.allclose(normalize_state(np.zeros(4)), 0.5)


def test_action_sequences_count():
    assert action_sequences(5).shape==(32,5)


def test_value_and_candidate_scores_are_finite():
    s=np.array([[0.0,0.0,0.03,0.0],[0.1,-0.1,-0.04,0.2]])
    v=finite_horizon_value(s,horizon=3)
    q=exact_candidate_scores(s,horizon=3)
    assert v.shape==(2,) and q.shape==(2,2)
    assert np.isfinite(v).all() and np.isfinite(q).all()
    assert (v>=0).all() and (q>=0).all()
