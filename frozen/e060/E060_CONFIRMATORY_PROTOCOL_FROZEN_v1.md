# E-060C1 frozen confirmatory protocol

Status: **project-internal freeze before confirmatory outcome evaluation**. This is not an externally time-stamped preregistration.

## Question
Does control-aware candidate ranking improve closed-loop stabilization on the canonical cart-pole benchmark when two interferometric physical estimators have nearly identical global static accuracy?

## Fixed models
- `noise_aware_mse.pt`: 2600 training steps; pointwise field + noisy-candidate MSE.
- `control_aware_rank.pt`: same initialization family and 2600 training steps; pointwise field + noisy candidate-action ranking, rank weight 0.02.

Calibration eligibility was fixed before confirmatory evaluation: both models must have R2 >= 0.95, |Delta R2| <= 0.001 and global RMSE ratio <= 1.02. Candidate-state RMSE is *not* a matching criterion because the scientific question is whether pointwise candidate error and decision ordering can dissociate.

## Benchmark
Project-maintained implementation of the standard frictionless CartPole equations associated with Barto, Sutton & Anderson (1983). State: cart position/velocity and pole angle/angular velocity. Actions: fixed left/right force. Canonical failure boundaries are |x| > 2.4 and |theta| > 12 degrees. The reference candidate score is a short-horizon nonlinear enumerative MPC value with a discrete-LQR terminal heuristic; this reference is deterministic and identical across compared methods.

## Deployment conditions
Nominal is secondary. Moderate and strong conditions jointly perturb plant parameters, sensed state and the interferometric estimator using the immutable values in `conditions.json`.

## Primary endpoint
For each held-out seed, compute the failure-padded 500-step stabilization loss in moderate and strong conditions for both methods, average the two condition losses within seed, and form the paired difference (control-aware rank minus noise-aware MSE).

Primary success requires both:
1. 95% paired percentile-bootstrap CI (20,000 resamples; seed 606060) lies strictly below zero;
2. relative reduction from the noise-aware MSE mean is at least 10%.

## Secondary endpoints
Nominal stabilization loss; survival steps; 500-step success; action agreement with the exact reference controller; exact one-step decision regret; first-divergence time and exact action gap. No secondary endpoint can rescue a failed primary endpoint.

## Sample and integrity
128 held-out seeds are committed in `confirmatory_seeds.txt`. Checkpoints, conditions, configuration, seed list and this protocol are SHA-256 committed in `E060_FROZEN_MANIFEST.sha256`. Development seeds/outcomes are excluded from confirmatory inference.
