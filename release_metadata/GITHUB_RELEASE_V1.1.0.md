# GitHub Release v1.1.0 — Reproducibility release for closed-loop PNN regulation

## Release form
- **Tag:** `v1.1.0`
- **Target:** `main`
- **Release title:** `v1.1.0 - Closed-loop regulation reproducibility update`
- **Pre-release:** No
- **Latest release:** Yes

## Paste-ready release notes

This release updates the reproducibility archive accompanying the manuscript **“Control-aware training of physical neural networks for closed-loop regulation.”**

### New in v1.1.0

- Adds **E-060C1**, a frozen canonical Cart-Pole stabilization study using 128 new held-out seeds.
- Two equal-budget interferometric estimators had nearly identical global static accuracy (RMSE 0.279056 vs 0.278873), while the control-aware model had worse candidate-state pointwise RMSE (0.2047 vs 0.0527).
- Despite that pointwise disadvantage, control-aware ranking reduced the predefined moderate+strong failure-padded stabilization loss from 8.9882 to 4.0783: **54.63% relative reduction**, paired-bootstrap 95% CI **[-5.4999, -4.3071]**.
- Adds the finite-horizon first-divergence theory result, Cart-Pole source data, frozen configuration/seed integrity records and Fig. 7 source data.
- Expands the statistical sensitivity and claim/source-data audits to the E-060 result.

### Previously retained confirmatory record

- E-010C1: matched-static interferometric behavioral separation, with the task-utility trade-off retained.
- E-020C1: 17.86% interferometric robustness improvement.
- E-030C1: 4.14% oscillator matched-static effect retained as a threshold failure because it did not meet the frozen 5% minimum effect.
- E-031C1: 28.63% oscillator robustness improvement.
- E-050C1: 11.95% task-matched viability-cost reduction at equal cumulative workload.
- E-051C1: exact common-state decision-regret reductions of 49.24% and 18.64% on the interferometric and oscillator substrates.
- E-040: external perturbation-severity grounding only; not hardware validation.

### Archive identity

Persistent Zenodo concept DOI: **10.5281/zenodo.22068583**. Zenodo may assign a new version-specific DOI when this release is uploaded; the concept DOI remains stable.

MIT License applies to project-maintained research software. No third-party research source code or third-party binary experimental files are redistributed.
