# Language and terminology audit — alpha.8

Date: 2026-08-23
Scope: Nature Communications working manuscript.

## Field terminology decisions

- **Physical neural network (PNN)** is retained as the established field term for trainable physical computing systems. The project-specific numerical models are described more narrowly as **physics-constrained estimators**, **physics-constrained models**, or **physical substrates**, because the present work does not report a device fabricated or operated by the author.
- Project numerical runs are called **evaluations**, **held-out evaluations**, **computations**, or **simulations**. The adjective **experimental** is reserved for measurements reported in cited hardware studies and for generic references to experiments in the literature.
- **Homeostasis** is used strictly in the control-theoretic sense of regulating engineering state variables within a viable operating region. The manuscript makes no claim about consciousness, desire, feeling, life, or moral status.
- **Viability cost** is the scalar control objective used to quantify departure from the viable operating region.
- **Action gap** is the difference between the exact best and second-best candidate-action scores.
- **Decision regret** is the exact score penalty of the action selected by an approximate estimator relative to the exact reference controller.
- **Exact reference controller** replaces “teacher” to avoid importing supervised-learning semantics into the control benchmark.
- **Candidate-state RMSE** denotes root-mean-square viability-estimation error evaluated on one-step candidate successor states.
- **Boundary-aware training** emphasizes approximation quality near the viability boundary.
- **Robust control-aware training** augments action-ranking protection using physics-motivated deployment perturbations.
- **Physics-motivated deployment perturbations** denotes project-defined input, readout, and parameter perturbations motivated by known physical non-idealities. They are not direct samples from hardware operated in this study.
- **Frozen** and **confirmatory** describe project-internal governance only. The hashes were not externally timestamped or preregistered before execution, and the manuscript states this explicitly.

## Academic-English decisions

- US English spelling is used consistently (behavior, optimization).
- Claims are phrased in terms of what the data establish, avoiding “prove” for empirical results and avoiding universal statements across all physical systems.
- “Improved regulation” is used only when the relevant comparison supports it; E-010 is described as closed-loop behavioral separation because its nominal viability gain carried a task-utility trade-off.
- E-030 remains a project-internal threshold failure despite a confidence interval excluding zero.
- Hardware-implying phrases such as “interferometric experiments” were replaced by “interferometric evaluations” for project-generated numerical evidence.
- External published measurements remain explicitly labeled experimental.
- The manuscript distinguishes static regression accuracy from closed-loop behavioral fidelity and avoids using “accuracy” alone where RMSE, viability cost, action agreement, or decision regret is the actual metric.

## Claim boundary

The manuscript supports a computational training/evaluation principle for physics-constrained approximators embedded in feedback control. It does **not** claim direct hardware validation of the proposed optimizer, artificial life, consciousness, subjective feeling, or universal superiority of control-aware training.
