# GitHub Release metadata - proposed archival release

## Release form

- **Tag:** `v1.0.0`
- **Target:** `main`
- **Release title:** `v1.0.0 - Reproducibility release for control-aware PNN training`
- **Pre-release:** No, once this is the version intended to accompany the submitted/public manuscript
- **Latest release:** Yes

GitHub releases are tag-based and GitHub automatically provides source-code ZIP and tar.gz archives for the tagged commit. If desired, additionally attach the cleaned `PIHA_NC_Public_v1.0.0.zip` as a named release asset, but do not attach the internal Project, Submission or ResearchNotes packages.

## Release notes - paste-ready

### Reproducibility release

This release contains the research software, held-out evaluation configurations, source data and integrity records accompanying the manuscript **“Control-aware training of physical neural networks for homeostatic regulation.”**

The repository implements two independent physics-constrained model families: a coherent interferometric estimator and a nonlinear coupled-Duffing-oscillator estimator. It includes the evaluation assets underlying the manuscript's main computational claims, together with the retained negative result and external measurement-scale audit.

Key archived analyses include:

- **E-010C1:** matched-static interferometric evaluation showing that nearly identical static regression accuracy can yield different closed-loop trajectories; the nominal viability gain carried a task-utility trade-off and is not presented as task-matched Pareto dominance.
- **E-020C1:** interferometric robustness comparison; robust control-aware training reduced viability cost under moderate+strong perturbations by 17.86% relative to the predefined boundary-aware comparator.
- **E-030C1:** nonlinear-oscillator nominal matched-static replication; the 4.14% effect did not meet the project-internal 5% minimum practical-effect criterion and is retained as a threshold failure.
- **E-031C1:** nonlinear-oscillator robustness evaluation; viability cost under perturbation decreased by 28.63% without a task-utility penalty.
- **E-050C1:** task-matched Pareto evaluation; the interferometric robustness advantage retained an 11.95% viability-cost reduction at equal cumulative workload.
- **E-051C1:** common reference-state mechanism audit; exact one-step decision regret decreased by 49.24% on the interferometric substrate and 18.64% on the oscillator substrate.
- **E-040:** descriptive external audit of reported experimental PNN deployment errors used only to assess perturbation-scale plausibility; it is not a hardware validation of the proposed method.

### Reproducibility and provenance

- No third-party research source code is copied or vendored into the repository.
- Runtime scientific libraries are declared as dependencies.
- Third-party binary experimental data are not redistributed.
- Held-out seed lists, configurations, project checkpoints, per-seed source data and integrity manifests required for the reported analyses are included where redistribution is appropriate.
- Project-internal protocol freezing is documented, but no claim of externally time-stamped preregistration is made.
- Generative-AI assistance is documented transparently in the repository and manuscript; the author remains responsible for the scientific content.

### License

MIT License for project-maintained research software. Source-data provenance and any third-party rights are documented separately.
