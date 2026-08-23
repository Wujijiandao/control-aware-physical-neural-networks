# Reproducibility status — v0.3.0-alpha.9

This release separates exploratory/development work from project-internally frozen held-out confirmatory experiments. Scientific claims in the manuscript are backed by frozen checkpoints, seed commitments and fixed aggregation scripts under `frozen/` and `results/`.

## Evidence namespaces

- `archive/`, `results/*smoke*`, development-family/checkpoint searches: exploratory or engineering evidence only.
- `frozen/e010`, `frozen/e020`, `frozen/e030`, `frozen/e031`, `frozen/e050`, `frozen/e051`: committed confirmatory assets.
- `results/e010_confirmatory`, `e020_confirmatory*`, `e030_confirmatory`, `e031_confirmatory`, `e050_confirmatory`, `e051_confirmatory`: confirmatory outcomes.
- E-040 is a project-authored factual extraction and analysis of published experimental measurements, not a confirmatory deployment of the proposed optimizer.

## Reproduction

```bash
python -m venv .venv
# activate the environment
pip install -e '.[test]'
pytest -q
```

For a source-tree-only check without editable installation, use:

```bash
PYTHONPATH=src python -m pytest -q
```

Confirmatory runners verify frozen asset hashes. Deterministic sharding used in long E-030/E-031/E-050/E-051 jobs changes only execution scheduling, not seeds, models, endpoints or analysis. Aggregators reject incomplete/duplicate frozen-key coverage.

## Software provenance

All research code in this package is project-maintained; no third-party research source repository is copied or vendored. NumPy, PyTorch, pandas, Matplotlib and related packages are runtime dependencies. Material generative-AI assistance in code drafting/debugging is disclosed in `docs/AI_ASSISTANCE.md` and does not alter the no-vendored-third-party-source boundary. Third-party binary experimental source data are not redistributed.
