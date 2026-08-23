# PIHA control-aware physical homeostasis research software

This directory contains **project-maintained** research software for the single-paper Nature Communications study.

## Provenance rule

No third-party research source code is vendored, copied, patched, or mirrored here. Runtime libraries such as NumPy, PyTorch and pandas are normal declared dependencies only. Public experimental measurements may later be analysed as external data, but their source, license, exact files and SHA-256 hashes must be recorded before confirmatory use.

## Maintained physical substrates

- `InterferometricOracle`: coherent multi-path phase encoding, interference and intensity readout.
- `NonlinearOscillatorOracle`: 24-node damped coupled Duffing-type dynamical system with bounded state-to-force transduction and finite-time physical readout.

Both implementations are maintained within this project without copying or vendoring third-party research source code; disclosed generative-AI assistance is documented separately and all generated suggestions were subject to human review and execution-based verification.

## Evidence levels

The repository intentionally retains exploratory failures, development runs and frozen confirmatory outputs separately. Current confirmatory record:

- E-010C1: interferometric matched-static primary success;
- E-020C1: interferometric robustness primary success;
- E-030C1: oscillator matched-static **primary threshold failure** (4.14% < frozen 5% minimum despite CI excluding zero);
- E-031C1: oscillator robustness primary success;
- E-050C1: task-matched Pareto primary success on the frozen E-020 pair;
- E-051C1: common-state exact decision-regret mechanism success on both physical substrates.

E-040 is external descriptive grounding and is not counted as a confirmatory test of the optimizer. Do not reinterpret development outputs as confirmatory evidence.

## Reproducibility

```bash
python -m venv .venv
# activate environment
pip install -e '.[test]'
pytest -q
```

A source-tree-only test invocation is also supported:

```bash
PYTHONPATH=src python -m pytest -q
```

Frozen confirmatory runners verify their checkpoint/seed manifests. E-030 and E-031 use deterministic seed sharding only to satisfy wall-clock limits; sharding does not alter the frozen scientific task.


## E-040 external empirical grounding

`experiments/e040_external_measurement_audit.py` analyses a project-authored extraction of published experimental measurements. No third-party research code or binary measurement data are bundled. E-040 is a perturbation-severity audit, not hardware validation of the proposed optimizer.


## Archival release metadata

An archival `v1.0.0` release is public at https://github.com/Wujijiandao/control-aware-physical-neural-networks/releases/tag/v1.0.0 and persistently archived at Zenodo DOI `10.5281/zenodo.22068583` (https://doi.org/10.5281/zenodo.22068583). `release_metadata/` is retained as an audit trail of the release metadata. The local alpha.9 package is a post-archive submission mirror; the DOI-bound v1.0.0 record remains the citable software archive.
