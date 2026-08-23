# Control-aware physical neural-network regulation research software

Project-maintained research software for the manuscript **“Control-aware training of physical neural networks for closed-loop regulation.”**

## Maintained physics-constrained estimators
- `InterferometricOracle`: coherent multi-path phase encoding, interference and intensity readout.
- `NonlinearOscillatorOracle`: 24-node damped coupled Duffing-type dynamical system.

## Confirmatory record
- E-010C1: matched-static interferometric behavioral separation.
- E-020C1: interferometric robustness success.
- E-030C1: oscillator matched-static threshold failure retained.
- E-031C1: oscillator robustness success.
- E-050C1: task-matched Pareto success.
- E-051C1: exact decision-regret mechanism success on both substrates.
- **E-060C1:** canonical Cart-Pole task-generalization success; 54.63% reduction in the predefined moderate+strong failure-padded stabilization-loss endpoint.
- E-040: descriptive external measurement-scale grounding only, not hardware validation.

## Reproducibility
```bash
python -m venv .venv
# activate environment
pip install -e '.[test]'
pytest -q
```

Source-tree invocation:
```bash
PYTHONPATH=src python -m pytest -q
```

Frozen confirmatory runners verify their checkpoint/seed/configuration manifests. Project-internal freezing is not claimed as externally time-stamped preregistration.

## Public archive
Target GitHub release: **v1.1.0**.  
Persistent Zenodo concept DOI: **10.5281/zenodo.22068583**.

No third-party research source code is copied or vendored. Generative-AI assistance is documented in the manuscript and repository documentation; the author remains responsible for the scientific work.
