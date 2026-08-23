# Reproducibility — v1.1.0

## Environment
Python dependencies are declared in `pyproject.toml` / `requirements.txt`.

```bash
python -m venv .venv
# activate the environment
pip install -e '.[test]'
pytest -q
```

For a source-tree-only run:
```bash
PYTHONPATH=src python -m pytest -q
```

## Confirmatory integrity
Frozen experiment directories contain immutable-in-project checkpoints, seed commitments, configurations and SHA-256 manifests. These controls establish internal provenance; they are not described as external preregistration.

E-060C1 adds the canonical Cart-Pole benchmark. Its project-maintained dynamics implementation is in `src/piha/cartpole.py`; the confirmatory shard/aggregate runners and frozen protocol are included. No Gym/Gymnasium runtime implementation is required.

## Public release identity
- GitHub release to publish: `v1.1.0`
- Persistent Zenodo concept DOI: `10.5281/zenodo.22068583`

If Zenodo generates a version-specific DOI for v1.1.0, preserve it in the repository release metadata while continuing to use the concept DOI where a persistent all-version software identifier is desired.
