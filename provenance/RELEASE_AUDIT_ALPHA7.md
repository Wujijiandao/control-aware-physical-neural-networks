# Release audit — v0.3.0-alpha.8

Date: 2026-08-23

- Project software tests: **25/25 passed** using `PYTHONPATH=src python -m pytest -q`.
- Cleaned public-package tests: **23/23 passed** with imports and test assets resolved only from the public release tree.
- Reproducibility dependency audit corrected the package metadata to declare NumPy, PyTorch, pandas and the pytest test extra.
- A-060 statistical reporting sensitivity, A-061 manuscript-claim traceability, and A-062 submission source-data/seed-key audits were rerun after the final manuscript wording changes.
- Final manuscript build: `pdflatex -> bibtex8 -> pdflatex -> pdflatex`, successful; **17 A4 pages**; no undefined citations/references in the final pass.
- Final PDF was rendered and visually inspected at the title/abstract, statistics/internal-freeze disclosure, generative-AI disclosure, and late-results/bibliography regions; no observed clipping or overlap.
- “Frozen/confirmatory” is explicitly scoped as project-internal governance, not externally timestamped preregistration.
- Material generative-AI assistance is disclosed in Methods and public provenance; “project-original” is limited to the no-copied/no-vendored-third-party-research-source criterion.
- Novelty positioning was stress-tested against recent robustness/control-oriented PNN work, including Sunada et al. (PRL 2025) and Ashtiani et al. (Nature 2026); the manuscript claim is narrowed to downstream closed-loop action-consequence training rather than generic robust PNN training.
- Public package excludes Springer Nature template files, internal research notes, submission-governance audits A-061/A-062, third-party research source code, and third-party binary experimental measurement files.
- Submission package excludes internal development directories and LaTeX build caches/logs.
- Remaining submission blockers are metadata/deposit items: corresponding-author email, final funding/acknowledgements text, and persistent public code/data DOI.
