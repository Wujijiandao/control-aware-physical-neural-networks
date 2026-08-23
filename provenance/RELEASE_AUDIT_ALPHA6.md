# Release audit — v0.3.0-alpha.6

Date: 2026-08-23

- Project software tests: 22/22 passed with `PYTHONPATH=src`.
- Cleaned public-package tests: 22/22 passed with imports resolved only from the public package.
- Final manuscript build: pdflatex -> bibtex8 -> pdflatex -> pdflatex, successful; 16 pages; 12 resolved bibliography entries.
- Final PDF rendered and inspected on title/abstract, mechanism/external-audit and bibliography pages; no observed clipping/overlap/broken glyphs.
- Project content manifest and each release-directory manifest were regenerated after content freeze and verified before ZIP creation.
- Public package excludes Springer Nature template files, internal research notes/development-result clutter, third-party research source code and third-party binary experimental measurement files.
- Submission package excludes internal development directories and LaTeX build caches/logs.
