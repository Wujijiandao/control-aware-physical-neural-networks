# Non-third-party software policy

## Scope

This policy applies to `software/src`, `software/experiments`, and `software/tests`.

## Rules

1. No source file from GitHub, Zenodo, supplementary-code archives, vendor SDK examples, or other external research repositories may be copied into the project.
2. Published equations may be independently implemented. The relevant paper must be cited in the research notebook/manuscript when scientifically material.
3. External Python packages are permitted only as declared dependencies; their source is not bundled.
4. External experimental data are permitted only as data, never as a hidden software dependency. Their origin, license, retrieval date, version/DOI, and SHA-256 must be recorded.
5. Any future code contribution whose provenance is uncertain is quarantined until reviewed.
6. `vendor/`, copied repositories, binary wheels, and embedded site-packages are forbidden.
7. Generative-AI assistance in code drafting/debugging is permitted only with human review, execution-based verification, and explicit disclosure; The provenance claim is limited to no copied/vendored third-party research source repositories; it does not mean that every line was typed without disclosed generative-AI assistance.

## Audit criterion

The engineering package is considered source-original when every tracked source file has project provenance and the automated provenance test finds no vendored-code directories or bundled package archives.
