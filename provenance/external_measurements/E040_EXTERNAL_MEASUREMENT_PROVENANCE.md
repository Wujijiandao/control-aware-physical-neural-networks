# E-040 external measurement provenance and reuse policy

Version: v0.3.0-alpha.5  
Date: 2026-08-23

## Purpose

E-040 is an **external empirical stress-envelope audit**, not a claim that the project method was deployed on third-party hardware. It asks whether the severity and qualitative structure of the project-defined physical perturbations are plausible relative to experimentally reported deployment errors in real physical neural networks (PNNs).

## Source A — Xu et al., Nature Communications (2026)

- Citation: Tengji Xu et al., *Physical neural networks using sharpness-aware training*, Nature Communications 17, 1766 (2026).
- DOI: `10.1038/s41467-026-68470-9`
- Article: open access under CC BY 4.0.
- Public code repository: `https://github.com/cuhkhuangslab/Sharpness-Aware-Training`
- Repository license: MIT for the repository code.
- Public source-data location reported by the article: Zenodo DOI `10.5281/zenodo.17905440`.
- GitHub source-data directory inspected: `Figure-raw data/`.
- Example public source-data file independently identified: `Figure-raw data/Figure2d.xlsx`, GitHub blob SHA `7af45c3299e71dbac12e10fa156514a0c00d07a2`, size 58,702 bytes.

### Important license boundary

The project does **not** infer that a code repository's MIT license automatically licenses every associated experimental data file. For alpha.5, no third-party binary source-data file is redistributed. The project-owned anchor table contains only factual numerical values explicitly reported in the open-access article text, together with citation and extraction notes.

## Source B — Xu et al., Optica (2024)

- Citation: Tengji Xu et al., *Control-free and efficient integrated photonic neural networks via hardware-aware training and pruning*, Optica 11, 1039–1049 (2024).
- DOI: `10.1364/OPTICA.523225`
- Publisher page identifies publication under the Optica Open Access Publishing Agreement.
- For alpha.5, only factual values explicitly stated in the article are transcribed. No third-party code or binary data are redistributed.

## Derived project-owned data

File: `software/data/external/e040_published_measurement_anchors.csv`

This CSV is a project-authored factual extraction table. It records:

- source DOI and source key;
- physical platform and task;
- physical perturbation;
- reported baseline and robustness-oriented method;
- explicitly reported nominal/reference accuracy when available;
- explicitly reported stressed accuracy;
- chance level for the 10-class task;
- a textual extraction note.

Blank reference fields mean the article text used for extraction did not provide a sufficiently explicit nominal value for that method/condition. The analysis script must not fill such values by inference.

## Interpretation boundary

E-040 can support the statement that real PNN deployment errors can induce accuracy losses comparable in scale to, or larger than, the project-defined synthetic stress regimes. It does **not** support the statement that control-aware training itself has been experimentally validated on these external devices.
