# Reference verification audit - alpha.8

Date checked: 2026-08-23

Purpose: verify that the numbered reference list supports the claims for which it is cited, that bibliographic metadata are internally consistent, and that the submission-ready LaTeX follows current Nature Communications reference instructions.

## Submission-format correction

Current Nature Communications guidance states that TeX/LaTeX submissions should contain numerical references directly in the main `.tex` file; if BibTeX is used during preparation, the generated `.bbl` reference list should be pasted into the main `.tex` and the `\\bibliography` / `\\bibliographystyle` commands removed. The alpha.8 manuscript therefore embeds the complete reference list. `references.bib` is retained only as an internal audit/preparation record and is excluded from the cleaned Submission release package.

Nature reference-style checks applied:
- references numbered by first appearance;
- one publication per reference number;
- article titles included;
- first author + `et al.` for six or more authors;
- all authors listed for fewer than six authors;
- standard abbreviated journal names in the embedded list;
- complete volume/page or article-number information where available;
- DOI URL retained for the 2026 Nature Photonics online-first item that does not yet have volume/page metadata in the source checked.

## Verified bibliography

| Key | Verified citation | DOI / identifier | Claim role | Status |
|---|---|---|---|---|
| Shen2017 | Shen, Y. et al. *Deep learning with coherent nanophotonic circuits*. Nat. Photon. 11, 441-446 (2017). | 10.1038/nphoton.2017.93 | Early coherent photonic neural network | Verified |
| Feldmann2021 | Feldmann, J. et al. *Parallel convolutional processing using an integrated photonic tensor core*. Nature 589, 52-58 (2021). | 10.1038/s41586-020-03070-1 | Integrated photonic tensor processing | Verified |
| Wright2022 | Wright, L. G. et al. *Deep physical neural networks trained with backpropagation*. Nature 601, 549-555 (2022). | 10.1038/s41586-021-04223-6 | General physical-neural-network training across physical systems | Verified |
| Pai2023 | Pai, S. et al. *Experimentally realized in situ backpropagation for deep learning in photonic neural networks*. Science 380, 398-404 (2023). | 10.1126/science.ade8450 | In-situ photonic backpropagation | Verified |
| Momeni2025 | Momeni, A. et al. *Training of physical neural networks*. Nature 645, 53-61 (2025). | 10.1038/s41586-025-09384-2 | Current PNN-training framework/review | Verified |
| Wanjura2024 | Wanjura, C. C. & Marquardt, F. *Fully nonlinear neuromorphic computing with linear wave scattering*. Nat. Phys. 20, 1434-1440 (2024). | 10.1038/s41567-024-02534-9 | Nonlinear computation through parameter-modulated linear wave scattering | Verified |
| Xu2024Optica | Xu, T. et al. *Control-free and efficient integrated photonic neural networks via hardware-aware training and pruning*. Optica 11, 1039-1049 (2024). | 10.1364/OPTICA.523225 | Hardware-aware robustness and E-040 external anchor | Verified |
| Xu2026 | Xu, T. et al. *Physical neural networks using sharpness-aware training*. Nat. Commun. 17, 1766 (2026). | 10.1038/s41467-026-68470-9 | Sharpness-aware PNN robustness and E-040 external anchor | Verified |
| Sunada2025 | Sunada, S. et al. *Blending optimal control and biologically plausible learning for noise-robust physical neural networks*. Phys. Rev. Lett. 134, 017301 (2025). | 10.1103/PhysRevLett.134.017301 | Optimal-control/noise-robust PNN baseline context | Verified |
| VanAssche2026 | Van Assche, R. et al. *Real-time optical signal equalization with a silicon photonic spatially distributed reservoir computer*. Nat. Photon. (2026). | 10.1038/s41566-026-01968-2 | Real-time integrated photonic processing | Verified; online-first citation retained with DOI |
| Ashtiani2026 | Ashtiani, F., Idjadi, M. H. & Kim, K. *Integrated photonic neural network with on-chip backpropagation training*. Nature 651, 927-932 (2026). | 10.1038/s41586-026-10262-8 | On-chip backpropagation | Verified |
| Keramati2014 | Keramati, M. & Gutkin, B. *Homeostatic reinforcement learning for integrating reward collection and physiological stability*. eLife 3, e04811 (2014). | 10.7554/eLife.04811 | Normative homeostatic-control/RL motivation | Verified |
| Man2019 | Man, K. & Damasio, A. *Homeostasis and soft robotics in the design of feeling machines*. Nat. Mach. Intell. 1, 446-452 (2019). | 10.1038/s42256-019-0103-7 | Homeostasis as an engineering/agent-design motivation | Verified |
| Foret2021 | Foret, P., Kleiner, A., Mobahi, H. & Neyshabur, B. *Sharpness-aware minimization for efficiently improving generalization*. In International Conference on Learning Representations (2021). | ICLR 2021 | Definition of the SAM principle used for the independently implemented baseline | Verified; corrected from `@article`-style description to conference proceeding |

## Claim-to-reference audit

1. **PNNs demonstrated across physical platforms**: Shen2017, Feldmann2021, Wright2022, Pai2023 and Momeni2025 collectively support this positioning. Wright2022 is the strongest general citation.
2. **Linear wave scattering can yield nonlinear computation when inputs modulate physical parameters**: Wanjura2024 directly supports this claim.
3. **Robust PNN training through hardware-aware / sharpness-aware / optimal-control approaches**: Xu2024Optica, Xu2026 and Sunada2025 support the three named categories respectively.
4. **Real-time integrated photonic processing / on-chip backpropagation**: VanAssche2026 and Ashtiani2026 support these two examples; the sentence has been worded so that each citation supports its own half of the claim.
5. **Homeostatic regulation as prior agent/control motivation**: Keramati2014 and Man2019 support the motivation; the manuscript explicitly avoids extending them into claims of consciousness or artificial life.
6. **E-040 measured external anchors**: only Xu2026 and Xu2024Optica are used. The manuscript states that numerical values were transcribed from reported experimental results and does not imply that the proposed optimizer was deployed on those devices.
7. **SAM baseline**: Foret2021 is cited only for the underlying sharpness-aware minimization principle; the manuscript states that the baseline implementation is independent.

## Remaining author-side checks before final submission

- Recheck Van Assche et al. immediately before submission in case issue/volume/page metadata have been assigned after the present audit.
- If any cited work receives a correction, update the reference and the affected statement.
- Do not add secondary review citations where the claim is already supported by a direct primary source unless they materially improve positioning.
