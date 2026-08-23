# Zenodo / GitHub archival workflow and metadata

## Recommended route: GitHub -> Zenodo integration

1. Create and populate the public GitHub repository.
2. Connect the GitHub account to Zenodo.
3. In Zenodo, open the GitHub integration page, click **Sync now**, find the repository and enable its toggle.
4. Before the final GitHub release, place the final `CITATION.cff` at repository root.
5. For explicit Zenodo-specific metadata, copy `ZENODO_V1.0.0.json` to repository root and rename it `.zenodo.json`.
6. Publish GitHub Release `v1.0.0`.
7. Zenodo should ingest/archive the enabled release automatically. Verify the record and DOI.
8. Record both the **version-specific DOI** and the concept/software record where Zenodo provides them.
9. Insert the appropriate DOI into the manuscript Code/Data availability text and into `CITATION.cff` if desired.

## Important metadata precedence

Zenodo currently supports both `CITATION.cff` and `.zenodo.json`. If both are present, **Zenodo uses `.zenodo.json` for GitHub-release archiving and ignores `CITATION.cff` for that ingestion**. Keep `CITATION.cff` anyway because GitHub uses it to render repository citation information.

Therefore, after editing either file, ensure creator name, title, version and license agree across both.

## Paste-ready Zenodo fields for manual review

- **Resource type / upload type:** Software
- **Title:** `Control-aware training of physical neural networks for homeostatic regulation: research software`
- **Creator:** `Zhang, Yuzhan`
- **Affiliation:** `Independent Researcher`
- **Version:** `1.0.0`
- **Language:** English (`eng`)
- **Access:** Open
- **License:** MIT
- **Funding / grants:** None
- **Keywords:** physical neural networks; physical computing; closed-loop control; homeostatic control; photonic computing; nonlinear dynamical systems; robust optimization; reproducible research
- **Description:** use the `description` field in `ZENODO_V1.0.0.json`.

## Related identifiers

Do not invent the article DOI before the article has one. After a journal DOI or citable preprint DOI exists, add it to the Zenodo record as a related identifier with a relation such as **is supplement to** / the closest current Zenodo relationship offered for the associated article.

## DOI timing

Under the GitHub integration route, Zenodo does not provide a DOI to place in metadata *before* the GitHub release is archived. If a DOI must be reserved in advance, use a **manual Zenodo deposit** rather than the GitHub-integration-only workflow.

## Metadata quality checks before releasing

- Exact creator spelling: `Zhang, Yuzhan`.
- Do not add an academic institution; current status is Independent Researcher.
- Do not add grants or funding identifiers.
- Ensure `.zenodo.json` is valid JSON.
- Ensure the release tag/version matches `1.0.0`.
- Ensure the repository LICENSE is MIT.
- Ensure no journal-template files, internal notes or third-party binary research data are present in the public repository.
