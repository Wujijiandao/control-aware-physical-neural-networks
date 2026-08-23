#!/usr/bin/env python3
"""E-040 external empirical stress-envelope audit.

Project-original analysis code. It operates on a project-authored factual extraction
of published experimental measurements. It does not import or execute third-party
research code and does not redistribute third-party binary source data.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = ROOT / "data" / "external" / "e040_published_measurement_anchors.csv"
E020 = ROOT / "results" / "e020_confirmatory_secondary" / "e020c1_all_summary.csv"
OUTDIR = ROOT / "results" / "e040_external_grounding"


def fnum(value: str):
    value = (value or "").strip()
    return None if value == "" else float(value)


def load_anchors():
    with ANCHORS.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def analyse_anchor(row):
    chance = float(row["chance_accuracy"])
    baseline = float(row["baseline_stressed_accuracy"])
    robust = float(row["robust_stressed_accuracy"])
    b_ref = fnum(row["baseline_reference_accuracy"])
    r_ref = fnum(row["robust_reference_accuracy"])

    out = dict(row)
    out["baseline_excess_fraction"] = (baseline - chance) / (1.0 - chance)
    out["robust_excess_fraction"] = (robust - chance) / (1.0 - chance)
    out["stressed_accuracy_uplift_pp"] = 100.0 * (robust - baseline)
    out["baseline_retention_to_reference"] = None
    out["robust_retention_to_reference"] = None
    out["baseline_relative_loss_from_reference"] = None
    out["robust_relative_loss_from_reference"] = None
    if b_ref is not None:
        out["baseline_retention_to_reference"] = (baseline - chance) / (b_ref - chance)
        out["baseline_relative_loss_from_reference"] = 1.0 - out["baseline_retention_to_reference"]
    if r_ref is not None:
        out["robust_retention_to_reference"] = (robust - chance) / (r_ref - chance)
        out["robust_relative_loss_from_reference"] = 1.0 - out["robust_retention_to_reference"]
    return out


def load_e020_boundary():
    rows = []
    with E020.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["method"] == "boundary_aware":
                rows.append(row)
    by_condition = {r["condition"]: float(r["mean_D"]) for r in rows}
    nominal = by_condition["nominal"]
    return {
        "nominal_mean_D": nominal,
        "moderate_mean_D": by_condition["moderate"],
        "strong_mean_D": by_condition["strong"],
        "moderate_cost_inflation": by_condition["moderate"] / nominal - 1.0,
        "strong_cost_inflation": by_condition["strong"] / nominal - 1.0,
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    analysed = [analyse_anchor(r) for r in load_anchors()]
    with_ref = [r for r in analysed if r["baseline_retention_to_reference"] is not None]

    baseline_losses = [r["baseline_relative_loss_from_reference"] for r in with_ref]
    robust_losses = [r["robust_relative_loss_from_reference"] for r in with_ref]
    baseline_excess = [r["baseline_excess_fraction"] for r in analysed]
    robust_excess = [r["robust_excess_fraction"] for r in analysed]
    uplifts = [r["stressed_accuracy_uplift_pp"] for r in analysed]
    e020 = load_e020_boundary()

    summary = {
        "experiment": "E-040",
        "status": "EXTERNAL_EMPIRICAL_GROUNDING_DESCRIPTIVE",
        "n_published_measurement_anchors": len(analysed),
        "n_reference_normalized_anchors": len(with_ref),
        "external_baseline_relative_loss_range": [min(baseline_losses), max(baseline_losses)],
        "external_baseline_relative_loss_median": median(baseline_losses),
        "external_robust_relative_loss_range": [min(robust_losses), max(robust_losses)],
        "external_robust_relative_loss_median": median(robust_losses),
        "external_baseline_stressed_excess_accuracy_range": [min(baseline_excess), max(baseline_excess)],
        "external_robust_stressed_excess_accuracy_range": [min(robust_excess), max(robust_excess)],
        "external_stressed_accuracy_uplift_pp_range": [min(uplifts), max(uplifts)],
        "e020_boundary_aware": e020,
        "interpretation": (
            "Published experimental PNN perturbations produce large deployment losses. "
            "For anchors with explicit references, baseline chance-adjusted excess-accuracy loss spans "
            "roughly 20-63%, while the E-020 boundary-aware synthetic moderate/strong viability-cost "
            "inflation is roughly 21-60%. The metrics are not equivalent; this is a severity-envelope "
            "grounding check, not experimental validation of the project optimizer."
        ),
    }

    metric_fields = [
        "anchor_id", "source_key", "doi", "platform", "task", "perturbation",
        "baseline_method", "robust_method", "baseline_reference_accuracy",
        "robust_reference_accuracy", "baseline_stressed_accuracy", "robust_stressed_accuracy",
        "chance_accuracy", "baseline_excess_fraction", "robust_excess_fraction",
        "stressed_accuracy_uplift_pp", "baseline_retention_to_reference",
        "robust_retention_to_reference", "baseline_relative_loss_from_reference",
        "robust_relative_loss_from_reference", "evidence_scope", "extraction_note",
    ]
    with (OUTDIR / "e040_anchor_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=metric_fields)
        w.writeheader()
        for row in analysed:
            w.writerow({k: "" if row.get(k) is None else row.get(k) for k in metric_fields})

    (OUTDIR / "e040_external_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
