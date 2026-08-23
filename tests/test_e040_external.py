import csv
from pathlib import Path


def test_e040_anchor_table_is_explicit_and_nonempty():
    root = Path(__file__).resolve().parents[1]
    p = root / "data/external/e040_published_measurement_anchors.csv"
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 7
    for r in rows:
        assert r["doi"]
        assert r["evidence_scope"] == "published experimental measurement"
        assert 0.0 <= float(r["baseline_stressed_accuracy"]) <= 1.0
        assert 0.0 <= float(r["robust_stressed_accuracy"]) <= 1.0
        assert float(r["chance_accuracy"]) == 0.10


def test_e040_does_not_bundle_third_party_binary_data():
    root = Path(__file__).resolve().parents[1]
    ext = root / "data/external"
    forbidden = {".xlsx", ".xls", ".pth", ".pt", ".zip", ".tar", ".gz", ".npz", ".npy"}
    assert not [p for p in ext.rglob("*") if p.is_file() and p.suffix.lower() in forbidden]
