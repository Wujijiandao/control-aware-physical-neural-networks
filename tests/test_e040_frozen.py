from pathlib import Path
import hashlib, json


def test_e040_frozen_external_audit_manifest():
    root = Path(__file__).resolve().parents[1]
    f = root / "frozen/e040"
    cfg = json.loads((f / "audit_config.json").read_text())
    assert cfg["experiment_id"] == "E-040"
    assert cfg["confirmatory_primary_test"] is False
    assert cfg["n_anchors"] == 7
    for line in (f / "EXTERNAL_AUDIT_MANIFEST.sha256").read_text().splitlines():
        h, name = line.split("  ", 1)
        assert hashlib.sha256((f / name).read_bytes()).hexdigest() == h
