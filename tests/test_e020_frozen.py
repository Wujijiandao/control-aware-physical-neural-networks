from pathlib import Path
import hashlib


def test_e020_frozen_manifest_integrity():
    root=Path(__file__).resolve().parents[1]
    frozen=root/'frozen/e020'
    manifest=frozen/'FREEZE_MANIFEST.sha256'
    assert manifest.exists()
    for line in manifest.read_text().splitlines():
        if not line.strip(): continue
        h,name=line.split('  ',1)
        p=frozen/name
        assert p.exists()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h


def test_e020_seed_count():
    root=Path(__file__).resolve().parents[1]
    seeds=[int(x) for x in (root/'frozen/e020/confirmatory_seeds.txt').read_text().split()]
    assert len(seeds)==192
    assert len(set(seeds))==192
