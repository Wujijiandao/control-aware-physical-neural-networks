from pathlib import Path
import hashlib, json


def test_e030_frozen_manifest_integrity():
    root=Path(__file__).resolve().parents[1]
    frozen=root/'frozen/e030'
    for line in (frozen/'FREEZE_MANIFEST.sha256').read_text().splitlines():
        if not line.strip(): continue
        h,name=line.split('  ',1)
        p=frozen/name
        assert p.exists()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h


def test_e030_seed_commitment_and_config():
    root=Path(__file__).resolve().parents[1]
    frozen=root/'frozen/e030'
    seeds=(frozen/'confirmatory_seeds.txt').read_bytes()
    expected=(frozen/'SEED_COMMITMENT.txt').read_text().splitlines()[0].split()[-1]
    assert hashlib.sha256(seeds).hexdigest()==expected
    vals=[int(x) for x in seeds.decode().split()]
    cfg=json.loads((frozen/'confirmatory_config.json').read_text())
    assert cfg['experiment_id']=='E-030C1'
    assert cfg['n_seeds']==192
    assert len(vals)==len(set(vals))==192
    static=json.loads((frozen/'static_selection.json').read_text())
    assert static['matching']['rmse_ratio'] < 1.02
    assert static['matching']['candidate_rmse_ratio'] < 1.05
