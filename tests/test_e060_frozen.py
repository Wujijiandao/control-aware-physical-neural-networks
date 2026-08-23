from pathlib import Path
import hashlib,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];F=ROOT/'frozen/e060';R=ROOT/'results/e060_confirmatory'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def test_e060_frozen_manifest_and_seed_count():
    for line in (F/'E060_FROZEN_MANIFEST.sha256').read_text().splitlines():
        expected,name=line.split(None,1);assert sha(F/name.strip())==expected
    seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()]
    assert len(seeds)==128 and len(set(seeds))==128

def test_e060_primary_result_matches_frozen_rule():
    cfg=json.loads((F/'audit_config.json').read_text());res=json.loads((R/'e060_primary_result.json').read_text())
    assert res['n_seeds']==cfg['confirmatory_n_seeds']==128
    assert res['ci95'][1] < 0
    assert res['relative_reduction'] >= .10
    assert res['primary_success'] is True

def test_e060_raw_complete_unique_coverage():
    d=pd.read_csv(R/'e060_confirmatory_raw.csv')
    assert len(d)==128*2*3
    assert not d.duplicated(['seed','method','condition']).any()
    assert set(d['method'])=={'noise_aware_mse','control_aware_rank'}
    assert set(d['condition'])=={'nominal','moderate','strong'}
