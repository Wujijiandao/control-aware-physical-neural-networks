from pathlib import Path
import hashlib,json,pandas as pd
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'frozen/e050';R=ROOT/'results/e050_confirmatory'
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def test_e050_freeze_manifest_and_coverage():
    for line in (F/'FREEZE_MANIFEST.sha256').read_text().splitlines():
        if not line.strip():continue
        h,n=line.split('  ',1);assert _sha(F/n)==h
    cfg=json.loads((F/'confirmatory_config.json').read_text());seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()]
    assert len(seeds)==cfg['n_seeds']==96 and len(set(seeds))==96
    raw=pd.read_csv(R/'e050c1_raw.csv');assert len(raw)==96*2*2*8
    assert not raw.duplicated(['seed','method','condition','lambda_task']).any()
def test_e050_primary_success_is_sealed():
    s=json.loads((R/'e050c1_summary.json').read_text())
    assert s['status']=='CONFIRMATORY_COMPLETED_ONCE'
    assert s['primary_success'] is True
    assert s['bootstrap_ci95'][1] < 0
    assert s['relative_reduction'] >= s['minimum_relative_reduction']
