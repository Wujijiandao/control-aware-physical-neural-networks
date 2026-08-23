from pathlib import Path
import hashlib,json,pandas as pd
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'frozen/e051';R=ROOT/'results/e051_confirmatory'
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def test_e051_freeze_manifest_and_coverage():
    for line in (F/'FREEZE_MANIFEST.sha256').read_text().splitlines():
        if not line.strip():continue
        h,n=line.split('  ',1);assert _sha(F/n)==h
    cfg=json.loads((F/'confirmatory_config.json').read_text());seeds=[int(x) for x in (F/'confirmatory_seeds.txt').read_text().split()]
    assert len(seeds)==cfg['n_seeds']==96 and len(set(seeds))==96
    raw=pd.read_csv(R/'e051c1_raw.csv');assert len(raw)==96*2*2*2
    assert not raw.duplicated(['seed','experiment','method','condition']).any()
    assert (raw['mean_regret']>=-1e-12).all()
def test_e051_coprimary_success_is_sealed():
    s=json.loads((R/'e051c1_summary.json').read_text())
    assert s['status']=='CONFIRMATORY_COMPLETED_ONCE' and s['full_mechanism_success'] is True
    for exp in ('E020','E031'):
        q=s['co_primary'][exp]
        assert q['success'] is True and q['bootstrap_ci95'][1] < 0 and q['relative_reduction'] >= 0.10
