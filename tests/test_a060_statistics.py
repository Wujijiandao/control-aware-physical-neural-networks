from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_a060_outputs_exist_and_cover_headline_comparisons():
    csv = ROOT/'results/a060_statistical_audit/confirmatory_sign_test_sensitivity.csv'
    js = ROOT/'results/a060_statistical_audit/audit_summary.json'
    assert csv.exists() and js.exists()
    df = pd.read_csv(csv)
    assert set(df['experiment']) == {
        'E-010C1','E-020C1','E-030C1','E-031C1','E-050C1','E-051C1-E020','E-051C1-E031'
    }
    assert (df['sign_test_p_two_sided'] < 0.001).all()
    assert (df['holm_p_global_sensitivity'] < 0.001).all()
    meta = json.loads(js.read_text())
    assert meta['independent_unit'].startswith('held-out stochastic simulation seed')
