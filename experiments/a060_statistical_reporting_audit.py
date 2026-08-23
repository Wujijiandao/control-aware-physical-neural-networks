from pathlib import Path
from math import comb
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUTDIR = RESULTS / 'a060_statistical_audit'
OUTDIR.mkdir(parents=True, exist_ok=True)


def exact_two_sided_sign_p(delta):
    x = np.asarray(delta, dtype=float)
    x = x[np.isfinite(x) & (x != 0)]
    n = int(x.size)
    nneg = int(np.sum(x < 0))
    npos = int(np.sum(x > 0))
    k = max(nneg, npos)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail), n, nneg, npos


def holm_adjust(pairs):
    ordered = sorted(pairs, key=lambda z: z[1])
    m = len(ordered)
    out = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered, 1):
        cur = min(1.0, (m - rank + 1) * p)
        running = max(running, cur)
        out[name] = running
    return out

specs = [
    ('E-010C1', RESULTS/'e010_confirmatory/e010c1_raw_per_seed.csv', 'delta_mean_D', 'static-matched interferometric behavior'),
    ('E-020C1', RESULTS/'e020_confirmatory/e020c1_primary_paired.csv', 'delta', 'interferometric robustness'),
    ('E-030C1', RESULTS/'e030_confirmatory/e030c1_paired.csv', 'delta_mean_D', 'oscillator nominal matched-static replication'),
    ('E-031C1', RESULTS/'e031_confirmatory/e031c1_primary_paired.csv', 'delta', 'oscillator robustness'),
    ('E-050C1', RESULTS/'e050_confirmatory/e050c1_primary_paired.csv', 'delta', 'task-matched robustness'),
    ('E-060C1', RESULTS/'e060_confirmatory/e060c1_primary_paired.csv', 'delta', 'canonical cart-pole task generalization'),
]
rows = []
for name, path, col, hypothesis in specs:
    df = pd.read_csv(path)
    p, n, nneg, npos = exact_two_sided_sign_p(df[col])
    rows.append(dict(experiment=name, hypothesis=hypothesis, n=n, mean_delta=float(df[col].mean()),
                     median_delta=float(df[col].median()), negative_pairs=nneg, positive_pairs=npos,
                     sign_test_p_two_sided=p))

# E-051 contains two prespecified co-primary substrate comparisons.
e51 = pd.read_csv(RESULTS/'e051_confirmatory/e051c1_primary_paired.csv')
for substrate, g in e51.groupby('experiment'):
    name = f'E-051C1-{substrate}'
    p, n, nneg, npos = exact_two_sided_sign_p(g['delta'])
    rows.append(dict(experiment=name, hypothesis=f'common-state decision regret ({substrate} substrate)', n=n,
                     mean_delta=float(g['delta'].mean()), median_delta=float(g['delta'].median()),
                     negative_pairs=nneg, positive_pairs=npos, sign_test_p_two_sided=p))

aud = pd.DataFrame(rows)
aud['holm_p_global_sensitivity'] = aud['experiment'].map(holm_adjust(list(zip(aud['experiment'], aud['sign_test_p_two_sided']))))

# Holm adjustment only within the two E-051 co-primary comparisons.
e51mask = aud['experiment'].str.startswith('E-051C1-')
e51pairs = list(zip(aud.loc[e51mask, 'experiment'], aud.loc[e51mask, 'sign_test_p_two_sided']))
e51adj = holm_adjust(e51pairs)
aud['holm_p_e051_coprimary'] = np.nan
for name, p in e51adj.items():
    aud.loc[aud['experiment'] == name, 'holm_p_e051_coprimary'] = p

aud.to_csv(OUTDIR/'confirmatory_sign_test_sensitivity.csv', index=False)

meta = {
    'audit_id': 'A-060',
    'status': 'POSTHOC_REPORTING_SENSITIVITY_NOT_USED_FOR_ORIGINAL_SUCCESS_CLASSIFICATION',
    'rationale': 'Nature Communications statistical reporting guidance requests named tests, n, sidedness and P values. Original confirmatory classifications used project-internally frozen-before-evaluation paired bootstrap CIs plus practical effect thresholds, not P-value thresholds.',
    'test': 'exact two-sided paired sign test on non-zero per-seed paired differences',
    'alpha_reference': 0.05,
    'global_holm_scope': 'post hoc sensitivity across eight headline paired comparisons only; not a redefinition of the sequential confirmatory program',
    'e051_holm_scope': 'post hoc sensitivity across the two E-051 co-primary substrate comparisons',
    'independent_unit': 'held-out stochastic simulation seed/episode; not a physical device replicate',
    'rows': aud.to_dict(orient='records'),
}
(OUTDIR/'audit_summary.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print(aud.to_string(index=False))
