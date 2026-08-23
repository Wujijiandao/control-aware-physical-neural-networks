from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
MANUSCRIPT = PROJ/'submission/manuscript.tex'
OUTDIR = ROOT/'results/a061_claim_traceability'
OUTDIR.mkdir(parents=True, exist_ok=True)
text = MANUSCRIPT.read_text(encoding='utf-8')

sources = {
    'E010': json.loads((ROOT/'results/e010_confirmatory/e010c1_summary.json').read_text()),
    'E020': json.loads((ROOT/'results/e020_confirmatory/e020c1_primary_summary.json').read_text()),
    'E030': json.loads((ROOT/'results/e030_confirmatory/e030c1_summary.json').read_text()),
    'E031': json.loads((ROOT/'results/e031_confirmatory/e031c1_primary_summary.json').read_text()),
    'E050': json.loads((ROOT/'results/e050_confirmatory/e050c1_summary.json').read_text()),
    'E051': json.loads((ROOT/'results/e051_confirmatory/e051c1_summary.json').read_text()),
    'E060': json.loads((ROOT/'results/e060_confirmatory/e060_primary_result.json').read_text()),
}
claims = [
    ('E-010C1 relative reduction', sources['E010']['primary']['relative_reduction']*100, '14.79\\%', 'software/results/e010_confirmatory/e010c1_summary.json'),
    ('E-020C1 relative reduction', sources['E020']['relative_reduction']*100, '17.86\\%', 'software/results/e020_confirmatory/e020c1_primary_summary.json'),
    ('E-030C1 relative reduction', sources['E030']['relative_reduction']*100, '4.14\\%', 'software/results/e030_confirmatory/e030c1_summary.json'),
    ('E-031C1 relative reduction', sources['E031']['relative_reduction']*100, '28.63\\%', 'software/results/e031_confirmatory/e031c1_primary_summary.json'),
    ('E-050C1 relative reduction', sources['E050']['relative_reduction']*100, '11.95\\%', 'software/results/e050_confirmatory/e050c1_summary.json'),
    ('E-051C1 interferometric regret reduction', sources['E051']['co_primary']['E020']['relative_reduction']*100, '49.24\\%', 'software/results/e051_confirmatory/e051c1_summary.json'),
    ('E-051C1 oscillator regret reduction', sources['E051']['co_primary']['E031']['relative_reduction']*100, '18.64\\%', 'software/results/e051_confirmatory/e051c1_summary.json'),
    ('E-060C1 Cart-Pole stabilization reduction', sources['E060']['relative_reduction']*100, '54.63\\%', 'software/results/e060_confirmatory/e060_primary_result.json'),
]
rows=[]
for claim, value, token, source in claims:
    rows.append({
        'claim': claim,
        'computed_percent': value,
        'expected_manuscript_token': token,
        'token_present': token in text,
        'source': source,
    })
# Ensure the retained E-030 failure is not silently converted to success.
rows.append({
    'claim':'E-030C1 retained threshold failure',
    'computed_percent': sources['E030']['relative_reduction']*100,
    'expected_manuscript_token':'threshold failure',
    'token_present':'threshold failure' in text,
    'source':'software/results/e030_confirmatory/e030c1_summary.json',
})
out=pd.DataFrame(rows)
out.to_csv(OUTDIR/'claim_traceability.csv', index=False)
assert out['token_present'].all(), out.loc[~out['token_present']]
(OUTDIR/'README.md').write_text(
    '# Claim traceability audit\n\nThis audit recomputes headline percentages from frozen JSON summaries and checks that the corresponding rounded tokens and the retained E-030 failure classification occur in the manuscript. It is a data-to-text integrity check, not a new scientific analysis.\n',
    encoding='utf-8')
print(out.to_string(index=False))
