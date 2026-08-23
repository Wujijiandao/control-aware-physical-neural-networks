from pathlib import Path
import hashlib
import pandas as pd
import json

ROOT=Path(__file__).resolve().parents[1]
PROJ=ROOT.parent
SUB=PROJ/'submission/source_data'
OUT=ROOT/'results/a062_source_data_audit'
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

pairs = [
    ('Fig1_E010C1_raw_per_seed.csv', ROOT/'results/e010_confirmatory/e010c1_raw_per_seed.csv'),
    ('E020C1_primary_paired.csv', ROOT/'results/e020_confirmatory/e020c1_primary_paired.csv'),
    ('E030C1_paired.csv', ROOT/'results/e030_confirmatory/e030c1_paired.csv'),
    ('E031C1_primary_paired.csv', ROOT/'results/e031_confirmatory/e031c1_primary_paired.csv'),
    ('E050C1_primary_paired.csv', ROOT/'results/e050_confirmatory/e050c1_primary_paired.csv'),
    ('E051C1_primary_paired.csv', ROOT/'results/e051_confirmatory/e051c1_primary_paired.csv'),
    ('E060_primary_paired.csv', ROOT/'results/e060_confirmatory/e060c1_primary_paired.csv'),
]
rows=[]
for dstname, src in pairs:
    dst=SUB/dstname
    rows.append({'submission_file':dstname,'source':str(src.relative_to(PROJ)),'exists':dst.exists(),
                 'byte_identical_sha256': dst.exists() and sha(dst)==sha(src),'sha256':sha(dst) if dst.exists() else None})

# Unique seed checks on headline source files.
seed_files = ['Fig1_E010C1_raw_per_seed.csv','E020C1_primary_paired.csv','E030C1_paired.csv','E031C1_primary_paired.csv','E050C1_primary_paired.csv','E060_primary_paired.csv']
seed_checks=[]
for name in seed_files:
    df=pd.read_csv(SUB/name)
    seed_checks.append({'file':name,'rows':len(df),'seed_unique':bool(df['seed'].is_unique),'missing_cells':int(df.isna().sum().sum())})
# E051 has two rows per seed because two substrate co-primary comparisons.
df=pd.read_csv(SUB/'E051C1_primary_paired.csv')
seed_checks.append({'file':'E051C1_primary_paired.csv','rows':len(df),
                    'seed_experiment_key_unique':bool(~df.duplicated(['seed','experiment']).any()),
                    'missing_cells':int(df.isna().sum().sum())})

pd.DataFrame(rows).to_csv(OUT/'copy_identity.csv', index=False)
pd.DataFrame(seed_checks).to_csv(OUT/'seed_integrity.csv', index=False)
summary={
    'audit_id':'A-062',
    'all_headline_copies_byte_identical': all(r['byte_identical_sha256'] for r in rows),
    'all_seed_keys_valid': all((x.get('seed_unique',x.get('seed_experiment_key_unique'))) and x['missing_cells']==0 for x in seed_checks),
    'n_submission_source_files': len(list(SUB.glob('*'))),
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
assert summary['all_headline_copies_byte_identical']
assert summary['all_seed_keys_valid']
print(json.dumps(summary, indent=2))
