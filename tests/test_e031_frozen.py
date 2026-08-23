from pathlib import Path
import hashlib,json

def test_e031_frozen_manifest_and_quality_floor():
 root=Path(__file__).resolve().parents[1];f=root/'frozen/e031'
 for line in (f/'FREEZE_MANIFEST.sha256').read_text().splitlines():
  if not line.strip():continue
  h,n=line.split('  ',1);assert hashlib.sha256((f/n).read_bytes()).hexdigest()==h
 s=json.loads((f/'static_training_summary.json').read_text())
 assert s['boundary_aware']['global']['r2']>=.99
 assert s['robust_control_aware']['global']['r2']>=.99

def test_e031_seed_count_and_commitment():
 root=Path(__file__).resolve().parents[1];f=root/'frozen/e031';b=(f/'confirmatory_seeds.txt').read_bytes();exp=(f/'SEED_COMMITMENT.txt').read_text().splitlines()[0].split()[-1]
 vals=[int(x) for x in b.decode().split()];assert hashlib.sha256(b).hexdigest()==exp;assert len(vals)==len(set(vals))==192
