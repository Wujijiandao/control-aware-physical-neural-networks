#!/usr/bin/env python3
from pathlib import Path
import hashlib, sys
root=Path(__file__).resolve().parent
manifest=root/"MANIFEST.sha256"
errors=[]; count=0
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    expected, rel=line.split("  ",1)
    p=root/rel
    if not p.is_file(): errors.append((rel,"MISSING")); continue
    got=hashlib.sha256(p.read_bytes()).hexdigest(); count += 1
    if got != expected: errors.append((rel,got))
if errors:
    for rel,got in errors: print(f"FAIL {rel}: {got}")
    sys.exit(1)
print(f"OK: {count} files verified")
