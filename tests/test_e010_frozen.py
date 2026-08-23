from pathlib import Path
import hashlib, json


def test_e010_frozen_seed_commitment_and_config():
    root = Path(__file__).resolve().parents[1]
    frozen = root / "frozen" / "e010"
    cfg = json.loads((frozen / "confirmatory_config.json").read_text())
    seeds = (frozen / "confirmatory_seeds.txt").read_bytes()
    expected = (frozen / "SEED_COMMITMENT.txt").read_text().splitlines()[0].split()[-1]
    assert hashlib.sha256(seeds).hexdigest() == expected
    assert cfg["experiment_id"] == "E-010C1"
    assert cfg["n_seeds"] == 256
    vals = [int(x) for x in seeds.decode().split()]
    assert len(vals) == len(set(vals)) == 256
