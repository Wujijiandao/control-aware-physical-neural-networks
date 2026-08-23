import unittest
from pathlib import Path

class ProvenancePolicyTests(unittest.TestCase):
    def test_no_vendor_directories_or_archives(self):
        root = Path(__file__).resolve().parents[1]
        forbidden_dirs = {"vendor", "site-packages", "node_modules"}
        for p in root.rglob("*"):
            if p.is_dir():
                self.assertNotIn(p.name, forbidden_dirs)
            elif p.suffix.lower() in {".whl", ".tar", ".gz", ".zip"}:
                self.fail(f"bundled archive forbidden in software package: {p}")

if __name__ == "__main__": unittest.main()
