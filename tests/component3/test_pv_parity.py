"""Tests for Component 3 numeric parity check (T3.5)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_pv_parity_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    module_path = src_dir / "component3" / "pv_parity.py"
    spec = importlib.util.spec_from_file_location("component3_pv_parity_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PVParityTests(unittest.TestCase):
    def test_numeric_parity_exceeds_threshold_for_ten_claims(self) -> None:
        mod = _load_pv_parity_module()
        report = mod.run_numeric_parity_check(num_claims=10, threshold=0.9)

        self.assertEqual(report["num_claims"], 10)
        self.assertGreater(report["num_edges"], 0)
        self.assertGreater(report["pearson_corr"], 0.9)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["per_claim"]), 10)

    def test_numeric_parity_rejects_invalid_claim_count(self) -> None:
        mod = _load_pv_parity_module()
        with self.assertRaises(ValueError):
            mod.run_numeric_parity_check(num_claims=0, threshold=0.9)


if __name__ == "__main__":
    unittest.main()
