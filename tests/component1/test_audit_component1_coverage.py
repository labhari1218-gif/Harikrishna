"""Tests for scripts/audit_component1_coverage.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    pd = None  # type: ignore[assignment]


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root, repo_root / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    module_path = repo_root / "scripts" / "audit_component1_coverage.py"
    spec = importlib.util.spec_from_file_location("audit_component1_coverage_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(pd is not None, "pandas unavailable")
class AuditCoverageTests(unittest.TestCase):
    def test_audit_split_counts_missing_claim_rows_and_field_issues(self) -> None:
        mod = _load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_root = Path(tmp_dir) / "logs" / "component1"
            split_dir = logs_root / "train"
            split_dir.mkdir(parents=True, exist_ok=True)
            pairs_path = split_dir / "pairs.jsonl"
            rows = [
                {
                    "schema_version": 2,
                    "record_type": "pair",
                    "claim_id": "train_0",
                    "evidence_id": "e0",
                    "pool": "A",
                    "raw_triple": ["A", "r", "B"],
                    "probs": {"entail": 0.8, "contra": 0.1, "neutral": 0.1},
                    "derived": {"rel": 0.9},
                },
                {
                    "claim_id": "train_1",
                    "evidence_id": "e1",
                    "pool": "A",
                    "raw_triple": ["A", "r", "B"],
                    "probs": {"entail": 0.5, "contra": 0.5, "neutral": 0.5},  # malformed sum
                    "derived": {},  # missing rel
                },
            ]
            with pairs_path.open("w", encoding="utf-8") as fp:
                for row in rows:
                    fp.write(json.dumps(row) + "\n")

            claims_df = pd.DataFrame({"Sentence": ["c0", "c1", "c2"], "Label": [[1], [0], [1]]})
            subgraphs_df = pd.DataFrame(
                {
                    "walked": [
                        {"connected": [["A", "r", "B"]], "walkable": []},
                        {"connected": [["A", "r", "B"]], "walkable": []},
                        {"connected": [], "walkable": []},
                    ]
                }
            )

            with patch.object(mod, "get_df", return_value=claims_df), patch.object(
                mod, "get_subgraphs", return_value=subgraphs_df
            ):
                report = mod._audit_split(
                    split="train",
                    logs_root=logs_root,
                    prob_sum_tol=1e-3,
                    embedding_entities={"A", "B"},
                )

            self.assertEqual(report["claims"]["total"], 3)
            self.assertEqual(report["claims"]["with_at_least_1_pair_row"], 2)
            self.assertEqual(report["claims"]["without_pair_rows"], 1)
            self.assertEqual(report["claims"]["without_pair_rows_with_zero_subgraph_triples"], 1)
            self.assertEqual(report["claims"]["missing_rel"], 1)
            self.assertEqual(report["claims"]["malformed_probs"], 1)
            self.assertEqual(report["claims"]["missing_schema_version"], 1)


if __name__ == "__main__":
    unittest.main()

