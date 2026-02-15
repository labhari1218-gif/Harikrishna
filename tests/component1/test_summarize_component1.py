"""
test_summarize_component1.py - Unit tests for summarize_component1 script.
"""

import json
import importlib.util
from pathlib import Path


def _load_summarize_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "summarize_component1.py"
    spec = importlib.util.spec_from_file_location("summarize_component1", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_fallback_to_esi_prod(tmp_path):
    summarize = _load_summarize_module()

    claims_jsonl = tmp_path / "claims.jsonl"
    examples_dir = tmp_path / "examples"

    claims = [
        {
            "claim_id": "test_0",
            "claim_text": "Test claim 0",
            "sufficiency": {"esi_prod": 0.05, "neutral_rate_A": 0.2},
            "counter_retention": {"max_contra_all": 0.8}
        },
        {
            "claim_id": "test_1",
            "claim_text": "Test claim 1",
            "sufficiency": {"esi_prod": 0.7, "neutral_rate_A": 0.5},
            "counter_retention": {"max_contra_all": 0.3}
        }
    ]

    with open(claims_jsonl, "w") as f:
        for claim in claims:
            f.write(json.dumps(claim) + "\n")

    loaded_claims = summarize.load_claims_jsonl(claims_jsonl)
    summary = summarize.compute_summary_statistics(loaded_claims)
    assert summary["mean_esi_geom"] == (0.05 + 0.7) / 2

    summarize.save_examples(loaded_claims, examples_dir, top_n=2)
    lowest_esi_file = examples_dir / "lowest_esi.json"
    assert lowest_esi_file.exists()
    with open(lowest_esi_file, "r") as f:
        lowest_esi = json.load(f)

    assert [c["claim_id"] for c in lowest_esi] == ["test_0", "test_1"]
