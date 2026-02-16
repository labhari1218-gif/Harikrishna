"""
test_logging_utils.py - Unit tests for logging_utils module

Tests CR@k correctness and save_examples ESI_geom usage.
"""

import json
import tempfile
from pathlib import Path

import pytest

from component1.evidence import EvidenceItem, PVResult
from component1.sufficiency_metrics import compute_cr_at_k, compute_esi
from component1.logging_utils import (
    PairLogConfig,
    save_examples,
    write_claim_log,
    write_empty_evidence_pair_sentinel,
    write_pair_log,
)


def _make_triple(
    evidence_id: str,
    subj: str,
    rel: str,
    obj: str,
    p_ent: float,
    p_con: float,
    p_neu: float = 0.0,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        kind="kg_triple",
        content={"triple": [subj, rel, obj]},
        retrieval_score=0.0,
        pv=PVResult(
            p_entail=float(p_ent),
            p_contra=float(p_con),
            p_neutral=float(p_neu),
            rel=float(p_ent + p_con),
            pol=float(p_ent - p_con),
        ),
        meta={},
    )


def test_cr_at_k_correctness():
    """Test CR@k computes correctly with known top-k contradictions."""
    # Create pool with known p_contra values
    pool = [
        EvidenceItem(
            evidence_id=f"ev_{i}",
            kind="kg_triple",
            content={"triple": ["a", "b", "c"]},
            retrieval_score=0.0,
            pv=PVResult(p_entail=0.1, p_neutral=0.1, p_contra=p_contra, rel=0.8, pol=-1.0),
            meta={}
        )
        for i, p_contra in enumerate([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    ]
    
    # Counter set contains top 2 contradictions (indices 0, 1)
    C = [pool[0], pool[1]]
    
    # Compute CR@k using keyword args
    result = compute_cr_at_k(pool_items=pool, counter_items=C, ks=[5, 10])
    
    # Top-5 contradictions: indices 0,1,2,3,4
    # C contains: indices 0,1
    # CR@5 = 2/5 = 0.4
    assert result["cr_at_5"] == pytest.approx(2/5, abs=1e-6)
    
    # Top-10 contradictions: all 10
    # C contains: indices 0,1
    # CR@10 = 2/10 = 0.2
    assert result["cr_at_10"] == pytest.approx(2/10, abs=1e-6)
    
    # Max contra in pool should be 0.9
    assert result["max_contra_all"] == pytest.approx(0.9, abs=1e-6)
    
    # Max contra in C should be 0.9 (from ev_0)
    assert result["max_contra_C"] == pytest.approx(0.9, abs=1e-6)


def test_cr_at_k_rejects_positional():
    """Test that compute_cr_at_k rejects positional arguments (keyword-only enforcement)."""
    pool = [EvidenceItem(
        evidence_id="ev_0",
        kind="kg_triple",
        content={"triple": ["a", "b", "c"]},
        retrieval_score=0.0,
        pv=PVResult(p_entail=0.1, p_neutral=0.1, p_contra=0.9, rel=0.8, pol=-1.0),
        meta={}
    )]
    C = [pool[0]]
    
    # Should raise TypeError when using positional args
    with pytest.raises(TypeError, match="takes 0 positional arguments"):
        compute_cr_at_k(pool, C, [5, 10])


def test_cr_at_k_small_pool():
    """Test CR@k handles small pools correctly (pool size < k)."""
    # Create small pool with only 3 items
    pool = [
        EvidenceItem(
            evidence_id=f"ev_{i}",
            kind="kg_triple",
            content={"triple": ["a", "b", "c"]},
            retrieval_score=0.0,
            pv=PVResult(p_entail=0.1, p_neutral=0.1, p_contra=p_contra, rel=0.8, pol=-1.0),
            meta={}
        )
        for i, p_contra in enumerate([0.9, 0.7, 0.5])
    ]
    
    C = [pool[0], pool[1]]  # Top 2
    
    result = compute_cr_at_k(pool_items=pool, counter_items=C, ks=[5, 10])
    
    # CR@5: pool only has 3 items, so denom = min(5, 3) = 3
    # Top-3 from pool: all 3, C has 2 of them
    # CR@5 = 2/3
    assert result["cr_at_5"] == pytest.approx(2/3, abs=1e-6)
    
    # CR@10: pool only has 3 items, so denom = min(10, 3) = 3
    # CR@10 = 2/3
    assert result["cr_at_10"] == pytest.approx(2/3, abs=1e-6)


def test_compute_esi_is_symmetric_over_a_plus_c():
    claim_entities = ["A", "C"]

    # Strong A, weak C
    a_strong = [
        _make_triple("a_strong", "A", "r1", "C", p_ent=0.90, p_con=0.05),
    ]
    c_weak = [
        _make_triple("c_weak", "X", "r2", "Y", p_ent=0.02, p_con=0.03),
    ]
    strong_a_score = compute_esi(a_strong + c_weak, claim_entities)["esi_geom"]

    # Weak A, strong C (historically penalized when using A-only)
    a_weak = [
        _make_triple("a_weak", "A", "r1", "B", p_ent=0.05, p_con=0.05),
    ]
    c_strong = [
        _make_triple("c_strong", "B", "r2", "C", p_ent=0.05, p_con=0.85),
    ]
    strong_c_score = compute_esi(a_weak + c_strong, claim_entities)["esi_geom"]

    # Weak A, weak C
    c_weak_disconnected = [
        _make_triple("c_weak_disconnected", "X", "r3", "Y", p_ent=0.02, p_con=0.03),
    ]
    weak_both_score = compute_esi(a_weak + c_weak_disconnected, claim_entities)["esi_geom"]

    assert strong_a_score > 0.70
    assert strong_c_score > 0.70
    assert weak_both_score < 0.50


def test_write_claim_log_uses_a_plus_c_for_esi_components():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "claims.jsonl"

        # A alone only covers one anchor; C supplies missing anchor and path.
        a_item = _make_triple("a_1", "A", "r_left", "B", p_ent=0.05, p_con=0.05)
        c_item = _make_triple("c_1", "B", "r_right", "C", p_ent=0.05, p_con=0.85)

        write_claim_log(
            claim_id="train_fix_ac",
            claim_text="A is related to C",
            label="FALSE",
            claim_entities=["A", "C"],
            A=[a_item],
            S=[],
            C=[c_item],
            pool=[a_item, c_item],
            min_A=1,
            contra_tau=0.6,
            out_file=out,
        )

        row = json.loads(out.read_text(encoding="utf-8").strip())
        suff = row["sufficiency"]
        assert suff["coverage_A"] == pytest.approx(1.0, abs=1e-12)
        assert suff["connectivity_A"] == pytest.approx(1.0, abs=1e-12)


def test_save_examples_uses_esi_geom():
    """Test save_examples uses esi_geom and doesn't crash on missing esi."""
    # Create temp claims JSONL with esi_geom (no legacy esi field)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        claims_jsonl = tmpdir / "claims.jsonl"
        examples_dir = tmpdir / "examples"
        
        # Write test claims with esi_geom
        claims = [
            {
                "claim_id": "test_0",
                "claim_text": "Test claim 0",
                "sufficiency": {"esi_geom": 0.1, "esi_prod": 0.05, "neutral_rate_A": 0.2},
                "counter_retention": {"max_contra_all": 0.8}
            },
            {
                "claim_id": "test_1",
                "claim_text": "Test claim 1",
                "sufficiency": {"esi_geom": 0.9, "esi_prod": 0.7, "neutral_rate_A": 0.5},
                "counter_retention": {"max_contra_all": 0.3}
            },
            {
                "claim_id": "test_2",
                "claim_text": "Test claim 2",
                "sufficiency": {"esi_geom": 0.5, "esi_prod": 0.3, "neutral_rate_A": 0.9},
                "counter_retention": {"max_contra_all": 0.6}
            }
        ]
        
        with open(claims_jsonl, 'w') as f:
            for claim in claims:
                f.write(json.dumps(claim) + '\n')
        
        # Call save_examples - should NOT crash despite missing "esi" field
        save_examples("train", claims_jsonl, examples_dir, top_n=2)
        
        # Verify files were created
        assert (examples_dir / "lowest_esi.json").exists()
        assert (examples_dir / "highest_contradiction.json").exists()
        
        # Verify lowest ESI is sorted correctly by esi_geom
        with open(examples_dir / "lowest_esi.json") as f:
            lowest_esi = json.load(f)
        
        # Should be sorted ascending by esi_geom: test_0 (0.1), test_2 (0.5)
        assert len(lowest_esi) == 2
        assert lowest_esi[0]["claim_id"] == "test_0"
        assert lowest_esi[1]["claim_id"] == "test_2"


def test_save_examples_fallback_to_esi_prod():
    """Test save_examples falls back to esi_prod when esi_geom is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        claims_jsonl = tmpdir / "claims.jsonl"
        examples_dir = tmpdir / "examples"
        
        # Claims with ONLY esi_prod (no esi_geom)
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
        
        with open(claims_jsonl, 'w') as f:
            for claim in claims:
                f.write(json.dumps(claim) + '\n')
        
        # Should NOT crash - falls back to esi_prod
        save_examples("train", claims_jsonl, examples_dir, top_n=2)
        
        # Verify sorting works with esi_prod fallback
        with open(examples_dir / "lowest_esi.json") as f:
            lowest_esi = json.load(f)
        
        # Should be sorted by esi_prod: test_0 (0.05), test_1 (0.7)
        assert len(lowest_esi) == 2
        assert lowest_esi[0]["claim_id"] == "test_0"
        assert lowest_esi[1]["claim_id"] == "test_1"


def test_write_claim_log_includes_entity_set_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "claims.jsonl"

        ev = EvidenceItem(
            evidence_id="ev_0",
            kind="kg_triple",
            content={"triple": ["A", "rel", "B"]},
            retrieval_score=0.0,
            pv=PVResult(p_entail=0.7, p_contra=0.1, p_neutral=0.2, rel=0.8, pol=0.6),
            meta={}
        )

        write_claim_log(
            claim_id="train_0",
            claim_text="A is related to B",
            label="TRUE",
            claim_entities=["A", "B"],
            A=[ev],
            S=[],
            C=[],
            pool=[ev],
            min_A=1,
            contra_tau=0.6,
            out_file=out,
        )

        row = json.loads(out.read_text().strip())
        assert row["entity_set"] == ["A", "B"]
        assert row["Entity_set"] == ["A", "B"]


def test_write_pair_log_includes_pool_assignment():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "pairs.jsonl"

        ev = EvidenceItem(
            evidence_id="ev_1",
            kind="kg_triple",
            content={"triple": ["A", "rel", "C"]},
            retrieval_score=0.0,
            pv=PVResult(p_entail=0.2, p_contra=0.6, p_neutral=0.2, rel=0.8, pol=-0.4),
            meta={}
        )

        write_pair_log(
            claim_id="train_0",
            evidence_item=ev,
            claim_text="A is not related to C",
            premise_text="A has rel C.",
            model_name="dummy-model",
            verbalizer_id="v_test",
            config=PairLogConfig(mode="all"),
            out_file=out,
            evidence_assignment="S",
        )

        row = json.loads(out.read_text().strip())
        assert row["schema_version"] == 2
        assert row["record_type"] == "pair"
        assert row["pool"] == "S"
        assert row["evidence_assignment"] == "S"


def test_write_pair_log_serializes_sentence_payload():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "pairs.jsonl"

        ev = EvidenceItem(
            evidence_id="ev_sentence_1",
            kind="sentence",
            content={"text": "Roman Atwood is a content creator.", "doc_id": "Roman_Atwood", "sent_id": 1},
            retrieval_score=0.0,
            pv=PVResult(p_entail=0.8, p_contra=0.1, p_neutral=0.1, rel=0.9, pol=0.7),
            meta={},
        )

        write_pair_log(
            claim_id="train_1",
            evidence_item=ev,
            claim_text="Roman Atwood is a content creator.",
            premise_text="Roman Atwood is a content creator.",
            model_name="dummy-model",
            verbalizer_id="v_test",
            config=PairLogConfig(mode="all"),
            out_file=out,
            evidence_assignment="A",
        )

        row = json.loads(out.read_text().strip())
        assert row["raw_triple"] is None
        assert row["raw_sentence"] == "Roman Atwood is a content creator."
        assert row["sentence_page"] == "Roman_Atwood"
        assert row["sentence_line"] == 1


def test_write_empty_evidence_pair_sentinel():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "pairs.jsonl"
        write_empty_evidence_pair_sentinel(
            claim_id="train_404",
            claim_text="No evidence claim",
            model_name="dummy-model",
            verbalizer_id="v_test",
            out_file=out,
        )
        row = json.loads(out.read_text().strip())
        assert row["schema_version"] == 2
        assert row["record_type"] == "claim_sentinel"
        assert row["empty_evidence_sentinel"] is True
        assert row["claim_id"] == "train_404"
        assert row["pool"] == "S"
        assert row["probs"] == {"entail": 0.0, "contra": 0.0, "neutral": 1.0}
        assert row["derived"]["rel"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
