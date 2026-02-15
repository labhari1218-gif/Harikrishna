"""Component 1 pair-log ingestion utilities for Component 3."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

VALID_POOLS = {"A", "S", "C"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_pool(pool_value: Any) -> str:
    pool = str(pool_value or "A").upper()
    if pool not in VALID_POOLS:
        return "A"
    return pool


def _iter_raw_subgraph_triples(subgraph_row: Any) -> list[list[str]]:
    if hasattr(subgraph_row, "to_dict"):
        subgraph_row = subgraph_row.to_dict()

    triples: list[Any] = []
    if isinstance(subgraph_row, dict):
        if "triples" in subgraph_row and isinstance(subgraph_row["triples"], list):
            triples = list(subgraph_row["triples"])
        elif "walked" in subgraph_row and isinstance(subgraph_row["walked"], dict):
            walked = subgraph_row["walked"]
            triples = list(walked.get("connected", [])) + list(walked.get("walkable", []))
        elif "connected" in subgraph_row or "walkable" in subgraph_row:
            triples = list(subgraph_row.get("connected", [])) + list(subgraph_row.get("walkable", []))
        elif "subgraph" in subgraph_row and isinstance(subgraph_row["subgraph"], list):
            triples = list(subgraph_row["subgraph"])
    elif isinstance(subgraph_row, list):
        triples = list(subgraph_row)

    deduped: list[list[str]] = []
    seen: set[tuple[str, str, str]] = set()
    for triple in triples:
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        triple_key = (str(triple[0]), str(triple[1]), str(triple[2]))
        if triple_key in seen:
            continue
        seen.add(triple_key)
        deduped.append([triple_key[0], triple_key[1], triple_key[2]])
    return deduped


def _fallback_rows_from_subgraph(claim_id: str, subgraph_row: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_triples = _iter_raw_subgraph_triples(subgraph_row)
    for idx, triple in enumerate(raw_triples):
        rows.append(
            {
                "evidence_id": f"{claim_id}_fallback_{idx}",
                "raw_triple": [triple[0], triple[1], triple[2]],
                "pool": "A",
                "p_ent": 0.0,
                "p_con": 0.0,
                "p_neu": 1.0,
                "rel": 0.0,
            }
        )
    return rows


def _parse_component1_pair_row(row: dict[str, Any], row_idx: int) -> tuple[str, dict[str, Any]] | None:
    claim_id = str(row.get("claim_id") or "").strip()
    if not claim_id:
        return None

    raw_triple = row.get("raw_triple")
    has_raw_triple = isinstance(raw_triple, (list, tuple)) and len(raw_triple) >= 3
    raw_sentence = str(row.get("raw_sentence") or row.get("premise_text") or "").strip()
    if (not has_raw_triple) and (not raw_sentence):
        return None

    probs = row.get("probs") if isinstance(row.get("probs"), dict) else {}
    derived = row.get("derived") if isinstance(row.get("derived"), dict) else {}

    p_ent = _safe_float(probs.get("entail"), 0.0)
    p_con = _safe_float(probs.get("contra"), 0.0)
    p_neu = _safe_float(probs.get("neutral"), max(0.0, 1.0 - (p_ent + p_con)))
    rel = _safe_float(derived.get("rel"), p_ent + p_con)
    pool = _normalize_pool(row.get("pool", row.get("evidence_assignment", "A")))

    evidence_id = str(row.get("evidence_id") or f"{claim_id}_pair_{row_idx}")
    page = row.get("page")
    if page is None:
        page = row.get("sentence_page")
    line = row.get("line")
    if line is None:
        line = row.get("sentence_line")
    evidence_row = {
        "evidence_id": evidence_id,
        "raw_triple": [str(raw_triple[0]), str(raw_triple[1]), str(raw_triple[2])] if has_raw_triple else None,
        "raw_sentence": raw_sentence if raw_sentence else None,
        "text": raw_sentence if raw_sentence else None,
        "page": page,
        "line": line,
        "pool": pool,
        "p_ent": p_ent,
        "p_con": p_con,
        "p_neu": p_neu,
        "rel": rel,
    }
    return claim_id, evidence_row


def _pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def load_component1_evidence_rows(
    split: str,
    claims_df,
    subgraphs_df,
    logs_root: str | Path,
    missing_policy: str = "hybrid_fallback",
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """
    Load C1 pair logs and align them with FactKG rows for Component 3.

    Returns:
        tuple: (evidence_rows, coverage_stats)
            evidence_rows is a list aligned to claim row index; each element is
            a list of dict rows in PV dataset format.
    """
    supported_missing_policies = {"hybrid_fallback", "strict"}
    if missing_policy not in supported_missing_policies:
        raise ValueError(
            f"Unsupported missing policy `{missing_policy}`. "
            f"Expected one of {sorted(supported_missing_policies)}."
        )

    if len(claims_df) != len(subgraphs_df):
        raise ValueError(
            f"claims_df/subgraphs_df length mismatch for split `{split}`: "
            f"{len(claims_df)} vs {len(subgraphs_df)}."
        )

    pairs_path = Path(logs_root) / split / "pairs.jsonl"
    if hasattr(claims_df, "columns") and "claim_id" in claims_df.columns:
        expected_claim_ids = [str(v) for v in claims_df["claim_id"]]
    else:
        expected_claim_ids = [f"{split}_{idx}" for idx in range(len(claims_df))]
    expected_claim_id_set = set(expected_claim_ids)

    rows_by_claim_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_rows_total = 0
    pair_rows_parsed = 0
    pair_rows_dropped = 0
    pair_rows_nonzero_pv = 0

    if pairs_path.exists():
        with pairs_path.open("r", encoding="utf-8") as handle:
            for row_idx, line in enumerate(handle):
                payload = line.strip()
                if not payload:
                    continue
                pair_rows_total += 1
                try:
                    row = json.loads(payload)
                except json.JSONDecodeError:
                    pair_rows_dropped += 1
                    continue
                if not isinstance(row, dict):
                    pair_rows_dropped += 1
                    continue
                parsed = _parse_component1_pair_row(row, row_idx=row_idx)
                if parsed is None:
                    pair_rows_dropped += 1
                    continue
                claim_id, evidence_row = parsed
                rows_by_claim_id[claim_id].append(evidence_row)
                pair_rows_parsed += 1
                if (
                    abs(float(evidence_row["p_ent"])) > 0.0
                    or abs(float(evidence_row["p_con"])) > 0.0
                    or abs(float(evidence_row["rel"])) > 0.0
                ):
                    pair_rows_nonzero_pv += 1

    aligned_rows: list[list[dict[str, Any]]] = []
    claims_with_c1_pairs = 0
    claims_missing_c1_pairs = 0
    claims_using_fallback = 0
    claims_with_empty_evidence = 0
    fallback_edge_rows = 0

    for row_idx, claim_id in enumerate(expected_claim_ids):
        c1_rows = rows_by_claim_id.get(claim_id, [])
        if c1_rows:
            aligned_rows.append(c1_rows)
            claims_with_c1_pairs += 1
            continue

        claims_missing_c1_pairs += 1
        if missing_policy == "strict":
            aligned_rows.append([])
            claims_with_empty_evidence += 1
            continue

        fallback_rows = _fallback_rows_from_subgraph(claim_id=claim_id, subgraph_row=subgraphs_df.iloc[row_idx])
        aligned_rows.append(fallback_rows)
        claims_using_fallback += 1
        fallback_edge_rows += len(fallback_rows)
        if not fallback_rows:
            claims_with_empty_evidence += 1

    extra_claim_ids = sorted(set(rows_by_claim_id.keys()) - expected_claim_id_set)
    total_claims = len(expected_claim_ids)

    coverage_stats = {
        "split": split,
        "logs_root": str(logs_root),
        "pairs_file": str(pairs_path),
        "pairs_file_exists": pairs_path.exists(),
        "missing_policy": missing_policy,
        "claims_total": total_claims,
        "claims_with_component1_pairs": claims_with_c1_pairs,
        "claims_missing_component1_pairs": claims_missing_c1_pairs,
        "claims_using_fallback": claims_using_fallback,
        "claims_with_empty_evidence": claims_with_empty_evidence,
        "claims_with_component1_pairs_pct": _pct(claims_with_c1_pairs, total_claims),
        "claims_missing_component1_pairs_pct": _pct(claims_missing_c1_pairs, total_claims),
        "claims_using_fallback_pct": _pct(claims_using_fallback, total_claims),
        "claims_with_empty_evidence_pct": _pct(claims_with_empty_evidence, total_claims),
        "pair_rows_total": pair_rows_total,
        "pair_rows_parsed": pair_rows_parsed,
        "pair_rows_dropped": pair_rows_dropped,
        "pair_rows_nonzero_pv": pair_rows_nonzero_pv,
        "pair_rows_nonzero_pv_pct": _pct(pair_rows_nonzero_pv, max(pair_rows_parsed, 1)),
        "fallback_edge_rows": fallback_edge_rows,
        "extra_claim_ids_in_pairs": len(extra_claim_ids),
        "extra_claim_id_examples": extra_claim_ids[:10],
    }
    return aligned_rows, coverage_stats


def _fallback_rows_from_fever_evidence(claim_id: str, claim_row: Any) -> list[dict[str, Any]]:
    if hasattr(claim_row, "to_dict"):
        claim_row = claim_row.to_dict()
    if not isinstance(claim_row, dict):
        return []

    evidence_rows = claim_row.get("evidence_rows")
    if not isinstance(evidence_rows, list):
        return []

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(evidence_rows):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or row.get("raw_sentence") or "").strip()
        if not text:
            continue
        p_ent = _safe_float(row.get("p_ent"), 0.0)
        p_con = _safe_float(row.get("p_con"), 0.0)
        rel = _safe_float(row.get("rel"), p_ent + p_con)
        rows.append(
            {
                "evidence_id": str(row.get("evidence_id") or f"{claim_id}_fallback_{idx}"),
                "raw_triple": row.get("raw_triple"),
                "raw_sentence": text,
                "text": text,
                "pool": _normalize_pool(row.get("pool", "A")),
                "p_ent": p_ent,
                "p_con": p_con,
                "p_neu": _safe_float(row.get("p_neu"), max(0.0, 1.0 - (p_ent + p_con))),
                "rel": rel,
                "is_gold": bool(row.get("is_gold", True)),
                "is_fallback": True,
                "page": row.get("page"),
                "line": row.get("line"),
            }
        )
    return rows


def load_component1_evidence_rows_fever(
    split: str,
    claims_df,
    logs_root: str | Path,
    missing_policy: str = "hybrid_fallback",
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Load C1 pair logs aligned to FEVER rows, with fallback to FEVER evidence_rows."""

    supported_missing_policies = {"hybrid_fallback", "strict"}
    if missing_policy not in supported_missing_policies:
        raise ValueError(
            f"Unsupported missing policy `{missing_policy}`. "
            f"Expected one of {sorted(supported_missing_policies)}."
        )

    if not hasattr(claims_df, "columns"):
        raise TypeError("`claims_df` must be a dataframe-like object with FEVER claim rows.")

    if "claim_id" in claims_df.columns:
        expected_claim_ids = [str(v) for v in claims_df["claim_id"]]
    else:
        expected_claim_ids = [f"{split}_{idx}" for idx in range(len(claims_df))]
    expected_claim_id_set = set(expected_claim_ids)

    pairs_path = Path(logs_root) / split / "pairs.jsonl"
    rows_by_claim_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_rows_total = 0
    pair_rows_parsed = 0
    pair_rows_dropped = 0
    pair_rows_nonzero_pv = 0

    if pairs_path.exists():
        with pairs_path.open("r", encoding="utf-8") as handle:
            for row_idx, line in enumerate(handle):
                payload = line.strip()
                if not payload:
                    continue
                pair_rows_total += 1
                try:
                    row = json.loads(payload)
                except json.JSONDecodeError:
                    pair_rows_dropped += 1
                    continue
                if not isinstance(row, dict):
                    pair_rows_dropped += 1
                    continue
                parsed = _parse_component1_pair_row(row, row_idx=row_idx)
                if parsed is None:
                    pair_rows_dropped += 1
                    continue
                claim_id, evidence_row = parsed
                rows_by_claim_id[claim_id].append(evidence_row)
                pair_rows_parsed += 1
                if (
                    abs(float(evidence_row["p_ent"])) > 0.0
                    or abs(float(evidence_row["p_con"])) > 0.0
                    or abs(float(evidence_row["rel"])) > 0.0
                ):
                    pair_rows_nonzero_pv += 1

    aligned_rows: list[list[dict[str, Any]]] = []
    claims_with_c1_pairs = 0
    claims_missing_c1_pairs = 0
    claims_using_fallback = 0
    claims_with_empty_evidence = 0
    fallback_edge_rows = 0

    for row_idx, claim_id in enumerate(expected_claim_ids):
        c1_rows = rows_by_claim_id.get(claim_id, [])
        if c1_rows:
            aligned_rows.append(c1_rows)
            claims_with_c1_pairs += 1
            continue

        claims_missing_c1_pairs += 1
        if missing_policy == "strict":
            aligned_rows.append([])
            claims_with_empty_evidence += 1
            continue

        fallback_rows = _fallback_rows_from_fever_evidence(claim_id=claim_id, claim_row=claims_df.iloc[row_idx])
        aligned_rows.append(fallback_rows)
        claims_using_fallback += 1
        fallback_edge_rows += len(fallback_rows)
        if not fallback_rows:
            claims_with_empty_evidence += 1

    extra_claim_ids = sorted(set(rows_by_claim_id.keys()) - expected_claim_id_set)
    total_claims = len(expected_claim_ids)

    coverage_stats = {
        "split": split,
        "logs_root": str(logs_root),
        "pairs_file": str(pairs_path),
        "pairs_file_exists": pairs_path.exists(),
        "missing_policy": missing_policy,
        "claims_total": total_claims,
        "claims_with_component1_pairs": claims_with_c1_pairs,
        "claims_missing_component1_pairs": claims_missing_c1_pairs,
        "claims_using_fallback": claims_using_fallback,
        "claims_with_empty_evidence": claims_with_empty_evidence,
        "claims_with_component1_pairs_pct": _pct(claims_with_c1_pairs, total_claims),
        "claims_missing_component1_pairs_pct": _pct(claims_missing_c1_pairs, total_claims),
        "claims_using_fallback_pct": _pct(claims_using_fallback, total_claims),
        "claims_with_empty_evidence_pct": _pct(claims_with_empty_evidence, total_claims),
        "pair_rows_total": pair_rows_total,
        "pair_rows_parsed": pair_rows_parsed,
        "pair_rows_dropped": pair_rows_dropped,
        "pair_rows_nonzero_pv": pair_rows_nonzero_pv,
        "pair_rows_nonzero_pv_pct": _pct(pair_rows_nonzero_pv, max(pair_rows_parsed, 1)),
        "fallback_edge_rows": fallback_edge_rows,
        "extra_claim_ids_in_pairs": len(extra_claim_ids),
        "extra_claim_id_examples": extra_claim_ids[:10],
    }
    return aligned_rows, coverage_stats
