#!/usr/bin/env python3
"""Audit Component 1 pair-log coverage and field health for strict Component 3 runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import pickle
from pathlib import Path
import sys
from typing import Any

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from constants import DATA_PATH, EMBEDDINGS_FILENAME
from datasets import get_df, get_subgraphs

VALID_POOLS = {"A", "S", "C"}


def _parse_splits(raw: str) -> list[str]:
    splits = [token.strip() for token in raw.split(",") if token.strip()]
    valid = {"train", "val", "test"}
    invalid = [split for split in splits if split not in valid]
    if invalid:
        raise ValueError(f"Invalid split(s): {invalid}. Valid splits: {sorted(valid)}")
    return splits


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_raw_triple(raw_triple: Any) -> bool:
    if not isinstance(raw_triple, (list, tuple)) or len(raw_triple) < 3:
        return False
    subj = str(raw_triple[0]).strip()
    rel = str(raw_triple[1]).strip()
    obj = str(raw_triple[2]).strip()
    return bool(subj and rel and obj)


def _subgraph_triple_count(subgraph_row: Any) -> int:
    if hasattr(subgraph_row, "to_dict"):
        subgraph_row = subgraph_row.to_dict()
    if not isinstance(subgraph_row, dict):
        return 0

    walked = subgraph_row.get("walked", {})
    connected = walked.get("connected", []) if isinstance(walked, dict) else []
    walkable = walked.get("walkable", []) if isinstance(walked, dict) else []
    seen: set[tuple[str, str, str]] = set()
    for triple in list(connected) + list(walkable):
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        key = (str(triple[0]), str(triple[1]), str(triple[2]))
        seen.add(key)
    return int(len(seen))


def _load_embedding_entities(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    with path.open("rb") as fp:
        loaded = pickle.load(fp)
    if not isinstance(loaded, dict):
        return None
    return {str(key) for key in loaded.keys()}


def _audit_split(
    *,
    split: str,
    logs_root: Path,
    prob_sum_tol: float,
    embedding_entities: set[str] | None,
) -> dict[str, Any]:
    claims_df = get_df(split).reset_index(drop=True)
    subgraphs_df = get_subgraphs(split, "direct_filled").reset_index(drop=True)
    expected_claim_ids = [f"{split}_{idx}" for idx in range(len(claims_df))]
    expected_claim_id_set = set(expected_claim_ids)

    pairs_path = logs_root / split / "pairs.jsonl"
    rows_total = 0
    rows_bad_json = 0
    rows_missing_claim_id = 0
    rows_empty_evidence_sentinel = 0
    rows_missing_schema_version = 0

    claim_row_counts: dict[str, int] = defaultdict(int)
    claims_missing_pv_fields: set[str] = set()
    claims_missing_pool_labels: set[str] = set()
    claims_malformed_probs: set[str] = set()
    claims_missing_rel: set[str] = set()
    claims_missing_triple_text_fields: set[str] = set()
    claims_missing_entity_embeddings: set[str] = set()
    claims_empty_evidence_sentinel: set[str] = set()
    claims_missing_schema_version: set[str] = set()

    if pairs_path.exists():
        with pairs_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                rows_total += 1
                try:
                    row = json.loads(payload)
                except json.JSONDecodeError:
                    rows_bad_json += 1
                    continue
                if not isinstance(row, dict):
                    rows_bad_json += 1
                    continue

                claim_id = str(row.get("claim_id") or "").strip()
                if not claim_id:
                    rows_missing_claim_id += 1
                    continue
                claim_row_counts[claim_id] += 1

                if row.get("schema_version") is None:
                    rows_missing_schema_version += 1
                    claims_missing_schema_version.add(claim_id)

                is_empty_sentinel = bool(row.get("empty_evidence_sentinel")) or (
                    str(row.get("record_type") or "").strip().lower() == "claim_sentinel"
                )
                if is_empty_sentinel:
                    rows_empty_evidence_sentinel += 1
                    claims_empty_evidence_sentinel.add(claim_id)

                pool = str(row.get("pool") or row.get("evidence_assignment") or "").upper()
                if pool not in VALID_POOLS:
                    claims_missing_pool_labels.add(claim_id)

                probs = row.get("probs")
                if not isinstance(probs, dict):
                    claims_missing_pv_fields.add(claim_id)
                    claims_malformed_probs.add(claim_id)
                else:
                    ent = _safe_float(probs.get("entail"))
                    con = _safe_float(probs.get("contra"))
                    neu = _safe_float(probs.get("neutral"))
                    if ent is None or con is None or neu is None:
                        claims_missing_pv_fields.add(claim_id)
                        claims_malformed_probs.add(claim_id)
                    else:
                        prob_sum = ent + con + neu
                        if (
                            ent < 0.0
                            or ent > 1.0
                            or con < 0.0
                            or con > 1.0
                            or neu < 0.0
                            or neu > 1.0
                            or abs(prob_sum - 1.0) > float(prob_sum_tol)
                        ):
                            claims_malformed_probs.add(claim_id)

                derived = row.get("derived")
                rel = derived.get("rel") if isinstance(derived, dict) else None
                rel_value = _safe_float(rel)
                if rel_value is None:
                    claims_missing_rel.add(claim_id)
                    claims_missing_pv_fields.add(claim_id)

                raw_triple = row.get("raw_triple")
                has_raw_triple = _valid_raw_triple(raw_triple)
                raw_sentence = str(row.get("raw_sentence") or row.get("premise_text") or "").strip()
                if not has_raw_triple and not raw_sentence:
                    claims_missing_triple_text_fields.add(claim_id)

                if embedding_entities is not None and has_raw_triple:
                    subj = str(raw_triple[0]).strip()
                    obj = str(raw_triple[2]).strip()
                    if subj not in embedding_entities or obj not in embedding_entities:
                        claims_missing_entity_embeddings.add(claim_id)

    claims_with_pairs = {cid for cid, cnt in claim_row_counts.items() if cnt > 0}
    claims_missing_pairs = sorted(expected_claim_id_set - claims_with_pairs)
    claims_extra_pairs = sorted(claims_with_pairs - expected_claim_id_set)

    missing_pairs_with_zero_subgraph = 0
    missing_pairs_with_nonzero_subgraph = 0
    for claim_id in claims_missing_pairs:
        try:
            idx = int(claim_id.rsplit("_", 1)[1])
        except (TypeError, ValueError, IndexError):
            continue
        if idx < 0 or idx >= len(subgraphs_df):
            continue
        triple_count = _subgraph_triple_count(subgraphs_df.iloc[idx])
        if triple_count <= 0:
            missing_pairs_with_zero_subgraph += 1
        else:
            missing_pairs_with_nonzero_subgraph += 1

    def _count_expected(values: set[str]) -> int:
        return int(len(values & expected_claim_id_set))

    report = {
        "split": split,
        "paths": {
            "pairs_jsonl": str(pairs_path),
        },
        "rows": {
            "total": int(rows_total),
            "bad_json": int(rows_bad_json),
            "missing_claim_id": int(rows_missing_claim_id),
            "empty_evidence_sentinel": int(rows_empty_evidence_sentinel),
            "missing_schema_version": int(rows_missing_schema_version),
        },
        "claims": {
            "total": int(len(expected_claim_ids)),
            "with_at_least_1_pair_row": int(len(claims_with_pairs & expected_claim_id_set)),
            "without_pair_rows": int(len(claims_missing_pairs)),
            "without_pair_rows_with_zero_subgraph_triples": int(missing_pairs_with_zero_subgraph),
            "without_pair_rows_with_nonzero_subgraph_triples": int(missing_pairs_with_nonzero_subgraph),
            "missing_pv_fields": _count_expected(claims_missing_pv_fields),
            "missing_pool_labels": _count_expected(claims_missing_pool_labels),
            "malformed_probs": _count_expected(claims_malformed_probs),
            "missing_rel": _count_expected(claims_missing_rel),
            "missing_triple_text_fields": _count_expected(claims_missing_triple_text_fields),
            "missing_entity_embeddings": _count_expected(claims_missing_entity_embeddings),
            "empty_evidence_sentinel": _count_expected(claims_empty_evidence_sentinel),
            "missing_schema_version": _count_expected(claims_missing_schema_version),
            "extra_claim_ids_in_pairs": int(len(claims_extra_pairs)),
        },
        "examples": {
            "missing_pair_claim_ids": claims_missing_pairs[:25],
            "extra_pair_claim_ids": claims_extra_pairs[:25],
            "missing_pv_fields_claim_ids": sorted((claims_missing_pv_fields & expected_claim_id_set))[:25],
            "missing_pool_labels_claim_ids": sorted((claims_missing_pool_labels & expected_claim_id_set))[:25],
            "malformed_probs_claim_ids": sorted((claims_malformed_probs & expected_claim_id_set))[:25],
            "missing_rel_claim_ids": sorted((claims_missing_rel & expected_claim_id_set))[:25],
            "missing_triple_text_claim_ids": sorted((claims_missing_triple_text_fields & expected_claim_id_set))[:25],
            "missing_entity_embedding_claim_ids": sorted((claims_missing_entity_embeddings & expected_claim_id_set))[:25],
            "empty_sentinel_claim_ids": sorted((claims_empty_evidence_sentinel & expected_claim_id_set))[:25],
            "missing_schema_version_claim_ids": sorted((claims_missing_schema_version & expected_claim_id_set))[:25],
        },
    }
    if embedding_entities is not None:
        report["entity_embedding_check"] = {
            "enabled": True,
            "embedding_entities_total": int(len(embedding_entities)),
        }
    else:
        report["entity_embedding_check"] = {
            "enabled": False,
            "reason": "embeddings file missing or unreadable",
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Component1 pair-log coverage and schema health.")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--logs-root", type=str, default="logs/component1")
    parser.add_argument("--embeddings-path", type=str, default=str(Path(DATA_PATH) / EMBEDDINGS_FILENAME))
    parser.add_argument("--prob-sum-tol", type=float, default=1e-3)
    parser.add_argument("--out-json", type=str, default=None, help="Optional aggregate output path.")
    args = parser.parse_args()

    logs_root = Path(args.logs_root)
    splits = _parse_splits(args.splits)
    embedding_entities = _load_embedding_entities(Path(args.embeddings_path))

    aggregate: dict[str, Any] = {
        "logs_root": str(logs_root),
        "splits": {},
        "embeddings_path": str(args.embeddings_path),
        "prob_sum_tol": float(args.prob_sum_tol),
    }

    for split in splits:
        split_report = _audit_split(
            split=split,
            logs_root=logs_root,
            prob_sum_tol=float(args.prob_sum_tol),
            embedding_entities=embedding_entities,
        )
        aggregate["splits"][split] = split_report

        out_path = logs_root / split / "coverage_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(split_report, indent=2), encoding="utf-8")

        claims = split_report["claims"]
        print(
            f"[{split}] total={claims['total']} with_pairs={claims['with_at_least_1_pair_row']} "
            f"without_pairs={claims['without_pair_rows']} missing_pv={claims['missing_pv_fields']} "
            f"missing_pool={claims['missing_pool_labels']} malformed_probs={claims['malformed_probs']} "
            f"missing_rel={claims['missing_rel']} missing_text={claims['missing_triple_text_fields']} "
            f"missing_emb={claims['missing_entity_embeddings']}"
        )
        print(f"[{split}] wrote {out_path}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"Wrote aggregate report to: {out_path}")


if __name__ == "__main__":
    main()

