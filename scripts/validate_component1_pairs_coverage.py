#!/usr/bin/env python3
"""Validate Component 1 pair-log coverage for Component 3 ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from component3.c1_pairs_loader import load_component1_evidence_rows
from datasets import get_df, get_subgraphs


def _parse_splits(raw: str) -> list[str]:
    splits = [token.strip() for token in raw.split(",") if token.strip()]
    valid = {"train", "val", "test"}
    invalid = [split for split in splits if split not in valid]
    if invalid:
        raise ValueError(f"Invalid split(s): {invalid}. Valid splits: {sorted(valid)}")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate C1 pair-log coverage for Component 3.")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--logs-root", type=str, default="logs/component1")
    parser.add_argument("--subgraph-type", type=str, default="direct_filled")
    parser.add_argument("--missing-policy", type=str, default="hybrid_fallback")
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    splits = _parse_splits(args.splits)
    report = {"splits": {}, "logs_root": args.logs_root, "missing_policy": args.missing_policy}

    for split in splits:
        claims_df = get_df(split)
        subgraphs_df = get_subgraphs(split, args.subgraph_type)
        _rows, coverage = load_component1_evidence_rows(
            split=split,
            claims_df=claims_df,
            subgraphs_df=subgraphs_df,
            logs_root=args.logs_root,
            missing_policy=args.missing_policy,
        )
        report["splits"][split] = coverage
        print(
            f"[{split}] claims={coverage['claims_total']} "
            f"pair_rows={coverage['pair_rows_parsed']}/{coverage['pair_rows_total']} "
            f"missing_claims={coverage['claims_missing_component1_pairs']} "
            f"fallback_claims={coverage['claims_using_fallback']} "
            f"fallback_pct={coverage['claims_using_fallback_pct']:.2%}"
        )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote coverage report to: {out_path}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

