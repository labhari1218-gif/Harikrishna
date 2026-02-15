#!/usr/bin/env python3
"""Offline cache precompute for Component 3 FactKG/FEVER datasets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _cache_tag(model_name: str, max_seq_len: int) -> str:
    tag = f"{model_name}_len{int(max_seq_len)}"
    return tag.replace("/", "_").replace("\\", "_").replace("-", "_")


def _normalize_splits(raw_splits: str) -> list[str]:
    allowed = {"train", "val", "test"}
    splits = [part.strip().lower() for part in str(raw_splits).split(",") if part.strip()]
    if not splits:
        raise ValueError("Argument `--splits` produced an empty split list.")
    invalid = [split for split in splits if split not in allowed]
    if invalid:
        raise ValueError(f"Unsupported split(s): {invalid}. Expected subset of {sorted(allowed)}.")
    return splits


def _manifest_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_fever_split_df(data_root: Path, split: str):
    import pandas as pd

    split_to_file = {
        "train": "fever_train.pkl",
        "val": "fever_dev.pkl",
        "test": "fever_test.pkl",
    }
    if split not in split_to_file:
        raise ValueError(f"Unsupported FEVER split `{split}`.")
    path = data_root / split_to_file[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FEVER artifact: {path}. "
            "Run `python scripts/prepare_fever_dataset.py` first."
        )
    return pd.read_pickle(path)


def _precompute_factkg(args: argparse.Namespace, splits: list[str]) -> dict[str, Any]:
    from component3.c1_pairs_loader import load_component1_evidence_rows
    from component3.pv_dataset import FactKGPVDatasetGraph
    from datasets import get_df, get_subgraphs

    summary: dict[str, Any] = {"splits": {}}
    for split_idx, split in enumerate(splits):
        claims_df = get_df(split).reset_index(drop=True)
        subgraphs_df = get_subgraphs(split, args.subgraph_type).reset_index(drop=True)

        if args.use_component1_pairs:
            evidence_rows, coverage = load_component1_evidence_rows(
                split=split,
                claims_df=claims_df,
                subgraphs_df=subgraphs_df,
                logs_root=args.component1_logs_root,
                missing_policy=args.missing_pv_policy,
            )
        else:
            evidence_rows = subgraphs_df
            coverage = {
                "split": split,
                "source": "raw_subgraphs",
                "claims_total": len(claims_df),
            }

        dataset = FactKGPVDatasetGraph(
            df=claims_df,
            evidence=evidence_rows,
            claim_triple_model_name=args.encoder_model_name,
            claim_max_length=args.max_seq_len,
            claim_triple_cache_path=args.factkg_claim_triple_cache_path,
            auto_precompute=False,
            require_claim_triple_cache=False,
            precompute_batch_size=args.batch_size,
        )
        before = len(dataset.claim_triple_embeddings)
        dataset.precompute_claim_triple_embeddings(force=(args.force and split_idx == 0))
        after = len(dataset.claim_triple_embeddings)
        summary["splits"][split] = {
            "claims": len(claims_df),
            "cache_entries_before": before,
            "cache_entries_after": after,
            "cache_entries_added": max(0, after - before),
            "coverage": coverage,
        }

    summary["cache_path"] = str(args.factkg_claim_triple_cache_path)
    return summary


def _precompute_fever(args: argparse.Namespace, splits: list[str]) -> dict[str, Any]:
    from component3.c1_pairs_loader import load_component1_evidence_rows_fever
    from component3.pv_dataset_fever import FeverPVDatasetGraph

    data_root = Path(args.fever_data_root)
    cache_tag = _cache_tag(args.encoder_model_name, args.max_seq_len)
    sentence_cache_path = Path(args.fever_sentence_cache_path or (data_root / f"sentence_embeddings_{cache_tag}.pkl"))
    claim_sentence_cache_path = Path(
        args.fever_claim_sentence_cache_path or (data_root / f"claim_sentence_embeddings_{cache_tag}.pkl")
    )

    summary: dict[str, Any] = {"splits": {}}
    for split_idx, split in enumerate(splits):
        claims_df = _load_fever_split_df(data_root, split).reset_index(drop=True)
        if args.use_component1_pairs:
            evidence_rows, coverage = load_component1_evidence_rows_fever(
                split=split,
                claims_df=claims_df,
                logs_root=args.fever_component1_logs_root,
                missing_policy=args.missing_pv_policy,
            )
        else:
            evidence_rows = claims_df["evidence_rows"].tolist() if "evidence_rows" in claims_df.columns else []
            coverage = {
                "split": split,
                "source": "fever_raw_evidence_rows",
                "claims_total": len(claims_df),
            }

        dataset = FeverPVDatasetGraph(
            df=claims_df,
            evidence=evidence_rows,
            sentence_cache_path=sentence_cache_path,
            claim_sentence_cache_path=claim_sentence_cache_path,
            sentence_model_name=args.encoder_model_name,
            max_seq_len=args.max_seq_len,
            include_s_pool=args.fever_include_s_pool,
            auto_precompute=False,
            precompute_batch_size=args.batch_size,
        )
        sentence_before = len(dataset.sentence_embeddings)
        claim_sentence_before = len(dataset.claim_sentence_embeddings)
        force_cache_reset = bool(args.force and split_idx == 0)
        dataset.precompute_sentence_embeddings(force=force_cache_reset)
        dataset.precompute_claim_sentence_embeddings(force=force_cache_reset)
        sentence_after = len(dataset.sentence_embeddings)
        claim_sentence_after = len(dataset.claim_sentence_embeddings)
        summary["splits"][split] = {
            "claims": len(claims_df),
            "sentence_cache_before": sentence_before,
            "sentence_cache_after": sentence_after,
            "sentence_cache_added": max(0, sentence_after - sentence_before),
            "claim_sentence_cache_before": claim_sentence_before,
            "claim_sentence_cache_after": claim_sentence_after,
            "claim_sentence_cache_added": max(0, claim_sentence_after - claim_sentence_before),
            "coverage": coverage,
        }

    summary["sentence_cache_path"] = str(sentence_cache_path)
    summary["claim_sentence_cache_path"] = str(claim_sentence_cache_path)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute Component 3 embedding caches")
    parser.add_argument("--dataset", type=str, choices=["factkg", "fever"], required=True)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--encoder-model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--component1-logs-root", type=str, default="logs/component1")
    parser.add_argument("--fever-component1-logs-root", type=str, default="logs/component1_fever")
    parser.add_argument("--missing-pv-policy", type=str, default="hybrid_fallback")
    parser.add_argument("--use-component1-pairs", dest="use_component1_pairs", action="store_true")
    parser.add_argument("--no-use-component1-pairs", dest="use_component1_pairs", action="store_false")
    parser.set_defaults(use_component1_pairs=True)

    parser.add_argument("--subgraph-type", type=str, default="direct_filled")
    parser.add_argument("--factkg-claim-triple-cache-path", type=str, default="data/claim_triple_embeddings.pkl")

    parser.add_argument("--fever-data-root", type=str, default="data/fever")
    parser.add_argument("--fever-include-s-pool", action="store_true")
    parser.add_argument("--fever-sentence-cache-path", type=str, default=None)
    parser.add_argument("--fever-claim-sentence-cache-path", type=str, default=None)

    parser.add_argument("--manifest-path", type=str, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    splits = _normalize_splits(args.splits)

    if args.max_seq_len < 1:
        raise ValueError("Argument `--max-seq-len` must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("Argument `--batch-size` must be >= 1.")

    started_at = datetime.now(timezone.utc).isoformat()
    if args.dataset == "factkg":
        summary = _precompute_factkg(args, splits)
    else:
        summary = _precompute_fever(args, splits)
    completed_at = datetime.now(timezone.utc).isoformat()

    fingerprint_input = {
        "dataset": args.dataset,
        "splits": splits,
        "encoder_model_name": args.encoder_model_name,
        "max_seq_len": int(args.max_seq_len),
        "batch_size": int(args.batch_size),
        "force": bool(args.force),
        "use_component1_pairs": bool(args.use_component1_pairs),
        "missing_pv_policy": args.missing_pv_policy,
    }
    manifest = {
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "config": fingerprint_input,
        "fingerprint": _manifest_fingerprint(fingerprint_input),
        "summary": summary,
    }

    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else Path("runs") / f"precompute_{args.dataset}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "fingerprint": manifest["fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()
