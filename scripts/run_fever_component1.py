#!/usr/bin/env python3
"""Run Component 1 (PV + ESM) on FEVER sentence evidence.

This script consumes normalized FEVER pickles produced by
``scripts/prepare_fever_dataset.py`` and writes claim/pair logs under
``logs/component1_fever/<split>``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import pandas as pd
from tqdm import tqdm

from component1 import EvidenceItem, EvidenceStateManager, ESMConfig, evidence_to_text, VERBALIZER_VERSION
from component1.cache import PVCache
from component1.evidence import stable_hash
from component1.logging_utils import PairLogConfig, write_claim_log, write_pair_log
from component1.pv import PVConfig, PVScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


LABEL_NAME_BY_ID = {
    0: "REFUTES",
    1: "SUPPORTS",
    2: "NOT ENOUGH INFO",
}


def _canonical_split(split: str) -> str:
    # Keep compatibility with FEVER terminology but align runtime outputs to train/val/test.
    if split == "dev":
        return "val"
    return split


def _load_fever_df(split: str, data_root: Path) -> pd.DataFrame:
    split_to_file = {
        "train": "fever_train.pkl",
        "val": "fever_dev.pkl",
        "test": "fever_test.pkl",
        "blind_test": "fever_blind_test.pkl",
    }
    if split not in split_to_file:
        raise ValueError(f"Unsupported split `{split}`.")

    path = data_root / split_to_file[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FEVER artifact: {path}. Run scripts/prepare_fever_dataset.py first."
        )
    df = pd.read_pickle(path).reset_index(drop=True)
    logger.info("Loaded %s rows from %s", len(df), path)
    return df


def _coerce_claim_label(row: pd.Series) -> str:
    label_text = row.get("label_text")
    if isinstance(label_text, str) and label_text.strip():
        return label_text

    label_value = row.get("Label", -1)
    try:
        label_int = int(label_value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    return LABEL_NAME_BY_ID.get(label_int, "UNKNOWN")


def _build_sentence_evidence_pool(claim_id: str, row: pd.Series) -> list[EvidenceItem]:
    evidence_rows = row.get("evidence_rows", [])
    if not isinstance(evidence_rows, list):
        return []

    pool: list[EvidenceItem] = []
    for idx, ev in enumerate(evidence_rows):
        if not isinstance(ev, dict):
            continue

        evidence_text = str(ev.get("text") or "").strip()
        if not evidence_text:
            continue

        evidence_id = str(ev.get("evidence_id") or f"{claim_id}_ev_{idx}")
        doc_id = ev.get("page")
        sent_id = ev.get("line")

        pool.append(
            EvidenceItem(
                evidence_id=evidence_id,
                kind="sentence",
                content={"text": evidence_text, "doc_id": doc_id, "sent_id": sent_id},
                retrieval_score=0.0,
                pv=None,
                meta={
                    "is_gold": bool(ev.get("is_gold", True)),
                    "source": ev.get("source", "fever_gold"),
                },
            )
        )
    return pool


def _process_claim(
    *,
    claim_id: str,
    claim_text: str,
    label: str,
    claim_entities: list[str],
    evidence_pool: list[EvidenceItem],
    pv_scorer: Optional[PVScorer],
    pv_cache: PVCache,
    esm_config: ESMConfig,
    pv_config: PVConfig,
    pair_log_config: PairLogConfig,
    claim_log_file: Path,
    pair_log_file: Path,
    cache_only: bool,
) -> bool:
    if not evidence_pool:
        logger.warning("Claim %s has no FEVER evidence rows; skipping.", claim_id)
        return False

    claim_hash = stable_hash(claim_text)
    premise_texts: list[str] = []

    for item in evidence_pool:
        premise_text, _raw_repr = evidence_to_text(item)
        premise_texts.append(premise_text)

    to_score_indices: list[int] = []
    to_score_premises: list[str] = []

    for idx, item in enumerate(evidence_pool):
        premise_text = premise_texts[idx]
        evidence_hash = stable_hash(premise_text)
        cached_pv = pv_cache.get(
            claim_hash,
            evidence_hash,
            pv_config.model_name,
            pv_config.max_length,
            VERBALIZER_VERSION,
            claim_id=claim_id,
        )
        if cached_pv is not None:
            item.pv = cached_pv
        else:
            to_score_indices.append(idx)
            to_score_premises.append(premise_text)

    if to_score_premises:
        if cache_only or pv_scorer is None:
            raise RuntimeError(
                f"Cache-only mode missing {len(to_score_premises)} PV scores for claim {claim_id}."
            )
        pv_results = pv_scorer.pv_score_many(claim_text, to_score_premises)
        for i, pv_result in enumerate(pv_results):
            idx = to_score_indices[i]
            item = evidence_pool[idx]
            item.pv = pv_result

            evidence_hash = stable_hash(to_score_premises[i])
            pv_cache.put(
                claim_hash,
                evidence_hash,
                pv_config.model_name,
                pv_config.max_length,
                VERBALIZER_VERSION,
                pv_result,
            )

    partition_result = EvidenceStateManager.partition(evidence_pool, esm_config)
    A = partition_result["A"]
    S = partition_result["S"]
    C = partition_result["C"]

    write_claim_log(
        claim_id=claim_id,
        claim_text=claim_text,
        label=label,
        claim_entities=claim_entities,
        A=A,
        S=S,
        C=C,
        pool=evidence_pool,
        min_A=esm_config.min_A,
        contra_tau=esm_config.contra_tau,
        out_file=claim_log_file,
    )

    if pair_log_config.mode != "none":
        assignment_map: dict[str, str] = {}
        for item in A:
            assignment_map[item.evidence_id] = "A"
        for item in S:
            assignment_map[item.evidence_id] = "S"
        for item in C:
            assignment_map[item.evidence_id] = "C"

        for idx, item in enumerate(evidence_pool):
            write_pair_log(
                claim_id=claim_id,
                evidence_item=item,
                claim_text=claim_text,
                premise_text=premise_texts[idx],
                model_name=pv_config.model_name,
                verbalizer_id=VERBALIZER_VERSION,
                config=pair_log_config,
                out_file=pair_log_file,
                evidence_assignment=assignment_map.get(item.evidence_id, "unknown"),
            )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Component 1 PV+ESM on FEVER sentence evidence")
    parser.add_argument("--split", required=True, choices=["train", "val", "dev", "test", "blind_test"])
    parser.add_argument("--data-root", type=str, default="data/fever")
    parser.add_argument("--limit-claims", type=int, default=None)

    parser.add_argument("--out-dir", type=str, default="logs/component1_fever")

    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-base-mnli")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--non-blocking-transfers",
        action="store_true",
        help="Enable non-blocking host-to-device PV batch transfer when CUDA is active",
    )

    parser.add_argument("--min-A", type=int, default=5)
    parser.add_argument("--max-A", type=int, default=20)
    parser.add_argument("--max-C", type=int, default=5)
    parser.add_argument("--contra-tau", type=float, default=0.6)

    parser.add_argument("--cache-dir", type=str, default="cache/pv_sqlite")
    parser.add_argument("--pair-log-mode", type=str, default="all", choices=["none", "all", "A_only", "A_and_C", "sample"])
    parser.add_argument("--pair-log-sample-rate", type=float, default=0.1)
    parser.add_argument("--cache-only", action="store_true")

    args = parser.parse_args()
    split = _canonical_split(args.split)

    out_dir = Path(args.out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    claim_log_file = out_dir / "claims.jsonl"
    pair_log_file = out_dir / "pairs.jsonl"

    # Remove stale finalized outputs up-front so a failed run cannot leave old
    # logs that look like fresh output.
    if claim_log_file.exists():
        claim_log_file.unlink()
    if pair_log_file.exists():
        pair_log_file.unlink()

    claim_log_tmp = out_dir / "claims.jsonl.tmp"
    pair_log_tmp = out_dir / "pairs.jsonl.tmp"
    if claim_log_tmp.exists():
        claim_log_tmp.unlink()
    if pair_log_tmp.exists():
        pair_log_tmp.unlink()
    claim_log_tmp.touch()
    if args.pair_log_mode != "none":
        pair_log_tmp.touch()

    pv_config = PVConfig(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        non_blocking_transfers=args.non_blocking_transfers,
    )
    pv_scorer: Optional[PVScorer] = None
    if not args.cache_only:
        pv_scorer = PVScorer(pv_config)
    else:
        logger.info("Cache-only mode enabled: skipping PV model initialization.")

    pv_cache = PVCache(args.cache_dir, read_only=args.cache_only)
    esm_config = ESMConfig(min_A=args.min_A, max_A=args.max_A, max_C=args.max_C, contra_tau=args.contra_tau)
    pair_log_config = PairLogConfig(mode=args.pair_log_mode, sample_rate=args.pair_log_sample_rate)

    claims_df = _load_fever_df(split, Path(args.data_root))
    if args.limit_claims is not None:
        claims_df = claims_df.head(int(args.limit_claims)).reset_index(drop=True)
        logger.info("Limited to %s claims", len(claims_df))

    claim_error_count = 0
    processed_with_evidence = 0
    skipped_no_evidence = 0

    try:
        for i in tqdm(range(len(claims_df)), desc="Processing FEVER claims"):
            row = claims_df.iloc[i]
            claim_id = str(row.get("claim_id") or f"{split}_{i}")
            claim_text = str(row.get("Sentence") or "")
            label = _coerce_claim_label(row)

            entity_set = row.get("Entity_set", [])
            if not isinstance(entity_set, list):
                entity_set = []

            evidence_pool = _build_sentence_evidence_pool(claim_id=claim_id, row=row)

            try:
                claim_written = _process_claim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    label=label,
                    claim_entities=entity_set,
                    evidence_pool=evidence_pool,
                    pv_scorer=pv_scorer,
                    pv_cache=pv_cache,
                    esm_config=esm_config,
                    pv_config=pv_config,
                    pair_log_config=pair_log_config,
                    claim_log_file=claim_log_tmp,
                    pair_log_file=pair_log_tmp,
                    cache_only=args.cache_only,
                )
                if claim_written:
                    processed_with_evidence += 1
                else:
                    skipped_no_evidence += 1
            except Exception:
                claim_error_count += 1
                logger.exception("Error processing claim %s", claim_id)
                continue
    finally:
        pv_cache.flush()
        pv_cache.close()

    cache_stats = pv_cache.get_stats()
    logger.info("PV Cache Statistics: hits=%s misses=%s", cache_stats["hits"], cache_stats["misses"])
    logger.info("Claims written: %s", processed_with_evidence)
    logger.info("Claims skipped (no evidence): %s", skipped_no_evidence)
    logger.info("Claim processing errors: %s", claim_error_count)

    if claim_error_count > 0:
        if claim_log_file.exists():
            claim_log_file.unlink()
        if pair_log_file.exists():
            pair_log_file.unlink()
        logger.error("Run finished with claim-level errors; temp logs kept for inspection and final logs removed.")
        raise RuntimeError(f"FEVER Component 1 failed with {claim_error_count} errors.")

    claim_log_tmp.replace(claim_log_file)
    if args.pair_log_mode != "none":
        pair_log_tmp.replace(pair_log_file)
    elif pair_log_file.exists():
        pair_log_file.unlink()

    logger.info("Processing complete")
    logger.info("Claim-level logs: %s", claim_log_file)
    if args.pair_log_mode != "none":
        logger.info("Pair-level logs: %s", pair_log_file)


if __name__ == "__main__":
    main()
