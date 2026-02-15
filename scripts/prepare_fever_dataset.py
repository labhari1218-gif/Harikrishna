#!/usr/bin/env python3
"""Download and normalize FEVER data for this repository.

Outputs dataset artifacts compatible with Component 3 loaders under ``data/fever``.

Primary source files:
- https://fever.ai/download/fever/train.jsonl
- https://fever.ai/download/fever/shared_task_dev.jsonl
- https://fever.ai/download/fever/shared_task_test.jsonl
- https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db

Notes:
- FEVER official test labels are hidden; this script writes:
  - ``fever_test.pkl`` from official dev (labeled) for internal evaluation.
  - ``fever_blind_test.pkl`` from official test (unlabeled) for submission workflows.
- ``Label`` is numeric and follows ``label_mode`` mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import unicodedata
import urllib.request
from urllib.parse import unquote
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

FEVER_URLS = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "test": "https://fever.ai/download/fever/shared_task_test.jsonl",
}
FEVER_DB_URL = "https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db"

BINARY_LABEL_MAP = {
    "REFUTES": 0,
    "SUPPORTS": 1,
}
THREE_WAY_LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
}


@dataclass(frozen=True)
class SplitBuildStats:
    split: str
    rows_total: int
    rows_written: int
    rows_skipped_label: int
    rows_missing_evidence_text: int
    rows_empty_evidence_after_resolution: int


class FeverSentenceResolver:
    """Resolve FEVER (wiki_page, line_id) references into sentence text."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection = sqlite3.connect(str(db_path))
        self._page_cache: dict[str, dict[int, str]] = {}

    @staticmethod
    def _normalize_doc_id(doc_id: str) -> str:
        # Matches FEVER/DrQA DB normalization behavior.
        return unicodedata.normalize("NFD", str(doc_id))

    @classmethod
    def _candidate_doc_ids(cls, page_title: str) -> list[str]:
        raw = str(page_title).strip()
        decoded = unquote(raw)
        variants = [
            raw,
            raw.replace(" ", "_"),
            raw.replace("_", " "),
            decoded,
            decoded.replace(" ", "_"),
            decoded.replace("_", " "),
        ]
        candidates: list[str] = []
        seen: set[str] = set()
        for value in variants:
            normalized = cls._normalize_doc_id(value)
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(normalized)
        return candidates

    def close(self) -> None:
        self.connection.close()

    def _load_page_sentences(self, page_title: str) -> dict[int, str]:
        if page_title in self._page_cache:
            return self._page_cache[page_title]

        cursor = self.connection.cursor()
        cursor.execute("SELECT lines FROM documents WHERE id = ?", (page_title,))
        result = cursor.fetchone()
        cursor.close()

        lines_map: dict[int, str] = {}
        if result is not None and result[0]:
            blob = str(result[0])
            for raw_line in blob.split("\n"):
                if not raw_line:
                    continue
                parts = raw_line.split("\t")
                if len(parts) < 2:
                    continue
                try:
                    line_idx = int(parts[0])
                except ValueError:
                    continue
                sentence_text = parts[1].strip()
                if sentence_text:
                    lines_map[line_idx] = sentence_text

        self._page_cache[page_title] = lines_map
        return lines_map

    def get_sentence(self, page_title: str, line_idx: int) -> str | None:
        for candidate in self._candidate_doc_ids(page_title):
            lines_map = self._load_page_sentences(candidate)
            sentence = lines_map.get(int(line_idx))
            if sentence:
                return sentence
        return None


def _download_file(url: str, output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path} already exists")
        return
    print(f"[download] {url} -> {output_path}")
    with urllib.request.urlopen(url) as response, output_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _extract_evidence_groups(row: dict[str, Any]) -> list[list[tuple[str, int]]]:
    raw_groups = row.get("evidence")
    if not isinstance(raw_groups, list):
        return []

    groups: list[list[tuple[str, int]]] = []
    for group in raw_groups:
        if not isinstance(group, list):
            continue
        pairs: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for item in group:
            if not isinstance(item, list) or len(item) < 4:
                continue
            page = item[2]
            line = item[3]
            if page is None or line is None:
                continue
            page_text = str(page)
            try:
                line_int = int(line)
            except (TypeError, ValueError):
                continue
            key = (page_text, line_int)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
        if pairs:
            groups.append(pairs)
    return groups


def _split_by_hash(records: list[dict[str, Any]], train_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("Argument `train_ratio` must be in (0, 1).")

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []

    for row in records:
        claim_id = str(row["claim_id"])
        digest = hashlib.sha256(f"{seed}:{claim_id}".encode("utf-8")).hexdigest()
        score = int(digest[:16], 16) / float(16 ** 16)
        if score < train_ratio:
            train_records.append(row)
        else:
            val_records.append(row)

    return train_records, val_records


def _build_records(
    *,
    split_name: str,
    rows: list[dict[str, Any]],
    label_map: dict[str, int] | None,
    resolver: FeverSentenceResolver | None,
    allow_missing_sentences: bool,
    max_claims: int,
) -> tuple[list[dict[str, Any]], SplitBuildStats]:
    rows_out: list[dict[str, Any]] = []
    rows_total = 0
    rows_skipped_label = 0
    rows_missing_evidence_text = 0
    rows_empty_evidence_after_resolution = 0

    for raw in rows:
        if max_claims > 0 and rows_total >= max_claims:
            break

        rows_total += 1
        label_text = raw.get("label")
        label_id = -1
        if label_map is not None:
            if label_text not in label_map:
                rows_skipped_label += 1
                continue
            label_id = int(label_map[str(label_text)])

        claim_int_id = int(raw.get("id", rows_total - 1))
        claim_id = f"{split_name}_{claim_int_id}"
        claim_text = str(raw.get("claim", "")).strip()

        evidence_groups_pairs = _extract_evidence_groups(raw)
        unique_pairs: list[tuple[str, int]] = []
        seen_pairs: set[tuple[str, int]] = set()
        for group in evidence_groups_pairs:
            for pair in group:
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                unique_pairs.append(pair)

        evidence_rows: list[dict[str, Any]] = []
        pair_to_evidence_id: dict[tuple[str, int], str] = {}
        for idx, (page, line) in enumerate(unique_pairs):
            sentence_text: str | None = None
            if resolver is not None:
                sentence_text = resolver.get_sentence(page_title=page, line_idx=line)

            if not sentence_text:
                rows_missing_evidence_text += 1
                if allow_missing_sentences:
                    sentence_text = f"{page.replace('_', ' ')} [line {line}]"
                else:
                    continue

            evidence_id = f"{claim_id}_ev_{idx}"
            pair_to_evidence_id[(page, line)] = evidence_id
            evidence_rows.append(
                {
                    "evidence_id": evidence_id,
                    "text": sentence_text,
                    "page": page,
                    "line": int(line),
                    "is_gold": True,
                    "pool": "A",
                    "source": "fever_gold",
                }
            )

        evidence_groups: list[list[str]] = []
        for group in evidence_groups_pairs:
            evidence_ids = [pair_to_evidence_id[pair] for pair in group if pair in pair_to_evidence_id]
            if evidence_ids:
                evidence_groups.append(evidence_ids)

        if label_map is not None and not evidence_rows and label_text in {"SUPPORTS", "REFUTES"}:
            rows_empty_evidence_after_resolution += 1

        rows_out.append(
            {
                "claim_id": claim_id,
                "fever_id": claim_int_id,
                "Sentence": claim_text,
                "Label": label_id,
                "label_text": label_text,
                "evidence_rows": evidence_rows,
                "evidence_groups": evidence_groups,
                "types": [],
                "Entity_set": [],
                "metadata": {
                    "split": split_name,
                    "verifiable": raw.get("verifiable"),
                    "raw_evidence_group_count": len(evidence_groups_pairs),
                },
            }
        )

    stats = SplitBuildStats(
        split=split_name,
        rows_total=rows_total,
        rows_written=len(rows_out),
        rows_skipped_label=rows_skipped_label,
        rows_missing_evidence_text=rows_missing_evidence_text,
        rows_empty_evidence_after_resolution=rows_empty_evidence_after_resolution,
    )
    return rows_out, stats


def _write_pickle(rows: list[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    print(f"[write] {path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FEVER data for Component 3")
    parser.add_argument("--data-root", type=Path, default=Path("data/fever"))
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--download-db", action="store_true")
    parser.add_argument("--label-mode", choices=["binary", "three_way"], default="binary")
    parser.add_argument("--val-source", choices=["train_split", "dev"], default="train_split")
    parser.add_argument(
        "--allow-val-test-overlap",
        action="store_true",
        help="Allow val/test overlap when --val-source=dev (disabled by default to prevent leakage).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--max-claims-per-split", type=int, default=0)
    parser.add_argument("--allow-missing-sentences", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root: Path = args.data_root
    raw_dir = args.raw_dir or (data_root / "raw")
    db_path = args.db_path or (data_root / "fever.db")

    # Download official FEVER claim files.
    _download_file(FEVER_URLS["train"], raw_dir / "train.jsonl", overwrite=args.overwrite)
    _download_file(FEVER_URLS["dev"], raw_dir / "shared_task_dev.jsonl", overwrite=args.overwrite)
    _download_file(FEVER_URLS["test"], raw_dir / "shared_task_test.jsonl", overwrite=args.overwrite)
    if args.download_db:
        _download_file(FEVER_DB_URL, db_path, overwrite=args.overwrite)

    resolver: FeverSentenceResolver | None = None
    if db_path.exists():
        print(f"[open] fever.db: {db_path}")
        resolver = FeverSentenceResolver(db_path=db_path)
    elif not args.allow_missing_sentences:
        raise FileNotFoundError(
            f"FEVER sentence database not found at {db_path}. "
            "Pass --download-db to fetch it, or --allow-missing-sentences to continue with placeholders."
        )

    label_map = BINARY_LABEL_MAP if args.label_mode == "binary" else THREE_WAY_LABEL_MAP
    if args.label_mode == "three_way":
        print(
            "[warn] Prepared FEVER labels in three_way mode. "
            "Current Component 3 trainer is binary-only and will reject label=2 (NEI)."
        )

    try:
        train_rows_raw = _iter_jsonl(raw_dir / "train.jsonl")
        dev_rows_raw = _iter_jsonl(raw_dir / "shared_task_dev.jsonl")
        test_rows_raw = _iter_jsonl(raw_dir / "shared_task_test.jsonl")

        max_claims = int(args.max_claims_per_split)
        train_records_full, train_stats = _build_records(
            split_name="train",
            rows=train_rows_raw,
            label_map=label_map,
            resolver=resolver,
            allow_missing_sentences=args.allow_missing_sentences,
            max_claims=max_claims,
        )
        dev_records_full, dev_stats = _build_records(
            split_name="dev",
            rows=dev_rows_raw,
            label_map=label_map,
            resolver=resolver,
            allow_missing_sentences=args.allow_missing_sentences,
            max_claims=max_claims,
        )
        blind_records, blind_stats = _build_records(
            split_name="test",
            rows=test_rows_raw,
            label_map=None,
            resolver=resolver,
            allow_missing_sentences=args.allow_missing_sentences,
            max_claims=max_claims,
        )
    finally:
        if resolver is not None:
            resolver.close()

    if args.val_source == "train_split":
        train_records, val_records = _split_by_hash(
            records=train_records_full,
            train_ratio=float(args.train_ratio),
            seed=int(args.seed),
        )
        test_records = dev_records_full
        val_origin = "train_split"
    else:
        if not args.allow_val_test_overlap:
            raise ValueError(
                "Refusing to build FEVER artifacts with val/test overlap. "
                "When --val-source=dev, both validation and test would come from "
                "shared_task_dev.jsonl. Re-run with --allow-val-test-overlap if this is intentional."
            )
        train_records = train_records_full
        val_records = dev_records_full
        test_records = dev_records_full
        val_origin = "dev"

    out_root = data_root
    _write_pickle(train_records, out_root / "fever_train.pkl")
    _write_pickle(val_records, out_root / "fever_dev.pkl")
    _write_pickle(test_records, out_root / "fever_test.pkl")
    _write_pickle(blind_records, out_root / "fever_blind_test.pkl")

    metadata = {
        "label_mode": args.label_mode,
        "label_map": label_map,
        "val_source": val_origin,
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "db_path": str(db_path),
        "allow_val_test_overlap": bool(args.allow_val_test_overlap),
        "stats": {
            "train_raw": train_stats.__dict__,
            "dev_raw": dev_stats.__dict__,
            "blind_test_raw": blind_stats.__dict__,
            "train_out_rows": len(train_records),
            "val_out_rows": len(val_records),
            "test_out_rows": len(test_records),
            "blind_test_out_rows": len(blind_records),
        },
    }
    metadata_path = out_root / "fever_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[write] {metadata_path}")


if __name__ == "__main__":
    main()
