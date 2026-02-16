#!/usr/bin/env python3
"""Summarize backtracking efficacy from run artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


def _infer_split(claim_id: str, default_split: str) -> str:
    text = str(claim_id).strip()
    if "_" in text:
        prefix = text.split("_", 1)[0].strip().lower()
        if prefix in {"train", "val", "test"}:
            return prefix
    return default_split


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def analyze_run(
    *,
    run_dir: Path,
    useful_delta: float,
    default_split: str,
) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
    actions_path = run_dir / "recovery_actions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing recovery actions file: {actions_path}")

    claims_by_split: dict[str, set[str]] = defaultdict(set)
    triggered_by_split: dict[str, set[str]] = defaultdict(set)
    actions_by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_claim: dict[str, dict[str, Any]] = {}
    claim_split: dict[str, str] = {}

    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            claim_id = str(row.get("claim_id"))
            split = _infer_split(claim_id, default_split=default_split)
            claim_split[claim_id] = split
            claims_by_split[split].add(claim_id)
            predictions_by_claim[claim_id] = row
            if bool(row.get("backtracking_triggered", False)):
                triggered_by_split[split].add(claim_id)

    with actions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            claim_id = str(row.get("claim_id"))
            split = claim_split.get(claim_id, _infer_split(claim_id, default_split=default_split))
            claim_split[claim_id] = split
            actions_by_claim[claim_id].append(row)
            if bool(row.get("triggered", False)):
                triggered_by_split[split].add(claim_id)

    split_report: dict[str, Any] = {}
    for split, claim_ids in sorted(claims_by_split.items()):
        triggered_ids = triggered_by_split.get(split, set())
        useful_count = 0
        flip_to_correct_count = 0
        flip_metrics_available_claims = 0
        wrong_before_triggered = 0
        wrong_before_available_claims = 0
        rounds_total = 0
        rounds_useful = 0
        rounds_negative = 0
        reverted_rounds = 0
        tentative_metrics_available_rounds = 0
        tentative_flips_total = 0
        tentative_flip_to_correct_total = 0
        tentative_flip_to_correct_but_reverted_total = 0

        for claim_id in triggered_ids:
            claim_actions = actions_by_claim.get(claim_id, [])
            useful_for_claim = False
            flip_to_correct_for_claim = False
            has_flip_fields = False

            prediction_row = predictions_by_claim.get(claim_id, {})
            if prediction_row and "label" in prediction_row:
                pred_before_claim_raw = prediction_row.get("pred_before_backtracking")
                if pred_before_claim_raw is None:
                    pred_before_claim_raw = prediction_row.get("pred")
                if pred_before_claim_raw is not None:
                    wrong_before_available_claims += 1
                    pred_before_claim = _safe_int(pred_before_claim_raw)
                    label_claim = _safe_int(prediction_row.get("label"))
                    if pred_before_claim != label_claim:
                        wrong_before_triggered += 1

            for row in claim_actions:
                margin_before = _safe_float(row.get("margin_before"), 0.0)
                margin_after = _safe_float(row.get("margin_after"), 0.0)
                delta = margin_after - margin_before
                rounds_total += 1
                if delta > useful_delta:
                    rounds_useful += 1
                    useful_for_claim = True
                if delta < 0.0:
                    rounds_negative += 1
                if bool(row.get("reverted", row.get("do_no_harm_reverted", False))):
                    reverted_rounds += 1

                if "pred_before" in row and "pred_after" in row and "label" in row:
                    has_flip_fields = True
                    pred_before = _safe_int(row.get("pred_before"))
                    pred_after = _safe_int(row.get("pred_after"))
                    label = _safe_int(row.get("label"))
                    if pred_before != pred_after and pred_after == label and pred_before != label:
                        flip_to_correct_for_claim = True

                if "pred_before" in row and "label" in row:
                    tentative_pred_raw = row.get("tentative_pred")
                    if tentative_pred_raw is None:
                        tentative_pred_raw = row.get("pred_after")
                    if tentative_pred_raw is not None:
                        tentative_metrics_available_rounds += 1
                        pred_before = _safe_int(row.get("pred_before"))
                        tentative_pred = _safe_int(tentative_pred_raw)
                        label = _safe_int(row.get("label"))
                        if pred_before != tentative_pred:
                            tentative_flips_total += 1
                            if tentative_pred == label and pred_before != label:
                                tentative_flip_to_correct_total += 1
                                if bool(row.get("reverted", row.get("do_no_harm_reverted", False))):
                                    tentative_flip_to_correct_but_reverted_total += 1
            if useful_for_claim:
                useful_count += 1
            if has_flip_fields:
                flip_metrics_available_claims += 1
            if flip_to_correct_for_claim:
                flip_to_correct_count += 1

        triggered_total = len(triggered_ids)
        split_total = len(claim_ids)
        split_report[split] = {
            "claims_total": int(split_total),
            "triggered_claims": int(triggered_total),
            "trigger_rate": (float(triggered_total) / float(split_total)) if split_total > 0 else None,
            "wrong_before_triggered": int(wrong_before_triggered),
            "wrong_before_triggered_rate": (
                float(wrong_before_triggered) / float(triggered_total)
            ) if triggered_total > 0 else None,
            "wrong_before_available_claims": int(wrong_before_available_claims),
            "wrong_before_triggered_rate_on_available": (
                float(wrong_before_triggered) / float(wrong_before_available_claims)
            ) if wrong_before_available_claims > 0 else None,
            "useful_delta_threshold": float(useful_delta),
            "useful_recovery_claims": int(useful_count),
            "useful_recovery_rate": (float(useful_count) / float(triggered_total)) if triggered_total > 0 else None,
            "flip_to_correct_claims": int(flip_to_correct_count),
            "flip_to_correct_rate": (
                float(flip_to_correct_count) / float(triggered_total)
            ) if triggered_total > 0 and flip_metrics_available_claims > 0 else None,
            "flip_metrics_available_claims": int(flip_metrics_available_claims),
            "rounds_total": int(rounds_total),
            "rounds_useful": int(rounds_useful),
            "rounds_useful_rate": (float(rounds_useful) / float(rounds_total)) if rounds_total > 0 else None,
            "rounds_negative": int(rounds_negative),
            "rounds_negative_rate": (float(rounds_negative) / float(rounds_total)) if rounds_total > 0 else None,
            "rounds_reverted": int(reverted_rounds),
            "rounds_reverted_rate": (float(reverted_rounds) / float(rounds_total)) if rounds_total > 0 else None,
            "tentative_metrics_available_rounds": int(tentative_metrics_available_rounds),
            "tentative_flips_total": int(tentative_flips_total),
            "tentative_flips_rate": (
                float(tentative_flips_total) / float(tentative_metrics_available_rounds)
            ) if tentative_metrics_available_rounds > 0 else None,
            "tentative_flip_to_correct_total": int(tentative_flip_to_correct_total),
            "tentative_flip_to_correct_rate": (
                float(tentative_flip_to_correct_total) / float(tentative_metrics_available_rounds)
            ) if tentative_metrics_available_rounds > 0 else None,
            "tentative_flip_to_correct_but_reverted_total": int(tentative_flip_to_correct_but_reverted_total),
            "tentative_flip_to_correct_but_reverted_rate": (
                float(tentative_flip_to_correct_but_reverted_total) / float(tentative_flip_to_correct_total)
            ) if tentative_flip_to_correct_total > 0 else None,
        }

    return {
        "run_dir": str(run_dir),
        "predictions_path": str(predictions_path),
        "recovery_actions_path": str(actions_path),
        "splits": split_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute backtracking efficacy metrics.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing predictions.jsonl and recovery_actions.jsonl")
    parser.add_argument("--useful-delta", type=float, default=0.01, help="Margin delta threshold for useful recovery")
    parser.add_argument("--default-split", type=str, default="test", choices=["train", "val", "test"], help="Split label for claim IDs without split prefix")
    parser.add_argument("--out-json", type=str, default="", help="Optional output JSON path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    report = analyze_run(
        run_dir=run_dir,
        useful_delta=float(args.useful_delta),
        default_split=str(args.default_split),
    )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(str(out_path))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
