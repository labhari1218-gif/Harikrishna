#!/usr/bin/env python3
"""Compare Component 3 run metrics against published/internal SOTA targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PUBLISHED_SOTA_OVERALL_ACC_PCT = 86.82
INTERNAL_BERT_BASELINE_ACC_PCT = 93.49


def _to_pct(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value <= 1.0:
        return value * 100.0
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get_overall_acc_pct(metrics_payload: dict[str, Any]) -> float | None:
    overall = (
        metrics_payload.get("test_metrics", {}).get("overall", {})
        if isinstance(metrics_payload, dict)
        else {}
    )
    return _to_pct(overall.get("accuracy"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Component 3 metrics to SOTA/baselines.")
    parser.add_argument("--run-metrics", type=str, required=True)
    parser.add_argument("--baseline-metrics", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    run_path = Path(args.run_metrics)
    run_payload = _load_json(run_path)
    run_acc_pct = _safe_get_overall_acc_pct(run_payload)

    baseline_path = Path(args.baseline_metrics) if args.baseline_metrics else None
    baseline_acc_pct = None
    if baseline_path is not None and baseline_path.exists():
        baseline_payload = _load_json(baseline_path)
        baseline_acc_pct = _safe_get_overall_acc_pct(baseline_payload)

    report = {
        "run_metrics": str(run_path),
        "run_overall_accuracy_pct": run_acc_pct,
        "published_sota_overall_accuracy_pct": PUBLISHED_SOTA_OVERALL_ACC_PCT,
        "internal_bert_baseline_accuracy_pct": INTERNAL_BERT_BASELINE_ACC_PCT,
        "delta_vs_published_sota_pct_points": (
            None if run_acc_pct is None else (run_acc_pct - PUBLISHED_SOTA_OVERALL_ACC_PCT)
        ),
        "delta_vs_internal_bert_baseline_pct_points": (
            None if run_acc_pct is None else (run_acc_pct - INTERNAL_BERT_BASELINE_ACC_PCT)
        ),
        "beats_published_sota": (
            None if run_acc_pct is None else (run_acc_pct > PUBLISHED_SOTA_OVERALL_ACC_PCT)
        ),
        "beats_internal_bert_baseline": (
            None if run_acc_pct is None else (run_acc_pct > INTERNAL_BERT_BASELINE_ACC_PCT)
        ),
        "baseline_metrics": str(baseline_path) if baseline_path is not None else None,
        "baseline_overall_accuracy_pct": baseline_acc_pct,
        "delta_vs_baseline_run_pct_points": (
            None
            if (run_acc_pct is None or baseline_acc_pct is None)
            else (run_acc_pct - baseline_acc_pct)
        ),
        "no_collapse_gate": run_payload.get("no_collapse_gate"),
    }

    text = json.dumps(report, indent=2)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote SOTA comparison report: {out_path}")
    print(text)


if __name__ == "__main__":
    main()

