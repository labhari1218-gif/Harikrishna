"""Diagnostic metrics for Component 4 backtracking."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
        if normalized in {"none", "null", ""}:
            return None
    return bool(value)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _extract_active_rel_count(claim_row: Mapping[str, Any], rel_threshold: float) -> int:
    if "active_rel_count" in claim_row:
        return int(claim_row["active_rel_count"])

    triples = claim_row.get("active_triples")
    if not isinstance(triples, Sequence) or isinstance(triples, (str, bytes)):
        triples = claim_row.get("triples", [])
    if not isinstance(triples, Sequence) or isinstance(triples, (str, bytes)):
        return 0

    count = 0
    for triple in triples:
        row = _as_mapping(triple)
        pool = str(row.get("pool", "A")).upper()
        if pool != "A":
            continue
        rel = row.get("rel")
        if rel is None:
            rel = float(row.get("p_ent", 0.0)) + float(row.get("p_con", 0.0))
        if float(rel) > rel_threshold:
            count += 1
    return count


def compute_starvation_rate(
    claim_rows: Sequence[Mapping[str, Any]],
    *,
    min_a: int = 5,
    rel_threshold: float = 0.3,
) -> dict[str, Any]:
    total = 0
    starved = 0
    for claim_row in claim_rows:
        total += 1
        high_rel_count = _extract_active_rel_count(claim_row, rel_threshold=rel_threshold)
        if high_rel_count < int(min_a):
            starved += 1
    return {
        "rate": _safe_rate(starved, total),
        "starved_claims": int(starved),
        "total_claims": int(total),
        "min_a": int(min_a),
        "rel_threshold": float(rel_threshold),
    }


def _extract_recovered_gold_counts(action_row: Mapping[str, Any]) -> tuple[int, int]:
    if "recovered_gold_count" in action_row and "recovered_total" in action_row:
        return int(action_row["recovered_gold_count"]), int(action_row["recovered_total"])

    recovered_ids = action_row.get("recovered_evidence_ids")
    gold_ids = action_row.get("gold_evidence_ids")
    if (
        isinstance(recovered_ids, Sequence)
        and not isinstance(recovered_ids, (str, bytes))
        and isinstance(gold_ids, Sequence)
        and not isinstance(gold_ids, (str, bytes))
    ):
        recovered_set = {str(value) for value in recovered_ids}
        gold_set = {str(value) for value in gold_ids}
        return len(recovered_set & gold_set), len(recovered_set)

    recovered_rows = action_row.get("recovered_evidence")
    if isinstance(recovered_rows, Sequence) and not isinstance(recovered_rows, (str, bytes)):
        total = 0
        gold = 0
        for recovered in recovered_rows:
            row = _as_mapping(recovered)
            selected = _coerce_optional_bool(row.get("selected", True))
            if not selected:
                continue
            total += 1
            if _coerce_optional_bool(row.get("is_gold", False)):
                gold += 1
        return gold, total

    return 0, 0


def compute_recovered_gold_rate(recovery_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recovered_gold = 0
    recovered_total = 0
    for row in recovery_rows:
        gold, total = _extract_recovered_gold_counts(row)
        recovered_gold += int(gold)
        recovered_total += int(total)
    return {
        "rate": _safe_rate(recovered_gold, recovered_total),
        "recovered_gold": int(recovered_gold),
        "recovered_total": int(recovered_total),
    }


def _extract_correctness(row: Mapping[str, Any], prefix: str) -> bool | None:
    key = f"{prefix}_correct"
    if key in row:
        return _coerce_optional_bool(row[key])

    label = row.get("label")
    if label is None:
        return None

    pred_key_candidates = [
        f"{prefix}_pred",
        f"{prefix}_prediction",
    ]
    for pred_key in pred_key_candidates:
        if pred_key in row:
            return str(row[pred_key]) == str(label)

    if prefix == "before":
        old_pred = row.get("old_pred")
        if old_pred is not None:
            return str(old_pred) == str(label)
    if prefix == "after":
        new_pred = row.get("new_pred")
        if new_pred is not None:
            return str(new_pred) == str(label)
    return None


def compute_recovery_impact(recovery_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = 0
    before_correct = 0
    after_correct = 0
    triggered = 0

    for row in recovery_rows:
        if _coerce_optional_bool(row.get("triggered", False)):
            triggered += 1
        before = _extract_correctness(row, prefix="before")
        after = _extract_correctness(row, prefix="after")
        if before is None or after is None:
            continue
        total += 1
        before_correct += int(before)
        after_correct += int(after)

    accuracy_before = _safe_rate(before_correct, total)
    accuracy_after = _safe_rate(after_correct, total)
    delta = None
    if accuracy_before is not None and accuracy_after is not None:
        delta = float(accuracy_after - accuracy_before)

    return {
        "delta_accuracy": delta,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "evaluated_claims": int(total),
        "triggered_claims": int(triggered),
        "trigger_rate": _safe_rate(triggered, len(recovery_rows)),
    }


def compute_component4_diagnostics(
    *,
    claim_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    min_a: int = 5,
    rel_threshold: float = 0.3,
) -> dict[str, Any]:
    starvation = compute_starvation_rate(
        claim_rows,
        min_a=min_a,
        rel_threshold=rel_threshold,
    )
    recovered_gold = compute_recovered_gold_rate(recovery_rows)
    impact = compute_recovery_impact(recovery_rows)
    return {
        "component": "component4",
        "generated_at_utc": _now_iso(),
        "metrics": {
            "starvation_rate": starvation,
            "recovered_gold_rate": recovered_gold,
            "recovery_impact": impact,
        },
    }


def write_diagnostics_json(report: Mapping[str, Any], *, run_id: str, output_root: str = "runs") -> Path:
    output_dir = Path(output_root) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "diagnostics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def run_component4_diagnostics(
    *,
    run_id: str,
    claim_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    output_root: str = "runs",
    min_a: int = 5,
    rel_threshold: float = 0.3,
) -> tuple[Path, dict[str, Any]]:
    report = compute_component4_diagnostics(
        claim_rows=claim_rows,
        recovery_rows=recovery_rows,
        min_a=min_a,
        rel_threshold=rel_threshold,
    )
    path = write_diagnostics_json(report, run_id=run_id, output_root=output_root)
    return path, report
