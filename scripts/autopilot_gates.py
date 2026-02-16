#!/usr/bin/env python3
"""Gate and preflight checks for Component 3 autopilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bool_ok(value: bool, ok_msg: str, fail_msg: str, reasons: list[str]) -> None:
    reasons.append(ok_msg if value else fail_msg)


def _compute_safety_gate(
    *,
    strict_clean: bool,
    sentinel_exception_confirmed: bool,
    embeddings_ok: bool,
    cache_ok: bool,
) -> bool:
    return bool(embeddings_ok and cache_ok and (strict_clean or sentinel_exception_confirmed))


def _required_artifacts(run_dir: Path, *, bt_enabled: bool) -> tuple[list[str], list[str]]:
    required = [
        "config.yaml",
        "metrics.json",
        "predictions.jsonl",
        "run_console.log",
    ]
    if bt_enabled:
        required.extend(
            [
                "recovery_actions.jsonl",
                "recovery_candidates.jsonl",
            ]
        )
    missing: list[str] = []
    for rel in required:
        path = run_dir / rel
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(rel)

    if bt_enabled:
        bt_effectiveness = sorted(run_dir.glob("backtracking_effectiveness_*.json"))
        if not bt_effectiveness:
            missing.append("backtracking_effectiveness_*.json")
    return required, missing


def _extract_overall_accuracy(metrics: dict[str, Any]) -> float:
    test_metrics = metrics.get("test_metrics", {}) if isinstance(metrics, dict) else {}
    overall = test_metrics.get("overall", {}) if isinstance(test_metrics, dict) else {}
    return float(overall.get("accuracy", 0.0))


def _extract_train_acc(stage_metrics: dict[str, Any]) -> float:
    stages = stage_metrics.get("stages", []) if isinstance(stage_metrics, dict) else []
    best = 0.0
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        if "train_accuracy_best" in stage:
            best = max(best, float(stage.get("train_accuracy_best", 0.0)))
            continue
        if "train_accuracy_best_pct" in stage:
            best = max(best, float(stage.get("train_accuracy_best_pct", 0.0)) / 100.0)
    return best


def _sentinel_claim_ids(pairs_path: Path) -> set[str]:
    sentinels: set[str] = set()
    if not pairs_path.exists():
        return sentinels
    with pairs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            is_sentinel = bool(row.get("empty_evidence_sentinel")) or (
                str(row.get("record_type") or "").strip().lower() == "claim_sentinel"
            )
            if not is_sentinel:
                continue
            claim_id = str(row.get("claim_id") or "").strip()
            if claim_id:
                sentinels.add(claim_id)
    return sentinels


def _triple_count_from_subgraph_row(row: Any) -> int:
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    if not isinstance(row, dict):
        return 0
    walked = row.get("walked", {}) if isinstance(row.get("walked", {}), dict) else {}
    connected = walked.get("connected", []) if isinstance(walked, dict) else []
    walkable = walked.get("walkable", []) if isinstance(walked, dict) else []
    seen: set[tuple[str, str, str]] = set()
    for triple in list(connected) + list(walkable):
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        s, r, o = str(triple[0]).strip(), str(triple[1]).strip(), str(triple[2]).strip()
        if not s or not r or not o:
            continue
        seen.add((s, r, o))
    return int(len(seen))


def _claim_id_to_index(claim_id: str) -> int | None:
    try:
        return int(str(claim_id).rsplit("_", 1)[1])
    except Exception:
        return None


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    from component3.c1_pairs_loader import load_component1_evidence_rows
    from datasets import get_df, get_subgraphs

    splits = ("train", "val", "test")
    logs_root = Path(args.logs_root)
    report: dict[str, Any] = {
        "mode": "preflight",
        "logs_root": str(logs_root),
        "embeddings_path": str(args.embeddings_path),
        "claim_triple_cache_path": str(args.claim_triple_cache_path),
        "splits": {},
        "strict_clean": True,
        "sentinel_only_zero_triple_exception_confirmed": True,
        "safety_gate_pass": True,
        "pass": True,
        "reasons": [],
    }

    global_reasons: list[str] = []

    # Embedding/cache checks.
    embeddings_path = Path(args.embeddings_path)
    cache_path = Path(args.claim_triple_cache_path)
    embeddings_ok = embeddings_path.exists() and embeddings_path.stat().st_size > 0
    cache_ok = cache_path.exists() and cache_path.stat().st_size > 0
    if embeddings_ok:
        with embeddings_path.open("rb") as fp:
            loaded = pickle.load(fp)
        embeddings_ok = isinstance(loaded, dict) and len(loaded) > 0
    if cache_ok:
        with cache_path.open("rb") as fp:
            loaded = pickle.load(fp)
        cache_ok = isinstance(loaded, dict) and len(loaded) > 0

    _bool_ok(
        embeddings_ok,
        "embeddings_present_nonempty",
        "embeddings_missing_or_empty",
        global_reasons,
    )
    _bool_ok(
        cache_ok,
        "claim_triple_cache_present_nonempty",
        "claim_triple_cache_missing_or_empty",
        global_reasons,
    )

    strict_clean = True
    sentinel_exception_ok = True

    for split in splits:
        split_report: dict[str, Any] = {"split": split, "checks": {}, "reasons": []}
        coverage_path = logs_root / split / "coverage_report.json"
        split_report["coverage_report_path"] = str(coverage_path)
        if not coverage_path.exists():
            split_report["checks"]["coverage_report_exists"] = False
            split_report["reasons"].append("coverage_report_missing")
            strict_clean = False
            sentinel_exception_ok = False
            report["splits"][split] = split_report
            continue

        coverage = _read_json(coverage_path)
        split_report["coverage_report"] = coverage

        # Coverage-level strict checks.
        claims = coverage.get("claims", {}) if isinstance(coverage, dict) else {}
        rows = coverage.get("rows", {}) if isinstance(coverage, dict) else {}

        checks = {
            "coverage_report_exists": True,
            "without_pair_rows_with_nonzero_subgraph_triples_zero": int(
                claims.get("without_pair_rows_with_nonzero_subgraph_triples", 0)
            )
            == 0,
            "missing_pv_fields_zero": int(claims.get("missing_pv_fields", 0)) == 0,
            "missing_pool_labels_zero": int(claims.get("missing_pool_labels", 0)) == 0,
            "malformed_probs_zero": int(claims.get("malformed_probs", 0)) == 0,
            "missing_rel_zero": int(claims.get("missing_rel", 0)) == 0,
            "missing_entity_embeddings_zero": int(claims.get("missing_entity_embeddings", 0)) == 0,
            "missing_schema_version_zero": int(claims.get("missing_schema_version", 0)) == 0
            and int(rows.get("missing_schema_version", 0)) == 0,
        }

        # Loader-level strict checks (fallback/schema counters from C3 ingestion path).
        claims_df = get_df(split).reset_index(drop=True)
        subgraphs_df = get_subgraphs(split, args.subgraph_type).reset_index(drop=True)
        _rows, loader_coverage = load_component1_evidence_rows(
            split=split,
            claims_df=claims_df,
            subgraphs_df=subgraphs_df,
            logs_root=logs_root,
            missing_policy="hybrid_fallback",
        )
        split_report["loader_coverage"] = loader_coverage
        checks.update(
            {
                "claims_using_fallback_zero": int(loader_coverage.get("claims_using_fallback", 0)) == 0,
                "fallback_edge_rows_zero": int(loader_coverage.get("fallback_edge_rows", 0)) == 0,
                "pair_rows_non_v2_schema_zero": int(loader_coverage.get("pair_rows_non_v2_schema", 0)) == 0,
                "pair_rows_missing_schema_version_zero": int(
                    loader_coverage.get("pair_rows_missing_schema_version", 0)
                )
                == 0,
            }
        )

        # Sentinel allowance check against zero-triple subgraphs.
        pairs_path = logs_root / split / "pairs.jsonl"
        sentinel_ids = sorted(_sentinel_claim_ids(pairs_path))
        nonzero_sentinels: list[dict[str, Any]] = []
        for claim_id in sentinel_ids:
            idx = _claim_id_to_index(claim_id)
            if idx is None or idx < 0 or idx >= len(subgraphs_df):
                nonzero_sentinels.append(
                    {
                        "claim_id": claim_id,
                        "reason": "invalid_claim_index",
                    }
                )
                continue
            triple_count = _triple_count_from_subgraph_row(subgraphs_df.iloc[idx])
            if triple_count > 0:
                nonzero_sentinels.append(
                    {
                        "claim_id": claim_id,
                        "triple_count": triple_count,
                    }
                )

        checks["sentinel_only_zero_triple"] = len(nonzero_sentinels) == 0
        split_report["sentinel_claims_total"] = len(sentinel_ids)
        split_report["sentinel_nonzero_subgraph_claims"] = nonzero_sentinels[:50]

        split_report["checks"] = checks
        for key, ok in checks.items():
            if not ok:
                split_report["reasons"].append(f"{key}_failed")

        # strict_clean tracks strict metadata/runtime cleanliness excluding the
        # sentinel exception pathway, which is evaluated separately.
        non_sentinel_checks = {
            key: value for key, value in checks.items() if key != "sentinel_only_zero_triple"
        }
        if not all(non_sentinel_checks.values()):
            strict_clean = False
        if not checks["sentinel_only_zero_triple"]:
            sentinel_exception_ok = False

        report["splits"][split] = split_report

    safety_pass = _compute_safety_gate(
        strict_clean=bool(strict_clean),
        sentinel_exception_confirmed=bool(sentinel_exception_ok),
        embeddings_ok=bool(embeddings_ok),
        cache_ok=bool(cache_ok),
    )
    report["strict_clean"] = bool(strict_clean)
    report["sentinel_only_zero_triple_exception_confirmed"] = bool(sentinel_exception_ok)
    report["safety_gate_pass"] = bool(safety_pass)
    report["pass"] = bool(safety_pass)
    report["reasons"] = global_reasons

    if args.out_json:
        _write_json(Path(args.out_json), report)
    return report


def _load_run_metrics(run_id: str) -> tuple[Path, dict[str, Any]]:
    run_dir = Path("runs") / str(run_id)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for run_id={run_id}: {metrics_path}")
    metrics = _read_json(metrics_path)
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected metrics dict at {metrics_path}")
    return run_dir, metrics


def _compare_predictions(run_a_dir: Path, run_b_dir: Path) -> dict[str, int]:
    pred_a_path = run_a_dir / "predictions.jsonl"
    pred_b_path = run_b_dir / "predictions.jsonl"
    if not pred_a_path.exists() or not pred_b_path.exists():
        return {"compared": 0, "improved": 0, "worsened": 0}

    def _load(path: Path) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                row = json.loads(payload)
                if not isinstance(row, dict):
                    continue
                cid = str(row.get("claim_id") or "").strip()
                if not cid:
                    continue
                try:
                    label = int(row.get("label"))
                    pred = int(row.get("pred"))
                except (TypeError, ValueError):
                    continue
                out[cid] = (label, pred)
        return out

    a = _load(pred_a_path)
    b = _load(pred_b_path)
    improved = 0
    worsened = 0
    compared = 0
    for cid, (label_a, pred_a) in a.items():
        if cid not in b:
            continue
        label_b, pred_b = b[cid]
        if label_a != label_b:
            continue
        compared += 1
        a_ok = pred_a == label_a
        b_ok = pred_b == label_b
        if (not a_ok) and b_ok:
            improved += 1
        elif a_ok and (not b_ok):
            worsened += 1
    return {"compared": compared, "improved": improved, "worsened": worsened}


def _load_bt_effectiveness(run_b_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(run_b_dir.glob("backtracking_effectiveness_*.json"))
    if not candidates:
        return None
    payload = _read_json(candidates[0])
    if not isinstance(payload, dict):
        return None
    return payload


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    stage = str(args.stage)
    report: dict[str, Any] = {
        "mode": "gate",
        "stage": stage,
        "pass": False,
        "reasons": [],
    }
    reasons: list[str] = []

    if stage in {"stage1_overfit200", "stage2_claim_only", "stage3_pvqagnn_no_bt"}:
        if not args.run_id:
            raise ValueError(f"--run-id is required for stage={stage}")
        run_dir, metrics = _load_run_metrics(args.run_id)
        report["run_id"] = str(args.run_id)
        report["run_dir"] = str(run_dir)

        bt_enabled = bool(metrics.get("runtime_flags", {}).get("backtracking_enabled", False))
        required, missing = _required_artifacts(run_dir, bt_enabled=bt_enabled)
        report["artifact_required"] = required
        report["artifact_missing"] = missing
        if missing:
            reasons.append(f"missing_artifacts:{','.join(missing)}")

        if stage == "stage1_overfit200":
            train_acc = _extract_train_acc(metrics)
            report["gate"] = "G1 Overfit200"
            report["threshold"] = 0.95
            report["train_acc"] = train_acc
            pass_gate = train_acc >= 0.95
            if not pass_gate:
                reasons.append(f"G1_FAIL_train_acc={train_acc:.6f}<0.95")
        elif stage == "stage2_claim_only":
            test_acc = _extract_overall_accuracy(metrics)
            report["gate"] = "G2 ClaimOnlyMedium"
            report["threshold"] = 0.60
            report["test_acc"] = test_acc
            mode = str(metrics.get("runtime_flags", {}).get("model_mode", ""))
            report["model_mode"] = mode
            pass_gate = (test_acc >= 0.60) and (mode == "claim_only")
            if test_acc < 0.60:
                reasons.append(f"G2_FAIL_test_acc={test_acc:.6f}<0.60")
            if mode != "claim_only":
                reasons.append(f"G2_FAIL_model_mode={mode}")
        else:
            test_acc = _extract_overall_accuracy(metrics)
            report["gate"] = "G3 PVQAGNN_NoBT_Medium"
            report["threshold"] = 0.65
            report["test_acc"] = test_acc
            mode = str(metrics.get("runtime_flags", {}).get("model_mode", ""))
            bt = bool(metrics.get("runtime_flags", {}).get("backtracking_enabled", False))
            report["model_mode"] = mode
            report["backtracking_enabled"] = bt
            pass_gate = (test_acc >= 0.65) and (mode == "pv_qagnn") and (not bt)
            if test_acc < 0.65:
                reasons.append(f"G3_FAIL_test_acc={test_acc:.6f}<0.65")
            if mode != "pv_qagnn":
                reasons.append(f"G3_FAIL_model_mode={mode}")
            if bt:
                reasons.append("G3_FAIL_backtracking_enabled")

        report["pass"] = bool((not missing) and pass_gate)

    elif stage == "stage4_bt_helping":
        if not args.run_a_id or not args.run_b_id:
            raise ValueError("--run-a-id and --run-b-id are required for stage4_bt_helping")
        run_a_dir, metrics_a = _load_run_metrics(args.run_a_id)
        run_b_dir, metrics_b = _load_run_metrics(args.run_b_id)
        report.update(
            {
                "run_a_id": str(args.run_a_id),
                "run_b_id": str(args.run_b_id),
                "run_a_dir": str(run_a_dir),
                "run_b_dir": str(run_b_dir),
                "gate": "G4 BT_Helping_Medium",
            }
        )

        req_a, miss_a = _required_artifacts(run_a_dir, bt_enabled=False)
        req_b, miss_b = _required_artifacts(run_b_dir, bt_enabled=True)
        report["artifact_required_run_a"] = req_a
        report["artifact_required_run_b"] = req_b
        report["artifact_missing_run_a"] = miss_a
        report["artifact_missing_run_b"] = miss_b
        if miss_a:
            reasons.append(f"run_a_missing_artifacts:{','.join(miss_a)}")
        if miss_b:
            reasons.append(f"run_b_missing_artifacts:{','.join(miss_b)}")

        acc_a = _extract_overall_accuracy(metrics_a)
        acc_b = _extract_overall_accuracy(metrics_b)
        delta_acc = acc_b - acc_a
        report["acc_a"] = acc_a
        report["acc_b"] = acc_b
        report["delta_acc"] = delta_acc

        pred_delta = _compare_predictions(run_a_dir, run_b_dir)
        report["prediction_delta"] = pred_delta

        bt_eff = _load_bt_effectiveness(run_b_dir)
        report["backtracking_effectiveness"] = bt_eff
        flip_to_correct = 0
        if isinstance(bt_eff, dict):
            split_payload = bt_eff.get("splits", {}) if isinstance(bt_eff.get("splits"), dict) else {}
            test_payload = split_payload.get("test", {}) if isinstance(split_payload.get("test"), dict) else {}
            flip_to_correct = int(test_payload.get("flip_to_correct_claims", 0))
        report["flip_to_correct_claims"] = int(flip_to_correct)

        improved = int(pred_delta.get("improved", 0))
        worsened = int(pred_delta.get("worsened", 0))
        cond_delta = delta_acc >= 0.001
        cond_flip = (flip_to_correct >= 10) and (worsened <= improved)
        report["condition_delta_acc_ge_0p001"] = cond_delta
        report["condition_flip_to_correct_and_net_not_worse"] = cond_flip

        if not cond_delta and not cond_flip:
            reasons.append(
                "G4_FAIL:(delta_acc<0.001) AND (flip_to_correct<10 OR worsened>improved)"
            )
        report["pass"] = bool((not miss_a) and (not miss_b) and (cond_delta or cond_flip))

    else:
        raise ValueError(f"Unsupported stage: {stage}")

    report["reasons"] = reasons
    if args.out_json:
        _write_json(Path(args.out_json), report)
    return report


def run_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path("runs") / str(args.run_id)
    required, missing = _required_artifacts(run_dir, bt_enabled=bool(args.bt_enabled))
    report = {
        "mode": "artifacts",
        "run_id": str(args.run_id),
        "run_dir": str(run_dir),
        "required": required,
        "missing": missing,
        "pass": len(missing) == 0,
    }
    if args.out_json:
        _write_json(Path(args.out_json), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Component 3 autopilot gate checker")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight", help="Run Stage 0 preflight checks")
    p_pre.add_argument("--logs-root", type=str, default="logs/component1")
    p_pre.add_argument("--subgraph-type", type=str, default="direct_filled")
    p_pre.add_argument("--embeddings-path", type=str, default="data/embeddings.pkl")
    p_pre.add_argument("--claim-triple-cache-path", type=str, default="data/claim_triple_embeddings.pkl")
    p_pre.add_argument("--out-json", type=str, default="")

    p_gate = sub.add_parser("gate", help="Run one gate check")
    p_gate.add_argument("--stage", type=str, required=True)
    p_gate.add_argument("--run-id", type=str, default="")
    p_gate.add_argument("--run-a-id", type=str, default="")
    p_gate.add_argument("--run-b-id", type=str, default="")
    p_gate.add_argument("--out-json", type=str, default="")

    p_art = sub.add_parser("artifacts", help="Artifact-only validation")
    p_art.add_argument("--run-id", type=str, required=True)
    p_art.add_argument("--bt-enabled", action="store_true")
    p_art.add_argument("--out-json", type=str, default="")

    args = parser.parse_args()

    if args.cmd == "preflight":
        report = run_preflight(args)
    elif args.cmd == "gate":
        report = run_gate(args)
    elif args.cmd == "artifacts":
        report = run_artifacts(args)
    else:
        raise ValueError(f"Unsupported cmd={args.cmd}")

    print(json.dumps(report, indent=2))
    raise SystemExit(0 if bool(report.get("pass", False)) else 1)


if __name__ == "__main__":
    main()
