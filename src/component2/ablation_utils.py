"""Shared utilities for Component 2 ablation runs."""

from __future__ import annotations

import csv
import datetime as dt
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .bridge_rescue import BridgeScore
from .esi import estimate_esi_geom_ac
from .run_m3 import (
    _is_component1_claim_metrics_row,
    _load_claims_from_component1_logs,
    _peek_first_json_row,
)
from .types import ClaimRecord, EvidenceTriple, claim_label_to_index


@dataclass(frozen=True)
class SubsetSelection:
    """Selected subset details for deterministic replay."""

    claims: Tuple[ClaimRecord, ...]
    indices: Tuple[int, ...]


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary classification metrics over labeled rows only."""

    num_labeled: int
    accuracy: float
    macro_f1: float


def make_ablation_run_dir(output_root: Path, run_id: str | None = None) -> Path:
    """Create a unique ablation run directory."""

    if run_id is None:
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_id}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def git_commit_hash(repo_root: Path) -> str | None:
    """Best-effort git commit hash discovery."""

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    commit = out.strip()
    return commit if commit else None


def load_claims_for_ablation(claims_jsonl: Path) -> Tuple[List[ClaimRecord], str]:
    """Load claims from either Component 1 logs+pairs or normalized triples JSONL."""

    if not claims_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {claims_jsonl}")

    first_row = _peek_first_json_row(claims_jsonl)
    if isinstance(first_row, dict) and _is_component1_claim_metrics_row(first_row):
        # Load all rows from Component 1 logs with pair reconstruction.
        return _load_claims_from_component1_logs(claims_jsonl, max_claims=10**9), "component1_logs"

    from .io_utils import load_claims_jsonl  # Local import to avoid circulars.

    return load_claims_jsonl(claims_jsonl, max_claims=None), "triples_jsonl"


def select_deterministic_subset(
    claims: Sequence[ClaimRecord],
    n_claims: int,
    seed: int,
) -> SubsetSelection:
    """Shuffle indices with seed and take first N."""

    if n_claims < 1:
        raise ValueError(f"n_claims must be >= 1, got {n_claims}")
    if len(claims) == 0:
        raise ValueError("No claims are available for subset selection.")

    indices = list(range(len(claims)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    selected_indices = indices[: min(int(n_claims), len(indices))]
    selected_claims = tuple(claims[i] for i in selected_indices)
    return SubsetSelection(claims=selected_claims, indices=tuple(selected_indices))


def write_selected_claim_ids(
    out_path: Path,
    selected_claims: Sequence[ClaimRecord],
) -> None:
    """Write selected claim IDs, one per line."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for claim in selected_claims:
            handle.write(f"{claim.claim_id}\n")


def pool_counts(triples: Sequence[EvidenceTriple]) -> Dict[str, int]:
    """Return |A|, |S|, |C| counts."""

    counts = {"A": 0, "S": 0, "C": 0}
    for triple in triples:
        if triple.pool in counts:
            counts[triple.pool] += 1
    return counts


def estimate_esi_geom(
    triples: Sequence[EvidenceTriple],
    anchors: Sequence[str],
    eps: float = 0.05,
) -> float:
    """Estimate ESI_geom on combined A+C pools using Component 1-compatible formula."""

    return float(estimate_esi_geom_ac(triples=triples, anchors=anchors, eps=eps))


def active_component_map(triples: Sequence[EvidenceTriple]) -> Dict[str, int]:
    """Connected component ID per node in Active graph (undirected)."""

    adjacency: Dict[str, set[str]] = {}
    for triple in triples:
        if triple.pool != "A":
            continue
        subj, _, obj = triple.raw_triple
        adjacency.setdefault(subj, set()).add(obj)
        adjacency.setdefault(obj, set()).add(subj)

    mapping: Dict[str, int] = {}
    comp_id = 0
    for node in adjacency:
        if node in mapping:
            continue
        queue = [node]
        mapping[node] = comp_id
        while queue:
            cur = queue.pop()
            for nxt in adjacency.get(cur, ()):  # pragma: no branch
                if nxt not in mapping:
                    mapping[nxt] = comp_id
                    queue.append(nxt)
        comp_id += 1
    return mapping


def bridge_edge_connects_different_components(edge: BridgeScore, component_map: Mapping[str, int]) -> bool:
    """Whether bridge edge endpoints are in different pre-recovery Active components."""

    src = str(edge.source)
    dst = str(edge.target)
    src_comp = component_map.get(src, f"missing:{src}")
    dst_comp = component_map.get(dst, f"missing:{dst}")
    return src_comp != dst_comp


def summarize_bridge_top_values(scores_s: Sequence[BridgeScore]) -> Tuple[float | None, List[float]]:
    """Return top1 and top5 BridgeBonus' values for S edges."""

    ranked = sorted(scores_s, key=lambda row: row.bridge_bonus_capped, reverse=True)
    top5 = [float(row.bridge_bonus_capped) for row in ranked[:5]]
    top1 = top5[0] if top5 else None
    return top1, top5


def baseline_predict_from_component1(triples: Sequence[EvidenceTriple]) -> Tuple[str, float, float]:
    """Pre-Component2 baseline: pooled A/C mass voting with deterministic confidence."""

    candidates = [triple for triple in triples if triple.pool in {"A", "C"}]
    if not candidates:
        candidates = list(triples)
    if not candidates:
        return "SUPPORTED", 0.5, 0.5

    support_mass = 0.0
    refute_mass = 0.0
    for triple in candidates:
        if triple.pool == "A":
            support_mass += float(triple.p_ent)
            refute_mass += 0.5 * float(triple.p_con)
        elif triple.pool == "C":
            support_mass += 0.5 * float(triple.p_ent)
            refute_mass += float(triple.p_con)
        else:
            support_mass += 0.5 * float(triple.p_ent)
            refute_mass += 0.5 * float(triple.p_con)

    total = support_mass + refute_mass
    if total <= 0.0:
        prob_supported = 0.5
    else:
        prob_supported = float(support_mass / total)
    pred_label = "SUPPORTED" if prob_supported >= 0.5 else "REFUTED"
    confidence = float(max(prob_supported, 1.0 - prob_supported))
    return pred_label, prob_supported, confidence


def _f1_for_class(y_true: Sequence[int], y_pred: Sequence[int], cls: int) -> float:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def compute_binary_metrics(
    gold_labels: Sequence[str | None],
    pred_labels: Sequence[str],
) -> BinaryMetrics:
    """Compute accuracy and macro-F1 over labeled rows."""

    gold_idx: List[int] = []
    pred_idx: List[int] = []
    for gold, pred in zip(gold_labels, pred_labels):
        gold_v = claim_label_to_index(gold)
        pred_v = claim_label_to_index(pred)
        if gold_v is None or pred_v is None:
            continue
        gold_idx.append(gold_v)
        pred_idx.append(pred_v)

    if not gold_idx:
        return BinaryMetrics(num_labeled=0, accuracy=0.0, macro_f1=0.0)

    correct = sum(1 for g, p in zip(gold_idx, pred_idx) if g == p)
    accuracy = float(correct) / float(len(gold_idx))
    macro_f1 = (_f1_for_class(gold_idx, pred_idx, 0) + _f1_for_class(gold_idx, pred_idx, 1)) / 2.0
    return BinaryMetrics(num_labeled=len(gold_idx), accuracy=accuracy, macro_f1=float(macro_f1))


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)

    i = 0
    n = len(arr)
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation, returning None for degenerate inputs."""

    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if len(x) < 2:
        return None

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return None

    rx = _rankdata(x_arr.tolist())
    ry = _rankdata(y_arr.tolist())

    x_std = float(np.std(rx))
    y_std = float(np.std(ry))
    if x_std == 0.0 or y_std == 0.0:
        return None

    corr = float(np.corrcoef(rx, ry)[0, 1])
    if not np.isfinite(corr):
        return None
    return corr


class PlotWriter:
    """Plot helper with matplotlib fallback to CSV + script."""

    def __init__(self, plots_dir: Path):
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self._plt = None
        self._has_matplotlib = False
        try:
            import matplotlib.pyplot as plt  # type: ignore

            self._plt = plt
            self._has_matplotlib = True
        except Exception:
            self._write_fallback_script()

    @property
    def has_matplotlib(self) -> bool:
        return self._has_matplotlib

    def _write_fallback_script(self) -> None:
        script_path = self.plots_dir / "plot_from_csv.py"
        if script_path.exists():
            return
        script_path.write_text(
            """#!/usr/bin/env python3\n"
            "import argparse\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--csv', type=Path, required=True)\n"
            "parser.add_argument('--x', type=str, required=True)\n"
            "parser.add_argument('--y', type=str, required=True)\n"
            "parser.add_argument('--out', type=Path, required=True)\n"
            "args = parser.parse_args()\n"
            "df = pd.read_csv(args.csv)\n"
            "plt.figure(figsize=(6, 4))\n"
            "plt.plot(df[args.x], df[args.y], marker='o', linewidth=1)\n"
            "plt.xlabel(args.x)\n"
            "plt.ylabel(args.y)\n"
            "plt.tight_layout()\n"
            "args.out.parent.mkdir(parents=True, exist_ok=True)\n"
            "plt.savefig(args.out, dpi=150)\n"
            """,
            encoding="utf-8",
        )

    def _write_csv(self, path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(header))
            for row in rows:
                writer.writerow(list(row))

    def risk_coverage(self, ablation_id: str, points: Sequence[Mapping[str, object]]) -> Path:
        png_path = self.plots_dir / f"risk_coverage_{ablation_id}.png"
        csv_path = self.plots_dir / f"risk_coverage_{ablation_id}.csv"

        rows = [
            (
                float(row["coverage"]),
                float(row["risk"]),
                1.0 - float(row["risk"]),
                float(row["abstain_threshold"]),
            )
            for row in points
        ]
        self._write_csv(csv_path, ["coverage", "risk", "accuracy", "abstain_threshold"], rows)

        if not self._has_matplotlib or self._plt is None:
            return csv_path

        plt = self._plt
        plt.figure(figsize=(6, 4))
        x = [row[0] for row in rows]
        y = [row[1] for row in rows]
        plt.plot(x, y, marker="o", markersize=2, linewidth=1)
        plt.xlabel("Coverage")
        plt.ylabel("Risk")
        plt.title(f"Risk-Coverage ({ablation_id})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()
        return png_path

    def accuracy_vs_coverage(self, ablation_id: str, points: Sequence[Mapping[str, object]]) -> Path:
        png_path = self.plots_dir / f"accuracy_vs_coverage_{ablation_id}.png"
        csv_path = self.plots_dir / f"accuracy_vs_coverage_{ablation_id}.csv"

        rows = [(float(row["coverage"]), 1.0 - float(row["risk"])) for row in points]
        self._write_csv(csv_path, ["coverage", "accuracy"], rows)

        if not self._has_matplotlib or self._plt is None:
            return csv_path

        plt = self._plt
        plt.figure(figsize=(6, 4))
        x = [row[0] for row in rows]
        y = [row[1] for row in rows]
        plt.plot(x, y, marker="o", markersize=2, linewidth=1)
        plt.xlabel("Coverage")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs Coverage ({ablation_id})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()
        return png_path

    def histogram(self, ablation_id: str, metric: str, values: Sequence[float]) -> Path:
        png_path = self.plots_dir / f"histogram_{metric}_{ablation_id}.png"
        csv_path = self.plots_dir / f"histogram_{metric}_{ablation_id}.csv"

        vals = [float(v) for v in values]
        self._write_csv(csv_path, [metric], [(v,) for v in vals])

        if not self._has_matplotlib or self._plt is None:
            return csv_path

        plt = self._plt
        plt.figure(figsize=(6, 4))
        if vals:
            plt.hist(vals, bins=min(30, max(5, int(math.sqrt(len(vals))))), edgecolor="black", alpha=0.8)
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center", transform=plt.gca().transAxes)
        pretty_name = metric.replace("_", " ")
        plt.xlabel(pretty_name)
        plt.ylabel("Count")
        plt.title(f"Histogram: {pretty_name} ({ablation_id})")
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()
        return png_path
