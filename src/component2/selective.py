"""Selective prediction helpers for abstention and risk-coverage evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class RiskCoveragePoint:
    """One point on the risk-coverage curve."""

    answered: int
    coverage: float
    errors: int
    risk: float
    abstain_threshold: float


@dataclass(frozen=True)
class OperatingPointMetrics:
    """Metrics at a fixed abstention threshold tau."""

    tau_abstain: float
    answered: int
    total: int
    coverage: float
    errors: int
    risk: float
    accuracy_on_answered: float
    abstain_rate: float


def abstain_score(probs: Sequence[float], esi_geom: float, alpha: float = 0.5) -> float:
    """Compute abstention score from confidence and sufficiency."""

    if len(probs) == 0:
        raise ValueError("probs must not be empty")
    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")

    probs_arr = np.array(probs, dtype=np.float64)
    if not np.all(np.isfinite(probs_arr)):
        raise ValueError("probs must contain finite values")
    if np.any((probs_arr < 0.0) | (probs_arr > 1.0)):
        raise ValueError("probs must be in [0, 1]")

    confidence = float(np.max(probs_arr))
    esi = float(esi_geom)
    if not np.isfinite(esi):
        raise ValueError("esi_geom must be finite")
    if esi < 0.0 or esi > 1.0:
        raise ValueError("esi_geom must be in [0, 1]")
    return alpha * (1.0 - confidence) + (1.0 - alpha) * (1.0 - esi)


def selective_operating_point(
    abstain_scores: Sequence[float],
    is_correct: Sequence[bool],
    tau_abstain: float,
) -> OperatingPointMetrics:
    """Evaluate coverage/risk at a fixed abstention threshold."""

    if len(abstain_scores) != len(is_correct):
        raise ValueError("abstain_scores and is_correct must have equal length")
    tau = float(tau_abstain)
    if not np.isfinite(tau):
        raise ValueError("tau_abstain must be finite")

    total = len(abstain_scores)
    if total == 0:
        return OperatingPointMetrics(
            tau_abstain=tau,
            answered=0,
            total=0,
            coverage=0.0,
            errors=0,
            risk=0.0,
            accuracy_on_answered=0.0,
            abstain_rate=0.0,
        )

    scores = np.array(abstain_scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("abstain_scores must be finite")
    correct = np.array([bool(flag) for flag in is_correct], dtype=np.bool_)

    answered_mask = scores <= tau
    answered = int(np.sum(answered_mask))
    if answered <= 0:
        coverage = 0.0
        risk = 0.0
        accuracy_on_answered = 0.0
        errors = 0
    else:
        errors = int(np.sum(~correct[answered_mask]))
        coverage = float(answered) / float(total)
        risk = float(errors) / float(answered)
        accuracy_on_answered = float(answered - errors) / float(answered)
    abstain_rate = 1.0 - coverage

    return OperatingPointMetrics(
        tau_abstain=tau,
        answered=answered,
        total=total,
        coverage=coverage,
        errors=errors,
        risk=risk,
        accuracy_on_answered=accuracy_on_answered,
        abstain_rate=abstain_rate,
    )


def risk_coverage_curve(
    abstain_scores: Sequence[float],
    is_correct: Sequence[bool],
) -> List[RiskCoveragePoint]:
    """Build risk-coverage points by answering lowest-abstention claims first."""

    if len(abstain_scores) != len(is_correct):
        raise ValueError("abstain_scores and is_correct must have equal length")
    total = len(abstain_scores)
    if total == 0:
        return []
    scores = np.array(abstain_scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("abstain_scores must be finite")

    rows = sorted(zip(scores.tolist(), is_correct), key=lambda row: row[0])
    points: List[RiskCoveragePoint] = []
    errors = 0
    for answered, (score, correct) in enumerate(rows, start=1):
        if not bool(correct):
            errors += 1
        risk = float(errors) / float(answered)
        coverage = float(answered) / float(total)
        points.append(
            RiskCoveragePoint(
                answered=answered,
                coverage=coverage,
                errors=errors,
                risk=risk,
                abstain_threshold=float(score),
            )
        )
    return points


def compute_aurc(points: Sequence[RiskCoveragePoint]) -> float:
    """Compute trapezoidal area under a risk-coverage curve."""

    if not points:
        return 0.0

    area = 0.0
    prev_cov = 0.0
    prev_risk = 0.0
    for point in points:
        cov = float(point.coverage)
        risk = float(point.risk)
        area += (cov - prev_cov) * (risk + prev_risk) * 0.5
        prev_cov = cov
        prev_risk = risk
    return float(area)


def evaluate_selective_prediction(
    abstain_scores: Sequence[float],
    is_correct: Sequence[bool],
    tau_abstain: float | None = None,
    compute_curve: bool = True,
) -> dict:
    """Compute optional risk-coverage sweep and fixed-threshold metrics."""

    response = {
        "num_examples": len(abstain_scores),
        "aurc": None,
        "points": [],
        "operating_point": None,
    }

    if compute_curve:
        points = risk_coverage_curve(abstain_scores=abstain_scores, is_correct=is_correct)
        aurc = compute_aurc(points)
        response["aurc"] = aurc
        response["points"] = [
            {
                "answered": row.answered,
                "coverage": row.coverage,
                "errors": row.errors,
                "risk": row.risk,
                "abstain_threshold": row.abstain_threshold,
            }
            for row in points
        ]

    if tau_abstain is not None:
        op = selective_operating_point(
            abstain_scores=abstain_scores,
            is_correct=is_correct,
            tau_abstain=tau_abstain,
        )
        response["operating_point"] = {
            "tau_abstain": op.tau_abstain,
            "answered": op.answered,
            "total": op.total,
            "coverage": op.coverage,
            "errors": op.errors,
            "risk": op.risk,
            "accuracy_on_answered": op.accuracy_on_answered,
            "abstain_rate": op.abstain_rate,
        }

    return response
