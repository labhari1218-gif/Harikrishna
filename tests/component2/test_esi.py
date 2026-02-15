"""Unit tests for A+C ESI estimation helpers."""

from __future__ import annotations

import unittest

from component2.esi import compute_esi_metrics_ac
from component2.types import EvidenceTriple


def _triple(
    evidence_id: str,
    subj: str,
    rel: str,
    obj: str,
    pool: str,
    p_ent: float,
    p_con: float,
    p_neu: float,
) -> EvidenceTriple:
    return EvidenceTriple(
        evidence_id=evidence_id,
        raw_triple=(subj, rel, obj),
        pool=pool,
        p_ent=p_ent,
        p_con=p_con,
        p_neu=p_neu,
    )


class EsiMetricsTests(unittest.TestCase):
    def test_esi_high_when_a_is_strong_and_c_is_weak(self) -> None:
        triples = (
            _triple("a1", "A", "r1", "C", "A", 0.90, 0.05, 0.05),
            _triple("c1", "X", "r2", "Y", "C", 0.02, 0.03, 0.95),
        )
        metrics = compute_esi_metrics_ac(triples=triples, anchors=("A", "C"))
        self.assertGreater(metrics["esi_geom"], 0.70)
        self.assertGreaterEqual(metrics["esi_geom"], 0.0)
        self.assertLessEqual(metrics["esi_geom"], 1.0)

    def test_esi_high_when_a_is_weak_and_c_is_strong(self) -> None:
        triples = (
            _triple("a1", "A", "r1", "B", "A", 0.05, 0.05, 0.90),
            _triple("c1", "B", "r2", "C", "C", 0.05, 0.85, 0.10),
        )
        metrics = compute_esi_metrics_ac(triples=triples, anchors=("A", "C"))
        self.assertGreater(metrics["coverage_A"], 0.99)
        self.assertGreater(metrics["connectivity_A"], 0.99)
        self.assertGreater(metrics["esi_geom"], 0.70)

    def test_esi_low_when_both_a_and_c_are_weak(self) -> None:
        triples = (
            _triple("a1", "A", "r1", "B", "A", 0.05, 0.05, 0.90),
            _triple("c1", "X", "r2", "Y", "C", 0.02, 0.03, 0.95),
        )
        metrics = compute_esi_metrics_ac(triples=triples, anchors=("A", "C"))
        self.assertLess(metrics["coverage_A"], 0.51)
        self.assertLess(metrics["esi_geom"], 0.50)


if __name__ == "__main__":
    unittest.main()
