"""Typed records used by Component 2."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


POOL_TO_ID = {"A": 0, "S": 1, "C": 2}


@dataclass(frozen=True)
class EvidenceTriple:
    """Single evidence triple with PV scores and pool assignment."""

    evidence_id: str
    raw_triple: Tuple[str, str, str]
    pool: str
    p_ent: float
    p_con: float
    p_neu: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def relation(self) -> str:
        return self.raw_triple[1]

    @property
    def rel(self) -> float:
        return float(self.p_ent + self.p_con)

    @property
    def pol(self) -> float:
        return float(self.p_ent - self.p_con)


@dataclass(frozen=True)
class ClaimRecord:
    """Claim-level input with seed entities and evidence triples."""

    claim_id: str
    claim_text: str
    label: Optional[str]
    entity_set: Tuple[str, ...]
    triples: Tuple[EvidenceTriple, ...]
    sufficiency: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    """Edge in the reasoning graph."""

    evidence_id: str
    source: int
    target: int
    relation: str
    relation_id: int
    pool: str
    pool_id: int
    p_ent: float
    p_con: float
    p_neu: float
    rel: float
    pol: float


@dataclass
class EvidenceGraph:
    """Per-claim graph used by the reasoner."""

    claim_id: str
    nodes: List[str]
    node_to_idx: Dict[str, int]
    edges: List[GraphEdge]
    anchors: List[str]
    anchor_indices: List[int]


def _extract_probs(raw: Dict[str, Any]) -> Tuple[float, float, float]:
    evidence_id = str(raw.get("evidence_id", "unknown"))

    def _validate_prob(name: str, value: Any) -> float:
        try:
            prob = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid {name} for evidence_id={evidence_id}: expected numeric, got {value!r}"
            ) from exc
        if not math.isfinite(prob):
            raise ValueError(f"Invalid {name} for evidence_id={evidence_id}: value must be finite")
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"Invalid {name} for evidence_id={evidence_id}: value {prob} must be in [0, 1]")
        return prob

    if {"p_ent", "p_con", "p_neu"}.issubset(raw.keys()):
        return (
            _validate_prob("p_ent", raw["p_ent"]),
            _validate_prob("p_con", raw["p_con"]),
            _validate_prob("p_neu", raw["p_neu"]),
        )

    probs = raw.get("probs")
    if not isinstance(probs, dict):
        raise ValueError(
            f"Missing probability fields for evidence_id={evidence_id}. "
            "Expected p_ent/p_con/p_neu or probs.{entail,contra,neutral}."
        )

    if {"entail", "contra", "neutral"}.issubset(probs.keys()):
        p_ent_raw = probs["entail"]
        p_con_raw = probs["contra"]
        p_neu_raw = probs["neutral"]
    elif {"p_ent", "p_con", "p_neu"}.issubset(probs.keys()):
        p_ent_raw = probs["p_ent"]
        p_con_raw = probs["p_con"]
        p_neu_raw = probs["p_neu"]
    else:
        raise ValueError(
            f"Incomplete probs for evidence_id={evidence_id}. "
            "Expected keys entail/contra/neutral or p_ent/p_con/p_neu."
        )

    return (
        _validate_prob("p_ent", p_ent_raw),
        _validate_prob("p_con", p_con_raw),
        _validate_prob("p_neu", p_neu_raw),
    )


def evidence_from_dict(raw: Dict[str, Any], index: int = 0) -> EvidenceTriple:
    """Normalize a dictionary into an EvidenceTriple."""

    triple_raw = raw.get("raw_triple") or raw.get("triple")
    if not triple_raw or len(triple_raw) != 3:
        raise ValueError(f"Invalid raw_triple at index {index}: {triple_raw}")

    p_ent, p_con, p_neu = _extract_probs(raw)

    evidence_id = raw.get("evidence_id") or f"triple_{index}"
    pool = str(raw.get("pool", "A")).upper()
    if pool not in POOL_TO_ID:
        raise ValueError(f"Unknown pool '{pool}' for evidence_id={evidence_id}")

    return EvidenceTriple(
        evidence_id=str(evidence_id),
        raw_triple=(str(triple_raw[0]), str(triple_raw[1]), str(triple_raw[2])),
        pool=pool,
        p_ent=p_ent,
        p_con=p_con,
        p_neu=p_neu,
        metadata={k: v for k, v in raw.items() if k not in {"evidence_id", "raw_triple", "triple", "pool", "p_ent", "p_con", "p_neu", "probs"}},
    )


def claim_from_dict(raw: Dict[str, Any]) -> ClaimRecord:
    """Normalize a claim dictionary into ClaimRecord."""

    claim_id = str(raw.get("claim_id", "")).strip()
    if not claim_id:
        raise ValueError("claim_id is required")
    claim_text = str(raw.get("claim_text", raw.get("Sentence", "")))
    label = raw.get("label")

    entity_set_raw = raw.get("entity_set")
    if entity_set_raw is None:
        entity_set_raw = raw.get("Entity_set", [])
    if isinstance(entity_set_raw, str):
        entity_set = (entity_set_raw,)
    elif isinstance(entity_set_raw, (list, tuple, set)):
        entity_set = tuple(str(entity) for entity in entity_set_raw)
    else:
        raise ValueError(f"entity_set must be a sequence for claim_id={claim_id}")

    triples_raw = raw.get("triples", [])
    if not isinstance(triples_raw, (list, tuple)):
        raise ValueError(f"triples must be a sequence for claim_id={claim_id}")
    triples = tuple(evidence_from_dict(item, index=i) for i, item in enumerate(triples_raw))
    if len({triple.evidence_id for triple in triples}) != len(triples):
        raise ValueError(f"Duplicate evidence_id values detected for claim_id={claim_id}")

    sufficiency_raw = raw.get("sufficiency", {})
    if not isinstance(sufficiency_raw, dict):
        raise ValueError(f"sufficiency must be an object for claim_id={claim_id}")

    return ClaimRecord(
        claim_id=claim_id,
        claim_text=claim_text,
        label=None if label is None else str(label),
        entity_set=entity_set,
        triples=triples,
        sufficiency=dict(sufficiency_raw),
    )


def claims_from_iterable(rows: Iterable[Dict[str, Any]]) -> List[ClaimRecord]:
    """Convert raw dictionaries into normalized claim records."""

    return [claim_from_dict(row) for row in rows]


def claim_label_to_index(label: Optional[str]) -> Optional[int]:
    """Map dataset labels to binary class indices."""

    if label is None:
        return None

    normalized = label.strip().upper()
    if normalized in {"TRUE", "SUPPORTED", "SUPPORT", "1"}:
        return 0
    if normalized in {"FALSE", "REFUTED", "REFUTE", "0"}:
        return 1
    return None


def ensure_non_empty_claims(claims: Sequence[ClaimRecord]) -> None:
    """Raise an explicit error for empty claim batches."""

    if not claims:
        raise ValueError("No claims were provided.")
