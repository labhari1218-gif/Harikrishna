"""Persistent ESM memory for Component 4 backtracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, MutableMapping, Sequence

try:
    from ._graph_utils import (
        triple_evidence_id as _triple_evidence_id,
        triple_field as _triple_field,
        triple_pool as _triple_pool,
        triple_raw as _triple_raw,
        triple_rel as _triple_rel,
    )
except ImportError:  # pragma: no cover - fallback for direct module execution
    from component3._graph_utils import (
        triple_evidence_id as _triple_evidence_id,
        triple_field as _triple_field,
        triple_pool as _triple_pool,
        triple_raw as _triple_raw,
        triple_rel as _triple_rel,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RecoveryAttempt:
    """One recovery attempt for a suspended triple."""

    round_index: int
    selected: bool
    bridge_score_rank: int | None = None
    bridge_score_value: float | None = None
    note: str | None = None
    timestamp_utc: str = field(default_factory=_now_iso)


@dataclass
class TripleMemory:
    """Persistent per-triple ESM memory."""

    evidence_id: str
    initial_pool: str
    current_pool: str
    rel: float
    raw_triple: tuple[str, str, str] | None = None
    is_gold: bool | None = None
    suspended_at_round: int | None = None
    suspended_reason: str | None = None
    suspended_timestamp_utc: str | None = None
    recovery_attempts: list[RecoveryAttempt] = field(default_factory=list)
    promoted_to_active_at_round: int | None = None


@dataclass
class ClaimESMState:
    """In-memory + serializable ESM pools and history for one claim."""

    claim_id: str
    triples: dict[str, TripleMemory]
    created_at_utc: str = field(default_factory=_now_iso)
    updated_at_utc: str = field(default_factory=_now_iso)

    def pools(self) -> dict[str, list[str]]:
        rows = {"A": [], "S": [], "C": []}
        for evidence_id, triple in self.triples.items():
            pool = triple.current_pool
            rows.setdefault(pool, []).append(evidence_id)
        for pool_ids in rows.values():
            pool_ids.sort()
        return rows

    def to_record(self) -> dict[str, Any]:
        triples_payload = [asdict(row) for row in self.triples.values()]
        triples_payload.sort(key=lambda row: row["evidence_id"])
        return {
            "claim_id": self.claim_id,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "recovery_source_pool": "S",
            "fixed_graph_pools": ["A", "C"],
            "pools": self.pools(),
            "triples": triples_payload,
        }


class ESMMemoryStore:
    """Persistent ESM memory store that appends claim snapshots to JSONL."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.claim_states: MutableMapping[str, ClaimESMState] = {}

    @classmethod
    def for_run(cls, run_id: str, output_root: str = "runs") -> "ESMMemoryStore":
        output_path = Path(output_root) / str(run_id) / "esm_pools.jsonl"
        return cls(output_path=output_path)

    def initialize_claim(
        self,
        *,
        claim_id: str,
        triples: Sequence[Any],
        default_suspended_reason: str = "initial_esm_partition",
    ) -> ClaimESMState:
        by_id: dict[str, TripleMemory] = {}
        for idx, triple in enumerate(triples):
            evidence_id = _triple_evidence_id(triple, fallback_idx=idx)
            if evidence_id in by_id:
                raise ValueError(
                    f"Duplicate evidence_id detected while initializing ESM memory: {evidence_id}"
                )
            pool = _triple_pool(triple)
            if pool not in {"A", "S", "C"}:
                pool = "S"

            memory = TripleMemory(
                evidence_id=evidence_id,
                initial_pool=pool,
                current_pool=pool,
                rel=float(_triple_rel(triple)),
                raw_triple=_triple_raw(triple),
                is_gold=_triple_field(triple, "is_gold", None),
            )
            if pool == "S":
                memory.suspended_reason = str(default_suspended_reason)
                memory.suspended_at_round = 0
                memory.suspended_timestamp_utc = _now_iso()
            by_id[evidence_id] = memory

        state = ClaimESMState(claim_id=str(claim_id), triples=by_id)
        self.claim_states[state.claim_id] = state
        return state

    def get_claim_state(self, claim_id: str) -> ClaimESMState:
        key = str(claim_id)
        if key not in self.claim_states:
            raise KeyError(f"Claim not initialized in ESM memory: {claim_id}")
        return self.claim_states[key]

    def record_suspension(
        self,
        *,
        claim_id: str,
        evidence_id: str,
        reason: str,
        round_index: int,
    ) -> None:
        state = self.get_claim_state(claim_id)
        triple = state.triples[str(evidence_id)]
        triple.current_pool = "S"
        triple.suspended_reason = str(reason)
        triple.suspended_at_round = int(round_index)
        triple.suspended_timestamp_utc = _now_iso()
        state.updated_at_utc = _now_iso()

    def record_recovery_attempt(
        self,
        *,
        claim_id: str,
        evidence_id: str,
        round_index: int,
        selected: bool,
        bridge_score_rank: int | None = None,
        bridge_score_value: float | None = None,
        note: str | None = None,
    ) -> None:
        state = self.get_claim_state(claim_id)
        triple = state.triples[str(evidence_id)]
        has_s_origin = bool(triple.initial_pool == "S" or triple.suspended_at_round is not None)
        if triple.current_pool not in {"S", "A"} or not has_s_origin:
            raise ValueError(
                f"Recovery attempts are valid only for S-sourced triples. "
                f"evidence_id={evidence_id} pool={triple.current_pool}"
            )

        triple.recovery_attempts.append(
            RecoveryAttempt(
                round_index=int(round_index),
                selected=bool(selected),
                bridge_score_rank=bridge_score_rank,
                bridge_score_value=bridge_score_value,
                note=note,
            )
        )
        state.updated_at_utc = _now_iso()

    def promote_s_to_active(self, *, claim_id: str, evidence_id: str, round_index: int) -> None:
        state = self.get_claim_state(claim_id)
        triple = state.triples[str(evidence_id)]
        if triple.current_pool != "S":
            raise ValueError(
                "Only suspended triples can be promoted to active. "
                f"evidence_id={evidence_id} pool={triple.current_pool}"
            )
        triple.current_pool = "A"
        triple.promoted_to_active_at_round = int(round_index)
        state.updated_at_utc = _now_iso()

    def to_claim_record(self, claim_id: str) -> dict[str, Any]:
        state = self.get_claim_state(claim_id)
        return state.to_record()

    def write_claim_snapshot(self, claim_id: str) -> None:
        payload = self.to_claim_record(claim_id)
        self._append_record(payload)

    def write_all_snapshots(self, claim_ids: Iterable[str] | None = None) -> None:
        ids = list(claim_ids) if claim_ids is not None else list(self.claim_states.keys())
        for claim_id in ids:
            self.write_claim_snapshot(str(claim_id))

    def _append_record(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
