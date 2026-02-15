"""Adapter for injecting FactKG Entity_set anchors into Component 2 records."""

from __future__ import annotations

from dataclasses import dataclass, replace
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple
import warnings

from .types import ClaimRecord


DEFAULT_FACTKG_PICKLES: Dict[str, str] = {
    "train": "factkg_train.pickle",
    "val": "factkg_dev.pickle",
    "test": "factkg_test.pickle",
}


def _normalize_anchor_entity(entity: Any) -> str:
    value = str(entity).strip()
    if not value:
        return ""
    # FactKG entities are KG IDs; normalize whitespace-only variants to underscore IDs.
    return "_".join(value.split())


def _normalize_claim_text(claim_text: Any) -> str:
    return str(claim_text).strip()


def _extract_entity_set(payload: Any) -> Tuple[str, ...]:
    if isinstance(payload, dict):
        entities = payload.get("Entity_set")
    else:
        entities = None

    if entities is None:
        return ()
    if isinstance(entities, str):
        entities = [entities]
    if not isinstance(entities, (list, tuple, set)):
        return ()

    normalized = tuple(
        entity
        for entity in (_normalize_anchor_entity(raw) for raw in entities)
        if entity
    )
    return normalized


@dataclass(frozen=True)
class AnchorInjectionResult:
    """Anchor injection result for one claim."""

    claim: ClaimRecord
    anchor_source: str


class FactKGAnchorAdapter:
    """Inject FactKG Entity_set anchors by claim_id or claim_text."""

    def __init__(
        self,
        factkg_dir: Path = Path("data/factkg"),
        split_pickles: Mapping[str, str] | None = None,
    ):
        self.factkg_dir = Path(factkg_dir)
        self.split_pickles = dict(split_pickles or DEFAULT_FACTKG_PICKLES)
        self.claim_id_to_entity_set: Dict[str, Tuple[str, ...]] = {}
        self.claim_text_to_entity_set: Dict[str, Tuple[str, ...]] = {}
        self._load_diagnostics: Dict[str, Any] = {}
        self._loaded = False

    def _split_aliases(self, split: str) -> Tuple[str, ...]:
        split_norm = str(split).strip().lower()
        if split_norm == "val":
            return ("val", "dev")
        return (split_norm,)

    def _register_mapping(self, split: str, index: int, claim_text: Any, entity_set: Tuple[str, ...]) -> None:
        claim_text_key = _normalize_claim_text(claim_text)
        if claim_text_key:
            self.claim_text_to_entity_set.setdefault(claim_text_key, entity_set)

        for alias in self._split_aliases(split):
            claim_id = f"{alias}_{index}"
            self.claim_id_to_entity_set.setdefault(claim_id, entity_set)

    def load(self) -> None:
        """Load split pickles and build claim_id/claim_text -> Entity_set maps."""

        if self._loaded:
            return

        configured_pickles = []
        loaded_pickles = []
        missing_pickles = []
        invalid_pickles = []
        mapping_rows_loaded = 0

        for split, filename in self.split_pickles.items():
            path = self.factkg_dir / filename
            configured_pickles.append(str(path))
            if not path.exists():
                missing_pickles.append(str(path))
                continue
            loaded_pickles.append(str(path))

            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict):
                invalid_pickles.append(str(path))
                continue

            for idx, (claim_text, row) in enumerate(payload.items()):
                entity_set = _extract_entity_set(row)
                if not entity_set:
                    continue
                self._register_mapping(split=split, index=idx, claim_text=claim_text, entity_set=entity_set)
                mapping_rows_loaded += 1

        self._load_diagnostics = {
            "configured_pickles": tuple(configured_pickles),
            "loaded_pickles": tuple(loaded_pickles),
            "missing_pickles": tuple(missing_pickles),
            "invalid_pickles": tuple(invalid_pickles),
            "mapping_rows_loaded": mapping_rows_loaded,
        }

        self._loaded = True

        load_issues = []
        if missing_pickles:
            load_issues.append(f"missing={missing_pickles}")
        if invalid_pickles:
            load_issues.append(f"invalid={invalid_pickles}")
        if loaded_pickles and mapping_rows_loaded == 0:
            load_issues.append("no Entity_set mappings extracted")

        if load_issues:
            warnings.warn(
                "FactKGAnchorAdapter load issues detected "
                f"({'; '.join(load_issues)}). Claims without provided entity_set may "
                "fall back to heuristic anchors. Verify factkg_dir and split pickle names.",
                RuntimeWarning,
                stacklevel=2,
            )

    def load_diagnostics(self) -> Dict[str, Any]:
        """Return loading diagnostics for visibility in callers/tests."""

        self.load()
        return dict(self._load_diagnostics)

    def mapping_for_claim(self, claim_id: str, claim_text: str) -> Tuple[str, ...]:
        """Resolve anchors using claim_id first, then claim_text."""

        self.load()
        claim_id_key = str(claim_id).strip()
        if claim_id_key and claim_id_key in self.claim_id_to_entity_set:
            return self.claim_id_to_entity_set[claim_id_key]

        claim_text_key = _normalize_claim_text(claim_text)
        if claim_text_key and claim_text_key in self.claim_text_to_entity_set:
            return self.claim_text_to_entity_set[claim_text_key]
        return ()

    def inject_anchors(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Inject `entity_set` from FactKG pickles into a raw record.

        Priority:
        1) existing `entity_set`/`Entity_set` in `record` (anchor_source=provided)
        2) FactKG pickle mapping by claim_id/claim_text (anchor_source=pickle)
        3) leave empty and defer to heuristics (anchor_source=heuristic)
        """

        out: MutableMapping[str, Any] = dict(record)
        has_explicit_entity_set = "entity_set" in out or "Entity_set" in out
        if has_explicit_entity_set:
            out["_anchor_source"] = "provided"
            return dict(out)

        mapped = self.mapping_for_claim(
            claim_id=str(out.get("claim_id", "")),
            claim_text=str(out.get("claim_text", out.get("Sentence", ""))),
        )
        if mapped:
            out["entity_set"] = list(mapped)
            out["Entity_set"] = list(mapped)
            out["_anchor_source"] = "pickle"
            return dict(out)

        out.setdefault("entity_set", [])
        out["_anchor_source"] = "heuristic"
        return dict(out)

    def inject_claim(self, claim: ClaimRecord) -> AnchorInjectionResult:
        """Inject anchors into a normalized `ClaimRecord`."""

        if claim.entity_set:
            return AnchorInjectionResult(claim=claim, anchor_source="provided")

        mapped = self.mapping_for_claim(claim_id=claim.claim_id, claim_text=claim.claim_text)
        if mapped:
            return AnchorInjectionResult(
                claim=replace(claim, entity_set=tuple(mapped)),
                anchor_source="pickle",
            )

        return AnchorInjectionResult(claim=claim, anchor_source="heuristic")
