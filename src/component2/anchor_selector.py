"""Anchor selection for Component 2."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence

from .config import Component2Config, DEFAULT_CONFIG
from .types import EvidenceTriple


def normalize_entity(entity: str) -> str:
    """Normalize entity surface forms for matching."""

    lowered = entity.replace("_", " ").strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def is_valid_entity(entity: str) -> bool:
    """Reject blank entities that would produce degenerate anchors."""

    return bool(normalize_entity(entity))


class AnchorSelector:
    """Select anchors using retrieval seeds first, then text/entity fallbacks."""

    def __init__(self, config: Component2Config = DEFAULT_CONFIG):
        self.config = config

    @staticmethod
    def _graph_entities(triples: Sequence[EvidenceTriple]) -> List[str]:
        entities: List[str] = []
        seen = set()
        for triple in triples:
            subj, _, obj = triple.raw_triple
            if is_valid_entity(subj) and subj not in seen:
                seen.add(subj)
                entities.append(subj)
            if is_valid_entity(obj) and obj not in seen:
                seen.add(obj)
                entities.append(obj)
        return entities

    def _seed_anchors(self, seed_entities: Iterable[str], graph_entities: Sequence[str]) -> List[str]:
        graph_norm_to_raw = {normalize_entity(entity): entity for entity in graph_entities}
        anchors: List[str] = []
        for seed in seed_entities:
            normalized_seed = normalize_entity(str(seed))
            if not normalized_seed:
                continue
            matched = graph_norm_to_raw.get(normalized_seed)
            if matched and matched not in anchors:
                anchors.append(matched)
                if len(anchors) >= self.config.anchor_max_k:
                    break
        return anchors

    def _text_fallback_anchors(self, claim_text: str, graph_entities: Sequence[str]) -> List[str]:
        claim_norm = normalize_entity(claim_text)
        matches: List[str] = []
        for entity in graph_entities:
            entity_norm = normalize_entity(entity)
            if not entity_norm:
                continue
            if entity_norm in claim_norm:
                matches.append(entity)
                if len(matches) >= self.config.anchor_max_k:
                    break
        return matches

    def _degree_fallback(self, triples: Sequence[EvidenceTriple], graph_entities: Sequence[str]) -> List[str]:
        if not graph_entities:
            return []

        degree = Counter()
        first_index = {entity: idx for idx, entity in enumerate(graph_entities)}
        for triple in triples:
            subj, _, obj = triple.raw_triple
            degree[subj] += 1
            degree[obj] += 1

        ranked = sorted(
            graph_entities,
            key=lambda entity: (-degree[entity], first_index[entity]),
        )
        return ranked[: self.config.anchor_fallback_degree_k]

    @staticmethod
    def _extend_unique(base: List[str], candidates: Iterable[str], limit: int) -> List[str]:
        for candidate in candidates:
            if candidate in base:
                continue
            base.append(candidate)
            if len(base) >= limit:
                break
        return base

    def select_anchors(
        self,
        claim_text: str,
        triples: Sequence[EvidenceTriple],
        seed_entities: Sequence[str] | None = None,
    ) -> List[str]:
        """Return up to K anchors guaranteed to exist in the graph.

        Seed anchors are preferred, but if only one seed resolves we still try to
        supplement with text and degree fallbacks so recovery diagnostics can
        operate on anchor pairs.
        """

        graph_entities = self._graph_entities(triples)
        if not graph_entities:
            return []

        max_k = int(self.config.anchor_max_k)
        min_anchor_count = min(2, len(graph_entities), max_k)
        seed_entities = seed_entities or []
        anchors = self._seed_anchors(seed_entities, graph_entities)
        if len(anchors) >= min_anchor_count:
            return anchors[:max_k]

        text_matches = self._text_fallback_anchors(claim_text, graph_entities)
        anchors = self._extend_unique(anchors, text_matches, limit=max_k)
        if len(anchors) >= min_anchor_count:
            return anchors[:max_k]

        degree_candidates = self._degree_fallback(triples, graph_entities)
        anchors = self._extend_unique(anchors, degree_candidates, limit=max_k)
        return anchors[:max_k]
