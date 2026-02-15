"""Graph construction for Component 2 reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .types import EvidenceGraph, EvidenceTriple, GraphEdge, POOL_TO_ID


@dataclass
class RelationVocab:
    """Mutable relation vocabulary shared across built graphs."""

    relation_to_id: Dict[str, int]

    def relation_id(self, relation: str) -> int:
        if relation not in self.relation_to_id:
            self.relation_to_id[relation] = len(self.relation_to_id)
        return self.relation_to_id[relation]


class GraphBuilder:
    """Build per-claim entity graphs from evidence triples."""

    def __init__(self, include_pools: Tuple[str, ...] = ("A", "C")):
        self.include_pools = tuple(pool.upper() for pool in include_pools)
        self.relation_vocab = RelationVocab(relation_to_id={})

    def _node_index(self, entity: str, node_to_idx: Dict[str, int], nodes: List[str]) -> int:
        if entity not in node_to_idx:
            node_to_idx[entity] = len(nodes)
            nodes.append(entity)
        return node_to_idx[entity]

    def build(
        self,
        claim_id: str,
        triples: Sequence[EvidenceTriple],
        anchors: Sequence[str],
    ) -> EvidenceGraph:
        """Build an `EvidenceGraph` with A∪C edges by default."""

        nodes: List[str] = []
        node_to_idx: Dict[str, int] = {}
        edges: List[GraphEdge] = []

        for anchor in anchors:
            self._node_index(anchor, node_to_idx, nodes)

        for triple in triples:
            if triple.pool not in self.include_pools:
                continue

            subj, relation, obj = triple.raw_triple
            src_idx = self._node_index(subj, node_to_idx, nodes)
            dst_idx = self._node_index(obj, node_to_idx, nodes)
            rel_id = self.relation_vocab.relation_id(relation)

            edges.append(
                GraphEdge(
                    evidence_id=triple.evidence_id,
                    source=src_idx,
                    target=dst_idx,
                    relation=relation,
                    relation_id=rel_id,
                    pool=triple.pool,
                    pool_id=POOL_TO_ID[triple.pool],
                    p_ent=triple.p_ent,
                    p_con=triple.p_con,
                    p_neu=triple.p_neu,
                    rel=triple.rel,
                    pol=triple.pol,
                )
            )

        anchor_indices = [node_to_idx[anchor] for anchor in anchors if anchor in node_to_idx]

        return EvidenceGraph(
            claim_id=claim_id,
            nodes=nodes,
            node_to_idx=node_to_idx,
            edges=edges,
            anchors=list(anchors),
            anchor_indices=anchor_indices,
        )

    def relation_vocab_snapshot(self) -> Dict[str, int]:
        """Expose the relation vocabulary for logging/debugging."""

        return dict(self.relation_vocab.relation_to_id)
