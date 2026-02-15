"""Hybrid masked dual-stream graph reasoner for Component 2 M1."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Component2Config, DEFAULT_CONFIG
from .types import EvidenceGraph


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    if denom <= 0.0:
        return np.zeros_like(values)
    return exp / denom


class HybridMaskedDualStreamReasoner:
    """Numpy implementation of a relation-conditioned dual-stream reasoner."""

    def __init__(self, config: Component2Config = DEFAULT_CONFIG):
        self.config = config
        if self.config.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if self.config.num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if self.config.rel_emb_dim < 1:
            raise ValueError("rel_emb_dim must be >= 1")
        if self.config.message_passing_layers < 1:
            raise ValueError("message_passing_layers must be >= 1")
        if self.config.hidden_dim % self.config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = self.config.hidden_dim
        self.num_heads = self.config.num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.rng = np.random.default_rng(self.config.random_seed)

        scale = 0.08
        self.w_init = self.rng.normal(0.0, scale, size=(4, self.hidden_dim))
        self.w_rel_up = self.rng.normal(0.0, scale, size=(self.config.rel_emb_dim, self.hidden_dim))

        self.w_q = self.rng.normal(0.0, scale, size=(self.num_heads, self.head_dim, self.hidden_dim))
        self.w_k = self.rng.normal(0.0, scale, size=(self.num_heads, self.head_dim, self.hidden_dim))
        self.w_v = self.rng.normal(0.0, scale, size=(self.num_heads, self.head_dim, self.hidden_dim))

        self.pool_vec_sup = self.rng.normal(0.0, scale, size=(self.hidden_dim,))
        self.pool_vec_ref = self.rng.normal(0.0, scale, size=(self.hidden_dim,))

        self.w_cls = self.rng.normal(0.0, scale, size=(2, self.hidden_dim * 6))
        self.b_cls = np.zeros((2,), dtype=np.float64)

        self.rel_embeddings = np.zeros((0, self.config.rel_emb_dim), dtype=np.float64)

    def support_mask(self, p_ent: np.ndarray) -> np.ndarray:
        if self.config.disable_pv_masks:
            return np.ones_like(p_ent, dtype=np.float64)
        return _sigmoid(self.config.mask_sup_scale * p_ent + self.config.mask_sup_bias)

    def refute_mask(self, p_con: np.ndarray) -> np.ndarray:
        if self.config.disable_pv_masks:
            return np.ones_like(p_con, dtype=np.float64)
        return _sigmoid(self.config.mask_ref_scale * p_con + self.config.mask_ref_bias)

    def _ensure_relation_embeddings(self, relation_count: int) -> None:
        if relation_count <= self.rel_embeddings.shape[0]:
            return

        extra = relation_count - self.rel_embeddings.shape[0]
        new_rows = self.rng.normal(0.0, 0.08, size=(extra, self.config.rel_emb_dim))
        self.rel_embeddings = np.concatenate([self.rel_embeddings, new_rows], axis=0)

    def _entity_hash_vector(self, entity: str) -> np.ndarray:
        digest = hashlib.blake2b(
            entity.encode("utf-8"),
            digest_size=8,
            person=b"component2",
        ).digest()
        stable_hash = int.from_bytes(digest, byteorder="little", signed=False)
        hashed_seed = (stable_hash + self.config.random_seed) % (2 ** 32)
        local_rng = np.random.default_rng(hashed_seed)
        return local_rng.normal(0.0, 0.04, size=(self.hidden_dim,))

    def _init_node_states(self, graph: EvidenceGraph) -> np.ndarray:
        num_nodes = len(graph.nodes)
        if num_nodes == 0:
            return np.zeros((1, self.hidden_dim), dtype=np.float64)

        in_degree = np.zeros((num_nodes,), dtype=np.float64)
        out_degree = np.zeros((num_nodes,), dtype=np.float64)
        for edge in graph.edges:
            out_degree[edge.source] += 1.0
            in_degree[edge.target] += 1.0

        anchor_mask = np.zeros((num_nodes,), dtype=np.float64)
        for idx in graph.anchor_indices:
            if 0 <= idx < num_nodes:
                anchor_mask[idx] = 1.0

        base_feats = np.stack(
            [
                in_degree,
                out_degree,
                anchor_mask,
                np.ones((num_nodes,), dtype=np.float64),
            ],
            axis=1,
        )
        states = np.tanh(base_feats @ self.w_init)

        hash_vectors = np.stack([self._entity_hash_vector(entity) for entity in graph.nodes], axis=0)
        states = np.tanh(states + hash_vectors)
        return states

    def _edge_arrays(self, graph: EvidenceGraph) -> Dict[str, np.ndarray]:
        num_edges = len(graph.edges)
        if num_edges == 0:
            return {
                "source": np.zeros((0,), dtype=np.int64),
                "target": np.zeros((0,), dtype=np.int64),
                "relation": np.zeros((0,), dtype=np.int64),
                "p_ent": np.zeros((0,), dtype=np.float64),
                "p_con": np.zeros((0,), dtype=np.float64),
            }

        source = np.array([edge.source for edge in graph.edges], dtype=np.int64)
        target = np.array([edge.target for edge in graph.edges], dtype=np.int64)
        relation = np.array([edge.relation_id for edge in graph.edges], dtype=np.int64)
        p_ent = np.array([edge.p_ent for edge in graph.edges], dtype=np.float64)
        p_con = np.array([edge.p_con for edge in graph.edges], dtype=np.float64)

        return {
            "source": source,
            "target": target,
            "relation": relation,
            "p_ent": p_ent,
            "p_con": p_con,
        }

    def _incoming_groups(self, target_indices: np.ndarray, num_nodes: int) -> List[np.ndarray]:
        incoming: List[List[int]] = [[] for _ in range(num_nodes)]
        for edge_idx, target in enumerate(target_indices.tolist()):
            incoming[target].append(edge_idx)
        return [np.array(indices, dtype=np.int64) for indices in incoming]

    def _run_stream(
        self,
        node_states: np.ndarray,
        rel_hidden: np.ndarray,
        source_idx: np.ndarray,
        target_idx: np.ndarray,
        incoming_groups: Sequence[np.ndarray],
        edge_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        num_nodes = node_states.shape[0]
        num_edges = source_idx.shape[0]
        if num_edges == 0:
            return {
                "states": node_states,
                "edge_attention": np.zeros((0,), dtype=np.float64),
                "edge_base_attention": np.zeros((0,), dtype=np.float64),
            }

        edge_attention_by_head = np.zeros((self.num_heads, num_edges), dtype=np.float64)
        base_attention_by_head = np.zeros((self.num_heads, num_edges), dtype=np.float64)
        head_outputs: List[np.ndarray] = []

        for head in range(self.num_heads):
            logits = np.zeros((num_edges,), dtype=np.float64)
            values = np.zeros((num_edges, self.head_dim), dtype=np.float64)

            for edge_idx in range(num_edges):
                src = source_idx[edge_idx]
                dst = target_idx[edge_idx]
                rel_vec = rel_hidden[edge_idx]

                src_state = node_states[src] + rel_vec
                dst_state = node_states[dst] + rel_vec

                q = self.w_q[head] @ dst_state
                k = self.w_k[head] @ src_state
                v = self.w_v[head] @ src_state

                logits[edge_idx] = float(np.dot(q, k) / math.sqrt(self.head_dim))
                values[edge_idx] = v

            attention = np.zeros((num_edges,), dtype=np.float64)
            for edge_indices in incoming_groups:
                if edge_indices.size == 0:
                    continue
                local_logits = logits[edge_indices]
                attention[edge_indices] = _softmax(local_logits)

            weighted_attention = attention * edge_mask
            edge_attention_by_head[head] = weighted_attention
            base_attention_by_head[head] = attention

            node_messages = np.zeros((num_nodes, self.head_dim), dtype=np.float64)
            for edge_idx in range(num_edges):
                dst = target_idx[edge_idx]
                node_messages[dst] += weighted_attention[edge_idx] * values[edge_idx]

            head_outputs.append(node_messages)

        concatenated = np.concatenate(head_outputs, axis=1)
        stream_states = np.tanh(concatenated + 0.5 * node_states)

        edge_attention = np.mean(edge_attention_by_head, axis=0)
        edge_base_attention = np.mean(base_attention_by_head, axis=0)
        return {
            "states": stream_states,
            "edge_attention": edge_attention,
            "edge_base_attention": edge_base_attention,
        }

    def _hybrid_pool(self, node_states: np.ndarray, pool_vector: np.ndarray) -> np.ndarray:
        if node_states.size == 0:
            return np.zeros((self.hidden_dim * 3,), dtype=np.float64)

        mean_pool = np.mean(node_states, axis=0)
        max_pool = np.max(node_states, axis=0)

        scores = (node_states @ pool_vector) / math.sqrt(self.hidden_dim)
        weights = _softmax(scores)
        attention_pool = np.sum(weights[:, None] * node_states, axis=0)

        return np.concatenate([mean_pool, max_pool, attention_pool], axis=0)

    def _classify(self, pooled_sup: np.ndarray, pooled_ref: np.ndarray) -> Dict[str, np.ndarray]:
        final_repr = np.concatenate([pooled_sup, pooled_ref], axis=0)
        logits = (self.w_cls @ final_repr) + self.b_cls
        probs = _softmax(logits)
        return {"repr": final_repr, "logits": logits, "probs": probs}

    def forward_graph(self, graph: EvidenceGraph) -> Dict[str, object]:
        """Run one claim graph through the reasoner."""

        node_states = self._init_node_states(graph)
        edge_data = self._edge_arrays(graph)

        source_idx = edge_data["source"]
        target_idx = edge_data["target"]
        relation_ids = edge_data["relation"]
        p_ent = edge_data["p_ent"]
        p_con = edge_data["p_con"]

        num_nodes = node_states.shape[0]
        incoming_groups = self._incoming_groups(target_idx, num_nodes)

        relation_count = int(np.max(relation_ids) + 1) if relation_ids.size > 0 else 0
        self._ensure_relation_embeddings(relation_count)

        rel_hidden = np.zeros((relation_ids.shape[0], self.hidden_dim), dtype=np.float64)
        if relation_ids.size > 0:
            rel_vectors = self.rel_embeddings[relation_ids]
            rel_hidden = rel_vectors @ self.w_rel_up

        mask_sup = self.support_mask(p_ent)
        mask_ref = self.refute_mask(p_con)

        sup_stream: Dict[str, np.ndarray]
        ref_stream: Dict[str, np.ndarray]
        sup_states = node_states
        ref_states = node_states
        for _ in range(self.config.message_passing_layers):
            sup_stream = self._run_stream(
                node_states=sup_states,
                rel_hidden=rel_hidden,
                source_idx=source_idx,
                target_idx=target_idx,
                incoming_groups=incoming_groups,
                edge_mask=mask_sup,
            )
            ref_stream = self._run_stream(
                node_states=ref_states,
                rel_hidden=rel_hidden,
                source_idx=source_idx,
                target_idx=target_idx,
                incoming_groups=incoming_groups,
                edge_mask=mask_ref,
            )
            sup_states = sup_stream["states"]
            ref_states = ref_stream["states"]

        pooled_sup = self._hybrid_pool(sup_stream["states"], self.pool_vec_sup)
        pooled_ref = self._hybrid_pool(ref_stream["states"], self.pool_vec_ref)
        cls = self._classify(pooled_sup, pooled_ref)

        salience = []
        for edge_idx, edge in enumerate(graph.edges):
            salience.append(
                {
                    "evidence_id": edge.evidence_id,
                    "relation": edge.relation,
                    "source": graph.nodes[edge.source],
                    "target": graph.nodes[edge.target],
                    "mask_sup": float(mask_sup[edge_idx]),
                    "mask_ref": float(mask_ref[edge_idx]),
                    "attention_sup": float(sup_stream["edge_attention"][edge_idx]),
                    "attention_ref": float(ref_stream["edge_attention"][edge_idx]),
                    "base_attention_sup": float(sup_stream["edge_base_attention"][edge_idx]),
                    "base_attention_ref": float(ref_stream["edge_base_attention"][edge_idx]),
                    "salience": float(
                        max(sup_stream["edge_attention"][edge_idx], ref_stream["edge_attention"][edge_idx])
                    ),
                }
            )

        return {
            "logits": cls["logits"],
            "probs": cls["probs"],
            "repr": cls["repr"],
            "salience": salience,
            "mask_sup": mask_sup,
            "mask_ref": mask_ref,
            "pool_sup": pooled_sup,
            "pool_ref": pooled_ref,
        }

    def compute_loss(self, logits: np.ndarray, labels: Sequence[int]) -> float:
        """Compute mean cross-entropy loss for binary labels."""

        labels_arr = np.array(labels, dtype=np.int64)
        if logits.shape[0] != labels_arr.shape[0]:
            raise ValueError("labels length must match logits batch size")
        if np.any((labels_arr < 0) | (labels_arr > 1)):
            raise ValueError("labels must contain only class indices 0 or 1")

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        batch_indices = np.arange(labels_arr.shape[0])
        return float(-np.mean(log_probs[batch_indices, labels_arr]))

    def forward_batch(
        self,
        graphs: Sequence[EvidenceGraph],
        labels: Optional[Sequence[int]] = None,
    ) -> Dict[str, object]:
        """Run a batch of graphs and optionally compute loss."""

        if len(graphs) == 0:
            raise ValueError("graphs batch must not be empty")
        graph_outputs = [self.forward_graph(graph) for graph in graphs]
        logits = np.stack([output["logits"] for output in graph_outputs], axis=0)
        probs = np.stack([output["probs"] for output in graph_outputs], axis=0)

        response: Dict[str, object] = {
            "logits": logits,
            "probs": probs,
            "per_graph": graph_outputs,
            "salience": [output["salience"] for output in graph_outputs],
        }

        if labels is not None:
            response["loss"] = self.compute_loss(logits, labels)

        return response
