"""PV-aware graph dataset for Component 3."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer

from constants import DATA_PATH, EMBEDDINGS_FILENAME
from datasets import FactKGDatasetGraph

POOL_TO_ID = {"A": 0.0, "S": 1.0, "C": 2.0}


@dataclass(frozen=True)
class TripleRecord:
    """Normalized triple row used by the PV graph dataset."""

    evidence_id: str
    subject: str
    relation: str
    object: str
    pool: str
    p_ent: float
    p_con: float
    p_neu: float
    rel: float
    is_fallback: bool
    is_promoted: bool = False


class FactKGPVDatasetGraph(FactKGDatasetGraph):
    """FactKG graph dataset with claim-conditioned triple edge features."""

    def __init__(
        self,
        df,
        evidence,
        *,
        embedding_dict: dict | None = None,
        embeddings_path: str | Path | None = None,
        claim_triple_cache_path: str | Path | None = None,
        claim_triple_encoder: Callable[[str, str, str, str], Sequence[float]] | None = None,
        claim_triple_model_name: str = "bert-base-uncased",
        claim_triple_dim: int = 768,
        claim_max_length: int = 256,
        require_pv_metadata: bool = False,
        require_claim_triple_cache: bool = False,
        include_s_pool: bool = True,
        add_reverse_edges: bool = True,
        auto_precompute: bool = True,
        precompute_batch_size: int = 32,
    ) -> None:
        self.inputs = df["Sentence"]
        self.labels = [self._coerce_label(label) for label in df["Label"]]
        self.length = len(df)
        self.claim_ids = self._extract_claim_ids(df)
        self.claim_types = self._extract_claim_types(df)

        self.evidence_rows = self._extract_evidence_rows(evidence)
        if len(self.evidence_rows) != self.length:
            raise ValueError(
                f"Evidence length ({len(self.evidence_rows)}) does not match dataframe length ({self.length})."
            )

        self.embedding_dict = self._load_embedding_dict(
            embedding_dict=embedding_dict,
            embeddings_path=embeddings_path,
        )
        self.node_embedding_dim = self._infer_node_embedding_dim(self.embedding_dict)

        default_cache_path = Path(DATA_PATH) / "claim_triple_embeddings.pkl"
        self.claim_triple_cache_path = Path(claim_triple_cache_path or default_cache_path)
        self.claim_triple_dim = int(claim_triple_dim)
        self.claim_max_length = int(claim_max_length)
        self.require_pv_metadata = bool(require_pv_metadata)
        self.require_claim_triple_cache = bool(require_claim_triple_cache)
        self.include_s_pool = bool(include_s_pool)
        self.add_reverse_edges = bool(add_reverse_edges)
        self.claim_triple_encoder = claim_triple_encoder
        self.claim_triple_model_name = claim_triple_model_name
        self.precompute_batch_size = max(1, int(precompute_batch_size))
        self._tokenizer = None
        self._encoder_model = None
        self._encoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.claim_triple_embeddings = self._load_claim_triple_cache()
        self._all_triples_by_index = [self._parse_triples_for_index(idx) for idx in range(self.length)]
        self._ac_triples_by_index = [
            [triple for triple in triples if triple.pool in {"A", "C"}]
            for triples in self._all_triples_by_index
        ]
        self._all_s_triples_by_index = [
            [triple for triple in triples if triple.pool == "S"]
            for triples in self._all_triples_by_index
        ]
        if self.include_s_pool:
            self._s_triples_by_index = [list(rows) for rows in self._all_s_triples_by_index]
        else:
            self._s_triples_by_index = [[] for _ in range(self.length)]
        self._fallback_total = int(
            sum(
                1
                for triples in self._all_triples_by_index
                for triple in triples
                if bool(triple.is_fallback)
            )
        )
        (
            self._missing_embeddings_total,
            self._missing_embedding_entities_total,
        ) = self._summarize_missing_embeddings()
        if auto_precompute:
            self.precompute_claim_triple_embeddings()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        claim_text = self.inputs[idx]
        label = self.labels[idx]
        claim_id = self.claim_ids[idx]

        triples = self._get_ac_triples(idx)
        graph = self._build_graph(claim_id=claim_id, claim_text=claim_text, triples=triples)
        graph.dataset_index = torch.tensor([idx], dtype=torch.long)
        return claim_text, graph, label

    @staticmethod
    def _coerce_label(label_value) -> int:
        if isinstance(label_value, (list, tuple)):
            if len(label_value) == 0:
                raise ValueError("Encountered empty label container.")
            return int(label_value[0])
        return int(label_value)

    @staticmethod
    def _extract_claim_ids(df) -> list[str]:
        if "claim_id" in df.columns:
            return [str(v) for v in df["claim_id"]]
        return [str(idx) for idx in range(len(df))]

    @staticmethod
    def _extract_claim_types(df) -> list[list[str]] | None:
        if "types" not in df.columns:
            return None
        normalized: list[list[str]] = []
        for raw in df["types"]:
            if isinstance(raw, (list, tuple, set)):
                normalized.append([str(v) for v in raw])
            elif raw is None:
                normalized.append([])
            else:
                normalized.append([str(raw)])
        return normalized

    @staticmethod
    def _extract_evidence_rows(evidence) -> list:
        if hasattr(evidence, "columns"):
            for column_name in ("triples", "walked", "subgraph"):
                if column_name in evidence.columns:
                    return list(evidence[column_name])
            raise ValueError("Evidence dataframe must contain one of: `triples`, `subgraph`, `walked`.")
        if isinstance(evidence, Sequence):
            return list(evidence)
        raise TypeError("`evidence` must be a dataframe-like object or a sequence of per-claim evidence rows.")

    @staticmethod
    def _load_embedding_dict(
        embedding_dict: dict | None,
        embeddings_path: str | Path | None,
    ) -> dict:
        if embedding_dict is not None:
            return embedding_dict

        path = Path(embeddings_path or (Path(DATA_PATH) / EMBEDDINGS_FILENAME))
        if not path.exists():
            raise FileNotFoundError(
                "Precomputed embeddings are required for Component 3. "
                f"Missing file: {path}. Hash fallback is disabled by design."
            )
        with path.open("rb") as fp:
            loaded = pickle.load(fp)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected embedding dictionary at {path}, got {type(loaded).__name__}.")
        return loaded

    @staticmethod
    def _infer_node_embedding_dim(embedding_dict: dict) -> int:
        if not embedding_dict:
            raise ValueError("Embedding dictionary is empty; at least one entity embedding is required.")
        first_vector = next(iter(embedding_dict.values()))
        vector = np.asarray(first_vector, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("Encountered empty embedding vector in embedding dictionary.")
        return int(vector.size)

    def _load_claim_triple_cache(self) -> dict[tuple[str, str], np.ndarray]:
        if not self.claim_triple_cache_path.exists():
            return {}
        with self.claim_triple_cache_path.open("rb") as fp:
            loaded = pickle.load(fp)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Expected dictionary cache at {self.claim_triple_cache_path}, got {type(loaded).__name__}."
            )
        normalized: dict[tuple[str, str], np.ndarray] = {}
        for raw_key, raw_value in loaded.items():
            if isinstance(raw_key, tuple) and len(raw_key) == 2:
                key = (str(raw_key[0]), str(raw_key[1]))
            else:
                key = (str(raw_key), "")
            value = np.asarray(raw_value, dtype=np.float32).reshape(-1)
            if value.size != self.claim_triple_dim:
                continue
            normalized[key] = value
        return normalized

    def _write_claim_triple_cache(self) -> None:
        self.claim_triple_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.claim_triple_cache_path.open("wb") as fp:
            pickle.dump(self.claim_triple_embeddings, fp)

    def _parse_triple_row(self, raw_row, row_idx: int) -> TripleRecord:
        if isinstance(raw_row, dict):
            raw_triple = raw_row.get("raw_triple")
            if raw_triple is None:
                raw_triple = [
                    raw_row.get("subject"),
                    raw_row.get("relation"),
                    raw_row.get("object"),
                ]
            if not isinstance(raw_triple, (list, tuple)) or len(raw_triple) < 3:
                raise ValueError(f"Malformed triple at index {row_idx}: {raw_row}")
            subject, relation, object_ = str(raw_triple[0]), str(raw_triple[1]), str(raw_triple[2])

            has_metadata = any(key in raw_row for key in ("p_ent", "p_con", "p_neu", "rel"))
            if self.require_pv_metadata and not has_metadata:
                raise ValueError(
                    f"PV metadata required, but missing at triple index {row_idx}: {raw_row}"
                )

            pool = str(raw_row.get("pool", "A")).upper()
            if pool not in POOL_TO_ID:
                pool = "A"
            p_ent = float(raw_row.get("p_ent", 0.0))
            p_con = float(raw_row.get("p_con", 0.0))
            p_neu = float(raw_row.get("p_neu", max(0.0, 1.0 - (p_ent + p_con))))
            rel = float(raw_row.get("rel", p_ent + p_con))
            evidence_id = str(raw_row.get("evidence_id", f"triple_{row_idx}"))
            return TripleRecord(
                evidence_id=evidence_id,
                subject=subject,
                relation=relation,
                object=object_,
                pool=pool,
                p_ent=p_ent,
                p_con=p_con,
                p_neu=p_neu,
                rel=rel,
                is_fallback=(not has_metadata),
                is_promoted=bool(raw_row.get("is_promoted", False)),
            )

        if isinstance(raw_row, (list, tuple)) and len(raw_row) >= 3:
            if self.require_pv_metadata:
                raise ValueError(
                    f"PV metadata required, but received fallback list/tuple triple at index {row_idx}."
                )
            subject, relation, object_ = str(raw_row[0]), str(raw_row[1]), str(raw_row[2])
            return TripleRecord(
                evidence_id=f"triple_{row_idx}",
                subject=subject,
                relation=relation,
                object=object_,
                pool="A",
                p_ent=0.0,
                p_con=0.0,
                p_neu=1.0,
                rel=0.0,
                is_fallback=True,
                is_promoted=False,
            )

        raise ValueError(f"Unsupported triple row format at index {row_idx}: {type(raw_row).__name__}")

    def _iter_raw_triples(self, claim_evidence) -> Iterable:
        if isinstance(claim_evidence, dict):
            if "triples" in claim_evidence:
                return claim_evidence["triples"]
            if "walked" in claim_evidence and isinstance(claim_evidence["walked"], dict):
                walked = claim_evidence["walked"]
                connected = walked.get("connected", [])
                walkable = walked.get("walkable", [])
                return list(connected) + list(walkable)
            if "connected" in claim_evidence or "walkable" in claim_evidence:
                connected = claim_evidence.get("connected", [])
                walkable = claim_evidence.get("walkable", [])
                return list(connected) + list(walkable)
        return claim_evidence

    def _parse_triples_for_index(self, idx: int) -> list[TripleRecord]:
        raw_rows = self._iter_raw_triples(self.evidence_rows[idx])
        if raw_rows is None:
            return []

        parsed: list[TripleRecord] = []
        for row_idx, raw_row in enumerate(raw_rows):
            triple = self._parse_triple_row(raw_row, row_idx=row_idx)
            parsed.append(triple)
        return parsed

    def _summarize_missing_embeddings(self) -> tuple[int, int]:
        missing_mentions = 0
        missing_entities: set[str] = set()
        for triples in self._all_triples_by_index:
            for triple in triples:
                for entity in (triple.subject, triple.object):
                    if entity not in self.embedding_dict:
                        missing_mentions += 1
                        missing_entities.add(entity)
        return int(missing_mentions), int(len(missing_entities))

    def _get_ac_triples(self, idx: int) -> list[TripleRecord]:
        return list(self._ac_triples_by_index[idx])

    def _get_s_triples(self, idx: int) -> list[TripleRecord]:
        return list(self._s_triples_by_index[idx])

    @staticmethod
    def _triple_to_row(triple: TripleRecord) -> dict:
        row = asdict(triple)
        row["raw_triple"] = [triple.subject, triple.relation, triple.object]
        return row

    def get_recovery_triplets(self, idx: int) -> tuple[list[dict], list[dict]]:
        active = [self._triple_to_row(triple) for triple in self._get_ac_triples(idx)]
        suspended = [self._triple_to_row(triple) for triple in self._get_s_triples(idx)]
        return active, suspended

    def build_graph_from_recovery_rows(self, index: int, active_rows: Sequence[dict]) -> Data:
        if index < 0 or index >= self.length:
            raise IndexError(f"index out of range for recovery graph build: {index}")
        triples: list[TripleRecord] = []
        for row_idx, row in enumerate(active_rows):
            triple = self._parse_triple_row(row, row_idx=row_idx)
            if triple.pool in {"A", "C"}:
                triples.append(triple)
        claim_id = self.claim_ids[index]
        claim_text = self.inputs[index]
        graph = self._build_graph(claim_id=claim_id, claim_text=claim_text, triples=triples)
        graph.dataset_index = torch.tensor([index], dtype=torch.long)
        return graph

    def get_pool_summary(self) -> dict[str, float | int | bool]:
        total_claims = int(self.length)
        ac_total = int(sum(len(rows) for rows in self._ac_triples_by_index))
        s_available_total = int(sum(len(rows) for rows in self._all_s_triples_by_index))
        s_retained_total = int(sum(len(rows) for rows in self._s_triples_by_index))
        claims_with_s_available = int(sum(1 for rows in self._all_s_triples_by_index if rows))
        claims_with_s_retained = int(sum(1 for rows in self._s_triples_by_index if rows))
        return {
            "include_s_pool": bool(self.include_s_pool),
            "claims_total": total_claims,
            "ac_triples_total": ac_total,
            "s_pool_available_total": s_available_total,
            "s_pool_retained_total": s_retained_total,
            "claims_with_s_pool_available": claims_with_s_available,
            "claims_with_s_pool_retained": claims_with_s_retained,
            "fallback_total": int(self._fallback_total),
            "missing_embeddings_total": int(self._missing_embeddings_total),
            "missing_embedding_entities_total": int(self._missing_embedding_entities_total),
        }

    def _cache_key(self, claim_id: str, triple: TripleRecord) -> tuple[str, str]:
        return claim_id, triple.evidence_id

    def _ensure_encoder(self) -> None:
        if self.claim_triple_encoder is not None:
            return
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.claim_triple_model_name)
        if self._encoder_model is None:
            self._encoder_model = AutoModel.from_pretrained(self.claim_triple_model_name)
            self._encoder_model.to(self._encoder_device)
            self._encoder_model.eval()

    def _encode_claim_conditioned_triple(
        self,
        claim_text: str,
        subject: str,
        relation: str,
        object_: str,
    ) -> np.ndarray:
        if self.claim_triple_encoder is not None:
            vector = np.asarray(
                self.claim_triple_encoder(claim_text, subject, relation, object_),
                dtype=np.float32,
            ).reshape(-1)
        else:
            self._ensure_encoder()
            triple_text = f"{subject} {relation} {object_}"
            encoded = self._tokenizer(
                claim_text,
                triple_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.claim_max_length,
            )
            encoded = {k: v.to(self._encoder_device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._encoder_model(**encoded)
            vector = outputs.last_hidden_state[0, 0, :].detach().cpu().numpy().astype(np.float32)

        if vector.size != self.claim_triple_dim:
            raise ValueError(
                f"Claim-triple embedding dimension mismatch: expected {self.claim_triple_dim}, got {vector.size}."
            )
        return vector

    def _encode_claim_conditioned_triples_batch(
        self,
        rows: Sequence[tuple[str, str, str, str]],
    ) -> list[np.ndarray]:
        if not rows:
            return []
        if self.claim_triple_encoder is not None:
            vectors = [
                np.asarray(
                    self.claim_triple_encoder(claim_text, subject, relation, object_),
                    dtype=np.float32,
                ).reshape(-1)
                for claim_text, subject, relation, object_ in rows
            ]
        else:
            self._ensure_encoder()
            claims = [claim_text for claim_text, _, _, _ in rows]
            triples = [f"{subject} {relation} {object_}" for _, subject, relation, object_ in rows]
            encoded = self._tokenizer(
                claims,
                triples,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.claim_max_length,
            )
            encoded = {k: v.to(self._encoder_device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._encoder_model(**encoded)
            vectors = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
            vectors = [np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors]

        for vector in vectors:
            if vector.size != self.claim_triple_dim:
                raise ValueError(
                    f"Claim-triple embedding dimension mismatch: expected {self.claim_triple_dim}, got {vector.size}."
                )
        return vectors

    def _get_claim_triple_embedding(
        self,
        claim_id: str,
        claim_text: str,
        triple: TripleRecord,
    ) -> np.ndarray:
        key = self._cache_key(claim_id=claim_id, triple=triple)
        cached = self.claim_triple_embeddings.get(key)
        if cached is not None:
            return cached
        if self.require_claim_triple_cache:
            raise RuntimeError(
                "Missing claim-triple embedding in strict cache mode for "
                f"claim_id={claim_id}, evidence_id={triple.evidence_id}. "
                f"Populate {self.claim_triple_cache_path} via precompute before training."
            )

        encoded = self._encode_claim_conditioned_triple(
            claim_text=claim_text,
            subject=triple.subject,
            relation=triple.relation,
            object_=triple.object,
        )
        self.claim_triple_embeddings[key] = encoded
        return encoded

    def precompute_claim_triple_embeddings(self, force: bool = False) -> Path:
        if force:
            self.claim_triple_embeddings = {}

        pending: list[tuple[tuple[str, str], str, TripleRecord]] = []
        for idx in range(self.length):
            claim_text = self.inputs[idx]
            claim_id = self.claim_ids[idx]
            triples = self._get_ac_triples(idx)
            if self.include_s_pool:
                triples = triples + self._get_s_triples(idx)
            for triple in triples:
                key = self._cache_key(claim_id=claim_id, triple=triple)
                if not force and key in self.claim_triple_embeddings:
                    continue
                pending.append((key, claim_text, triple))

        pending.sort(key=lambda row: row[0])
        for start in range(0, len(pending), self.precompute_batch_size):
            batch = pending[start: start + self.precompute_batch_size]
            vectors = self._encode_claim_conditioned_triples_batch(
                [(claim_text, triple.subject, triple.relation, triple.object) for _, claim_text, triple in batch]
            )
            for (key, _claim_text, _triple), vector in zip(batch, vectors):
                self.claim_triple_embeddings[key] = vector

        self._write_claim_triple_cache()
        return self.claim_triple_cache_path

    def _node_embedding(self, entity: str) -> torch.Tensor:
        if entity not in self.embedding_dict:
            raise KeyError(
                f"Missing precomputed embedding for entity `{entity}`. "
                "Component 3 forbids hash-based fallback vectors."
            )
        vector = np.asarray(self.embedding_dict[entity], dtype=np.float32).reshape(-1)
        if vector.size != self.node_embedding_dim:
            raise ValueError(
                f"Entity embedding dimension mismatch for `{entity}`: "
                f"expected {self.node_embedding_dim}, got {vector.size}."
            )
        return torch.from_numpy(vector)

    def _empty_graph(self) -> Data:
        node_features = torch.zeros((1, self.node_embedding_dim), dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, self.claim_triple_dim + 5), dtype=torch.float32)
        edge_is_gold = torch.zeros((1,), dtype=torch.float32)
        edge_is_support = torch.zeros((1,), dtype=torch.float32)
        edge_is_counter = torch.zeros((1,), dtype=torch.float32)
        edge_is_fallback = torch.ones((1,), dtype=torch.float32)
        edge_is_promoted = torch.zeros((1,), dtype=torch.float32)
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_is_gold=edge_is_gold,
            edge_is_support=edge_is_support,
            edge_is_counter=edge_is_counter,
            edge_is_fallback=edge_is_fallback,
            edge_is_promoted=edge_is_promoted,
        )

    def _build_graph(self, claim_id: str, claim_text: str, triples: list[TripleRecord]) -> Data:
        if not triples:
            return self._empty_graph()

        node_to_index: dict[str, int] = {}
        node_features: list[torch.Tensor] = []
        edge_indices: list[list[int]] = []
        edge_features: list[torch.Tensor] = []
        edge_is_gold: list[float] = []
        edge_is_support: list[float] = []
        edge_is_counter: list[float] = []
        edge_is_fallback: list[float] = []
        edge_is_promoted: list[float] = []

        for triple in triples:
            if triple.subject not in node_to_index:
                node_to_index[triple.subject] = len(node_to_index)
                node_features.append(self._node_embedding(triple.subject))
            if triple.object not in node_to_index:
                node_to_index[triple.object] = len(node_to_index)
                node_features.append(self._node_embedding(triple.object))

            src = node_to_index[triple.subject]
            dst = node_to_index[triple.object]

            claim_triple_embedding = torch.from_numpy(
                self._get_claim_triple_embedding(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    triple=triple,
                )
            )
            pool_id = float(POOL_TO_ID.get(triple.pool, POOL_TO_ID["A"]))
            pv_meta = torch.tensor(
                [triple.p_ent, triple.p_con, triple.p_neu, triple.rel, pool_id],
                dtype=torch.float32,
            )
            edge_feature = torch.cat((claim_triple_embedding, pv_meta), dim=0)

            label_gold = 1.0 if triple.pool == "A" else 0.0
            label_support = 1.0 if triple.pool == "A" else 0.0
            label_counter = 1.0 if triple.pool == "C" else 0.0
            label_fallback = 1.0 if triple.is_fallback else 0.0
            label_promoted = 1.0 if triple.is_promoted else 0.0

            edge_indices.append([src, dst])
            edge_features.append(edge_feature)
            edge_is_gold.append(label_gold)
            edge_is_support.append(label_support)
            edge_is_counter.append(label_counter)
            edge_is_fallback.append(label_fallback)
            edge_is_promoted.append(label_promoted)
            if self.add_reverse_edges:
                edge_indices.append([dst, src])
                edge_features.append(edge_feature.clone())
                edge_is_gold.append(label_gold)
                edge_is_support.append(label_support)
                edge_is_counter.append(label_counter)
                edge_is_fallback.append(label_fallback)
                edge_is_promoted.append(label_promoted)

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        x = torch.stack(node_features).to(torch.float32)
        edge_attr = torch.stack(edge_features).to(torch.float32)
        edge_is_gold_tensor = torch.tensor(edge_is_gold, dtype=torch.float32)
        edge_is_support_tensor = torch.tensor(edge_is_support, dtype=torch.float32)
        edge_is_counter_tensor = torch.tensor(edge_is_counter, dtype=torch.float32)
        edge_is_fallback_tensor = torch.tensor(edge_is_fallback, dtype=torch.float32)
        edge_is_promoted_tensor = torch.tensor(edge_is_promoted, dtype=torch.float32)
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_is_gold=edge_is_gold_tensor,
            edge_is_support=edge_is_support_tensor,
            edge_is_counter=edge_is_counter_tensor,
            edge_is_fallback=edge_is_fallback_tensor,
            edge_is_promoted=edge_is_promoted_tensor,
        )
