"""PV-aware sentence graph dataset for FEVER in Component 3."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer

POOL_TO_ID = {"A": 0.0, "S": 1.0, "C": 2.0}


@dataclass(frozen=True)
class SentenceEvidenceRecord:
    """Normalized FEVER evidence row."""

    evidence_id: str
    text: str
    pool: str
    p_ent: float
    p_con: float
    p_neu: float
    rel: float
    is_gold: bool
    is_fallback: bool
    page: str | None
    line: int | None


class FeverPVDatasetGraph(Dataset):
    """Sentence-node graph dataset with PV metadata for FEVER."""

    def __init__(
        self,
        df,
        evidence,
        *,
        sentence_cache_path: str | Path | None = None,
        claim_sentence_cache_path: str | Path | None = None,
        sentence_encoder: Callable[[str], Sequence[float]] | None = None,
        claim_sentence_encoder: Callable[[str, str], Sequence[float]] | None = None,
        sentence_model_name: str = "bert-base-uncased",
        sentence_dim: int = 768,
        claim_sentence_dim: int = 768,
        max_seq_len: int = 256,
        include_s_pool: bool = False,
        auto_precompute: bool = False,
        precompute_batch_size: int = 32,
    ) -> None:
        self.inputs = list(df["Sentence"])
        self.labels = [self._coerce_label(v) for v in df["Label"]]
        self.length = len(df)
        self.claim_ids = self._extract_claim_ids(df)

        self.evidence_rows = self._extract_evidence_rows(evidence)
        if len(self.evidence_rows) != self.length:
            raise ValueError(
                f"Evidence length ({len(self.evidence_rows)}) does not match dataframe length ({self.length})."
            )

        self.sentence_model_name = sentence_model_name
        self.sentence_dim = int(sentence_dim)
        self.claim_sentence_dim = int(claim_sentence_dim)
        self.max_seq_len = int(max_seq_len)
        self.include_s_pool = bool(include_s_pool)
        self.precompute_batch_size = max(1, int(precompute_batch_size))

        self.sentence_encoder = sentence_encoder
        self.claim_sentence_encoder = claim_sentence_encoder

        self._tokenizer = None
        self._encoder_model = None
        self._encoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sentence_cache_path = Path(sentence_cache_path) if sentence_cache_path else None
        self.claim_sentence_cache_path = (
            Path(claim_sentence_cache_path) if claim_sentence_cache_path else None
        )

        self.sentence_embeddings, self.sentence_embedding_signatures = self._load_cache(
            self.sentence_cache_path,
            key_kind="single",
        )
        self.claim_sentence_embeddings, self.claim_sentence_embedding_signatures = self._load_cache(
            self.claim_sentence_cache_path,
            key_kind="pair",
        )
        self._active_records_cache = [self._parse_active_records_for_index(idx) for idx in range(self.length)]

        if auto_precompute:
            self.precompute_sentence_embeddings()
            self.precompute_claim_sentence_embeddings()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        claim_text = self.inputs[idx]
        label = self.labels[idx]
        claim_id = self.claim_ids[idx]

        evidence_records = self._get_active_records(idx)
        graph = self._build_graph(claim_id=claim_id, claim_text=claim_text, records=evidence_records)
        return claim_text, graph, label

    @staticmethod
    def _coerce_label(value) -> int:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Encountered empty label container.")
            return int(value[0])
        return int(value)

    @staticmethod
    def _extract_claim_ids(df) -> list[str]:
        if "claim_id" in df.columns:
            return [str(v) for v in df["claim_id"]]
        return [str(i) for i in range(len(df))]

    @staticmethod
    def _extract_evidence_rows(evidence) -> list[list[dict]]:
        if hasattr(evidence, "columns"):
            if "evidence_rows" not in evidence.columns:
                raise ValueError("Evidence dataframe must include `evidence_rows` column for FEVER.")
            return [list(v) if isinstance(v, list) else [] for v in evidence["evidence_rows"]]
        if isinstance(evidence, Sequence):
            rows = []
            for item in evidence:
                if isinstance(item, list):
                    rows.append(item)
                else:
                    rows.append([])
            return rows
        raise TypeError("`evidence` must be a dataframe-like object or sequence aligned to claims.")

    @staticmethod
    def _normalize_cache_key(raw_key, key_kind: str):
        if key_kind == "single":
            return str(raw_key)
        if isinstance(raw_key, tuple) and len(raw_key) == 2:
            return (str(raw_key[0]), str(raw_key[1]))
        return None

    def _load_cache(self, path: Path | None, key_kind: str) -> tuple[dict, dict]:
        if path is None or not path.exists():
            return {}, {}
        with path.open("rb") as fp:
            payload = pickle.load(fp)
        if not isinstance(payload, dict):
            return {}, {}

        raw_vectors = payload.get("vectors", payload)
        raw_signatures = payload.get("signatures", {})
        if not isinstance(raw_vectors, dict):
            return {}, {}

        normalized: dict = {}
        for raw_key, raw_val in raw_vectors.items():
            key = self._normalize_cache_key(raw_key, key_kind=key_kind)
            if key is None:
                continue
            if isinstance(raw_val, dict) and "vector" in raw_val:
                raw_val = raw_val["vector"]
            vec = np.asarray(raw_val, dtype=np.float32).reshape(-1)
            expected_dim = self.sentence_dim if key_kind == "single" else self.claim_sentence_dim
            if int(vec.size) != int(expected_dim):
                continue
            normalized[key] = vec

        signatures: dict = {}
        if isinstance(raw_signatures, dict):
            for raw_key, raw_signature in raw_signatures.items():
                key = self._normalize_cache_key(raw_key, key_kind=key_kind)
                if key is None:
                    continue
                if key not in normalized:
                    continue
                if not isinstance(raw_signature, str):
                    continue
                signatures[key] = raw_signature

        return normalized, signatures

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_int(value) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_pool(value) -> str:
        pool = str(value or "A").upper()
        if pool not in POOL_TO_ID:
            return "A"
        return pool

    def _parse_record(self, row: dict, row_idx: int) -> SentenceEvidenceRecord | None:
        if not isinstance(row, dict):
            return None

        evidence_id = str(row.get("evidence_id") or f"ev_{row_idx}")

        text = row.get("text")
        if text is None:
            text = row.get("raw_sentence")
        if text is None:
            text = row.get("premise_text")
        text = str(text or "").strip()
        if not text:
            return None

        pool = self._normalize_pool(row.get("pool", row.get("evidence_assignment", "A")))

        probs = row.get("probs") if isinstance(row.get("probs"), dict) else {}
        derived = row.get("derived") if isinstance(row.get("derived"), dict) else {}

        p_ent = self._safe_float(row.get("p_ent", probs.get("entail", 0.0)), 0.0)
        p_con = self._safe_float(row.get("p_con", probs.get("contra", 0.0)), 0.0)
        p_neu = self._safe_float(
            row.get("p_neu", probs.get("neutral", max(0.0, 1.0 - (p_ent + p_con)))),
            max(0.0, 1.0 - (p_ent + p_con)),
        )
        rel = self._safe_float(row.get("rel", derived.get("rel", p_ent + p_con)), p_ent + p_con)

        if "is_gold" in row:
            is_gold = bool(row.get("is_gold"))
        else:
            is_gold = pool == "A"

        is_fallback = bool(row.get("is_fallback", False))
        page = row.get("page") or row.get("sentence_page")
        line = row.get("line")
        if line is None:
            line = row.get("sentence_line")

        return SentenceEvidenceRecord(
            evidence_id=evidence_id,
            text=text,
            pool=pool,
            p_ent=float(p_ent),
            p_con=float(p_con),
            p_neu=float(p_neu),
            rel=float(rel),
            is_gold=bool(is_gold),
            is_fallback=bool(is_fallback),
            page=(str(page) if page is not None else None),
            line=self._safe_int(line),
        )

    def _parse_active_records_for_index(self, idx: int) -> list[SentenceEvidenceRecord]:
        parsed: list[SentenceEvidenceRecord] = []
        for row_idx, row in enumerate(self.evidence_rows[idx]):
            rec = self._parse_record(row, row_idx=row_idx)
            if rec is None:
                continue
            if rec.pool in {"A", "C"}:
                parsed.append(rec)
            elif self.include_s_pool:
                parsed.append(rec)
        return parsed

    def _get_active_records(self, idx: int) -> list[SentenceEvidenceRecord]:
        return list(self._active_records_cache[idx])

    def _ensure_encoder(self) -> None:
        if self.sentence_encoder is not None and self.claim_sentence_encoder is not None:
            return
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.sentence_model_name)
        if self._encoder_model is None:
            self._encoder_model = AutoModel.from_pretrained(self.sentence_model_name)
            self._encoder_model.to(self._encoder_device)
            self._encoder_model.eval()

    def _encode_sentence(self, text: str) -> np.ndarray:
        if self.sentence_encoder is not None:
            vector = np.asarray(self.sentence_encoder(text), dtype=np.float32).reshape(-1)
        else:
            self._ensure_encoder()
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_len,
            )
            encoded = {k: v.to(self._encoder_device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._encoder_model(**encoded)
            vector = outputs.last_hidden_state[0, 0, :].detach().cpu().numpy().astype(np.float32)

        if vector.size != self.sentence_dim:
            raise ValueError(
                f"Sentence embedding dim mismatch: expected {self.sentence_dim}, got {vector.size}."
            )
        return vector

    def _encode_sentence_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        if self.sentence_encoder is not None:
            vectors = [np.asarray(self.sentence_encoder(text), dtype=np.float32).reshape(-1) for text in texts]
        else:
            self._ensure_encoder()
            encoded = self._tokenizer(
                list(texts),
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_len,
            )
            encoded = {k: v.to(self._encoder_device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._encoder_model(**encoded)
            vectors = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
            vectors = [np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors]
        for vector in vectors:
            if vector.size != self.sentence_dim:
                raise ValueError(
                    f"Sentence embedding dim mismatch: expected {self.sentence_dim}, got {vector.size}."
                )
        return vectors

    def _encode_claim_sentence(self, claim_text: str, sentence_text: str) -> np.ndarray:
        if self.claim_sentence_encoder is not None:
            vector = np.asarray(self.claim_sentence_encoder(claim_text, sentence_text), dtype=np.float32).reshape(-1)
        else:
            self._ensure_encoder()
            encoded = self._tokenizer(
                claim_text,
                sentence_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_len,
            )
            encoded = {k: v.to(self._encoder_device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._encoder_model(**encoded)
            vector = outputs.last_hidden_state[0, 0, :].detach().cpu().numpy().astype(np.float32)

        if vector.size != self.claim_sentence_dim:
            raise ValueError(
                "Claim-sentence embedding dim mismatch: "
                f"expected {self.claim_sentence_dim}, got {vector.size}."
            )
        return vector

    def _encode_claim_sentence_batch(
        self,
        claim_texts: Sequence[str],
        sentence_texts: Sequence[str],
    ) -> list[np.ndarray]:
        if len(claim_texts) != len(sentence_texts):
            raise ValueError("Batch claim/sentence inputs must be the same length.")
        if not claim_texts:
            return []
        if self.claim_sentence_encoder is not None:
            vectors = [
                np.asarray(self.claim_sentence_encoder(claim_text, sentence_text), dtype=np.float32).reshape(-1)
                for claim_text, sentence_text in zip(claim_texts, sentence_texts)
            ]
        else:
            self._ensure_encoder()
            encoded = self._tokenizer(
                list(claim_texts),
                list(sentence_texts),
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_len,
            )
            encoded = {k: v.to(self._encoder_device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._encoder_model(**encoded)
            vectors = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
            vectors = [np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors]
        for vector in vectors:
            if vector.size != self.claim_sentence_dim:
                raise ValueError(
                    "Claim-sentence embedding dim mismatch: "
                    f"expected {self.claim_sentence_dim}, got {vector.size}."
                )
        return vectors

    def _sentence_signature(self, text: str) -> str:
        payload = (
            f"sentence|{self.sentence_model_name}|len={self.max_seq_len}|dim={self.sentence_dim}|{text}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _claim_sentence_signature(self, claim_text: str, sentence_text: str) -> str:
        payload = (
            "claim_sentence|"
            f"{self.sentence_model_name}|len={self.max_seq_len}|dim={self.claim_sentence_dim}|"
            f"{claim_text}||{sentence_text}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_sentence_embedding(self, record: SentenceEvidenceRecord) -> np.ndarray:
        cache_key = record.evidence_id
        expected_signature = self._sentence_signature(record.text)
        cached = self.sentence_embeddings.get(cache_key)
        if cached is not None and self.sentence_embedding_signatures.get(cache_key) == expected_signature:
            return cached
        encoded = self._encode_sentence(record.text)
        self.sentence_embeddings[cache_key] = encoded
        self.sentence_embedding_signatures[cache_key] = expected_signature
        return encoded

    def _get_claim_sentence_embedding(
        self,
        claim_id: str,
        claim_text: str,
        record: SentenceEvidenceRecord,
    ) -> np.ndarray:
        key = (claim_id, record.evidence_id)
        expected_signature = self._claim_sentence_signature(claim_text=claim_text, sentence_text=record.text)
        cached = self.claim_sentence_embeddings.get(key)
        if cached is not None and self.claim_sentence_embedding_signatures.get(key) == expected_signature:
            return cached
        encoded = self._encode_claim_sentence(claim_text=claim_text, sentence_text=record.text)
        self.claim_sentence_embeddings[key] = encoded
        self.claim_sentence_embedding_signatures[key] = expected_signature
        return encoded

    def _write_cache(self, vectors: dict, signatures: dict, path: Path | None) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "vectors": vectors,
            "signatures": signatures,
        }
        with path.open("wb") as fp:
            pickle.dump(payload, fp)

    def precompute_sentence_embeddings(self, force: bool = False) -> None:
        if force:
            self.sentence_embeddings = {}
            self.sentence_embedding_signatures = {}
        pending: list[tuple[str, str, str]] = []
        for idx in range(self.length):
            records = self._get_active_records(idx)
            for record in records:
                signature = self._sentence_signature(record.text)
                if (
                    record.evidence_id in self.sentence_embeddings
                    and self.sentence_embedding_signatures.get(record.evidence_id) == signature
                ):
                    continue
                pending.append((record.evidence_id, signature, record.text))

        pending.sort(key=lambda row: row[0])
        for start in range(0, len(pending), self.precompute_batch_size):
            batch = pending[start: start + self.precompute_batch_size]
            vectors = self._encode_sentence_batch([text for _, _, text in batch])
            for (evidence_id, signature, _text), vector in zip(batch, vectors):
                self.sentence_embeddings[evidence_id] = vector
                self.sentence_embedding_signatures[evidence_id] = signature
        self._write_cache(
            self.sentence_embeddings,
            self.sentence_embedding_signatures,
            self.sentence_cache_path,
        )

    def precompute_claim_sentence_embeddings(self, force: bool = False) -> None:
        if force:
            self.claim_sentence_embeddings = {}
            self.claim_sentence_embedding_signatures = {}
        pending: list[tuple[tuple[str, str], str, str, str]] = []
        for idx in range(self.length):
            claim_id = self.claim_ids[idx]
            claim_text = self.inputs[idx]
            records = self._get_active_records(idx)
            for record in records:
                key = (claim_id, record.evidence_id)
                signature = self._claim_sentence_signature(claim_text=claim_text, sentence_text=record.text)
                if (
                    key in self.claim_sentence_embeddings
                    and self.claim_sentence_embedding_signatures.get(key) == signature
                ):
                    continue
                pending.append((key, signature, claim_text, record.text))

        pending.sort(key=lambda row: row[0])
        for start in range(0, len(pending), self.precompute_batch_size):
            batch = pending[start: start + self.precompute_batch_size]
            claim_vectors = self._encode_claim_sentence_batch(
                [claim_text for _key, _sig, claim_text, _sentence_text in batch],
                [sentence_text for _key, _sig, _claim_text, sentence_text in batch],
            )
            for (key, signature, _claim_text, _sentence_text), vector in zip(batch, claim_vectors):
                self.claim_sentence_embeddings[key] = vector
                self.claim_sentence_embedding_signatures[key] = signature
        self._write_cache(
            self.claim_sentence_embeddings,
            self.claim_sentence_embedding_signatures,
            self.claim_sentence_cache_path,
        )

    def _empty_graph(self) -> Data:
        x = torch.zeros((1, self.sentence_dim), dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, self.claim_sentence_dim + 5), dtype=torch.float32)
        edge_is_gold = torch.zeros((1,), dtype=torch.float32)
        edge_is_support = torch.zeros((1,), dtype=torch.float32)
        edge_is_counter = torch.zeros((1,), dtype=torch.float32)
        edge_is_fallback = torch.ones((1,), dtype=torch.float32)
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_is_gold=edge_is_gold,
            edge_is_support=edge_is_support,
            edge_is_counter=edge_is_counter,
            edge_is_fallback=edge_is_fallback,
        )

    def _build_graph(self, claim_id: str, claim_text: str, records: list[SentenceEvidenceRecord]) -> Data:
        if not records:
            return self._empty_graph()

        node_features: list[torch.Tensor] = []
        claim_edge_features: list[torch.Tensor] = []
        for record in records:
            sent_vec = torch.from_numpy(self._get_sentence_embedding(record)).to(torch.float32)
            claim_vec = torch.from_numpy(
                self._get_claim_sentence_embedding(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    record=record,
                )
            ).to(torch.float32)
            node_features.append(sent_vec)
            claim_edge_features.append(claim_vec)

        n_nodes = len(records)
        edge_indices: list[list[int]] = []
        edge_features: list[torch.Tensor] = []
        edge_is_gold: list[float] = []
        edge_is_support: list[float] = []
        edge_is_counter: list[float] = []
        edge_is_fallback: list[float] = []

        for src_idx, src_record in enumerate(records):
            src_claim_vec = claim_edge_features[src_idx]
            src_pool_id = float(POOL_TO_ID.get(src_record.pool, POOL_TO_ID["A"]))
            src_pv_meta = torch.tensor(
                [src_record.p_ent, src_record.p_con, src_record.p_neu, src_record.rel, src_pool_id],
                dtype=torch.float32,
            )
            for dst_idx in range(n_nodes):
                edge_indices.append([src_idx, dst_idx])

                if src_idx == dst_idx:
                    edge_embedding = src_claim_vec
                else:
                    edge_embedding = 0.5 * (src_claim_vec + claim_edge_features[dst_idx])

                edge_features.append(torch.cat((edge_embedding, src_pv_meta), dim=0))
                is_gold = 1.0 if src_record.is_gold else 0.0
                if src_record.pool == "A":
                    is_support = 1.0
                    is_counter = 0.0
                elif src_record.pool == "C":
                    is_support = 0.0
                    is_counter = 1.0
                else:
                    is_support = is_gold
                    is_counter = 0.0
                edge_is_gold.append(is_gold)
                edge_is_support.append(is_support)
                edge_is_counter.append(is_counter)
                edge_is_fallback.append(1.0 if src_record.is_fallback else 0.0)

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        x = torch.stack(node_features).to(torch.float32)
        edge_attr = torch.stack(edge_features).to(torch.float32)
        edge_is_gold_tensor = torch.tensor(edge_is_gold, dtype=torch.float32)
        edge_is_support_tensor = torch.tensor(edge_is_support, dtype=torch.float32)
        edge_is_counter_tensor = torch.tensor(edge_is_counter, dtype=torch.float32)
        edge_is_fallback_tensor = torch.tensor(edge_is_fallback, dtype=torch.float32)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_is_gold=edge_is_gold_tensor,
            edge_is_support=edge_is_support_tensor,
            edge_is_counter=edge_is_counter_tensor,
            edge_is_fallback=edge_is_fallback_tensor,
        )
