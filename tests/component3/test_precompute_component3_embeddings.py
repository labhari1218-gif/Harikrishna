"""Tests for scripts/precompute_component3_embeddings.py helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "precompute_component3_embeddings.py"
    spec = importlib.util.spec_from_file_location("precompute_component3_embeddings_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_splits_and_cache_tag():
    module = _load_module()
    assert module._normalize_splits("train,val,test") == ["train", "val", "test"]
    assert module._cache_tag("roberta-base", 256) == "roberta_base_len256"


def test_manifest_fingerprint_is_stable():
    module = _load_module()
    payload = {"dataset": "factkg", "splits": ["train", "val"], "max_seq_len": 256}
    fp1 = module._manifest_fingerprint(payload)
    fp2 = module._manifest_fingerprint(json.loads(json.dumps(payload)))
    assert fp1 == fp2
