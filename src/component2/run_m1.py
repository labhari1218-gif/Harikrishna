"""Component 2 M1 smoke runner (CPU)."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Dict, List, Sequence
import warnings

import numpy as np

from .anchor_adapter_factkg import FactKGAnchorAdapter
from .anchor_selector import AnchorSelector
from .config import Component2Config, DEFAULT_CONFIG
from .graph_builder import GraphBuilder
from .io_utils import (
    ensure_existing_input_path,
    ensure_positive_max_claims,
    load_claims_jsonl,
    make_run_dir,
    write_json,
    write_jsonl,
)
from .reasoner import HybridMaskedDualStreamReasoner
from .types import ClaimRecord, claim_label_to_index, claim_from_dict, ensure_non_empty_claims


def _anchor_hash(anchors: List[str]) -> str:
    digest = hashlib.sha1()
    digest.update("||".join(anchors).encode("utf-8"))
    return digest.hexdigest()


def _make_anchor_adapter(config: Component2Config) -> FactKGAnchorAdapter:
    return FactKGAnchorAdapter(
        factkg_dir=Path(config.factkg_data_dir),
        split_pickles={
            "train": config.factkg_train_pickle,
            "val": config.factkg_val_pickle,
            "test": config.factkg_test_pickle,
        },
    )


def _synthetic_claims(num_claims: int = 5) -> List[ClaimRecord]:
    """Generate deterministic toy claims for smoke runs."""

    rows: List[Dict[str, object]] = []
    for idx in range(num_claims):
        rows.append(
            {
                "claim_id": f"synthetic_{idx}",
                "claim_text": "Barack Obama was born in Hawaii",
                "label": "SUPPORTED" if idx % 2 == 0 else "REFUTED",
                "entity_set": ["Barack_Obama", "Hawaii"],
                "triples": [
                    {
                        "evidence_id": f"{idx}_e1",
                        "raw_triple": ["Barack_Obama", "birthPlace", "Honolulu"],
                        "pool": "A",
                        "p_ent": 0.72,
                        "p_con": 0.06,
                        "p_neu": 0.22,
                    },
                    {
                        "evidence_id": f"{idx}_e2",
                        "raw_triple": ["Honolulu", "isPartOf", "Hawaii"],
                        "pool": "A",
                        "p_ent": 0.64,
                        "p_con": 0.05,
                        "p_neu": 0.31,
                    },
                    {
                        "evidence_id": f"{idx}_e3",
                        "raw_triple": ["Barack_Obama", "birthPlace", "Kenya"],
                        "pool": "C",
                        "p_ent": 0.05,
                        "p_con": 0.75,
                        "p_neu": 0.20,
                    },
                ],
                "sufficiency": {"esi_geom": 0.5},
            }
        )
    return [claim_from_dict(row) for row in rows]


def build_graphs(
    claims: Sequence[ClaimRecord],
    anchor_selector: AnchorSelector,
    graph_builder: GraphBuilder,
    anchor_adapter: FactKGAnchorAdapter,
    edge_warn_threshold: int,
):
    """Build graphs and labels from claim records."""

    graphs = []
    labels = []
    graph_metadata = []
    perf_guardrail_rows: List[Dict[str, object]] = []
    anchor_source_counts: Dict[str, int] = {"provided": 0, "pickle": 0, "heuristic": 0}
    claims_missing_pickle_mapping = 0
    for claim in claims:
        injection = anchor_adapter.inject_claim(claim)
        claim_with_anchors = injection.claim
        anchor_source = injection.anchor_source
        anchor_source_counts[anchor_source] = anchor_source_counts.get(anchor_source, 0) + 1
        if anchor_source == "heuristic":
            claims_missing_pickle_mapping += 1

        anchors = anchor_selector.select_anchors(
            claim_text=claim_with_anchors.claim_text,
            triples=claim_with_anchors.triples,
            seed_entities=claim_with_anchors.entity_set,
        )
        graph = graph_builder.build(
            claim_id=claim_with_anchors.claim_id,
            triples=claim_with_anchors.triples,
            anchors=anchors,
        )
        if len(graph.edges) > int(edge_warn_threshold):
            warnings.warn(
                "Component 2 numpy mode is running on a large graph "
                f"(claim_id={claim_with_anchors.claim_id}, edges={len(graph.edges)}). "
                "Consider a future torch mode or limit max_claims for faster runs."
            )
            perf_guardrail_rows.append(
                {
                    "claim_id": claim_with_anchors.claim_id,
                    "num_edges": len(graph.edges),
                }
            )

        graphs.append(graph)
        graph_metadata.append(
            {
                "anchor_source": anchor_source,
                "anchor_seed_entities": list(claim_with_anchors.entity_set),
                "anchors_hash": _anchor_hash(anchors),
            }
        )

        label_idx = claim_label_to_index(claim_with_anchors.label)
        if label_idx is not None:
            labels.append(label_idx)

    return (
        graphs,
        labels,
        graph_metadata,
        perf_guardrail_rows,
        claims_missing_pickle_mapping,
        anchor_source_counts,
    )


def run_smoke(
    input_jsonl: Path | None,
    output_root: Path,
    max_claims: int,
    config: Component2Config = DEFAULT_CONFIG,
) -> Path:
    """Run the M1 model on at most max_claims and write logs."""

    ensure_positive_max_claims(max_claims)
    ensure_existing_input_path(input_jsonl)
    run_dir = make_run_dir(output_root=output_root, milestone="m1")

    if input_jsonl is not None:
        claims = load_claims_jsonl(input_jsonl, max_claims=max_claims)
    else:
        claims = _synthetic_claims(num_claims=max_claims)
    ensure_non_empty_claims(claims)

    anchor_selector = AnchorSelector(config=config)
    graph_builder = GraphBuilder(include_pools=("A", "C"))
    reasoner = HybridMaskedDualStreamReasoner(config=config)
    anchor_adapter = _make_anchor_adapter(config=config)

    (
        graphs,
        labels,
        graph_metadata,
        perf_guardrail_rows,
        claims_missing_pickle_mapping,
        anchor_source_counts,
    ) = build_graphs(
        claims=claims,
        anchor_selector=anchor_selector,
        graph_builder=graph_builder,
        anchor_adapter=anchor_adapter,
        edge_warn_threshold=config.numpy_edge_warn_threshold,
    )
    outputs = reasoner.forward_batch(graphs, labels=labels if len(labels) == len(graphs) else None)

    predictions = []
    for graph, per_graph, meta in zip(graphs, outputs["per_graph"], graph_metadata):
        probs = per_graph["probs"]
        pred_idx = int(np.argmax(probs))
        pred_label = "SUPPORTED" if pred_idx == 0 else "REFUTED"

        top_salience = sorted(
            per_graph["salience"],
            key=lambda row: row["salience"],
            reverse=True,
        )[:3]

        predictions.append(
            {
                "claim_id": graph.claim_id,
                "prediction": pred_label,
                "prob_supported": float(probs[0]),
                "prob_refuted": float(probs[1]),
                "anchors": graph.anchors,
                "anchors_hash": str(meta["anchors_hash"]),
                "anchor_source": str(meta["anchor_source"]),
                "anchor_seed_entities": list(meta["anchor_seed_entities"]),
                "top_salience": top_salience,
            }
        )

    run_config = {
        "component2_config": config.to_snapshot_dict(),
        "runtime": {
            "input_jsonl": None if input_jsonl is None else str(input_jsonl),
            "max_claims": max_claims,
            "num_graphs": len(graphs),
            "include_pools": ["A", "C"],
            "output_dir": str(run_dir),
            "reasoner": "HybridMaskedDualStreamReasoner",
            "relation_embeddings": {
                "mode": "fixed_features_numpy",
                "trainable": False,
            },
            "anchors": {
                "source_priority": ["provided", "pickle", "heuristic"],
                "source_counts": anchor_source_counts,
                "claims_missing_pickle_mapping": claims_missing_pickle_mapping,
                "factkg_pickles": {
                    "train": str(Path(config.factkg_data_dir) / config.factkg_train_pickle),
                    "val": str(Path(config.factkg_data_dir) / config.factkg_val_pickle),
                    "test": str(Path(config.factkg_data_dir) / config.factkg_test_pickle),
                },
            },
            "mask_formulas": {
                "support": f"sigmoid({config.mask_sup_scale:g}*p_ent{config.mask_sup_bias:+g})",
                "refute": f"sigmoid({config.mask_ref_scale:g}*p_con{config.mask_ref_bias:+g})",
            },
            "performance_guardrails": {
                "numpy_edge_warn_threshold": int(config.numpy_edge_warn_threshold),
                "num_warnings": len(perf_guardrail_rows),
            },
        },
    }

    write_json(run_dir / "run_config.json", run_config)
    write_json(
        run_dir / "summary.json",
        {
            "num_claims": len(predictions),
            "loss": outputs.get("loss"),
            "anchor_source_counts": anchor_source_counts,
            "claims_missing_pickle_mapping": claims_missing_pickle_mapping,
            "num_perf_guardrail_warnings": len(perf_guardrail_rows),
            "perf_guardrail_claim_ids": [str(row["claim_id"]) for row in perf_guardrail_rows],
            "relation_vocab_size": len(graph_builder.relation_vocab_snapshot()),
        },
    )
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    return run_dir


def _config_with_mask_overrides(
    config: Component2Config,
    *,
    mask_sup_alpha: float | None,
    mask_sup_beta: float | None,
    mask_ref_alpha: float | None,
    mask_ref_beta: float | None,
) -> Component2Config:
    updates = {}
    if mask_sup_alpha is not None:
        updates["mask_sup_scale"] = float(mask_sup_alpha)
    if mask_sup_beta is not None:
        updates["mask_sup_bias"] = float(mask_sup_beta)
    if mask_ref_alpha is not None:
        updates["mask_ref_scale"] = float(mask_ref_alpha)
    if mask_ref_beta is not None:
        updates["mask_ref_bias"] = float(mask_ref_beta)
    if not updates:
        return config
    return replace(config, **updates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Component 2 M1 smoke runner")
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Optional claim JSONL input")
    parser.add_argument("--max-claims", type=int, default=DEFAULT_CONFIG.smoke_max_claims)
    parser.add_argument("--output-root", type=Path, default=Path("logs/component2"))
    parser.add_argument("--mask-sup-alpha", type=float, default=None)
    parser.add_argument("--mask-sup-beta", type=float, default=None)
    parser.add_argument("--mask-ref-alpha", type=float, default=None)
    parser.add_argument("--mask-ref-beta", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _config_with_mask_overrides(
        DEFAULT_CONFIG,
        mask_sup_alpha=args.mask_sup_alpha,
        mask_sup_beta=args.mask_sup_beta,
        mask_ref_alpha=args.mask_ref_alpha,
        mask_ref_beta=args.mask_ref_beta,
    )
    run_dir = run_smoke(
        input_jsonl=args.input_jsonl,
        output_root=args.output_root,
        max_claims=args.max_claims,
        config=config,
    )
    print(f"Component 2 M1 run completed: {run_dir}")


if __name__ == "__main__":
    main()
