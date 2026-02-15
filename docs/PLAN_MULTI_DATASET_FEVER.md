# Plan: Multi-Dataset Expansion (Starting with FEVER)

Date: 2026-02-15  
Scope: Extend current FactKG-first pipeline to support FEVER and future datasets without breaking current training/eval.

---

## 0. Implementation Status (2026-02-15)

Implemented in this cloned workspace:

1. FEVER dataset preparation/download script
- `scripts/prepare_fever_dataset.py`
- Downloads official FEVER JSONL files and builds normalized pickles under `data/fever/`.
- Supports binary/3-way label mapping and deterministic train/val split from FEVER train.

2. FEVER Component 1 runner (sentence evidence)
- `scripts/run_fever_component1.py`
- Runs PV+ESM over FEVER sentence evidence and writes Component-3-compatible pair logs under `logs/component1_fever/<split>/`.

3. FEVER Component 3 graph dataset + loader dispatch
- `src/component3/pv_dataset_fever.py`
- `src/component3/run_train.py` now supports `--dataset-name fever` while keeping `factkg` default.
- FEVER pair-log ingestion added in `src/component3/c1_pairs_loader.py`.

4. Evaluation dispatch
- `evaluate.py` now accepts `dataset_name` and adds FEVER-friendly macro/micro/per-class metrics while preserving FactKG behavior.

5. Smoke + tests
- FEVER prep smoke run completed with `--max-claims-per-split 5`.
- Targeted tests passed: `24 passed`.

Quick commands:

```bash
# 1) Prepare FEVER artifacts (full run)
conda run --no-capture-output -n fact_check_env python scripts/prepare_fever_dataset.py --download-db

# 2) Run FEVER Component 1 pair logs
conda run --no-capture-output -n fact_check_env python scripts/run_fever_component1.py --split train --pair-log-mode all
conda run --no-capture-output -n fact_check_env python scripts/run_fever_component1.py --split val --pair-log-mode all
conda run --no-capture-output -n fact_check_env python scripts/run_fever_component1.py --split test --pair-log-mode all

# 3) Train Component 3 on FEVER
conda run --no-capture-output -n fact_check_env python -m component3.run_train \
  --dataset-name fever \
  --fever-data-root data/fever \
  --fever-component1-logs-root logs/component1_fever \
  --batch-size 8 --max-seq-len 256
```

---

## 1. Current State Snapshot

From current workspace state:

- Component roadmap through C7 is marked complete in `Fact-or-Fiction/PLAN.md`.
- Active full training is running for:
  - `run_id=t33_full_seed57_rerun2_20260215_1321`
  - entrypoint: `python -m component3.run_train ...`
- Current data/eval stack is FactKG-coupled in multiple places:
  - `datasets.get_df()` loads `data/factkg/*.pickle`
  - labels are parsed from `Label[0]` (binary)
  - `evaluate.py` assumes FactKG `types` metadata (`existence`, `substitution`, `multi hop`, `multi claim`, `negation`, `single hop`)
  - graph/evidence flow assumes KG triples and A/S/C pools

Implication:

- FEVER support should be added via an explicit dataset abstraction layer, not by patching FactKG paths inline.

---

## 2. FEVER Fit and Main Gaps

## FEVER properties (target support)

- Claim verification with labels typically in:
  - `SUPPORTS`
  - `REFUTES`
  - `NOT ENOUGH INFO` (NEI)
- Evidence is sentence-level (Wikipedia), not KG triples.
- Multi-hop evidence exists as sets of sentences.

## Gaps vs current pipeline

1. Data schema mismatch
- Current: `(Sentence, Label, types, subgraph/walked triples)`
- FEVER: `(claim, label, evidence sentence sets, page/sentence ids)`

2. Label-space mismatch
- Current default training/eval is binary.
- FEVER requires either:
  - 3-way classifier (`SUPPORTS/REFUTES/NEI`), or
  - NEI filtering + binary mode (for an initial compatibility milestone).

3. Evidence representation mismatch
- Current C1/C2/C3 pipeline expects triples with PV scores and A/S/C partitioning.
- FEVER uses text evidence; requires a sentence-to-graph adapter layer.

4. Metric mismatch
- Current evaluation uses FactKG claim-type buckets from `types`.
- FEVER requires FEVER-style metrics and evidence-recall metrics.

---

## 3. Execution Strategy

Guiding rule:

- Keep FactKG behavior fully backward compatible.
- Add FEVER as a parallel dataset path behind a dataset registry.

---

## 4. Phased Plan

## Phase A: Dataset Abstraction (No behavior change for FactKG)

Goal:

- Introduce a common dataset interface so `train.py`, `evaluate.py`, and component runners can consume either FactKG or FEVER.

Tasks:

1. Add dataset registry and config
- New files:
  - `src/data/registry.py`
  - `src/data/contracts.py`
- Add `dataset_name` config key (default `factkg`).

2. Refactor existing FactKG loaders behind interface
- Wrap current `get_df/get_subgraphs/get_dataloader` as FactKG adapter.

3. Remove hard-coded FactKG assumptions from eval entrypoints
- `evaluate.py`: accept dataset-specific metadata provider instead of direct `get_df("test")`.

Acceptance:

- Existing FactKG commands run unchanged and produce identical outputs.

---

## Phase B: FEVER Ingestion + Normalization

Goal:

- Build FEVER preprocessing pipeline producing deterministic train/val/test artifacts in project format.

Tasks:

1. FEVER raw import
- New script:
  - `scripts/prepare_fever_dataset.py`
- Output:
  - `data/fever/fever_train.pkl`
  - `data/fever/fever_dev.pkl`
  - `data/fever/fever_test.pkl` (if labels available) or held-out format.

2. Unified sample schema
- Standardized columns:
  - `claim_id`
  - `Sentence` (claim text)
  - `Label`
  - `evidence_rows` (sentence IDs/text for FEVER)
  - `metadata` (dataset-specific fields)

3. Label policy
- Decide one of:
  - `fever_3way` (recommended final)
  - `fever_binary` (exclude NEI as interim)

Acceptance:

- FEVER dataset can be loaded through same top-level dataset API.

---

## Phase C: Evidence Path for FEVER (C1-Compatible)

Goal:

- Reuse Component 1 PV scoring/partitioning with FEVER sentence evidence.

Tasks:

1. FEVER evidence verbalization adapter
- For FEVER, evidence is already text; bypass triple verbalization.

2. PV scoring over sentence evidence
- Reuse NLI scoring (premise=evidence sentence, hypothesis=claim).

3. ESM partitioning for sentence items
- Keep A/S/C semantics with sentence-level evidence IDs.

4. Pair-log format compatibility
- Produce FEVER pair logs in a schema compatible with C3 ingestion (`claim_id`, `evidence_id`, `pool`, probs, rel).

Acceptance:

- FEVER claims generate C1-like pair logs and coverage reports.

---

## Phase D: Graph Construction for FEVER (C2/C3 bridge)

Goal:

- Provide graph objects for C2/C3 when evidence units are sentences.

Options:

1. Sentence-node graph (recommended first)
- Nodes = evidence sentences
- Edges = lexical/entity overlap, TF-IDF/BM25 links, or retrieval links
- Edge attrs include PV and claim-conditioned features

2. Entity-projected graph (later)
- Extract entities from FEVER evidence and build pseudo-triples/entity graph.

Tasks:

1. New FEVER graph builder:
- `src/fever/graph_builder.py`

2. FEVER dataset class mirroring C3 dataset contract:
- `src/fever/pv_dataset_fever.py`

3. Ensure `PV_QAGNN` accepts FEVER graph tensors with same edge attr tail conventions where needed.

Acceptance:

- `component3.run_train` can run with `dataset_name=fever` in smoke mode end-to-end.

---

## Phase E: Evaluation and Metrics

Goal:

- Add FEVER-specific evaluation while retaining FactKG metrics.

Tasks:

1. Metric adapters
- `src/eval/factkg_metrics.py`
- `src/eval/fever_metrics.py`

2. FEVER outputs
- Claim label metrics:
  - accuracy, macro-F1, per-class precision/recall/F1
- Evidence metrics:
  - evidence recall@k / strict FEVER score variant (if full evidence sets available)

3. Runtime switch
- `evaluate.py` dispatch by dataset name.

Acceptance:

- One command can evaluate FactKG or FEVER via config only.

---

## Phase F: Experiment Matrix

## Minimum FEVER milestones

1. FEVER smoke
- 1k train subset, binary mode, C1+C3 forward path.

2. FEVER baseline
- BERT-only and QA-GNN baseline.

3. FEVER PV-QAGNN
- full C3 model with two-stage schedule.

4. FEVER robustness
- component6/router-style hard/easy routing can be postponed until core FEVER path stabilizes.

## Recommended seeds

- Keep 3-seed setup: `42, 1337, 2026`

---

## 5. Proposed File-Level Changes

Likely new files:

- `src/data/contracts.py`
- `src/data/registry.py`
- `src/data/factkg_adapter.py`
- `src/data/fever_adapter.py`
- `src/fever/graph_builder.py`
- `src/fever/pv_dataset_fever.py`
- `src/eval/factkg_metrics.py`
- `src/eval/fever_metrics.py`
- `scripts/prepare_fever_dataset.py`
- `scripts/run_fever_component1.py`
- `scripts/run_fever_component3_train.sh`

Likely modified files:

- `datasets.py` (or replaced by adapter entrypoint)
- `evaluate.py`
- `src/component3/run_train.py` (dataset dispatch)
- `src/component3/run_validation.py` (dataset-aware eval mode)
- `constants.py` (dataset path registry)

---

## 6. Risk Controls

1. Do not mix FactKG and FEVER assumptions in same loader class.
2. Keep current FactKG default path as strict default until FEVER pipeline passes smoke tests.
3. Add schema validation at dataset boundary (required columns and label set).
4. Preserve `runs/<run_id>/config.yaml` with explicit `dataset_name`.

---

## 7. Immediate Next Steps (Ordered)

1. Implement Phase A (dataset abstraction) with zero regression on FactKG.
2. Add FEVER preprocessing script and normalized artifacts (Phase B).
3. Build FEVER C1 PV + ESM path (Phase C) and validate pair-log compatibility.
4. Add FEVER graph adapter and C3 smoke training (Phase D).
5. Add FEVER metrics and 3-seed evaluation harness (Phase E).
