# Component 1 Package - README

**Package**: `component1_package.tar.gz`  
**Date**: 2026-02-10  
**Version**: Final (Production-ready)

---

## Contents

### Source Code (`src/component1/`)
- `__init__.py` - Module exports
- `pv.py` - Premise Validation (PV) using NLI models
- `esm.py` - Evidence Selection & Masking (ESM) 
- `evidence.py` - Evidence pool management (A/S/C)
- `sufficiency_metrics.py` - ESI, RPI, CR, connectivity metrics
- `cache.py` - PV prediction caching
- `logging_utils.py` - Structured logging (claims.jsonl, pairs.jsonl)

### Tests (`tests/component1/`)
- `test_logging_utils.py` - Logging format tests
- `test_summarize_component1.py` - Summary generation tests

### Runner Script
- `scripts/run_component1_pv_esm.py` - Main pipeline script

### Documentation

#### Core Docs
- `building.md` - **START HERE**: Comprehensive development history
- `COMPONENT1_CODEX_FINAL_README.md` - Final implementation summary
- `COMPONENT1_INPUT_OUTPUT_EXPLAINED.md` - Data format guide
- `COMPONENT1_OUTPUT_ANALYSIS.md` - Output interpretation
- `COMPONENT1_PRESENTATION.md` - Overview for presentations
- `COMPONENT1_REVIEW_README.md` - Code review checklist

#### Development Docs (`docs/`)
- `CODEX_FIXES_COMPONENT1_V4.md` - V4 fixes log
- `CODEX_FIXES_POST_V4_REVIEW.md` - Post-V4 hardening
- `CODEX_REPORT_V4_HARDENING.md` - Hardening report
- `CODEX_RUNBOOK_COMPONENT1.md` - Runbook
- `HF_MODEL_STORAGE.md` - HuggingFace model guide
- `component1_reconnaissance.md` - Initial analysis

#### Methodology Docs
- `SUBGRAPH_RETRIEVAL_METHODOLOGY.md` - Retrieval explanation
- `CHATGPT_VALIDATION_PROMPT.md` - Validation guide
- `DEVLOG_COMPONENT1.md` - Development log

---

## Quick Start

### 1. Extract the Package
```bash
tar -xzf component1_package.tar.gz
cd Fact-or-Fiction
```

### 2. Install Dependencies
```bash
# Requires existing conda environment with:
# - transformers, torch, datasets, networkx, tqdm
# See building.md for complete setup
```

### 3. Run Component 1 Pipeline
```bash
# Full pipeline on 100 claims:
python3 scripts/run_component1_pv_esm.py \
  --split train \
  --limit_claims 100 \
  --pair_log_mode all \
  --out_dir logs/component1 \
  --model_name "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli" \
  --device cuda \
  --batch_size 8

# Cache-only mode (regenerate logs from cache):
python3 scripts/run_component1_pv_esm.py \
  --split train \
  --limit_claims 100 \
  --pair_log_mode all \
  --cache_only \
  --out_dir logs/component1_regenerated
```

---

## Output Format

### Claims Log (`claims.jsonl`)
Each claim has:
- `claim_id`, `claim_text`, `label`
- `entity_set` / `Entity_set` - Retrieval seed entities
- `counts` - Pool sizes {A, S, C}
- `sufficiency` - ESI, coverage, connectivity metrics
- `counter_retention` - CR, max contra scores
- `recovery` - RPI metrics
- `top_evidence` - Best supporting/contradicting triples

### Pairs Log (`pairs.jsonl`)
Each evidence triple has:
- `claim_id`, `evidence_id`
- `raw_triple` - [subject, relation, object]
- `pool` / `evidence_assignment` - "A", "S", or "C"
- `probs` - {entail, contra, neutral} from PV
- `premise_text`, `hypothesis_text` - Verbalized triple

---

## Key Features

### 1. Premise Validation (PV)
- Uses DeBERTa-v3-large NLI model
- Scores each triple for: entailment, contradiction, neutral
- Caches predictions for fast re-runs

### 2. Evidence Selection & Masking
- Assigns triples to pools:
  - **A (Active)**: High relevance (rel > threshold)
  - **S (Suspended)**: Low relevance
  - **C (Counter)**: Contradiction evidence
- Ensures counter-evidence is preserved

### 3. Sufficiency Metrics
- **ESI** (Evidence Sufficiency Index): geometric/product of PV scores
- **Coverage**: Fraction of claim entities covered
- **Connectivity**: Anchor reachability in graph
- **RPI** (Recovery Potential Index): Connectivity gain from S
- **CR** (Counter Retention): Presence of contradictions

---

## Known Configuration

### PV Model
```python
model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
verbalizer_id = "v3"
batch_size = 8 (adjust for GPU memory)
```

### ESM Thresholds
```python
rel_threshold = 0.5  # A vs S cutoff
neutral_guard = 0.8  # High neutral → S
counter_threshold = 0.15  # Min p_con for C
```

### Output Modes
- `pair_log_mode = "all"` - Logs all triples (A/S/C)
- `pair_log_mode = "active"` - Logs only A/C

---

## Integration with Component 2

Component 2 expects Component 1 logs with:
- ✅ `entity_set` field in each claim row
- ✅ `pool` field in each pair row
- ✅ `p_ent`, `p_con`, `p_neu` probabilities
- ✅ Sufficiency metrics (ESI, RPI, CR)

**Recent Fix (2026-02-10):**
Added `entity_set` and `pool` fields to logging for Component 2 compatibility.

---

## Validation & Testing

### Test Suite
```bash
# Run logging tests:
python3 -m unittest tests.component1.test_logging_utils -v

# Run summary tests:
python3 -m unittest tests.component1.test_summarize_component1 -v
```

### Validation Checklist
- [ ] PV cache hit rate > 95%
- [ ] All claims have A/S/C counts
- [ ] ESI values in [0, 1]
- [ ] Counter evidence preserved (CR > 0 when contradictions exist)
- [ ] Logs match expected schema

---

## Performance

### Throughput
- **With cache**: ~500-1000 claims/min
- **Without cache**: ~50-100 claims/min (depends on GPU)

### Resource Usage
- **GPU Memory**: ~4-6GB (DeBERTa-v3-large)
- **Disk**: ~10MB per 1000 claims (logs)
- **Cache**: ~5MB per 1000 claim-triple pairs

---

## Contact & Maintenance

**Last Updated**: 2026-02-10  
**Status**: Production-ready  
**Test Coverage**: 2 test files  

For comprehensive documentation, see `building.md`.
