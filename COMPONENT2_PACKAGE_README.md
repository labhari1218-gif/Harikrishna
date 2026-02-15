# Component 2 Package - README

**Package**: `component2_package.tar.gz`  
**Date**: 2026-02-10  
**Version**: M3 (Milestones 0-3 complete)

---

## Contents

### Source Code (`src/component2/`)
- `__init__.py` - Module exports
- `types.py` - Data structures and validation
- `config.py` - Configuration management
- `graph_builder.py` - Entity graph construction
- `anchor_selector.py` - Anchor entity selection
- `bridge_rescue.py` - PPR-based bridge scoring
- `recovery.py` - Policy-aware S→A recovery
- `reasoner.py` - Dual-stream masked GNN reasoner
- `salience.py` - Rationale extraction
- `selective.py` - Abstention and risk-coverage
- `io_utils.py` - I/O utilities
- `run_m1.py` - M1 smoke runner
- `run_m2.py` - M2 smoke runner (with bridge rescue)
- `run_m3.py` - M3 smoke runner (with selective prediction)

### Tests (`tests/component2/`)
- `test_types.py` - Data validation tests
- `test_graph_builder.py` - Graph construction tests
- `test_anchor_selector.py` - Anchor selection tests
- `test_bridge_rescue.py` - PPR and bridge scoring tests
- `test_recovery.py` - Recovery policy tests
- `test_reasoner.py` - Dual-stream reasoner tests
- `test_salience.py` - Rationale extraction tests
- `test_selective.py` - Abstention tests
- `test_io_utils.py` - I/O utility tests
- `test_run_m1.py` - M1 integration test
- `test_run_m2.py` - M2 integration test
- `test_run_m3.py` - M3 integration test

### Documentation (`docs/component2/`)
- `COMPONENT2_SPEC_AND_PLAN.md` - Complete specification
- `COMPONENT2_PLAN.md` - Implementation checklist
- `COMPONENT2_DECISIONS.md` - Design decisions log
- `COMPONENT2_CONFIG.md` - Configuration reference
- `PAPER_REFERENCES.md` - Paper-to-task mapping
- `DEV_NOTES_M0.md` - M0 implementation notes
- `DEV_NOTES_M1.md` - M1 implementation notes (dual-stream reasoner)
- `DEV_NOTES_M2.md` - M2 implementation notes (bridge rescue)
- `DEV_NOTES_M3.md` - M3 implementation notes (selective prediction)

### Root Documentation
- `AGENTS.md` - Development handoff guide

---

## Quick Start

### 1. Extract the Package
```bash
tar -xzf component2_package.tar.gz
cd Fact-or-Fiction
```

### 2. Run Tests
```bash
PYTHONPATH=src python3 -m unittest discover -s tests/component2 -v
```

**Expected**: All 15 tests pass

### 3. Run M3 Smoke Test
```bash
# With synthetic data:
PYTHONPATH=src python3 -m component2.run_m3 --max-claims 5

# With Component 1 logs (requires compatible format):
PYTHONPATH=src python3 -m component2.run_m3 \
  --input-jsonl logs/component1/train/claims.jsonl \
  --max-claims 100
```

---

## Known Issues & Fixes Needed

### ⚠️ Critical Issues (from latest M3 run)

1. **Single-Anchor Problem**
   - 100% of claims have only 1 anchor
   - Breaks connectivity-based recovery
   - **Fix**: Modify `anchor_selector.py` to ensure minimum 2 anchors

2. **PPR Non-Convergence**
   - All PPR runs fail to converge (tolerance too strict)
   - **Fix**: Relax `ppr_tolerance` from 1e-9 to 1e-6 in `config.py`

3. **Reasoner NOT Trained**
   - Current reasoner uses random numpy weights
   - Predictions are essentially random
   - **Fix**: Implement PyTorch training pipeline

See `m3_analysis.md` (in artifacts) for detailed analysis.

---

## Architecture Overview

### Component 2 Flow

```
Input: Claim + Evidence Triples (with PV scores + pools A/S/C)
  ↓
1. Anchor Selection (entities of interest)
  ↓
2. Graph Construction (A∪C for reasoning, A∪S∪C for bridge scoring)
  ↓
3. Bridge Rescue (PPR-based scoring)
  ↓
4. Recovery Decision (if ESI low & connectivity gain possible)
  ↓
5. Dual-Stream Reasoning (support vs refute)
  ↓
6. Selective Prediction (abstain on low-confidence/low-ESI)
  ↓
Output: SUPPORTED/REFUTED/ABSTAIN + rationale + diagnostics
```

### Key Design Patterns

- **Frozen dataclasses** for immutability
- **Policy-aware recovery** (bridge vs rel selection)
- **Dual-stream masking** (separate support/refute streams)
- **Reproducibility** (config snapshots, detailed logging)

---

## Component Dependencies

### Required from Component 1
- `entity_set` field in claims
- `pool` field in evidence triples
- `p_ent`, `p_con`, `p_neu` PV probabilities
- `esi_geom`, `rpi`, `cr` sufficiency metrics

### External Dependencies
- `numpy` (for reasoner computations)
- Python 3.10+

---

## Next Steps

### To Get Results

1. **Fix anchor/PPR issues** (see Known Issues)
2. **Re-implement in PyTorch** (current is numpy inference-only)
3. **Train on FactKG** (86K train examples)
4. **Compare to SOTA** (93.49% baseline)

### For Publishable Paper

1. Complete above
2. Run ablations (with/without PV, recovery, etc.)
3. Add learned recovery controller (Component 3)
4. Comprehensive evaluation

---

## Contact & Maintenance

**Last Updated**: 2026-02-10  
**Status**: M3 complete, needs training implementation  
**Test Coverage**: 15 tests, all passing  

For questions or updates, see `AGENTS.md`.
