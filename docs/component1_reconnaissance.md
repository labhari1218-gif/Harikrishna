# Component 1: Codebase Reconnaissance

**Date**: 2026-02-02  
**Purpose**: Document existing FactKG data structures and integration points for Component 1 (PV Scorer + ESM)

---

## Files Inspected

- `datasets.py`: Data loading utilities
- `constants.py`: Path and filename constants
- `data/factkg/*.pickle`: Claim data files
- `data/subgraphs/subgraphs_direct_filled_*.pkl`: Retrieved subgraph data

---

## FactKG Data Structure

### Claim Data (`data/factkg/factkg_{train|dev|test}.pickle`)

**Loading**: Via `get_df(data_split)` in `datasets.py` (lines 19-48)

**Structure**: Pandas DataFrame with columns:
```python
Sentence        str       # Claim text, e.g., "He had a successor named John E. Beck as well."
Label           list[bool]  # [True] or [False] (always single-element list)
Entity_set      list[str] # Detected entities, e.g., ['John_E._Beck']
Evidence        dict      # (train/val only) Entity -> relation paths
types           list[str] # Claim taxonomy, e.g., ['coll:model', 'existence']
```

**Dataset sizes**:
- Train: 86,367 claims
- Val (dev): 13,266 claims
- Test: 9,041 claims

**Label extraction**: `int(label[0])` → 0 (False) or 1 (True)

**Claim ID**: Use DataFrame index (0-based) as `claim_id`

---

## Subgraph Data Structure

### Subgraph Files (`data/subgraphs/subgraphs_direct_filled_{train|val|test}.pkl`)

**Loading**: Via `get_subgraphs(data_split, subgraph_type)` in `datasets.py` (lines 51-91)

**Structure**: Pandas DataFrame with columns:
```python
subgraph  dict   # {entity: [relations...]}  (not normalized to triples yet)
walked    dict   # {'connected': [...], 'walkable': [...]}
```

**Triple format** in `walked['walkable']`:
```python
# Each item is a list: [subject, relation, object]
['John_E._Beck', '~successor', 'Edward_E._Willard']
['Mobyland', 'foundingYear', '"2006"']
```

**Key observations**:
- `walked['walkable']` contains the actual triple lists
- Entities use underscores: `John_E._Beck`
- Relations may have tilde prefix for inverse: `~successor`
- Literal values are quoted: `"2006"`
- `walked['connected']` is mostly empty (0 items in samples inspected)

**Usage in existing code** (lines 336-352 in `datasets.py`):
```python
walked = self.subgraphs[idx]
if walked['connected'] != []:
    subgraph = walked['connected']
else:
    subgraph = walked['walkable']  # Most common path
```

---

## Entity/Relation Representation

### Entities
- Format: DBpedia URI tail with underscores
- Example: `W._Haydon_Burns`, `Mobyland`
- Retrieval: From `Entity_set` column in claim data

### Relations
- Format: Property name (may be inverse)
- Inverse prefix: `~` (e.g., `~successor` means "is successor of")
- Examples:
  - `successor`
  - `~predecessor`
  - `foundingYear`

### Verbalization for NLI
Component 1 will need to convert triples to natural language:
```python
# Input triple
['John_E._Beck', '~successor', 'Edward_E._Willard']

# Verbalize to premise text for NLI
"John E. Beck is the successor of Edward E. Willard"
```

**Requirements for `evidence_to_text()` helper**:
1. Replace underscores with spaces in entities
2. Strip tilde and adjust relation semantics
3. Handle quoted literals
4. Produce readable premise text

---

## Integration Points for Component 1

### Hook Strategy: NEW Scripts/Modules Only

**CRITICAL**: Do NOT modify existing pipeline. Component 1 operates independently.

### Data Access (Read-Only)
```python
# Load claim data
from datasets import get_df
df_claims = get_df("train")  # or "val", "test"

# Load subgraph data
from datasets import get_subgraphs
df_subgraphs = get_subgraphs("train", "direct_filled")

# Merge by index
for idx in range(len(df_claims)):
    claim_text = df_claims.iloc[idx]["Sentence"]
    label = int(df_claims.iloc[idx]["Label"][0])
    triples = df_subgraphs.iloc[idx]["walked"]["walkable"]
    
    # Component 1 processing...
```

### Output Artifacts (New Paths)
- **claim_id**: Use `{split}_{idx}` (e.g., `train_0`, `val_42`)
- **Outputs**: Write to `logs/component1/{split}/` (never overwrite training data)
- **Augmented pkl** (optional): Save as NEW file, e.g., `data/subgraphs/subgraphs_direct_filled_train_component1.pkl`

### Environment
- **Conda env**: `fact_check_env` (verified working with pandas, torch, transformers)
- **Python**: `python3` via `conda run -n fact_check_env`

---

## Component 1 Workflow (per claim)

```
Input: claim_idx from split
↓
1. Load claim_text, label from df_claims.iloc[idx]
2. Load triples from df_subgraphs.iloc[idx]["walked"]["walkable"]
3. Build EvidenceItem objects per triple:
   - evidence_id = f"{split}_{idx}_triple_{ti}"
   - content = {"triple": [s, r, o]}
   - verbalize to text
4. PV score (batched, cached):
   - premise = evidence_to_text(triple)
   - hypothesis = claim_text
   - p_entail, p_contra, p_neutral = PVScorer.score(...)
5. ESM partition (A/S/C):
   - Apply policy (min_A, max_A, max_C, contra_tau)
6. Log to JSONL:
   - claim-level: logs/component1/{split}/claims.jsonl
   - pair-level: logs/component1/{split}/pairs.jsonl (if enabled)
↓
Output: PV results + A/S/C membership
```

---

## Verification Plan (for Implementation Plan)

### Unit Tests
1. **ESM invariants**: `tests/component1/test_esm.py`
   - Test disjoint A/S/C sets
   - Test |A| >= min_A when possible
   - Test counter protection (if contra exists, C not empty)

2. **PV label mapping**: `tests/component1/test_pv.py`
   - Test entail/neutral/contra index mapping
   - Test fallback for unknown label structures

3. **Evidence text conversion**: `tests/component1/test_evidence.py`
   - Test triple verbalization
   - Test URI tail extraction
   - Test tilde handling

**Run**: `conda run -n fact_check_env pytest tests/component1/ -q`

### Integration Test
Run Component 1 on **100 claims** from train split:
```bash
conda run -n fact_check_env python3 scripts/run_component1_pv_esm.py \
  --split train \
  --limit_claims 100 \
  --pair_log_mode none \
  --cache_dir cache/pv_sqlite
```

**Verify outputs**:
- `logs/component1/train/claims.jsonl` exists (100 lines)
- Starved_A rate = 0 (min_A guard working)
- Cache stats printed (hits/misses)
- Summary created via `scripts/summarize_component1.py --split train`

### Baseline Training Unchanged
**Run existing training** to ensure no interference:
```bash
conda run -n fact_check_env python3 train.py  # (check existing args if needed)
```
Verify it runs without import errors or path conflicts.

---

## Next Steps

1. Skip STEP 1 (paper reading) since we have domain knowledge
2. Mark STEP 2 complete
3. Create implementation_plan.md with detailed changes
4. Implement STEP 3 modules
5. Implement STEP 4/5 scripts
6. Run STEP 6 tests

---
