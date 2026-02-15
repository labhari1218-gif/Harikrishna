# Component 1 Output Analysis for Component 2 Development

**Generated**: 2026-02-09  
**Purpose**: Detailed documentation of Component 1 outputs for Component 2 development  
**Datasets**: Train (86,367 claims), Val (13,266 claims)

---

## Overview

Component 1 generates two types of logs:

1. **`claims.jsonl`**: Per-claim metrics and A/S/C assignments
2. **`pairs.jsonl`**: Per-triple PV scores and box assignments (Active/Suspended/Counter)

---

##  Output 1: claims.jsonl (Per-Claim Summary)

**Location**: `logs/component1/train/claims.jsonl` and `logs/component1/val/claims.jsonl`

**Format**: One JSON object per line (JSONL)

### What Each Claim Entry Contains

```json
{
  "claim_id": "train_0",
  "claim_text": "He had a successor named John E. Beck as well.",
  "label": "TRUE",
  
  "counts": {
    "A": 12,    // Number of triples in Active set
    "S": 0,     // Number of triples in Suspended set
    "C": 1      // Number of triples in Counter set
  },
  
  "sufficiency": {
    "esi_geom": 0.43,              // PRIMARY METRIC: Evidence Sufficiency Index
    "coverage_A": 1.0,              // % of claim entities found in Active
    "connectivity_A": 1.0,          // Are entities connected in graph?
    "neutral_rate_A": 0.915         // % of uninformative evidence
  },
  
  "counter_retention": {
    "has_counter": true,            // Does this claim have contradictions?
    "counter_kept": 1,              // How many contradictions in C?
    "cr_at_5": 0.2,                 // Did we keep top-5 contradictions?
    "max_contra_C": 0.99,           // Strongest contradiction in C
    "counter_in_A": 0,              // Contradictions that leaked into A
    "max_contra_A": 0.006           // Highest p_contra in Active set
  },
  
  "pool_stats": {
    "pool_size_total": 13,          // Total evidence triples
    "mean_rel": 0.155,              // Average relevance across all triples
    "mean_pol": -0.002              // Average polarity (support vs contradict)
  },
  
  "top_evidence": {
    "by_rel": ["train_0_triple_2", "train_0_triple_0", ...],  // Ranked by relevance
    "by_contra": ["train_0_triple_6", ...]                     // Ranked by contradiction
  }
}
```

---

## 🔍 Output 2: pairs.jsonl (Per-Triple Details)

**Location**: `logs/component1/train/pairs.jsonl` and `logs/component1/val/pairs.jsonl`

**Format**: One JSON object per line (JSONL)

**Mode**: `A_and_C` (only logs Active and Counter evidence)

### What Each Triple Entry Contains

```json
{
  "claim_id": "train_0",
  "evidence_id": "train_0_triple_2",
  "evidence_assignment": "A",              // Which box: A, S, or C
  
  "raw_triple": [
    "John_E._Beck",                        // Subject
    "successor",                           // Relation
    "Person_X"                             // Object
  ],
  
  "premise_text": "John E. Beck has successor Person X.",  // Verbalized triple
  "hypothesis_text": "He had a successor named John E. Beck as well.",  // Claim
  
  "model_name": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
  "verbalizer_id": "v3",
  
  "probs": {
    "entail": 0.88,                        // Probability of entailment
    "contra": 0.01,                        // Probability of contradiction
    "neutral": 0.11                        // Probability of neutral
  },
  
  "logits": [-2.45, 2.13, -1.87],          // Raw model outputs
  
  "derived": {
    "rel": 0.89,                           // Relevance = entail + contra (1 - neutral)
    "pol": 0.87                            // Polarity = entail - contra (in [-1, 1])
  }
}
```

---

## 📋 Understanding A/S/C Assignment Logic

### Selection Order (CRITICAL)

Component 1 partitions evidence in this **specific order**:

```
1. Counter (C) selected FIRST
   ↓
2. Active (A) selected SECOND (from remainder after C)
   ↓
3. Suspended (S) gets everything left
```

### 📕 Box C (COUNTER) - Selected FIRST

**Selection Rule**:
```python
triple goes to C IF:
  probs.contra >= 0.6  (contradiction threshold)
  
Maximum: 5 triples (configurable via --max_C)
Sort by: contradiction probability (highest first)
```

**Example Triples**:
```json
{
  "evidence_id": "train_0_triple_6",
  "raw_triple": ["John_Beck", "never_held_office", "true"],
  "premise_text": "John Beck has never held office true.",
  "probs": {"entail": 0.01, "contra": 0.99, "neutral": 0.00},
  "derived": {"rel": 1.00, "pol": -0.98},
  "evidence_assignment": "C"
}
// WHY IN C? probs.contra (0.99) >= threshold (0.6)
```

### 📗 Box A (ACTIVE) - Selected SECOND

**Selection Rule**:
```python
triple goes to A IF:
  NOT in C (Counter selected first)
  AND ranked in top 20 by relevance (rel = entail + contra)
  
Minimum: 5 triples (--min_A, guaranteed unless pool too small)
Maximum: 20 triples (--max_A)
Sort by: relevance (highest first)
```

**Example Triples**:
```json
{
  "evidence_id": "train_0_triple_2",
  "raw_triple": ["Person_X", "successor", "John_E._Beck"],
  "premise_text": "Person X has successor John E. Beck.",
  "probs": {"entail": 0.88, "contra": 0.01, "neutral": 0.11},
  "derived": {"rel": 0.89, "pol": 0.87},
  "evidence_assignment": "A"
}
// WHY IN A? 
//   - NOT in C (probs.contra = 0.01 < 0.6)
//   - High relevance (rel = 0.89)
//   - Within top 20 by relevance
```

### 📘 Box S (SUSPENDED) - Gets Remainder

**Selection Rule**:
```python
triple goes to S IF:
  NOT in C
  AND NOT in A (ranked outside top 20 by relevance)
```

**Note**: With `--pair_log_mode A_and_C`, Suspended triples are NOT logged to `pairs.jsonl`

---

## 📊 Key Metrics Explained

### ESI_geom (Evidence Sufficiency Index)

**Formula**:
```
ESI_geom = ∛(coverage × connectivity × (1 - neutral_rate))
```

**Range**: 0 to 1 (higher = better sufficiency)

**Interpretation**:
- **ESI < 0.3**: Evidence starvation (insufficient)
- **ESI 0.3-0.6**: Moderate sufficiency
- **ESI > 0.6**: Good sufficiency

**Example**:
```
Claim: "He had a successor named John E. Beck"
Coverage: 1.0 (all entities "John", "Beck" found)
Connectivity: 1.0 (entities connected in graph)
Neutral Rate: 0.915 (91.5% of evidence is uninformative)

ESI_geom = ∛(1.0 × 1.0 × (1 - 0.915))
         = ∛(1.0 × 1.0 × 0.085)
         = ∛0.085
         = 0.43
```

**Diagnosis**: Moderate sufficiency - structurally good (coverage/connectivity), but semantically weak (high neutral rate)

---

### Relevance (rel)

**Formula**: `rel = p_entail + p_contra = 1 - p_neutral`

**Range**: 0 to 1 (higher = more relevant)

**Interpretation**:
- How much does this triple relate to the claim?
- Ignores whether it supports or contradicts
- Used for Active set selection (want high-relevance evidence)

---

### Polarity (pol)

**Formula**: `pol = p_entail - p_contra`

**Range**: -1 to +1
- **+1**: Strong support (entailment)
- **0**: Neutral
- **-1**: Strong contradiction

**Interpretation**:
- Distinguishes support vs contradiction
- Used for understanding evidence stance
- NOT primary selection criterion (relevance is)

---

### Coverage

**Formula**:
```
coverage = |claim_entities ∩ entities_in_Active| / |claim_entities|
```

**Range**: 0 to 1 (higher = better)

**Example**:
```
Claim entities: {"John", "Beck"}
Entities in Active: {"John_Beck", "Person_X", "1850"}
Normalized overlap: {"John", "Beck"} (both found)

Coverage = 2/2 = 1.0 (100%)
```

---

### Connectivity

**Formula**:
```
# For each pair of claim entities, check if connected in graph
# Graph edges from Active evidence triples
connectivity = (connected_pairs) / (total_pairs)
```

**Range**: 0 to 1 (higher = better)

**Special Cases**:
- If < 2 claim entities: `connectivity = 1.0` (trivially connected)

**Example**:
```
Claim entities: {"John", "Beck"}
Graph from Active:
  John_Beck --successor--> Person_X
  John_Beck --elected--> 1850

Both "John" and "Beck" are in same entity "John_Beck"
→ Connected = TRUE
→ Connectivity = 1.0
```

---

## 🔍 How to Use This Data for Component 2

### Step 1: Read claims.jsonl

```python
import json

claims = []
with open('logs/component1/train/claims.jsonl') as f:
    for line in f:
        claims.append(json.loads(line))

# Find claims with low ESI (evidence gaps)
low_esi_claims = [c for c in claims if c['sufficiency']['esi_geom'] < 0.3]

# Find claims with contradictions
claims_with_counter = [c for c in claims if c['counter_retention']['has_counter']]
```

### Step 2: Read pairs.jsonl for Detailed Analysis

```python
import json

# Build evidence map: claim_id -> list of triples with assignments
evidence_map = {}

with open('logs/component1/train/pairs.jsonl') as f:
    for line in f:
        pair = json.loads(line)
        claim_id = pair['claim_id']
        
        if claim_id not in evidence_map:
            evidence_map[claim_id] = []
        
        evidence_map[claim_id].append({
            'evidence_id': pair['evidence_id'],
            'triple': pair['raw_triple'],
            'verbalized': pair['premise_text'],
            'assignment': pair['evidence_assignment'],  # A, S, or C
            'rel': pair['derived']['rel'],
            'pol': pair['derived']['pol'],
            'p_entail': pair['probs']['entail'],
            'p_contra': pair['probs']['contra'],
            'p_neutral': pair['probs']['neutral']
        })

# Now for any claim, you can see all its evidence:
claim_id = "train_0"
for evidence in sorted(evidence_map[claim_id], key=lambda x: x['rel'], reverse=True):
    print(f"{evidence['assignment']}: {evidence['verbalized']}")
    print(f"   rel={evidence['rel']:.3f}, pol={evidence['pol']:.3f}")
```

### Step 3: Analyze Assignment Logic

For each claim, you can see:
1. **Which triples went to C**: `probs.contra >= 0.6`
2. **Which triples went to A**: High `rel`, not in C, within top-20
3. **Which triples went to S**: Low `rel`, not in C, outside top-20

---

## 📁 Expected Output Files

After Component 1 completes, you will have:

```
logs/component1/
├── train/
│   ├── claims.jsonl       ← 86,367 lines (one per claim)
│   ├── pairs.jsonl        ← ~1.3M lines (Active + Counter evidence only)
│   └── summary.json       ← Aggregate statistics (generated later)
└── val/
    ├── claims.jsonl       ← 13,266 lines
    ├── pairs.jsonl        ← ~200K lines
    └── summary.json
```

---

## ⏱️ Processing Status

**Current Run**:
- Split: Train (86,367 claims)
- Model: DeBERTa-v3-large-mnli-fever-anli-ling-wanli
- Device: CUDA (GPU)
- Batch Size: 16
- Pair Logging: A_and_C mode

**Expected Time**:
- Without cache: ~15-20 hours (GPU) for full train set
- With cache (2nd run): < 30 minutes

**Monitor Progress**:
```bash
# Check number of claims processed
wc -l logs/component1/train/claims.jsonl

# Check number of evidence pairs logged
wc -l logs/component1/train/pairs.jsonl

# View last processed claim
tail -n 1 logs/component1/train/claims.jsonl | jq '.'
```

---

## 🎯 Summary

**For Component 2**, you will have access to:

1. ✅ **Every claim** with its A/S/C counts and ESI metrics
2. ✅ **Every Active and Counter triple** with:
   - Raw KG triple
   - Verbalized text
   - PV scores (entailment, contradiction, neutral)
   - Box assignment (A or C)
   - Relevance and polarity scores
3. ✅ **Selection logic reasoning**: Why each triple went to which box

**This data enables Component 2 to**:
- Identify evidence gaps (low ESI claims)
- Analyze contradiction patterns (Counter set)
- Understand evidence quality (relevance, neutral rate)
- Build models that incorporate sufficiency signals

---

**End of Analysis Documentation**
