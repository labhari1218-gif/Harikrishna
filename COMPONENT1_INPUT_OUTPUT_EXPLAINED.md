# Component 1: Input/Output Flow & Data Storage

## 🔄 Complete Pipeline Flow

```
INPUT                    PROCESSING                    OUTPUT
─────                    ──────────                    ──────

📄 Claim                 
"He had a                🔍 1. RETRIEVE EVIDENCE
successor named          ─────────────────────
John E. Beck"            From subgraph → 13 triples
                         Example triples:
                         • (John_Beck, successor, Person_X)
                         • (John_Beck, elected, 1850)
                         • (Some_Entity, unrelated, fact)
                              ↓
                         🤖 2. PV SCORING (NLI)
                         ──────────────────────
                         For each triple:
                         • Verbalize: "John Beck was the successor of Person X"
                         • Score vs claim using DeBERTa NLI
                         • Get: rel, pol, p_contra
                         
                         Example scores:
                         Triple 1: rel=0.85, pol=0.75, p_contra=0.01
                         Triple 7: rel=0.95, pol=-0.82, p_contra=0.99 ← strong counter!
                              ↓
                         📊 3. ESM PARTITIONING
                         ──────────────────────
                         Split 13 triples into 3 sets:
                         
                         Step 1: Select Counter (C)
                         ─────────────────────────────
                         • Find p_contra >= 0.6
                         • Take top 2 by p_contra
                         → C = 1 triple (Triple 7, p_contra=0.99)
                         
                         Step 2: Select Active (A)
                         ─────────────────────────────
                         • From remaining 12 triples
                         • Take top by relevance
                         • Ensure min_A = 5
                         → A = 12 triples (all remaining)
                         
                         Step 3: Suspended (S)
                         ─────────────────────────────
                         • Whatever's left
                         → S = 0 triples
                              ↓
                         📐 4. COMPUTE METRICS
                         ─────────────────────
                         On Active set (A):
                         • Coverage = 100% (all claim entities found)
                         • Connectivity = 100% (entities connected)
                         • Neutral Rate = 91.5%
                         • ESI_geom = 0.4336
                         
                         On Counter set (C):
                         • CR@5 = 20% (kept 1 of top-5 contradictions)
                         • counter_in_A = 0 (no leaks)
                              ↓
                              ↓
                         💾 OUTPUT FILES
                         ───────────────

📁 claims.jsonl          One line per claim:
   {
     "claim_id": "train_0",
     "claim_text": "He had a successor named John E. Beck as well.",
     "label": "TRUE",
     
     "counts": {
       "A": 12,  ← Active
       "S": 0,   ← Suspended  
       "C": 1    ← Counter
     },
     
     "sufficiency": {
       "esi_geom": 0.4336,      ← PRIMARY METRIC
       "coverage_A": 1.0,
       "connectivity_A": 1.0,
       "neutral_rate_A": 0.915
     },
     
     "counter_retention": {
       "has_counter": true,
       "counter_kept": 1,
       "cr_at_5": 0.2,
       "max_contra_C": 0.9945,
       "counter_in_A": 0,       ← FIX 1: Uses contra_tau
       "max_contra_A": 0.006
     },
     
     "pool_stats": {
       "pool_size_total": 13,
       "pool_size_after_C": 12,
       "min_A_feasible": 5
     }
   }

📁 pairs.jsonl           One line per (claim, evidence) pair:
   {
     "claim_id": "train_0",
     "triple": "(John_Beck, successor, Person_X)",
     "set": "A",              ← Which box: A, S, or C
     "pv": {
       "rel": 0.8534,         ← Relevance
       "pol": 0.7521,         ← Polarity (support)
       "p_contra": 0.0134     ← Contradiction prob
     }
   }

📁 summary.json          Aggregate stats across all claims:
   {
     "num_claims": 20,
     "mean_A": 14.55,
     "mean_esi_geom": 0.4894,
     "starve_rate": 0.00%,
     ...
   }

📁 examples/             Top-10 claims by different criteria:
   • highest_contradiction.json
   • lowest_esi.json
   • most_neutral.json
```

---

## 📦 What Goes in Each Box (A, S, C)?

### 📗 BOX A (ACTIVE) - The Main Evidence

**Purpose**: Evidence used to assess claim sufficiency

**Selection Criteria**:
- Selected AFTER Counter (C) is chosen
- Top items by **relevance** score
- Default: top 20 items (max_A=20)
- Minimum: 5 items (min_A=5) unless pool too small

**What gets computed on A**:
- ✅ Coverage: % of claim entities found in A
- ✅ Connectivity: Are claim entities connected in graph?
- ✅ ESI_geom: Overall sufficiency index
- ✅ Neutral Rate: % of uninformative evidence

**Example items in A** (for claim "John E. Beck was successor"):
```
Triple 1: (John_Beck, elected, 1850) - rel=0.85, p_contra=0.01
Triple 2: (John_Beck, politician, true) - rel=0.75, p_contra=0.02
Triple 3: (Person_X, successor, John_Beck) - rel=0.95, p_contra=0.01
...12 items total
```

---

### 📕 BOX C (COUNTER) - Strong Contradictions

**Purpose**: Isolate strong contradictory evidence

**Selection Criteria**:
- Selected FIRST (before A)
- **p_contra >= 0.6** (contra_tau threshold) ← FIX 1 ensured consistency
- Maximum 2 items (max_C=2)
- Top items by p_contra score

**What gets computed on C**:
- ✅ CR@k: Counter Retention (did we keep top contradictions?)
- ✅ counter_in_A: How many contradictions leaked into A?
- ✅ max_contra_C: Strongest contradiction strength

**Example items in C** (for claim "John E. Beck was successor"):
```
Triple 7: (John_Beck, never_held_office, true) - p_contra=0.9945 ← Strong counter!
```

**Why select C first?**
- Prevents contradictions from being counted as supporting evidence
- Ensures we track counter-evidence retention
- Protects Active set from contamination

---

### 📘 BOX S (SUSPENDED) - Remaining Evidence

**Purpose**: Lower-priority evidence for future recovery

**Selection Criteria**:
- Everything NOT in A or C
- Typically lower relevance items
- **FIX 4**: Docstring now correctly says "remaining after C+A selection"

**What gets computed on S**:
- ✅ Bridge Count: Items that could connect claim entities
- ✅ RPI@k: Recovery Potential Index (would top-k from S help?)

**Example items in S**:
```
Triple 12: (Some_Entity, unrelated, fact) - rel=0.002, p_contra=0.1
(In our example: S is empty because all 12 remaining items fit in A)
```

---

## 📊 Real Example from Logs

**Claim**: "He had a successor named John E. Beck as well."  
**Label**: TRUE  
**Retrieved Evidence**: 13 triples from subgraph

### Partitioning Results:

| Set | Count | Purpose | Key Metrics |
|-----|-------|---------|-------------|
| **A** | 12 | Main evidence | ESI=0.43, Coverage=100%, Connectivity=100% |
| **S** | 0 | Low-priority | (none - all fit in A) |
| **C** | 1 | Contradictions | max_contra=0.99, CR@5=20% |

### Interpretation:

✅ **Good Coverage**: All claim entities found in Active set  
✅ **Good Connectivity**: Entities form connected graph  
⚠️ **High Neutral Rate**: 91.5% of evidence is uninformative (NLI neutral)  
⚠️ **Has Counter**: 1 strong contradiction (p_contra=0.99)  
✅ **No Leakage**: 0 contradictions in Active set (threshold: 0.6)

**Final ESI_geom = 0.4336** → Moderate sufficiency (claim entities covered but high neutral rate)

---

## 🔍 How to Inspect the Data

### View a specific claim:
```bash
# Extract claim train_0
cat logs/component1/train/claims.jsonl | grep "train_0" | python3 -m json.tool

# See what's in each set
cat logs/component1/train/pairs.jsonl | grep "train_0" | grep '"set": "A"'
cat logs/component1/train/pairs.jsonl | grep "train_0" | grep '"set": "C"'
```

### Check summary stats:
```bash
cat logs/component1/train/summary.json | python3 -m json.tool
```

### Find interesting cases:
```bash
# Lowest ESI (evidence starvation)
cat logs/component1/train/examples/lowest_esi.json | python3 -m json.tool

# Highest contradiction
cat logs/component1/train/examples/highest_contradiction.json | python3 -m json.tool
```

---

## 🎯 Key Takeaways

1. **Input**: Claim + Retrieved subgraph (triples)
2. **Processing**: 
   - Score each triple with NLI (rel, pol, p_contra)
   - Partition into A/S/C based on scores
   - Compute sufficiency metrics on each set
3. **Output**: 
   - Per-claim metrics (claims.jsonl)
   - Per-evidence details (pairs.jsonl, optional)
   - Summary statistics (summary.json)
   - Example cases (examples/ directory)

4. **The 3 Boxes (A/S/C)**:
   - **A (Active)**: Main supporting evidence, used for sufficiency
   - **C (Counter)**: Strong contradictions, isolated for analysis
   - **S (Suspended)**: Low-priority, available for recovery

5. **Critical Metrics**:
   - **ESI_geom**: Overall sufficiency (0-1, higher = better)
   - **Coverage**: % of claim entities found
   - **CR@k**: Did we retain top contradictions?
   - **counter_in_A**: Contamination check (should be 0)

---

All data is stored in **JSONL format** (JSON Lines) for easy streaming and analysis! 🚀
