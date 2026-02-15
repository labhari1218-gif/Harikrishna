# Component 2 (Fact-or-Fiction) — PV-Aware Graph Reasoner + Bridge Rescue (Spec + Plan)

This document is **standalone**: if someone opens it with zero context, they should understand:
- what Component 2 does,
- what it consumes from Component 1,
- what it outputs,
- how “partial evidence” is *rescued* (so it isn’t suppressed by PV),
- how to implement it cleanly inside the **Fact-or-Fiction** repository.

It also includes:
- a **checklist plan** (checkboxes you can tick as you implement),
- a **Codex interview workflow** so Codex asks you one question at a time and maintains the plan,
- a **paper-reading map**: *what to look for* in each paper + which GitHub folders to inspect.

---

## 1) The big picture

### 1.1 Problem we’re solving

FactKG is **binary** (SUPPORTED / REFUTED). But in real systems, you often face:

> “I *can* output SUPPORTED/REFUTED, but I don’t actually have enough evidence to be confident.”

Component 2 turns your Component 1 sensors (ESI / RPI / CR@k + A/S/C pools) into an **actual reasoning module** that can:

1) **Reason** over evidence graphs using a GNN,
2) **Downweight noise** using PV **soft masks** (not hard pruning),
3) **Use polarity** (support vs refute) explicitly,
4) **Rescue missing bridges** using diffusion/PPR scoring,
5) Optionally **recover** a small number of bridge edges from S → A and re-run once.

---

## 2) Contract: what Component 2 consumes and produces

### 2.1 Inputs from Component 1 (per claim)

For each evidence unit `e` (triple/sentence):

- PV distribution: `p_ent(e), p_con(e), p_neu(e)`
- Derived:
  - `rel(e) = p_ent(e) + p_con(e)`  (decision relevance “mass”)
  - `pol(e) = p_ent(e) - p_con(e)`  (direction)

Pools:
- `A` (Active)
- `C` (Counter; contradiction-leaning but must not vanish)
- `S` (Suspended; not used unless we “recover” bridges)

Diagnostics:
- `ESI_geom` (sufficiency sensor)
- `CR@k` (counter retention sensor)
- `RPI@k` (recovery potential sensor)
- `neutral_rate` / neutral dominance

### 2.2 Outputs of Component 2 (per claim)

**Core outputs:**
- predicted label: `{SUPPORTED, REFUTED}`
- confidence score: `conf ∈ [0,1]`
- **abstain decision** (optional selective prediction): `ABSTAIN?` (explained in §8)

**Evidence outputs (for interpretability and paper plots):**
- salience scores per edge/node
- top-`N` rationale edges / subgraph (what the model “used” most)
- if recovery triggered:
  - which edges moved from `S → A`
  - before/after ESI + before/after confidence

---

## 3) Graph representation in FactKG

### 3.1 What are the nodes?

In the **KG setting**, nodes are **entities**.

Given a triple:

```
(subject_entity, relation, object_entity)
```

you create nodes for:
- `subject_entity`
- `object_entity`

and an edge labeled by `relation`.

> Where do entities come from?
> - From the retrieved subgraph triples you already have.
> - Every unique entity ID/string appearing as a subject/object becomes a node.

**Implementation detail:**
- Build a mapping: `entity_id (string) -> node_index (int)`
- Do it per-claim for simplicity; if you want speed, you can cache or globalize later.

### 3.2 What is “relation embedding” in edge features?

Each edge has a **relation type** like `birthPlace`, `successor`, `award`, etc.

You represent each relation type with an embedding vector:
- `r_emb = Embedding[num_relations, d_rel](rel_id)`

Then the GNN uses `r_emb` to condition message passing on relation type.

Common patterns (choose one):
1) **Edge-conditioned message**:
   - `msg(u->v) = W_h h_u + W_r r_emb`
2) **Relation-aware attention**:
   - attention logit depends on `h_u, h_v, r_emb`
3) **R-GCN style**: relation-specific transforms

**Why this matters:** `successor` ≠ `birthPlace`. The GNN needs relation semantics.

### 3.3 Edge features for Component 2

For each edge/triple `e`, store:
- `rel_id` (for relation embedding)
- PV: `p_ent, p_con, p_neu`
- derived: `rel(e), pol(e)`
- pool label: A/S/C (as an integer feature)

---

## 4) Anchor entities — what they are and how to choose them

### 4.1 Definition

**Anchor entities = the claim’s key entities**.

They matter because “bridge rescue” asks:

> “Which edges help connect the claim’s anchors through multi-hop reasoning?”

If you can connect anchors with a meaningful chain, evidence is more likely sufficient.

### 4.2 How to choose anchors (best-practice hierarchy)

Use this priority order:

#### (A) Use the same seed entities that retrieval used (recommended)

In the Fact-or-Fiction repo, subgraph retrieval starts from a small set of entities derived from the claim.
Those “seed entities” are the *best* anchors because:
- they already map claim text → KG IDs correctly,
- they’re consistent with Component 1 coverage logic.

**Action:** find where retrieval is called and what “entity list” it uses (search the repo for keywords like `entities`, `entity_set`, `seed`, `retrieve`, `dbpedia`).

#### (B) Use FactKG-provided claim entities (if present in dataset files)

Some FactKG formats include pre-linked entities for each claim.
If the repo stores these, use them directly.

#### (C) Lightweight entity extraction fallback (if A/B missing)

If you only have claim text and retrieved triples:

1) Normalize entity labels from triples:
   - `Barack_Obama` → `barack obama`
2) Normalize claim text similarly.
3) Any triple-entity whose normalized label is a substring of claim text becomes an anchor.
4) If too many anchors, keep the top K by:
   - frequency in subgraph
   - or “degree” (how connected it is)

**Rule of thumb:** keep `K=2..5` anchors. Too many anchors makes bridge scoring noisy.

### 4.3 Anchor example (simple)

Claim:
> “Barack Obama was born in Hawaii.”

Anchors:
- `Barack_Obama`
- `Hawaii`

Why?
- Any meaningful proof chain must connect the person entity to the location entity.

---

## 5) Component 2 reasoning core (PV-aware + polarity-aware GNN)

### 5.1 Build the working evidence graph

Default:
- use `A ∪ C` as the reasoning graph (don’t forget counters!)
- keep `S` aside (only used by recovery)

### 5.2 Soft masks (do not hard-prune evidence)

PV gives you `p_ent, p_con, p_neu`.
We convert that into **masks** (weights) so evidence can be *downweighted* but not deleted.

#### Support mask vs refute mask

For an edge `e`:

```
m_sup(e) = sigmoid(a_s * p_ent(e) + b_s)
m_ref(e) = sigmoid(a_r * p_con(e) + b_r)
```

Interpretation:
- if an edge strongly *supports* the claim (`p_ent` high), it should contribute more to the **support stream**.
- if an edge strongly *contradicts* (`p_con` high), it should contribute more to the **refute stream**.

### 5.3 Dual-stream message passing (what “two streams” means)

Think of it like this:

- Stream 1 builds a representation of the graph “as if you were trying to SUPPORT the claim”
- Stream 2 builds a representation “as if you were trying to REFUTE the claim”

Both are computed on the same graph, but edges are weighted differently.

#### Example (conceptual)

If you have edges:
- e1: “Obama birthPlace Honolulu” (supporting)
- e2: “Honolulu isPartOf Hawaii” (supporting)
- e3: “Obama birthPlace Kenya” (contradiction)

Then:
- support-stream will weight e1,e2 high and e3 low
- refute-stream will weight e3 high (if `p_con` high)

### 5.4 Pooling and classifier head (the part you said you didn’t get)

After message passing, each node has an embedding.
To make a claim-level decision, you must compress node embeddings into one vector.

A minimal pooling recipe:

- `pool_sup`: pooled representation from **support-stream** node embeddings
- `pool_ref`: pooled representation from **refute-stream** node embeddings
- `pool_all`: pooled representation from an **unmasked (or lightly masked)** stream (optional)

Then combine:

```
repr = concat(pool_sup, pool_ref, pool_all)
logits = MLP(repr)  # 2 logits: [SUPPORTED, REFUTED]
probs = softmax(logits)
```

**Why concat helps:**  
The classifier sees: “here is the best support summary” + “here is the best refute summary” (+ optional overall signal).

So if both are weak → you can abstain (selective prediction).
If refute strong → REFUTED.
If support strong → SUPPORTED.

---

## 6) Bridge rescue (recommended default): diffusion / random-walk scoring

You asked to explain this carefully:

### 6.1 Why diffusion (PPR) instead of enumerating paths?

Path enumeration (k-shortest, BFS all paths) can blow up in dense graphs.
Diffusion gives a scalable approximation of:

> “How much does this edge lie on the *flow* between anchors?”

### 6.2 PPR setup

We build a weighted adjacency on the graph:

```
w(e) = ε + rel(e)
```

- `rel(e)=p_ent+p_con`
- `ε` is a tiny constant so no edge is zero-weight (e.g., 1e-3)

Compute **Personalized PageRank (PPR)** from the anchor set.

Intuition:
- start random walks from anchors
- edges with higher PV relevance are more likely to be walked
- PPR score `p[v]` becomes “how reachable/important node v is from anchors”

### 6.3 Bridge score for an edge

For edge `e = (u → v)`:

```
BridgeBonus(e) = p[u] * w(e) * p[v]
```

Interpretation:
- edge is important if both endpoints are important under anchor diffusion,
- and edge itself has decent PV relevance.

### 6.4 Neutrality safeguard (your key concern)

PV can be uncertain: an edge can be a cheap connector but semantically useless.

If PV says edge is mostly neutral, cap the bonus:

```
BridgeBonus'(e) = BridgeBonus(e) * (1 - p_neu(e))^γ
```

- if `p_neu(e)=0.9` and `γ=2`, multiplier is `(0.1)^2 = 0.01` → huge penalty.
- if `p_neu(e)=0.1`, multiplier `(0.9)^2 = 0.81` → mostly preserved.

This prevents “irrelevant shortcut connectors” from dominating.

### 6.5 Worked example (numbers)

Claim:
> “Barack Obama was born in Hawaii.”

Anchors:
- `Barack_Obama`, `Hawaii`

Retrieved triples (simplified):
- e1: Obama —birthPlace→ Honolulu  
  PV: p_ent=0.70, p_con=0.05, p_neu=0.25  → rel=0.75
- e2: Honolulu —isPartOf→ Hawaii  
  PV: p_ent=0.60, p_con=0.05, p_neu=0.35  → rel=0.65
- e3: Obama —spouse→ Michelle_Obama  
  PV: p_ent=0.05, p_con=0.02, p_neu=0.93  → rel=0.07 (mostly neutral)
- e4: Obama —birthPlace→ Kenya  
  PV: p_ent=0.02, p_con=0.75, p_neu=0.23  → rel=0.77 (contradiction edge)

Set ε=0.01:
- w(e1)=0.76, w(e2)=0.66, w(e3)=0.08, w(e4)=0.78

PPR from anchors gives (example):
- p[Obama]=0.40
- p[Honolulu]=0.20
- p[Hawaii]=0.35
- p[Michelle]=0.05
- p[Kenya]=0.08

BridgeBonus:
- e1: 0.40 * 0.76 * 0.20 = 0.0608
- e2: 0.20 * 0.66 * 0.35 = 0.0462
- e3: 0.40 * 0.08 * 0.05 = 0.0016  (tiny anyway)
- e4: 0.40 * 0.78 * 0.08 = 0.02496

Neutral safeguard with γ=2:
- e1 multiplier (1-0.25)^2 = 0.75^2 = 0.5625  → 0.0342
- e2 multiplier (1-0.35)^2 = 0.65^2 = 0.4225  → 0.0195
- e3 multiplier (1-0.93)^2 = 0.07^2 = 0.0049  → 0.0000078 (kills shortcut)
- e4 multiplier (1-0.23)^2 = 0.77^2 = 0.5929  → 0.0148

Now you clearly see:
- the “bridge” chain e1+e2 stays high,
- the irrelevant shortcut e3 is destroyed,
- the contradiction e4 stays meaningful (as it should).

---

## 7) Recovery from S → A (one-step, diagnostic-triggered)

### 7.1 When to trigger recovery

Use Component 1 sensors:

Trigger if:
- `ESI_geom < τ_esi`  (evidence looks insufficient)
AND
- `RPI@k > τ_rpi`     (recovering could help)
OR
- graph is disconnected between anchors

### 7.2 What to recover

Compute `BridgeBonus'(e)` for edges in **S** (you can compute it on the full graph A∪S∪C, but only move from S).

Recover:
- top-`k` edges in S by BridgeBonus’ (with constraints)
Constraints (important):
- prefer edges that connect **different components** (bridge actual gaps)
- optionally enforce relation diversity (avoid 5 similar edges)

Then:
- move recovered edges from `S → A`
- recompute ESI
- rerun GNN once

Log:
- whether recovery happened + which edges moved
- before/after ESI + confidence change

---

## 8) “Abstention” (selective prediction) on a binary dataset

FactKG is binary, so you can’t train a supervised NEI label easily.
But you can still do **selective prediction**:

> The model predicts SUPPORTED/REFUTED only when it is confident AND evidence looks sufficient.
> Otherwise it “abstains”.

### 8.1 What is an “abstain score”?

You define a scalar that measures “should I answer?”

A simple version:

```
abstain_score = α * (1 - max(probs)) + (1-α) * (1 - ESI_geom)
```

- if the classifier is uncertain → abstain_score high
- if ESI is low → abstain_score high

Then choose a threshold `τ`:
- abstain if `abstain_score > τ`

### 8.2 Risk–coverage (example)

Suppose 10 claims. You sort by abstain_score (lowest first).
Coverage = fraction you *do* answer.

Example:
- At 100% coverage (answer all 10): you get 2 wrong → risk = 20%
- If you abstain on 3 most “insufficient” cases:
  - you answer 7
  - only 0 wrong among answered → risk = 0%
  - coverage = 70%

A risk–coverage curve plots:
- x = coverage
- y = risk (error rate among answered)

Goal:
- keep risk low as you reduce coverage
- your method is better if it achieves lower risk at the same coverage.

This is exactly how you evaluate “insufficiency” without needing NEI labels.

---

## 9) Rationale / salience: “top N edges by salience” — what and why

### 9.1 What is salience?

Salience is “how much did this edge influence the prediction?”

You can define it via:
- attention weights in GAT (fast, not always faithful)
- gradient-based attribution (more faithful, slower)
- leave-one-out: remove edge and see confidence drop (slowest but most faithful)

### 9.2 Why output top-N edges?

Two reasons:
1) **Interpretability**: show what evidence chain the model relied on.
2) **Faithfulness test**: if you remove the top-N rationale edges, confidence should drop significantly.

### 9.3 Practical default (good balance)

Start with:
- Use learned attention weights *times* PV relevance as a proxy salience.
- Output top 10 edges.

Then for “paper grade” faithfulness:
- run a small subset with leave-one-out removal to validate.

---

## 10) Implementation plan (checkboxes)

### 10.1 Core engineering tasks

- [ ] **(T1)** Find/confirm where claim entities are stored / used in retrieval in Fact-or-Fiction repo
- [ ] **(T2)** Implement `AnchorSelector` with the hierarchy (A/B/C) in §4
- [ ] **(T3)** Implement `GraphBuilder` for A∪C (entities as nodes, triples as typed edges)
- [ ] **(T4)** Add edge features: relation id + PV features + pool id
- [ ] **(T5)** Implement `MaskedGAT` (or R-GCN) backbone
- [ ] **(T6)** Implement dual-stream masking (support/refute)
- [ ] **(T7)** Implement pooling and classifier head
- [ ] **(T8)** Implement `BridgeRescuePPR` (PPR + BridgeBonus’)
- [ ] **(T9)** Implement diagnostic-triggered 1-step recovery (S→A) + re-run
- [ ] **(T10)** Implement salience extraction + top-N rationale output
- [ ] **(T11)** Add selective prediction (abstention) + risk–coverage evaluation
- [ ] **(T12)** Add ablations: no PV mask / no bridge rescue / no recovery

### 10.2 Tests you should write (minimum)

- [ ] **(Test A)** anchors are non-empty and belong to the claim graph
- [ ] **(Test B)** PPR bridge scores are stable and monotonic in rel(e)
- [ ] **(Test C)** recovery actually increases connectivity on a constructed toy graph
- [ ] **(Test D)** model runs end-to-end on 5 claims (CPU) without crashing
- [ ] **(Test E)** rationale edge extraction returns exactly N edges and they exist in graph

---

## 11) How to use Codex effectively (interview + plan maintenance)

Goal: Codex should:
1) read this doc,
2) interview you *one question at a time*,
3) produce a detailed plan in a single MD file,
4) keep it updated with checkboxes that get ticked as tasks finish.

### 11.1 Files to create in your repo

Create a folder:

```
docs/component2/
```

Add these files:

1) `COMPONENT2_SPEC.md`  
   - paste **this** document (or a cleaned version)

2) `COMPONENT2_PLAN.md`  
   - Codex-maintained plan with checkboxes + logs

3) `COMPONENT2_DECISIONS.md`  
   - small “decision log” (what you chose & why)

### 11.2 The Codex “interview prompt” (copy-paste)

Use this as your first message to Codex (and attach/point it to `COMPONENT2_SPEC.md`):

```text
You are my implementation interviewer and project manager.

Read docs/component2/COMPONENT2_SPEC.md.

Then interview me ONE QUESTION AT A TIME to remove ambiguity and lock the exact implementation choices for Component 2 inside this repo.

Rules:
- Ask exactly ONE question each turn.
- The question must be concrete (choose between options, request a number, confirm a file path, etc).
- After I answer, you must update docs/component2/COMPONENT2_PLAN.md:
  - add or refine tasks,
  - add checkboxes,
  - add acceptance criteria for each task,
  - link tasks to repo paths/modules,
  - keep a “Progress” section that I can tick.
- Do NOT implement code until the plan is locked.
- Once the plan is locked, implement step-by-step and after each completed task, update the plan by ticking the checkbox and adding a short completion note.

Start by asking me:
(1) Which evidence format we will implement first (FactKG KG triples only, or also HoVer text),
and (2) where in this repo the claim entities / retrieval seed entities are stored.
```

### 11.3 Best-practice loop (Codex)

1) **Lock interface first** (what each module takes/returns)
2) **Write toy tests** (tiny graphs)
3) Implement in small PR-sized steps
4) Update `COMPONENT2_PLAN.md` every step

---

## 12) “Antigravity” best practices (how to use it alongside Codex)

Use Codex for **code**.
Use Antigravity (and/or a research assistant) for:
- turning papers into actionable “what to implement” notes,
- generating diagrams (bridge rescue + recovery loop),
- sanity checking the narrative.

**Rule:** decisions must land in your repo MD files, not inside chat threads.

Suggested workflow:
- Antigravity produces a 1–2 page summary of a paper,
- you paste the key “implementation takeaways” into `COMPONENT2_SPEC.md` or `DECISIONS.md`,
- Codex turns that into tasks + code.

---

## 13) Paper → implementation map (what to read & what code to inspect)

Below, each paper has:
- what to read for Component 2,
- what to steal conceptually,
- what repo to inspect (if available).

### 13.1 Fact-or-Fiction (your base system)

Read:
- the subgraph retrieval method (single-step vs multi-hop),
- how they represent triples and entities.

Code:
- search for retrieval scripts/modules, and identify:
  - where “seed entities” are computed,
  - where subgraphs are saved/loaded.

### 13.2 FactKG

Read for:
- reasoning types (multi-hop, negation, existence),
- what the dataset treats as “evidence” and label space.

Why it matters:
- informs anchor selection corner cases (existence/negation often involve fewer entities).

### 13.3 CO-GAT (masking)

Read for:
- node confidence score,
- “soft mask vs hard mask” results,
- how they inject masking before GAT.

Why it matters:
- validates “soft PV masking” story.

Repo:
- CO-GAT code exists (search for a GitHub repo under the authors).

### 13.4 GEAR (graph FV backbone)

Read for:
- graph construction patterns,
- how they aggregate evidence pieces.

Repo:
- thunlp/GEAR (look at `gear/` and `feature_extractor/` folders).

### 13.5 Reasoning Paths as Signals / SR-MFV style work

Read for:
- why “paths/progression” matter,
- ablations that compare static graph vs path-aware reasoning.

Takeaway:
- use it to motivate bridge rescue + recovery loop.

### 13.6 MRR-FV

Read for:
- why multi-hop chain building can explode,
- how they control search space.

Takeaway:
- strengthens your argument for diffusion instead of enumerating paths.

---

## 14) Where to start (recommended order)

If you want the fastest progress:

1) Implement **anchors + graph builder** (T1–T4)
2) Implement **masked dual-stream GAT** (T5–T7)
3) Implement **PPR bridge scoring** (T8)
4) Implement **recovery** (T9)
5) Add **rationale outputs + abstention** (T10–T11)
6) Add ablations (T12)

---

## 15) Quick sanity checks (what “good” looks like)

Before you chase accuracy:
- recovery triggers rarely, but helps on hard cases
- bridge edges moved from S→A tend to:
  - connect anchor components
  - reduce disconnectedness
- abstention improves risk–coverage (risk drops when abstaining on low-ESI cases)
- rationale edges include multi-hop chains for multi-hop claims

---

# Appendix A — Minimal pseudo-API (so you don’t get lost)

```text
component2/
  anchor_selector.py    -> anchors = select_anchors(claim, claim_entities, triples)
  graph_builder.py      -> G = build_graph(triples_AUC, pv_features, anchors)
  bridge_rescue.py      -> bonus = bridge_bonus(G_full, anchors, pv_rel, pv_neu)
  reasoner.py           -> (probs, salience) = forward(G_AUC, masks, bonus)
  recovery.py           -> if trigger: move top-k from S->A and rerun
  selective.py          -> abstain_score, risk_coverage_eval
```

---

# Appendix B — Links (kept in code blocks)

```text
Fact-or-Fiction repo:
https://github.com/Tobias-Opsahl/Fact-or-Fiction

FactKG:
https://arxiv.org/abs/2305.06590

HoVer:
https://aclanthology.org/2020.findings-emnlp.309/

CO-GAT:
https://arxiv.org/abs/2405.10481
https://github.com/neuir/co-gat

GEAR:
https://github.com/thunlp/GEAR
```
