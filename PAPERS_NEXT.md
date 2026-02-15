# PAPERS_NEXT.md — Research Reference Guide for Components 3–7

> **Purpose**: Map every paper to specific components + extract best practices for our PV-QA-GNN design.
> Papers downloaded to `docs/papers_next/` (9 of 13 — 3 behind paywalls).

---

## Component 3 — PV-Enhanced QA-GNN

### QA-GNN — Reasoning with Language Models and Knowledge Graphs (NAACL 2021)
- **Link**: [ACL Anthology — 2021.naacl-main.45](https://aclanthology.org/2021.naacl-main.45/)
- **Already in our codebase**: `models.py` implements `QAGNN` class
- **Key ideas we already use**: BERT CLS → claim embedding, GATConv over KG subgraph, cosine relevance scoring, concat(GNN_pool, claim_embed) → classifier
- **What we add on top**: Dual-stream PV edge masks, A/S/C-aware graph construction, bridge rescue from S pool

### CO-GAT — Confidential Graph Attention Network (IEEE T-BigData 2025)
- **Link**: [arXiv 2405.10481](https://arxiv.org/abs/2405.10481) | **PDF**: `docs/papers_next/CO-GAT_2405.10481.pdf`
- **Best practices extracted**:
  1. ✅ **Soft confidence gating**: Sigmoid CO-SCO ∈ [0,1] beats hard 0/1 masking by **+0.79%** → We use `σ(α·p + β)` soft masks
  2. ✅ **Multi-task loss**: Joint fact verification loss + evidence relevance prediction loss gives **+0.18%** → We add auxiliary evidence-quality loss
  3. ✅ **Two-stage warmup**: Stage 1 (lr=5e-5, train attention only) → Stage 2 (lr=2e-6, fine-tune all) → We freeze BERT first, then unfreeze pooler
  4. ✅ **Max length 256**: Tail truncation for claim-evidence pairs → Adopted
  5. ✅ **Early stopping patience ≤ 5** → We keep existing patience=3
  6. ✅ **Batch size 8 for large models** (they use 8 for BERT-large) → Fits our 8GB GPU constraint
  7. ✅ **Blank node as noise baseline**: Encode claim-only (no evidence) as reference → Interesting for future ablation

### GEAR — Graph-based Evidence Aggregating and Reasoning (ACL 2019)
- **Link**: [ACL Anthology — P19-1085](https://aclanthology.org/P19-1085/) | **PDF**: `docs/papers_next/GEAR_P19-1085.pdf`
- **Why it matters**: GEAR is the baseline to beat — single-stream fully-connected evidence graph + BERT
- **Key comparison**: GEAR uses one attention stream; we use two (support/refute) with PV-derived masks
- **GEAR achieves**: 67.10% FEVER score on FEVER dataset
- **Our advantage**: Dual-stream separation isolates conflicting signals rather than mixing them

### HESM — Hierarchical Evidence Set Modeling (EMNLP 2020)
- **Link**: [ACL Anthology — 2020.emnlp-main.627](https://aclanthology.org/2020.emnlp-main.627/) | **PDF**: `docs/papers_next/HESM_2020.emnlp-main.627.pdf`
- **Relevance**: Models evidence *sets* hierarchically with 3-way labels (SUP/REF/NEI) → maps to our A/S/C partition
- **Best practice**: Set encoder for variable-size evidence pools → informs our pooling design

### EvidenceNet — Gated Evidence Fusion (WWW 2022) ⚠️ PAYWALL
- **Link**: [ACM DOI — 10.1145/3485447.3512135](https://dl.acm.org/doi/10.1145/3485447.3512135)
- **Status**: Could not download (403). Using CO-GAT's gating mechanism instead.

---

## Component 4 — S-Only Backtracking

### ProgramFC — Program-Guided Iterative Fact-Checking (ACL 2023)
- **Link**: [ACL Anthology — 2023.acl-long.386](https://aclanthology.org/2023.acl-long.386/) | **PDF**: `docs/papers_next/ProgramFC_2023.acl-long.386.pdf`
- **Position against**: ProgramFC iterates forward through a program trace; we iterate *backward* (backtracking from S pool). Fundamentally different approach — recovers dropped evidence, not new evidence.

### MRR-FV — Multi-hop Retrieval + Reasoning (AAAI 2025) ⚠️ PAYWALL
- **Link**: [AAAI article/view/34802](https://ojs.aaai.org/index.php/AAAI/article/view/34802)
- **Status**: Could not download (403). Key contrast: they iterate forward-only; we allow backtracking.

---

## Component 5 — Learned Controller + Losses

### SaGP — Salience-Aware Graph Pruning (arXiv 2022)
- **Link**: [arXiv 2212.01060](https://arxiv.org/abs/2212.01060) | **PDF**: `docs/papers_next/SaGP_2212.01060.pdf`
- **Best practices for losses**:
  1. Salience-aware pruning → validates our rationale extraction (attention × PV) approach
  2. Noise control through graph sparsification → our S-only recovery is the inverse: add back what was wrongly pruned
  3. Rationale precision/recall metrics → use for evaluating our controller's recovery decisions

### Dynamic Debiasing via Counterfactual Reasoning (Knowledge-Based Systems 2025) ⚠️ PAYWALL
- **Link**: [ScienceDirect — S0950705125019136](https://www.sciencedirect.com/science/article/pii/S0950705125019136)
- **Status**: Could not download (403). Concept still applies: recovery utility loss = counterfactual "what if I hadn't dropped this evidence?"

---

## Component 6 — Claim-Type Router

### FactKG — Fact Verification via Reasoning on Knowledge Graphs (ACL 2023)
- **Link**: [ACL Anthology — 2023.acl-long.895](https://aclanthology.org/2023.acl-long.895/) | **PDF**: `docs/papers_next/FactKG_2023.acl-long.895.pdf`
- **Key for router**: FactKG defines 5 reasoning types (one-hop, multi-hop, negation, conjunction, existence) with explicit annotations → **direct supervision labels** for our router classifier
- **Key result**: BERT single-step baseline achieves ~93% overall → our target is to beat this on EVERY type

### HoVer — Many-Hop Fact Extraction and Claim Verification (EMNLP Findings 2020)
- **Link**: [ACL Anthology — 2020.findings-emnlp.309](https://aclanthology.org/2020.findings-emnlp.309/) | **PDF**: `docs/papers_next/HoVer_2020.findings-emnlp.309.pdf`
- **Use for**: Stress-testing router on 2-4 hop claims; validating that multi-hop budget allocation helps

---

## Component 7 — Evaluation + Robustness

### VitaminC — Contrastive Evidence Sensitivity (NAACL 2021)
- **Link**: [ACL Anthology — 2021.naacl-main.52](https://aclanthology.org/2021.naacl-main.52/) | **PDF**: `docs/papers_next/VitaminC_2021.naacl-main.52.pdf`
- **Use for**: Contrastive robustness test — swap one evidence triple → prediction should flip. Target: flip rate > 70%

### ProoFVer — Natural Logic Theorem Proving (TACL 2022)
- **Link**: [ACL Anthology — 2022.tacl-1.59](https://aclanthology.org/2022.tacl-1.59/) | **PDF**: `docs/papers_next/ProoFVer_2022.tacl-1.59.pdf`
- **Use for**: Faithfulness benchmark — compare our salience-based rationales against formal logic proofs

---

## Quick Reference Table

| Paper | PDF Downloaded? | Component | Primary Use |
|-------|:-:|-----------|-------------|
| QA-GNN | N/A (in code) | C3 | Foundation model (already implemented) |
| CO-GAT | ✅ | C3 | Gating + multi-task + warm-up best practices |
| GEAR | ✅ | C3 | Single-stream baseline to beat |
| HESM | ✅ | C3 | Set-level evidence modeling reference |
| EvidenceNet | ❌ paywall | C3 | Gating reference (covered by CO-GAT) |
| ProgramFC | ✅ | C4 | Contrast: iterative ≠ backtracking |
| MRR-FV | ❌ paywall | C4 | Contrast: no backtracking memory |
| SaGP | ✅ | C5 | Salience-aware losses + rationale metrics |
| Dynamic Debiasing | ❌ paywall | C5 | Counterfactual recovery motivation |
| FactKG | ✅ | C6 | Router supervision (5 reasoning types) |
| HoVer | ✅ | C6-C7 | Multi-hop stress test |
| VitaminC | ✅ | C7 | Contrastive robustness test |
| ProoFVer | ✅ | C7 | Faithfulness benchmark |
