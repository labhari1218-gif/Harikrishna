# ChatGPT Prompt: Diagnose Low Accuracy in Fact-or-Fiction (Component 3)

Use this prompt in ChatGPT (or another LLM) after sharing the repository/files.

---

You are reviewing a fact-checking project and must diagnose **why Component 3 is underperforming**.

## Goal
- Explain the low scores in plain language.
- Identify concrete code-level causes.
- Propose a prioritized fix plan to beat SOTA and improve explainability.

## Repository + Key Files
- `README.md`
- `src/component3/run_train.py`
- `src/component3/pv_dataset.py`
- `src/component3/pv_dataset_fever.py`
- `src/component3/c1_pairs_loader.py`
- `src/component3/pv_qagnn.py`
- `src/component3/backtracking.py`
- `src/component3/router.py`
- `src/component3/run_component7.py`
- `src/component1/esm.py`
- `src/component1/logging_utils.py`
- `scripts/t33_full_recovery_run.sh`
- `scripts/compare_component3_to_sota.py`

## Run Artifacts to Analyze
- `runs/t33_full_seed57_rerun7_tmux_20260215_1755/config.yaml`
- `runs/t33_full_seed57_rerun7_tmux_20260215_1755/train.log`
- `runs/t33_full_seed57_rerun7_tmux_20260215_1755/metrics.json`
- `runs/t33_full_seed57_rerun7_tmux_20260215_1755/sota_comparison.json`
- `runs/t33_tmux_20260215_095704/config.yaml`
- `runs/t33_tmux_20260215_095704/metrics.json`
- `logs/component1/train/claims.jsonl`
- `logs/component1/train/pairs.jsonl`
- `logs/component1/val/claims.jsonl`
- `logs/component1/val/pairs.jsonl`
- `logs/component1/test/claims.jsonl`
- `logs/component1/test/pairs.jsonl`

## What I Need From You
1. Reconstruct the **actual method path used in this run** (not planned architecture).
2. Clarify whether router/backtracking were actually active in training/eval.
3. Explain claim log vs pair log vs train log and what each proves.
4. Audit A/S/C handling end-to-end:
   - whether S is dropped in Component 3,
   - whether A/C are both used,
   - whether C is wrongly treated as negative in auxiliary evidence supervision.
5. Explain why substitution precision is very low and how label/base-rate imbalance affects this.
6. Check if SOTA comparison is fair or apples-to-oranges.
7. Give a concrete ablation matrix (small, medium, full runs) with expected impact.
8. Give a fix plan for explainability outputs: why Support/Refute/NEI (or binary alternatives) for each claim.

## Important Context
- I expected: graph reasoner uses A+C first, then backtracking can recover from S.
- I suspect current pipeline drops S and does not run backtracking.
- I suspect previous auxiliary edge loss used only A as positive and treated C as negative.
- I want high accuracy and trustworthy explanations.

## Output Format (strict)
1. **System Diagram of Current Reality** (what actually runs)
2. **Top 10 Root Causes** (severity-ordered, with file:line references)
3. **Metric Forensics** (why each weak slice fails, especially substitution/negation)
4. **Loss Objective Audit** (old vs corrected formulation)
5. **Router Necessity Decision** (needed now / later / not needed, with reason)
6. **Backtracking Integration Plan** (minimal viable implementation)
7. **SOTA Plan** (3-stage experiments with target numbers and stop criteria)
8. **Explainability Contract** (per-claim rationale schema and confidence fields)

Do not give generic advice. Use concrete references to the provided files/runs.
