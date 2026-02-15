# DEV NOTES M0 (Repo Orientation)

## 1) What was implemented

- Milestone scope was orientation only (no runtime code modules in M0).
- Evidence and anchor contracts were locked and documented in:
  - `docs/component2/COMPONENT2_DECISIONS.md`
  - `docs/component2/COMPONENT2_PLAN.md`
- Data source mapping captured:
  - Anchor seeds: `Entity_set` from `data/factkg/factkg_{train|dev|test}.pickle`
  - Claim metrics logs: `logs/component1/{train|val}/claims.jsonl`
  - Pair PV logs: `logs/component1/train/pairs.jsonl`

## 2) Exact defaults used (numeric) and sources

- `hidden_dim=256`, `num_heads=4`, `rel_emb_dim=64` from `COMPONENT2_DECISIONS.md` (Model sizing Option B).
- `m_sup=sigmoid(4*p_ent-2)`, `m_ref=sigmoid(4*p_con-2)` from `COMPONENT2_DECISIONS.md` (fixed mask Option A).
- `epsilon=0.001`, `gamma=2`, `tau_esi=0.3`, `tau_rpi_legacy=0.15`, `recovery_k=3` from `COMPONENT2_DECISIONS.md`.
- M0 did not execute train/inference code; defaults were only locked for M1+ implementation.

## 3) Mechanism walkthrough with tiny toy graph

Toy claim anchors: `Barack_Obama`, `Hawaii`.

Edges:
- `e1`: Obama-birthPlace-Honolulu (`p_ent=0.70`, `p_con=0.05`, `p_neu=0.25`)
- `e2`: Honolulu-isPartOf-Hawaii (`p_ent=0.60`, `p_con=0.05`, `p_neu=0.35`)
- `e3`: Obama-spouse-Michelle (`p_ent=0.05`, `p_con=0.02`, `p_neu=0.93`)
- `e4`: Obama-birthPlace-Kenya (`p_ent=0.02`, `p_con=0.75`, `p_neu=0.23`)

Mask values (locked formulas):
- `m_sup(e1)=0.689974`, `m_ref(e1)=0.141851`
- `m_sup(e2)=0.598688`, `m_ref(e2)=0.141851`
- `m_sup(e3)=0.141851`, `m_ref(e3)=0.127862`
- `m_sup(e4)=0.127862`, `m_ref(e4)=0.731059`

Bridge score preview (for M2, using assumed PPR node masses `p(Obama)=0.40`, `p(Honolulu)=0.20`, `p(Hawaii)=0.35`, `p(Michelle)=0.05`, `p(Kenya)=0.08`):
- `w(e)=epsilon + rel(e)` with `epsilon=0.001`
- `BridgeBonus(e1)=0.060080`, `BridgeBonus'(e1)=0.033795`
- `BridgeBonus(e2)=0.045570`, `BridgeBonus'(e2)=0.019253`
- `BridgeBonus(e3)=0.001420`, `BridgeBonus'(e3)=0.000007`
- `BridgeBonus(e4)=0.024672`, `BridgeBonus'(e4)=0.014628`

Connectivity delta preview (for recovery trigger in M2):
- If `A={e1,e4}` then anchors are disconnected: `Conn(A)=0`
- Add bridge edge `e2` from `S`: `Conn(A∪{e2})=1`
- `DeltaConn_bridge@1 = 1 - 0 = 1`

## 4) Why each decision was taken

- Seed entities (`Entity_set`) were chosen as anchors to remain aligned with retrieval-time entity linking (Decision: Anchors Option A).
- Soft masks were fixed (not learnable in v1) for deterministic auditability and easier debugging (Decision: fixed mask Option A).
- Diffusion defaults (`epsilon=0.001`, `gamma=2`) were locked to avoid unstable path explosion and suppress high-neutral connectors.

## 5) Logs/outputs produced and exact paths

- Orientation produced no new model outputs in M0.
- Verified source artifacts used for M0 context:
  - `logs/component1/train/claims.jsonl`
  - `logs/component1/train/pairs.jsonl`
  - `docs/component2/COMPONENT2_DECISIONS.md`

## 6) Tests added and what they validate

- No code was implemented in M0; therefore no M0-only tests were added.
- Test implementation starts in M1 and validates anchors/graph/reasoner/run-config snapshot behavior.

## 7) Alternatives considered and not chosen

- Extract anchors from free-form claim text only (rejected): inconsistent with retrieval seeds and vulnerable to surface-form mismatch.
- Enumerate paths for bridge scoring in M1 (rejected): deferred due combinatorial overhead; diffusion/PPR kept as locked default for M2.

## 8) Failure modes / limitations + next-milestone dependencies

- Current Component 1 logs in this workspace are split across claim-level metrics and pair-level PV lines; M1 parser must support normalized `triples` input plus flexible field names.
- No bridge recovery or abstention exists yet (expected until M2/M3).
- M1 depends on this milestone’s locked contracts for anchors (`Entity_set`), mask formulas, and model-size defaults.

Post-M3 audit note (2026-02-10):
- Later milestones refined implementation details beyond M0 scope (anchor augmentation guards, missing-anchor recovery connectivity handling, and relaxed PPR solver defaults). See `DEV_NOTES_M1.md`, `DEV_NOTES_M2.md`, and `DEV_NOTES_M3.md` for the final behavior.
