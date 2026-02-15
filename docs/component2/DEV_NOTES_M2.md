# DEV NOTES M2 (Bridge Rescue + Policy-Aware Recovery)

## 1) Implemented files/classes/functions/signatures

New code files:
- `src/component2/bridge_rescue.py`
  - `@dataclass(frozen=True) class BridgeScore`
  - `@dataclass(frozen=True) class PPRDiagnostics`
  - `class BridgeRescuePPR`
  - `BridgeRescuePPR.compute_node_scores(self, graph: EvidenceGraph) -> np.ndarray`
  - `BridgeRescuePPR.compute_node_scores_with_diagnostics(self, graph: EvidenceGraph) -> Tuple[np.ndarray, PPRDiagnostics]`
  - `BridgeRescuePPR.compute_bridge_scores(self, graph: EvidenceGraph) -> List[BridgeScore]`
- `src/component2/recovery.py`
  - `@dataclass(frozen=True) class RecoveryMetrics`
  - `@dataclass(frozen=True) class RecoveryResult`
  - `class BridgeRecoveryPolicy`
  - `BridgeRecoveryPolicy.evaluate(self, triples, anchors, bridge_scores, esi_geom) -> RecoveryMetrics`
  - `BridgeRecoveryPolicy.apply_recovery(self, triples, selected_bridge_ids) -> RecoveryResult`
  - `connectivity_for_active(triples, anchors) -> float`
- `src/component2/run_m2.py`
  - `run_smoke_m2(input_jsonl, output_root, max_claims, config=DEFAULT_CONFIG) -> Path`
  - CLI entrypoint: `python3 -m component2.run_m2`

Updated files:
- `src/component2/config.py` (added/updated bridge solver defaults: `ppr_alpha`, `ppr_max_iters`, `ppr_tolerance`)
- `src/component2/__init__.py` (exports `BridgeRescuePPR`, `BridgeRecoveryPolicy`)

New tests:
- `tests/component2/test_bridge_rescue.py`
- `tests/component2/test_recovery.py`
- `tests/component2/test_run_m2.py`

## 2) Exact numeric defaults + source

Locked from `docs/component2/COMPONENT2_DECISIONS.md` and surfaced via `src/component2/config.py`:
- `ppr_epsilon = 0.001`
- `neutral_cap_gamma = 2.0`
- `recovery_tau_esi = 0.3`
- `recovery_top_k = 3`
- `recovery_tau_rpi_legacy = 0.15` (logged for reference only; not used for trigger)

Newly needed and now locked for M2 PPR solver:
- `ppr_alpha = 0.15`
- `ppr_max_iters = 100`
- `ppr_tolerance = 1e-6`

Other inherited defaults used during M2 smoke:
- `mask_sup = sigmoid(4*p_ent - 2)`
- `mask_ref = sigmoid(4*p_con - 2)`
- `hidden_dim = 256`, `num_heads = 4`, `rel_emb_dim = 64`, `random_seed = 13`

## 3) Tiny toy walkthrough with intermediate numbers

Toy anchors: `A`, `C`.

Active edges:
- `a1`: `A-left-B` (`p_ent=0.65`, `p_con=0.05`, `p_neu=0.30`)
- `a2`: `C-right-D` (`p_ent=0.62`, `p_con=0.06`, `p_neu=0.32`)

Suspended edges:
- `s_bad1`: `B-r_noise-X` (`p_ent=0.83`, `p_con=0.05`, `p_neu=0.12`)
- `s_bad2`: `B-r_noise_2-Y` (`p_ent=0.82`, `p_con=0.05`, `p_neu=0.13`)
- `s_bad3`: `D-r_noise_3-W` (`p_ent=0.81`, `p_con=0.05`, `p_neu=0.14`)
- `s_bridge`: `B-r_bridge-C` (`p_ent=0.52`, `p_con=0.08`, `p_neu=0.12`)

Mask examples:
- `m_sup(s_bad1)=sigmoid(4*0.83-2)=0.789182`
- `m_ref(s_bad1)=sigmoid(4*0.05-2)=0.141851`
- `m_sup(s_bridge)=sigmoid(4*0.52-2)=0.519989`
- `m_ref(s_bridge)=sigmoid(4*0.08-2)=0.157095`

Bridge scores from smoke run (`logs/component2/m2/20260210_095015_447187/predictions.jsonl`):
- `BridgeBonus'(s_bridge)=0.0189136`
- `BridgeBonus'(s_bad1)=0.0103967`
- `BridgeBonus'(s_bad2)=0.0099324`
- `BridgeBonus'(s_bad3)=0.0046006`

Connectivity deltas (`k=3`):
- `Conn(A)=0.0` (anchors disconnected in Active graph)
- Rel-policy top-3 S (`s_bad1,s_bad2,s_bad3`) gives `Conn(A∪S_rel_top3)=0.0` so `RPI_rel@3=0.0`
- Bridge-policy top-3 S includes `s_bridge`, giving `Conn(A∪S_bridge_top3)=1.0`
- `DeltaConn_bridge@3 = RPI_bridge@3 = 1.0`

Trigger:
- `esi_geom=0.2 < 0.3` and `DeltaConn_bridge@3=1.0 > 0` → recovery executes.

PPR diagnostic example (same run):
- `converged=true`, `iterations=90`, `residual_l1=8.89e-07`
- `has_disconnected_anchors=false`, `anchor_pairs_connected=1/1`
- These diagnostics are now logged per-claim and aggregated in `summary.json`.

## 4) Why each decision was taken

- Bridge scoring on `A∪S∪C` with anchor-conditioned PPR was implemented to capture potential connectors in S before moving anything.
- Neutral cap `BridgeBonus' = BridgeBonus * (1-p_neu)^gamma` was applied directly to suppress high-neutral shortcut edges.
- Trigger uses `esi_geom < tau_esi AND DeltaConn_bridge@k > 0` (not rel-only RPI) to satisfy locked policy-aware recovery decision.
- Recovery selection prefers S edges linking different Active connected components, matching the bridge-intent constraint from decisions.

## 5) Logs/outputs produced + exact paths

M2 smoke run command:
- `PYTHONPATH=src python3 -m component2.run_m2 --max-claims 5 --output-root logs/component2`

Run directory:
- `logs/component2/m2/20260210_095015_447187/`

Files:
- `logs/component2/m2/20260210_095015_447187/run_config.json`
- `logs/component2/m2/20260210_095015_447187/summary.json`
- `logs/component2/m2/20260210_095015_447187/predictions.jsonl`

## 6) Tests added + what they validate

- `tests/component2/test_bridge_rescue.py`
  - validates epsilon-weighted bridge score formula and neutral-capped formula on toy graph
  - validates monotonicity when only `rel(e)` changes.
  - validates disconnected-anchor and non-convergence diagnostic flags.
  - validates default solver convergence behavior under updated PPR defaults.
- `tests/component2/test_recovery.py`
  - validates policy-aware case where rel top-k fails but bridge top-k reconnects anchors.
  - validates trigger gate behavior (`esi` high => no recovery).
  - validates missing-anchor connectivity semantics and recovery behavior when an S-edge introduces an anchor into Active graph.
- `tests/component2/test_run_m2.py`
  - validates end-to-end M2 run writes reproducibility/config/prediction logs.
  - validates recovery diagnostics (`rpi_rel_at_k`, `rpi_bridge_at_k`, `delta_conn_bridge_at_k`) and output change after rerun.
  - validates PPR diagnostics are present in per-claim and summary outputs.

Suite run:
- `PYTHONPATH=src python3 -m unittest discover -s tests/component2 -v`

## 7) Alternatives considered (>=2) + why not chosen

- Path enumeration over all candidate anchor paths:
  - rejected due combinatorial blow-up and unstable runtime on dense subgraphs.
- Directed-only connectivity for `Conn` metrics:
  - rejected because existing sufficiency/connectivity interpretation in this repo is undirected entity connectivity.
- Trigger based on legacy `rpi_at_k > 0.15`:
  - rejected for M2 trigger because locked decisions require bridge-policy `DeltaConn_bridge@k`.

## 8) Failure modes / limitations + next-milestone dependencies

- Current bridge selection priority is component-aware but still greedy; it may recover extra non-bridge S edges when `k=3`.
- M2 is deterministic numpy inference, not yet trained end-to-end on FactKG full data.
- Next milestone (M3) depends on M2 diagnostics fields to implement:
  - selective prediction/risk-coverage (T9),
  - rationale top-N extraction/faithfulness checks (T10).

## 9) Post-Review Hardening Update (2026-02-10)

- Added explicit PPR diagnostics in `src/component2/bridge_rescue.py`:
  - convergence status, iterations, and residual (`residual_l1`)
  - anchor-pair connectivity stats and disconnected-anchor flag.
- Logged PPR diagnostics in M2 outputs via `src/component2/run_m2.py`:
  - per-claim `ppr_diagnostics` in `predictions.jsonl`
  - aggregate counts in `summary.json` (`num_ppr_not_converged`, `num_disconnected_anchor_graphs`).
- Added/extended tests:
  - `tests/component2/test_bridge_rescue.py` now covers disconnected-anchor diagnostics and non-convergence flag behavior.
  - `tests/component2/test_run_m2.py` now validates diagnostic fields in output files.
- Latest validation run:
  - `PYTHONPATH=src python3 -m unittest discover -s tests/component2 -v` → 46 tests passed.
  - `PYTHONPATH=src python3 -m component2.run_m2 --max-claims 5 --output-root logs/component2`
  - Output path: `logs/component2/m2/20260210_095015_447187/`.

## 10) Post-M3 stability patch summary (2026-02-10)

- Recovery connectivity logic now treats missing anchors as disconnected (`conn=0` for that pair) instead of dropping them from pair accounting.
- Bridge selection now treats an S-edge connecting an in-graph node to a missing-anchor node as component-connecting, improving recovery ranking in that edge case.
- Combined with anchor-selection and PPR-solver fixes, this resolved the previous "single-anchor + non-converged PPR" failure mode observed in M3 runs.
