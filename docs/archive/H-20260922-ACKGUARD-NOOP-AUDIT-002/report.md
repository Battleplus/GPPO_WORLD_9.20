# ACK guard NOOP contract audit

handoff_id: H-20260922-ACKGUARD-NOOP-AUDIT-002

This is a zero-environment-step, zero-model-forward audit. The guard now retains every NOOP that is legal in the original public mask and ranks it with surviving allocation candidates using the existing probability and `(probability, -action_index)` tie-break. Task identity exclusion, continuation handling, the mask, ACK/lease behavior, reward, model, communication, and budget are unchanged.

The preserved initial output is `test-output-initial-18pass-15pass1fail.txt` (SHA-256: `3bb5b913c7e8cddf39b5958044b40d43d88ba3079ebc80141002ff904c30a013`); it records the focused 18-pass run and the old resume test's 15-pass/1-failure result. The old test is now renamed to `test_guard_contract_probe_confirms_noop_fix_without_execution` and asserts final action 24 plus `contract_compatible=true`.

The final related pure-function suite is recorded in `test-output-final.txt` with summary `35 passed`. It includes the original guard tests, the updated resume contract, and the NOOP contract tests.

## Historical decisions

The preserved two guard-on branches contain 20 decisions after fixed-scope assertions for exact branch identities and unique steps 1..10. Classification is A=12 provably unchanged, B=8 insufficient because full candidate probabilities are absent, and C=0 inconsistent or known mismatch. The eight B rows are the changed allocation rows where NOOP was legal; their new action cannot be established from selected probabilities alone. A is only a single-action proof from saved observations: if the first step is B, no later new-trajectory state or hidden equality follows. Therefore neither complete branch can be merged as an unchanged new-rule trajectory.

## Snapshot and identity limits

Historical hidden/cache non-pollution is B / `not_recorded`: snapshot identity confirms presence and current semantic digests, but the old process did not save an independent post-run comparison. The corrected first-pair gate remains `base_snapshot_unchanged=null`, `gate_status=insufficient_evidence`, `passed=false`. New and old guard identities are explicit, but full branch identity cannot be merged because the guard changed and the saved probability vector is incomplete.

## Resources and boundaries

The historical SQLite SHA-256 is `700b0c9173c8debd5cacb8066328dd2359bb913a128f3018710c8129de5e6223` and matches the supplied `700b0c9173c8debd5cacb8066328dd2359bb913a128f3018710c8129de5e6223`. Historical environment budget remains limit=384 reserved=20 verified=20 unknown=0 pending=0; audit calls are env.step=0, model forward=0, optimizer/world/offline updates=0, new attempt=0. Historical artifacts and SQLite were read only.

## Evidence

- Source identity: `source-identity.json`
- Per-decision and per-branch A/B/C table: `decision-evidence.json`
- Scope/reuse decision: `scope-decision.json`
- Hidden/cache evidence: `hidden-cache-evidence.json`
- Budget read-only audit: `budget-readonly-audit.json`
- Initial preserved test output: `test-output-initial-18pass-15pass1fail.txt` (SHA-256: `3bb5b913c7e8cddf39b5958044b40d43d88ba3079ebc80141002ff904c30a013`)
- Final test output: `test-output-final.txt` (SHA-256: `0b0a4933da14d397782c838be71dae2d7b3292381a966309cd5bac06c1d61304`)
- Final supplement: `final-supplement.md`
- Minimal diff: `minimal-diff.patch`

The complete result is `insufficient_evidence`; no environment replay, model inference, training, budget reservation, runner change, SQLite write, commit, push, release, or server action was performed.

## Next stage suggestion

If a later stage is explicitly authorized, treat the 12 A rows as saved-record invariants only. Require either complete historical candidate probabilities and independently saved hidden/cache snapshots, or a new guard-on execution with the corrected rule before merging the eight B rows or making any utility claim. This audit does not execute that stage.
