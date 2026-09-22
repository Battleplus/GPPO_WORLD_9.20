# ACK guard NOOP contract audit

handoff_id: H-20260922-ACKGUARD-NOOP-AUDIT-002

This is a zero-environment-step, zero-model-forward audit. The guard now retains every NOOP that is legal in the original public mask and ranks it with surviving allocation candidates using the existing probability and `(probability, -action_index)` tie-break. Task identity exclusion, continuation handling, the mask, ACK/lease behavior, reward, model, communication, and budget are unchanged.

The focused pure-function regression covers: no exclusion, NOOP as the original maximum, NOOP ordering after exclusion, only NOOP, probability ties, existing continuation identities, and stubs proving no environment step or model forward. It passed 18 tests. The existing resume regression was run once without modifying its source: 15 tests passed and the one expected failure remains `test_existing_guard_contract_gap_is_exposed_without_changing_guard`, which asserted the old buggy final action 0 instead of the corrected final action 24.

## Historical decisions

The preserved two guard-on branches contain 20 decisions. Classification is A=12 provably unchanged, B=8 insufficient because full candidate probabilities are absent, and C=0 inconsistent or known mismatch. The eight B rows are the changed allocation rows where NOOP was legal; their new action cannot be established from selected probabilities alone. Therefore neither complete branch can be merged as an unchanged new-rule trajectory.

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
- Test output: `test-output.txt` (SHA-256: `3bb5b913c7e8cddf39b5958044b40d43d88ba3079ebc80141002ff904c30a013`)
- Minimal diff: `minimal-diff.patch`

The complete result is `insufficient_evidence`; no environment replay, model inference, training, budget reservation, runner change, SQLite write, commit, push, release, or server action was performed.

## Next stage suggestion

If a later stage is explicitly authorized, treat the 12 A rows as saved-record invariants only. Require either complete historical candidate probabilities and independently saved hidden/cache snapshots, or a new guard-on execution with the corrected rule before merging the eight B rows or making any utility claim. This audit does not execute that stage.
