# ACK guard corrected protocol, phase 1

## Result

The previous preflight was rejected and is retained as report-rejected-intermediate.md. This corrected phase performed no environment step, model forward, optimizer/world/offline update, formal attempt creation, budget reservation, server action, commit, or push.

The 24-branch manifest is fixed to parent-00..07, W1, seed-1101, prefix-0, repeat 0/1/2. The two old 20-step guard branches remain in their historical output and are excluded. The 24 historical R controls total 299 steps and were not rerun.

## Gate status

Current read-only ledger: `limit=384, reserved=20, verified=20, unknown=0, pending=0`; available credit is 364, below the required 384. A future authorization must pin the same ledger after an approved total-limit extension to 404. No extension or new ledger was created.

Historical R compatibility is `insufficient_evidence` for the original control contract across 23 of 24 pairs. Each pair directly compares the R step=1 `public_observation_sha256` with the corresponding immutable snapshot `prefix_digest` from `preference-vector-label-train-v1/snapshot-identity.json`, validated by `allocation-branch-preflight-v2/snapshot-interface-check.json`; the old guard snapshot output is excluded. The new guard trajectory is never inferred from R. Complete historical action probabilities and independent hidden/cache non-pollution evidence remain insufficient for corrected-guard trajectory reuse. The reported utility is `0.8*0.5*G_task + 0.2*1.0*G_energy` at `gamma=0.99`; task/host confirmation is read from native step `info` task records/completion records or `task_outcome`, with `last_info` retained for postprocessing.

## Entry behavior

The runner rejects missing or unapproved authorization, any hash mismatch, a non-fresh output, duplicate/partial identities, an existing new run/attempt, an unknown/pending reservation, incompatible historical R controls, or insufficient credit before constructing models, environments, or a budget object. The pending authorization template cannot authorize a run. Once manually authorized, each step reserves before `env.step`, persists full evidence before budget completion, verifies reward and budget completion, records complete probabilities and before/after state digests, and terminally records failures while retaining the actual reservation state (`verified`, `unknown`, or `pending`) and any secondary finalization error.

See `executable-protocol.md`, `protocol.json`, `authorization-template.json`, `branch-manifest.json`, `historical-R-compatibility.json`, `source-input-identities.json`, `budget-proposal.json`, and the focused test output for the reviewable contract.
