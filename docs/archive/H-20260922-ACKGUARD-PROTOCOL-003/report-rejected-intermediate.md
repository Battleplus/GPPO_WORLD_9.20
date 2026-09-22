# ACK guard corrected protocol, phase 1

## Result

The corrected execution entry is implemented and statically audited. This phase performed no environment step, model forward, optimizer/world/offline update, formal attempt creation, budget reservation, server action, commit, or push.

The 24-branch manifest is fixed to parent-00..07, W1, seed-1101, prefix-0, repeat 0/1/2. The two old 20-step guard branches remain in their historical output and are excluded. The 24 historical R controls total 299 steps and were not rerun.

## Gate status

Current read-only ledger: `limit=384, reserved=20, verified=20, unknown=0, pending=0`; available credit is 364, below the required 384. A future authorization must pin the same ledger after an approved total-limit extension to 404. No extension or new ledger was created.

Historical R compatibility is `insufficient_evidence` for all 24 pairs. Native source, checkpoint, public initial snapshot, repeat/exogenous keys, reward/discount, and termination contracts are recorded. Complete historical action probabilities and independent hidden/cache non-pollution evidence are absent, so no R trajectory is promoted as a corrected-guard control.

## Entry behavior

The runner rejects missing or unapproved authorization, any hash mismatch, a non-fresh output, duplicate/partial identities, an existing new run/attempt, an unknown/pending reservation, or insufficient credit before constructing models, environments, or a budget object. Once authorized, each step reserves before `env.step`, verifies reward and budget completion, records complete probabilities and before/after state digests, and terminally records failures as `unknown` without retry or refund.

See `executable-protocol.md`, `protocol.json`, `branch-manifest.json`, `historical-R-compatibility.json`, `source-input-identities.json`, `budget-proposal.json`, and the focused test output for the reviewable contract.
