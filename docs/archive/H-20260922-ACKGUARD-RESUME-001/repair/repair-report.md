# ACK guard runner repair and resume preflight

The runner repair is static and the resume is blocked before model loading or environment execution.

- The first-pair gate now compares one dynamic guard branch with its reused historical R control while checking cumulative dynamic budget separately.
- Base snapshot checks use per-prefix semantic digests that include environment state, public observation, hidden state, and auxiliary cache state.
- Branch forward counters are recorded as per-branch deltas, and the first feedback is taken from the actual first step row.
- Resume output is isolated from the original run; completed branch IDs are skipped by identity and the original SQLite was opened read-only during preflight.
- Regression tests cover failed first-pair gates, the ordinary-run freshness gate, and missing/duplicate/corrupted completed-ledger records.
- Historical hidden/cache mutation proof is explicitly `not_recorded` because the old snapshot identity did not persist those hashes; current semantic hashes are not treated as historical proof.

The guard contract probe blocked execution: original action 24 with no continuation and NOOP probability 0.9 produced final action 0.
The guard module was intentionally not changed under this handoff.
This handoff records env=0, model=0, attempt=0; the old SQLite main-file hash remained unchanged.

