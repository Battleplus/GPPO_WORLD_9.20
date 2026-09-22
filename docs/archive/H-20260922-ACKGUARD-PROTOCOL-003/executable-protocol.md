# ACK guard corrected executable protocol

Protocol ID: `ackguard-corrected-protocol-20260922-v1`. Phase 1 prepares and audits the entry only; dynamic execution is not called.

The future run command is:

`python tools/run_ack_known_task_guard_corrected_20260922.py --mode run --authorization <reviewed-authorization.json> --manifest E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt\runs\finite-communication-ack-lease-fix-20260920\ackguard-corrected-protocol-20260922\branch-manifest.json --out E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt\runs\finite-communication-ack-lease-fix-20260920\ackguard-corrected-protocol-20260922\authorized-run`

The authorization must pin this protocol JSON, the corrected runner, the unchanged guard, the native 25-action source snapshot, the seed-1101 checkpoint, the prefix snapshot, and the same SQLite ledger. It must also prove a global environment-step limit of 404 with at least 384 available new-run credit, no pending/unknown reservations, and unused run/attempt identities. Every condition is checked before model or environment construction.

Each of 24 branches is restored from a deep copy. A decision row records every action probability, the public legal mask, original argmax, corrected choice, continuation identities, hidden/probe digests, and the pre-step environment digest. The budget reservation occurs immediately before `env.step`; a step exception, reward mismatch, logging failure, or finalization conflict records the actual reservation state (`verified`, `unknown`, or `pending`) and stops the run. There is no refund, retry, or zero fill.

The original 299-step R record is reusable as the matched R control only for the 23 of 24 pairs that pass the read-only compatibility gate. The gate directly compares each R step=1 `public_observation_sha256` with the immutable snapshot `prefix_digest`; the old guard snapshot output is excluded. That control statement is separate from corrected-guard trajectory reuse: the old guard treatment and a future corrected guard treatment are never inferred from R, and complete R action probabilities or post-run hidden/cache non-pollution evidence are not required for the original R control itself.

Primary metric: for each parent, average the three repeats, then give the eight parents equal weight for guard minus R utility at preference `(0.8,0.2)`, with `gamma=0.99`, where utility is `0.8*0.5*G_task + 0.2*1.0*G_energy`. Per-branch task completion and host confirmation come from the native step `info` schema (`completion_records`/task records or `task_outcome`) and the retained historical R definition; energy comes from `env.clock.resources`, and terminal counts/end reason come from native `info`. Step rows retain `actual_feedback` and branch rows retain `last_info` for postprocessing. Any negative result is retained; no retry is allowed.

Current preflight budget is `limit=384, reserved=20, verified=20, unknown=0, pending=0`, leaving 364 available. It cannot authorize the 384-step matrix. `budget-proposal.json` recommends an approval-only extension of the same ledger to total limit 404 while preserving the old 20; it was not applied.
