# Event-trigger-aware GPPO fair replication

Experiment ID: `event-trigger-aware-gppo-fair-replication-20260919-v1`  
Status: complete. The pre-registered acceptance gate is **not passed**; this is a retained negative result.

## Outcome

The primary comparison is C - B under W1/W2 with equal condition weight and equal training-seed weight. Physical on-time arrival rates were A `0.6723`, B `0.6701`, and C `0.6636`. C - B was `-0.651` percentage points, with the paired-parent 95% bootstrap interval `[-2.083, 0.781]` percentage points. The interval lower bound is not above zero and the required 1 percentage-point improvement was not reached. C - A was `-0.868` percentage points, interval `[-2.474, 0.694]` percentage points.

Two of three seeds had a strictly positive C - B difference, but that criterion alone does not pass the gate. Maximum recorded security violations and illegal actions were both zero. The supported conclusion is therefore only that this protocol and these measured seeds did not establish an improvement from event-trigger training.

## Per-seed W1/W2 rates

| seed | A | B | C | C - B (pp) | C - A (pp) |
|---:|---:|---:|---:|---:|---:|
| 1101 | 0.8802/0.4635 | 0.8802/0.4661 | 0.9089/0.4401 | 0.130 | 0.260 |
| 2203 | 0.8906/0.4948 | 0.8880/0.4870 | 0.8542/0.4609 | -2.995 | -3.516 |
| 3307 | 0.8385/0.4661 | 0.8385/0.4609 | 0.8672/0.4505 | 0.911 | 0.651 |

## Training and evaluation budget

All six groups completed 8192 verified environment steps. Policy optimizer calls were 63 per group (378 total, below the 64-call maximum per group); world optimizer updates were zero. Evaluation completed 1728 unique episodes using 24437 verified environment steps and zero optimizer calls. Evaluation cost was:

| config | env steps | actor calls | continuation steps | world forwards |
|---|---:|---:|---:|---:|
| A | 8142 | 8142 | 0 | 8142 |
| B | 8141 | 7389 | 752 | 8141 |
| C | 8154 | 7412 | 742 | 8154 |

The actor-call reduction in B/C is not reported as a reduction in total compute cost. Decision latency includes observation encoding, trigger decision, actor/world inference, hidden/cache action transition and action selection; environment stepping and persistence I/O are excluded. Pooled P95/P99 were `3.668`/`4.426` ms over `24437` decisions.

## Replay repair evidence

The preserved real failure prefix identifies episode 8, scenario `train-mixed-seed-851003-W2`, with the first divergence at step 3: action 1, `submit_command=false`, actor decision false, public new-command mask false, but `continuation_actions=[1]` and valid continuation feedback `reuse_existing`. Step 12 repeats the same class for action 22. The repair was therefore exactly the suspected semantic error: replay used the new-command mask to reject a legal continuation. It was not inferred from a masked action alone; the paired ledger explicitly restored decision type.

The repaired contract keeps current public command-mask checks for new actor actions, checks continuations only against the saved public continuation set and `by_action`, and uses the same hidden transition function online and during replay. It does not delete safety checks, turn continuation into a new submission, use NOOP fallback, call `env.step`, resample, read future truth, clear hidden state, or call optimizer during replay. The real prefix replay had 15 steps, zero environment steps, zero optimizer calls, exact discrete fields, and maximum stored-vs-rebuilt policy/world hidden difference 0.0. Targeted fixtures passed 14 tests in the launch check, including legal continuation with a false new mask, rejection of illegal new assignment and invalid continuation, and episode-boundary decision-type preservation.

## Provenance and publication

The three original WD checkpoint hashes, source snapshot manifest, new training/evaluation tape manifests, SQLite budget database, checkpoints, raw episode ledger, analysis and evidence hashes are in `artifact-manifest.json`. The old development validation, historical failure runs and old budget database remain excluded from the independent result. No tuning, seed replacement, tape replacement or repeated evaluation was performed.

Git publication is a separate step from local archival; signature retry and upload claims are reported separately in the final task response.
