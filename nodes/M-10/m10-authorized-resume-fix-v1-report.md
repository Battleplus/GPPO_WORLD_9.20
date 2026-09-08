# M-10 authorized resume / triggered continuation fix

Date: 2026-09-08  
Repository branch: `execute-r02-20260905`  
Code commit: `74f75d4` (`fix(m10): keep triggered critic context on continuation`)  
Server source snapshot: `/home/user1/m10-authorized-training-20260908-fix-v1`  

## Scope

This run was authorized to resume the paused work. It fixed one concrete runtime defect and then ran the existing bounded pilot and formal A/B/C protocol. It is not a weak-communication availability pass and does not replace the historical releases or negative results.

The original pilot at `/home/user1/m10-runs/20260908-authorized-resume/pilot` is preserved. It failed in C before completion with `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x0 and 8x128)`. The failing path was triggered continuation: a context-enabled policy critic received the base-only gated vector when risk was below threshold.

## Code change

`collect_rollout` and `evaluate_policy` now use the already computed full context vector for `value_only` whenever the policy has `context_dim`. The actor is still not called on continuation steps, no new command is submitted, and no second world-model inference is performed. The transition stores the same context-sized vector so PPO sequence evaluation has a stable input shape. Base policies are unchanged.

This preserves the existing execution contract: continuation maintains an accepted command; the controller, not the actor distribution, supplies the repeated action; lease/ACK/fencing gates remain in the environment.

## Verification

- Local targeted regression: `13 passed`.
- Local full suite: `164 passed` using a dedicated writable `--basetemp`; the initial default-temp run had 7 Windows permission errors, not code failures.
- Server direct regression (the server venv has no pytest): `DIRECT_REGRESSION_PASS`.
- Fixed pilot: A/B/C each 512 environment steps, 2 rollouts, 8 optimizer updates; C completed with 301 actor decisions and 211 continuation steps.
- Pilot ledger: 1,536 valid JSON lines, split 512 per fusion; no non-finite token was found by the audit.
- Formal: 9/9 variant-seed records and 9/9 checkpoints; every checkpoint state and optimizer tensor was finite and optimizer state was present.

## Fixed pilot evidence

The pilot is a correctness/throughput check only. Validation means are:

| Variant | completed | expired | rejected | actor decisions | continuation |
|---|---:|---:|---:|---:|---:|
| A Graph-5 Base | 1.875 | 4.125 | 3.9375 | 512 | 0 |
| B Graph-5 World | 2.000 | 4.000 | 3.6250 | 512 | 0 |
| C Graph-5 Triggered | 0.750 | 5.250 | 1.6250 | 301 | 211 |

These are 16 validation tapes for one training seed and are not a deployment claim.

## Formal protocol and result

Formal training used the existing `tools/run_m10_weak_comm.py` protocol: Graph-5, same composite communication setting and tapes, from-scratch A/B/C policies, seeds `1101,2203,3307`, `12,288` environment steps per variant-seed, 48 rollouts and 192 optimizer updates per run. The fixed world model was the pilot-generated model; it was not retrained during formal policy training. The formal run completed in about 49 minutes.

| Variant | completed mean | expired mean | rejected mean | return mean | actor calls / env steps |
|---|---:|---:|---:|---:|---:|
| A Graph-5 Base | 1.7500 | 4.2500 | 3.5417 | -1.2331 | 12288 / 12288 |
| B Graph-5 World | 1.7083 | 4.2917 | 3.3958 | -1.7629 | 12288 / 12288 |
| C Graph-5 Triggered | 1.0208 | 4.9792 | 1.6250 | -10.7559 | 7451.7 / 12288 |

Means are over the 16 final tapes within each evaluation and then the three training seeds; they do not establish statistical superiority. C reduced actor calls by about 39.3% but did not preserve task outcomes or return. Validation threshold selection found no candidate satisfying the frozen no-effect and zero-rejection constraint with lower actor cost, so the configured `0.45` threshold was retained as a negative-result evaluation. Safety gates were not disabled.

The fixed world model's recorded historical test metrics remain: reward RMSE `3.3253`, event BCE `0.4910`, and done BCE `0.5380`; its corresponding simple baseline metrics were reward RMSE `3.4007`, event BCE `0.1678`, and done BCE `0.2237`. These are not evidence that the model-triggered policy improves the task.

## Artifact and ledger limitation

The pilot wrapper recorded a raw `rollout-ledger.jsonl`. The formal runner recorded per-seed JSON records and checkpoints but did not emit a formal raw rollout ledger. This report therefore does not claim a formal event-level training ledger; the missing artifact is a reproducibility limitation, not silently reconstructed data.

Downloaded archive: `archive-m10-authorized-resume-fix-v1-20260908/`. Remote and local SHA-256 matched for all 23 formal files, the post-fix pilot key artifacts, and the original failed pilot ledger/model. The original failed pilot intentionally has no complete `pilot-results.json` or `run-complete.json`. Representative hashes:

- `formal-fix-v1/formal-results.json`: `b3502ac0da90c3b80ad8b41239c7b4d50c41297986d3a3ab8c5441de831ac35b`
- `formal-fix-v1/trigger-threshold-selection.json`: `2267740a8c2a2a37e2594f5e3b68284e9dd8abe15cda6278c9d1135d162cc135`
- `pilot-fix-v1/pilot-results.json`: `a1bb4718ab82458521a3484711ba58129be42154e47b50ab58e801d8fd1d9199`
- `pilot-fix-v1/world-model.pt`: `e31968ebe328b2deef46c8ba9d4150bf79e3a85d5aae5e293d391b9f84105ed3`
- `original-pilot/rollout-ledger.jsonl`: `737b5adb30c7b4be188ceeeb68e11ac6ffedc4ebc13b1674975af299cebbdbc8`

## Conclusion and limits

The concrete triggered-continuation shape defect is fixed and covered locally and on the server. The bounded formal matrix is a valid post-fix training artifact, not evidence of stable world-model or trigger benefit. In this run B did not improve the reported task outcomes over A, and C saved actor computation at the cost of worse outcomes; the negative result is retained.

The historical weak-communication results, old failed pilot, old releases, and prior negative findings remain valid only under their original source/protocol versions. Full weak-communication availability remains **not passed**. Real communication requirements, control period, mission scale, return-to-base, battery swap/charging scope, and human presentation activities remain outside this run and require separate confirmation.
