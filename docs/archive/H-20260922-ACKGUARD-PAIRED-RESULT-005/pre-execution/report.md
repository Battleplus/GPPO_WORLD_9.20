# ACK guard corrected protocol, phase 1

## Result

The previous preflight was rejected and is retained as report-rejected-intermediate.md. This corrected phase performed no environment step, model forward, optimizer/world/offline update, formal attempt creation, budget reservation, server action, commit, or push.

The 24-branch manifest is fixed to parent-00..07, W1, seed-1101, prefix-0, repeat 0/1/2. The two old 20-step guard branches remain in their historical output and are excluded. The 24 historical R controls total 299 steps and were not rerun.

The corrected runner now pins the H-004 bridge helper, provenance files, and verified remote receipt. At authorization time it rebuilds the strict failure row from current labels, branch results, manifest, and prefix snapshots, evaluates the current bridge schema, and re-runs compatibility; the stored qualified 24/24 JSON is not used as a trusted boolean. The real read-only bridge check returned strict `23/24` and bridge-qualified `24/24`.

## Gate status

Current read-only ledger: `limit=384, reserved=20, verified=20, unknown=0, pending=0`; available credit is 364, below the required 384. A future authorization must pin the same ledger after an approved total-limit extension to 404. No extension or new ledger was created.

Historical R compatibility is `insufficient_evidence` for the strict original control contract across 23 of 24 pairs. The dynamic authorization gate separately recomputes the H-004 saved-observation bridge for the strict failure row(s) from current labels, branch results, manifest, prefix snapshots, pinned helper/provenance hashes, and the verified remote receipt; it requires qualified 24/24 at authorization time and does not trust the saved qualified JSON boolean. Each pair directly compares the R step=1 `public_observation_sha256` with the corresponding immutable snapshot `prefix_digest` from `preference-vector-label-train-v1/snapshot-identity.json`, validated by `allocation-branch-preflight-v2/snapshot-interface-check.json`; the old guard snapshot output is excluded. The new guard trajectory is never inferred from R. Complete historical action probabilities and independent hidden/cache non-pollution evidence remain insufficient for corrected-guard trajectory reuse. The reported utility is `0.8*0.5*G_task + 0.2*1.0*G_energy` at `gamma=0.99`; task/host confirmation is read from native step `info` task records/completion records or `task_outcome`, with `last_info` retained for postprocessing.

## Entry behavior

The runner rejects missing or unapproved authorization, any hash mismatch, a non-fresh output, duplicate/partial identities, an existing new run/attempt, an unknown/pending reservation, incompatible historical R controls, or insufficient credit before constructing models, environments, or a budget object. The pending authorization template cannot authorize a run. Once manually authorized, each step reserves before `env.step`, persists full evidence before budget completion, verifies reward and budget completion, records complete probabilities and before/after state digests, and terminally records failures while retaining the actual reservation state (`verified`, `unknown`, or `pending`) and any secondary finalization error.

Verification: bridge authorization tests `6 passed`; corrected protocol regression `49 passed`; schema tests `5 passed`; runner and schema compilation passed. These checks used only saved files or stub runtime fixtures and performed zero environment steps, model forwards, updates, reservations in the authorized SQLite ledger, or formal attempts.

See `executable-protocol.md`, `protocol.json`, `authorization-template.json`, `branch-manifest.json`, `historical-R-compatibility.json`, `source-input-identities.json`, `budget-proposal.json`, `analysis-and-stop-rules.json`, `analyze_results.py`, and the focused test output for the reviewable contract.


集中审查补充：动态授权入口已接入绑定当前输入的 H-004 桥接重算。纯分析器已修复重复/缺失/身份/终止/计费反馈统计，独立10项合成测试通过。执行申请见 authorization-request.md，仍待明确批准同一SQLite从384扩至404；没有配对实验结果。
