# M-10 authorized resume fix v1 artifact index

Date: 2026-09-08  
Code: `74f75d4`  
Report: `039b9fd`  
Local archive: `archive-m10-authorized-resume-fix-v1-20260908/`

## Verified artifacts

The remote files were downloaded before comparison. All 23 formal files matched their remote SHA-256; pilot key files and the patched source also matched.

| Artifact | Local SHA-256 | Notes |
|---|---|---|
| `formal-fix-v1/formal-results.json` | `b3502ac0da90c3b80ad8b41239c7b4d50c41297986d3a3ab8c5441de831ac35b` | 9 variant-seed records |
| `formal-fix-v1/trigger-threshold-selection.json` | `2267740a8c2a2a37e2594f5e3b68284e9dd8abe15cda6278c9d1135d162cc135` | validation-only threshold audit |
| `pilot-fix-v1/pilot-results.json` | `a1bb4718ab82458521a3484711ba58129be42154e47b50ab58e801d8fd1d9199` | post-fix pilot |
| `pilot-fix-v1/rollout-ledger.jsonl` | `e0897acd819a73630997fafd690e3a163acc4dcfbbf5f23ca6f571189c4a1631` | 1,536 pilot transitions |
| `pilot-fix-v1/world-model.pt` | `e31968ebe328b2deef46c8ba9d4150bf79e3a85d5aae5e293d391b9f84105ed3` | fixed model consumed by formal B/C |

Formal checkpoints are under `formal-fix-v1/formal/checkpoints/` with three files for each of A, B and C. Each has finite model and optimizer tensors. Per-seed formal records are under `formal-fix-v1/formal/records/`. The original failed pilot is separately downloaded under `original-pilot/`; its missing completion/result files are expected evidence of the preserved failure.

## Accounting boundary

Formal records include environment steps, rollout count, optimizer updates, actor decisions and continuation steps. The formal runner did not emit raw training rollout/event ledger, world-model call counters, full decision-chain latency samples, communication byte/message audit, or per-event safety ledger. These are missing evidence, not reconstructed from another release. The original failed pilot is under `original-pilot/` in the same downloaded archive and is retained as a negative result.

## Archive status

The local archive is complete and hash-verified. GitHub push/release publication is a separate remote operation; if HTTPS remains unavailable, status must remain “remote pending” and no old release or `main` may be changed.
