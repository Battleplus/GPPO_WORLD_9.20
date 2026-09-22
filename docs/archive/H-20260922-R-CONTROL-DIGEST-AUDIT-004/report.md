# R-control digest audit

Target: `parent-00|W1|seed-1101|prefix-0|repeat-0|mode-R`.

The mismatch is classification A: schema-only. One saved observation recomputes the pilot digest `029f...` and historical digest `9696...`; branch identity, exogenous key, and times also match. Strict compatibility remains 23/24. With the pinned saved-observation bridge, compatibility is 24/24. No environment step, model forward, update, budget write, or new attempt was performed. Authorization remains pending.

See `root-cause-report.json`, `schema-comparison.json`, `bridge-evidence.json`, `compatibility-strict.json`, `compatibility-bridge-qualified.json`, `branch-statuses.json`, `source-input-identities.json`, `validation.json`, and `independent-verification.json`. The actual source patch relative to the frozen phase-1 runner is `minimal-diff.patch`; it contains the runner bridge hook, current-authority binding checks, the schema helper, and pure regression tests. The strict result remains `23/24`; the saved bridge-qualified result remains `24/24`.
