# First Pair Interface Check Erratum

The prior first-pair-interface-check.json is preserved but is not accepted by primary review. It encoded checks.base_snapshot_unchanged=true and passed=true even though historical_base_snapshot_check_verifiable=false.

The corrected semantics are checks.base_snapshot_unchanged=null, checks_known_passed=true for the independently verifiable fields, gate_status=insufficient_evidence, and overall passed=false. Unknown historical snapshot evidence cannot be promoted to a passing gate.

resume-prepare was not rerun and no prior resume JSON was overwritten. This erratum records env=0, model=0, attempt=0. The guard hash remains 6e73323f60647ea48fe05bd841aaa3e0631bd870eb786ad082a07ce007bb207b and the original SQLite hash remains 700b0c9173c8debd5cacb8066328dd2359bb913a128f3018710c8129de5e6223. The historical hidden/cache mutation status remains not_recorded.

Regression validation: 27 passed.
