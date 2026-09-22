# Final supplement: ACK guard NOOP audit

The original 18-pass focused output and 15-pass/1-failure resume output remain preserved in `test-output-initial-18pass-15pass1fail.txt` with SHA-256 `3bb5b913c7e8cddf39b5958044b40d43d88ba3079ebc80141002ff904c30a013`. The old failure asserted the pre-fix NOOP bug and was not used as a final contract.

The updated resume test now verifies `original_action=24`, `final_action=24`, `contract_compatible=true`, and zero environment/model execution. The final three-file related suite summary is `35 passed`.

The new continuation case proves that an original highest allocation excluded by continuation selects a higher-probability legal NOOP, while a tie between the remaining allocation and NOOP keeps the smaller action index. Historical A=12/B=8/C=0 remains conditional on saved observations. A does not establish later state or hidden equality when any earlier step is B; both historical branches remain `insufficient_evidence` and are not certified as corrected-rule trajectories.

Scope guards now reject missing, duplicated, or out-of-scope branch/step rows before producing the fixed two-branch conclusion. No model, environment, update, budget reservation, or new attempt was used.
