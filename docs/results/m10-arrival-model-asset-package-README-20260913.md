# M-10 arrival consequence asset package

Package scope: frozen arrival-to-region data, one bounded CPU training run,
recovery evidence, task-outcome audit, and the one-shot test/OOD evaluation.

Protocol: `world-gppo-9.11-arrival/0.1.0`.

The data was generated from simulator counterfactual branches.  It is not a
new blind test of the old service protocol and contains no real network
traffic.  Test/OOD were not used for model selection.  The model run used
CPU, four threads, seed 1101, eight-epoch maximum, validation patience 3,
4096-update maximum and a 3600-second wall-clock maximum; it stopped after
670 optimizer updates by validation patience.

The package does not claim strategy fusion, weak-communication usability,
GPU performance, or production safety.  Physical arrival is the current
research basis; host confirmation is a separate sensitivity basis.

The complete file association and SHA-256 values are in
`docs/results/m10-arrival-artifact-manifest-20260913.json`.
