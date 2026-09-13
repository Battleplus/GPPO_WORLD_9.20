# M-10 candidate-consequence feasibility report

Date: 2026-09-13  
Repository commit: `9785cdf8c9c49ba7dc674973ba6d14311a4062cc`  
Protocol: `world-gppo-9.11-consequence/0.2.0-feasibility-h6`  
Data: `m10-consequence-data-feasibility-h6-global27-v020-20260913`

## Scope and contract

This is a local CUDA feasibility run, not a formal blind test and not a weak-communication availability result. The frozen contract is Graph-5/25-action with a 27-value public global context (normalized time, event signal, and ACK-confirmed continuation actions). It retains the existing `continuous_service_until_deadline` semantics: arrival is not completion, and deadline risk is evaluated from the simulator task lifecycle at the end of the six-step window.

The model sees the serialized public Graph-5 snapshot only. Hidden state is used only by simulator branches to produce labels. A selected legal action is followed by five periodic NOOP decisions while accepted leases continue. Travel and deadline labels outside the observed window are masked; NOOP task-specific travel/service/deadline labels are masked, while total-system energy remains observable.

The data was generated with 32 parent episodes per split, four public-hash-selected prefix lengths (1, 2, 4, 6), at most 25 candidate branches per prefix, and shared realized exogenous streams. Train/validation/test/OOD are parent-episode disjoint. This development data is not presented as a new blind benchmark.

## Execution evidence

The initial CUDA smoke pilot completed 72 updates. Its planned interruption branch stopped at update 37 and resumed to update 72. The deterministic verifier found exact equality for sample order, per-update losses and gradient norms, final model state, optimizer state, RNG state, data order, and reloaded predictions (maximum reported difference 0).

The feasibility training used one model seed (1101), four CPU threads, one data-loader worker, and the RTX 3060 Laptop GPU. Runtime was Python 3.11.15, PyTorch 2.7.0+cu128, CUDA 12.8, NumPy 1.26.0, with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. It stopped by the registered validation patience rule after 9 epochs and 6,246 actual optimizer updates; cumulative training elapsed was 2,471.015 seconds. No non-finite loss, gradient, parameter, or optimizer state was observed.

There was one stale `running` status after an interrupted local process at update 5,722. No process with that run identity remained. The recovery checkpoint was preserved and explicitly resumed after a Windows PID-check compatibility fix. The compatibility record is retained; model, data, protocol, and core training identity were unchanged.

## Prediction gate

Model selection used validation only. The registered gate required at least two continuous heads to beat the best simple baseline, deadline Brier score to beat the train event-rate baseline, and at least two heads with candidate pairwise accuracy at least 0.55.

Validation results:

- Continuous-head baseline gate: **1/3**, so failed.
- Deadline Brier: model **0.2666**, train event-rate baseline **0.2499**; failed.
- Candidate ranking: 3 heads reached the 0.55 threshold, but this does not override the two failed conditions.
- Validation gate: **failed**.

Selected validation details:

| head | model error/calibration | train-mean/rate | public-physics | valid labels | masked/not applicable |
|---|---:|---:|---:|---:|---:|
| travel_time | MAE 0.4593; pairwise 0.7471 | MAE 0.7456 | MAE 0.3318 | 345 | 314 |
| service_progress | MAE 0.6821; pairwise 0.6764 | MAE 0.7307 | MAE 0.6536 | 531 | 128 |
| energy_delta | MAE 0.7215; pairwise 0.6284 | MAE 1.0308 | MAE 2.9656 | 659 | 0 |
| deadline_risk | Brier 0.2666; pairwise 0.3789 | Brier 0.2499 | Brier 0.3505 | 368 | 291 |

After the model was frozen, the held-out development test/OOD files were evaluated for reporting only. Test model MAE/Brier was travel 0.4401, service 0.6789, energy 0.7270, deadline Brier 0.2481; OOD was travel 0.3164, service 1.3898, energy 1.6248, deadline Brier 0.3007. OOD service candidate pairwise accuracy was 0.4696 and OOD deadline Brier was worse than its 0.2508 event-rate baseline. These are not policy gains.

## Decision and limits

The candidate-consequence model is **trained and technically loadable**, but its registered prediction gate is **not passed**. Therefore the planned traditional/GPPO/GPPO-History/fused policy matrix was not executed, no strategy gain is claimed, and no additional budget was added. The following remain unverified:

- candidate predictions improving scheduling beyond legal historical information;
- policy-level task completion, deadline, recovery, safety, communication, and latency gains;
- formal multi-seed strategy comparison;
- real network traffic or production weak-communication availability;
- equivalence to the unavailable original server environment.

This negative feasibility result does not show that world models are generally ineffective. It shows only that this frozen h6 candidate model did not satisfy the predeclared evidence gate on this development distribution.

