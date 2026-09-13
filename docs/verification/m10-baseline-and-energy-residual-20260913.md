# M-10 baseline completion and narrowed consequence-model audit

Date: 2026-09-13  
Repository: `WORLD-GPPO_9.11`  
Branch: `world-model-consequence-v1`  
Source HEAD before this change: `7803df56fec0207fb210f4080c67a47ca890aa1a`

## Scope and budget

This stage did not resume the failed consequence-fusion matrix. It ran one
bounded pilot and the pre-registered baseline comparison only. The pilot used
validation tapes; the formal comparison used 16 final-test tapes shared by all
groups. The learning budget was 8192 environment steps for each of seeds
1101, 2203, and 3307 in each of GPPO and GPPO-History: 49152 formal steps in
total. PPO used 32 rollouts and 4 optimizer steps per rollout, hence 128
optimizer updates per seed. No event trigger, reward shaping, or consequence
model was used.

The pilot and formal runs used the dedicated environment with CUDA on an
NVIDIA GeForce RTX 3060 Laptop GPU, one process, two Torch threads, and a
single data-loading path. The formal training elapsed times were 204.29,
214.09, 227.97 seconds for GPPO and 247.33, 258.38, 251.84 seconds for
History, respectively. These are local measurements and do not imply GPU
competition real-time performance.

## Formal baseline results

The final-test tape contains 16 episodes and six tasks per episode. The table
therefore reports mean completed tasks per episode, not a new 288-task claim.

| group | seed 1101 | seed 2203 | seed 3307 | mean completed tasks/episode | mean return | mean actor calls/episode |
|---|---:|---:|---:|---:|---:|---:|
| legal traditional scheduler | — | — | — | 1.9375 | 1.2291 | 16.8750 |
| GPPO-Graph5 | 1.6875 | 1.8125 | 1.5625 | 1.6875 | -2.2110 | 16.8750 |
| GPPO-History-Graph5 | 1.7500 | 1.8750 | 1.6875 | 1.7708 | -1.0440 | 16.7917 |

The historical-information group does not establish an improvement over the
traditional rule or GPPO on this 16-tape comparison. These are paired
simulation results on shared tapes, not independent-task confidence claims.
They are not comparable to the old A/B/C completion denominators without the
old tape and protocol.

All three groups recorded zero duplicate accepts and zero unauthorized,
fenced, expired, or unknown-command executions in the checked audit fields.
The communication byte counts in the run are canonical JSON UTF-8 audit
serialization proxies, not measured network traffic. Recovery entries use
simulator event to first legally received telemetry; a missing time remains
null and is not converted to zero.

## Energy target and baseline correction

The existing generator computes `energy_delta` as
`sum(resource.energy_before) - sum(resource.energy_after)` over the whole
six-step branch window. It is therefore a system-energy target across all
UAV resources. Travel and service labels are local to the selected UAV/task;
they do not change the scope of the energy label. The generator provenance
string was corrected for future data to say
`selected-uav-task + system-energy`; existing data values and hashes were not
rewritten.

The public physics baseline uses only serialized public Graph-5 fields,
the same six-step window, and the configured simulator units (`travel_power`
0.35, `service_power` 1.0, `idle_power` 0.05). It is a transparent physical
proxy, not an exact hidden-state replay: it estimates other UAV activity from
their public idle flags and assumes the selected legal action's travel/service
path. Consequently, the previous energy MAE comparison is a system-energy
comparison against a public proxy, not a candidate-UAV energy comparison.

The no-update audit used only train/validation and grouped candidates by
`(parent_episode_id, prefix_id)`. On validation, the previous absolute model
had energy MAE 0.7215 and pairwise accuracy 0.6284; the public-physics
zero-residual baseline had MAE 2.9656 and pairwise accuracy 0.6009. However,
the model's mean selected-candidate regret was 0.1040 while the physics proxy
was 0.0 on the comparable groups. A constant train residual correction made
validation MAE 3.6920, worse than zero residual. Thus the absolute MAE gain
does not establish better candidate selection. Ties and equal ground-truth
pairs are excluded from pair scoring; each comparable prefix remains a
correlated candidate set, not independent episodes.

Loss auditing confirmed each head is normalized over its valid mask before
the available-head losses are averaged. Fully masked heads are omitted, not
treated as zero error. The previous best checkpoint was selected by the
record-weighted validation total; this is retained as historical evidence and
is not silently reinterpreted as a head-balanced objective.

The six-step labels describe one selected legal action followed by five
periodic NOOP steps while accepted leases continue. They do not describe an
arbitrary future controller.

## Narrowed hypothesis and decision

The versioned plan is in
`configs/world-gppo-9.11-energy-residual-v0.1.0.json`. It proposes keeping
travel/service as auditable physics estimates, learning only an energy
residual against the public system-energy proxy, and excluding the deadline
head from fusion until it independently passes its development gate. The
proposed future run is explicitly bounded at 694 train records, 659 validation
records, one seed, at most 8 epochs, patience 3, 4096 optimizer updates, and
3600 seconds; it was **not started** here.

The no-update audit does not justify starting that residual experiment for
policy fusion now: the constant residual is harmful and the existing model's
candidate selection is not better than the proxy. The baseline comparison is
therefore complete as a negative/diagnostic stage; no formal fusion matrix was
started.

## Evidence files

- Formal baseline run: `E:/Z博士/9.2日/WORLD-GPPO_9.11-local-runs/m10-baseline-comparison-formal-20260913-v1/`
- Pilot run: `E:/Z博士/9.2日/WORLD-GPPO_9.11-local-runs/m10-baseline-comparison-pilot-20260913-v1/`
- No-update audit: `E:/Z博士/9.2日/WORLD-GPPO_9.11-local-runs/m10-energy-residual-no-update-audit-20260913-v1.json`
- Prior consequence gate and checkpoint remain unchanged under `m10-consequence-*`.

This report does not claim that weak-communication usability passed, that
real network traffic was measured, or that the old server state changed.
