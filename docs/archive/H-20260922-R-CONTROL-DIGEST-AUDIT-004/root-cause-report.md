# Root cause

Classification A applies: the target label and prefix digest use different canonical observation projections. The saved observation is identical. The pilot projection contains `flat`, `mask`, `version`, and `time`; the historical projection contains `flat`, `mask`, `version`, `public_entity_ids`, `continuation_actions`, and `trigger_flags`.

The pilot label digest is `029fdda5e21097ca33a4fdf738f9a0a5ada2a80d35440e4d3ea9e95100e25a87`. Recomputing the pilot projection from the saved observation reproduces it. The historical prefix digest is `969682e664ab722f982fd6e10af98291705ed2c311e155ae894d71733918e64f`. Recomputing the historical projection reproduces it. Identity and time checks pass: observation and label start are `2.0`, label end is `3.0`, and the full branch identity and exogenous key match.

The strict gate remains 23/24. The bridge-qualified view is 24/24 for R-control reuse only. No corrected trajectory is inferred or executed, and authorization remains `pending`.
