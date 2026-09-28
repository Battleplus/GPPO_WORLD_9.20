# W1 Oracle Pairing Observer Repair Archive

Date: 2026-09-28

This archive records the stopped oracle run, the offline root-cause audit, and the zero-call observer repair prepared for one future dynamic technical unit.

## Evidence status

- The stopped oracle run did not answer whether accurate action-consequence prediction is useful. Its seven branches remain a technical stop.
- The audit reproduced 983 coarse common identities and 149 mismatches. All 149 were explained: 135 telemetry cases compared branch-dependent downstream receipt records, while 8 command and 6 ACK cases were merged by coarse `command_id` grouping.
- Saved telemetry primitive fate was consistent. Missing command/ACK primitive addresses were not reconstructed, so the stopped gate was not retroactively passed.
- The completed repaired-runtime fair rerun is unaffected. The defect was in the oracle-specific verifier, not in the shared environment or communication primitive.

## Repair

`repair/communication_observer.py` wraps the frozen communication primitive at its real call boundary. It records branch, ordinal, source callsite, method, complete identity and parameters, and the primitive return value or exception. It delegates exactly once and preserves returned objects and exceptions.

The repaired verifier compares only exact common primitive calls. Branch-specific calls remain separate, duplicate calls retain multiplicity, parameter conflicts are explicit, and downstream delivery, expiration, acceptance, and lease results are not treated as random fate.

`repair/runner.py` is restricted to `validation-0000 / repeat-0`. It recomputes the qualifying window and actions from the frozen public state, performs no oracle materiality analysis, and cannot continue to the other 23 units.

## Verification

On WSL native ext4 with Python 3.11.16 and the frozen runtime dependencies, 34 zero-call tests passed. The final package manifest verified all 38 files byte-for-byte.

- Execution manifest SHA-256: `99643974f96681ebd1d70a24bfd1190218caf7868a08bb80fc5fdb38e98b9353`
- Package hashes SHA-256: `694e7fefabf7ac1139741b4cb742d520ac7ce6a5540ad3872b366a39e443bed4`
- Root-cause audit hashes SHA-256: `24a12e8cf7582258e85d4a1c8764588450897754e7f505df0e2ba0f941eef911`

## Authorization boundary

The single-unit request is `NOT_APPROVED`. This archive does not authorize environment construction, reset/step calls, model use, training, creation of an attempt, or continuation to the remaining oracle matrix.

## Archive scope

This Git archive contains the root-cause reports, evidence indexes, minimal observer and runner changes, zero-call tests, frozen protocol, unapproved budget, and final hashes. It is not a full backup or deployable runtime. Large raw step and branch logs, SQLite ledgers, checkpoints, credentials, and the raw one-shot authorization token are not included. Their local paths, sizes, and hashes are listed in `artifact-index.json`.

Historical 90 environment steps from the stopped oracle attempt, all earlier costs and negative results, and the 10 ms cost failure remain in force.
