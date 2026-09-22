# Limitations and next recommendation

The bridge validates one saved observation, the full branch identity, and the before/after times. It does not prove hidden/cache non-pollution after the historical run, and it does not certify a corrected guard trajectory. The original strict result and digest values remain preserved in this bundle.

Keep the bridge as qualified R-control evidence and retain the strict 23/24 result as the unqualified historical gate. The source change is recorded in `minimal-diff.patch` and is limited to the audit bridge plus its pure schema tests. For future collection, emit an explicit schema identifier with every observation digest or use one canonical projection at both label and prefix boundaries.
