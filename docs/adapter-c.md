# Optional pre-audit context adapter

Adapter C prepares prespecified legitimate audit events for the generic
`fp_persistence.audit()` API. It does not train a model, select a threshold,
inspect persistence outcomes, or choose an analysis window.

The caller must prespecify and document the training and audit periods,
legitimate-event definition, entity definition, amount field, coarse context,
minimum final-stratum support, and exposure bounds. Timestamps must carry an
explicit timezone and are normalized to UTC. Amount cutpoints are empirical
quintiles calculated from legitimate training events only.

`resolve_residual_strata()` constructs a disjoint fine-to-coarse partition on
the unresolved residual rows at each level:

1. UTC date, six-hour block, amount bin, and coarse context;
2. UTC date, six-hour block, and amount bin;
3. UTC date and six-hour block;
4. UTC date.

Non-final groups are accepted only when they meet the configured minimum on
the remaining rows. An undersized final-day residual raises an error. Accepted
fine groups never re-enter coarser grouping.

The adapter returns final-stratum labels; the released generic audit remains
responsible for the persistence and fixed-count reference calculations. The
helper `exact_legitimate_budget()` exposes the frozen paper-profile integer 2%
count for paper reproduction. It is not a default imposed by the generic API.

The example CLI is `examples/prepare_audit_inputs.py`. Event-level export is
disabled unless `--write-events` is supplied because it contains identifiers.
