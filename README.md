# fp-persistence-audit

[![DOI](https://zenodo.org/badge/1413689835.svg)](https://doi.org/10.5281/zenodo.23288628)

`fp-persistence-audit` audits whether false-positive decisions repeatedly concentrate on the same entities beyond a fixed-count, context-conditioned reference.

It is a retrospective, model-agnostic auditing tool. It accepts either externally supplied binary false-positive flags or externally generated scores converted to an explicit exact number of flags. It is not a fraud model.

## What it does not do

This package does not:

- train a fraud model or improve detection automatically;
- choose an optimal production threshold;
- establish causality;
- establish fairness or discrimination;
- prove consumer harm.

## Installation

From a local checkout:

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e '.[test]'
```

## Five-minute quick start

Each row is an event. Supply deterministic event IDs, entity IDs, legitimate-event labels, already resolved final context strata, and binary false-positive flags.

```python
import pandas as pd

from fp_persistence import ExposureSpec, PermutationSpec, audit

events = pd.DataFrame(
    {
        "event_id": [f"e{i:02d}" for i in range(12)],
        "entity_id": ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3,
        "legitimate": [1] * 12,
        "final_stratum": ["day-1"] * 6 + ["day-2"] * 6,
        "fp_flag": [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    }
)

result = audit(
    events,
    entity_col="entity_id",
    event_id_col="event_id",
    final_stratum_col="final_stratum",
    legitimate_col="legitimate",
    flag_col="fp_flag",
    exposure=ExposureSpec(min_legitimate_events=3, max_legitimate_events=3),
    permutation=PermutationSpec(replicates=500, seed=2026091601),
    metadata={
        "entity_semantics": "payment instrument",
        "context_definition": "prespecified example strata",
    },
)

print(
    {
        "eligible_entities": result.population.eligible_entity_count,
        "observed_persistence": result.estimates.observed_repeat_fp_proportion,
        "reference_expectation": result.estimates.reference_expected_repeat_fp_proportion,
        "excess_persistence_pp": result.estimates.excess_repeat_fp_percentage_points,
        "final_strata": result.diagnostics.final_stratum_count,
        "stratum_size_distribution": dict(
            result.diagnostics.final_stratum_size_distribution
        ),
    }
)
```

Entity semantics and context definitions materially affect interpretation and should always be documented.

## Score-based usage

Scores must be generated outside this package. The caller supplies the exact selection count:

```python
from fp_persistence import audit, flags_from_scores

selection = flags_from_scores(
    events.assign(score=[0.99, 0.95, 0.40, 0.90, 0.30, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.01]),
    score_col="score",
    event_id_col="event_id",
    legitimate_col="legitimate",
    n_select=3,
)

result = audit(
    events,
    entity_col="entity_id",
    event_id_col="event_id",
    final_stratum_col="final_stratum",
    legitimate_col="legitimate",
    flag_selection=selection,
    exposure=ExposureSpec(3, 3),
    permutation=PermutationSpec(500, 2026091601),
)
```

`n_select` is always explicit. The generic package does not infer a fraction and does not automatically impose the Persistence paper's 2% operating point.

## Interpreting the output

- **Observed persistence** is the proportion of exposure-eligible entities with at least two false-positive events.
- **Reference expectation** is the mean persistence after repeatedly permuting the fixed realized flag count within each supplied final context stratum.
- **Excess persistence** is observed persistence minus the reference expectation, reported in percentage points.
- **Eligible entity count** is the denominator defined by the supplied legitimate-exposure bounds.
- **Context diagnostics** report the final-stratum count and lossless stratum-size structure used by the reference procedure.

> **Excess persistence is conditional on the supplied entity definition and context stratification. Report and interpret it together with those definitions and structural diagnostics.**

An estimable zero is a completed audit whose defined persistence value is zero. `NOT_ESTIMABLE` instead means the completed, well-formed audit had zero exposure-eligible entities, so the scientific estimand is undefined. Validation failures and interrupted executions produce no completed result.

## Reproducibility

The public API requires deterministic event IDs and an explicit random seed. Permutations use canonical event ordering, so dataframe row order alone cannot change scientific outputs. The implementation is covered by synthetic tests, dataframe row-shuffle invariance tests, and an exact known-answer regression fixture.

No claim of external adoption or independent validation is made.

## Version status

Version `0.1.0` is the initial generic audit release. Paper-reproduction and RQ2 workflows are not part of this release.

The archived v0.1.0 release is available at [https://doi.org/10.5281/zenodo.23288629](https://doi.org/10.5281/zenodo.23288629).
