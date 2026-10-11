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

Each row is an event. The two scenarios below use the same 12 legitimate events, four entities, four total false-positive flags, and two flags inside each final context stratum. Only the entity-level concentration of those flags changes.

```python
import json

import pandas as pd

from fp_persistence import ExposureSpec, PermutationSpec, audit

events = pd.DataFrame(
    {
        "event_id": [f"e{i:02d}" for i in range(12)],
        "entity_id": ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3,
        "legitimate": [1] * 12,
        "final_stratum": ["day-1"] * 6 + ["day-2"] * 6,
    }
)

flags_dispersed = [
    1, 0, 0,  # A
    1, 0, 0,  # B
    1, 0, 0,  # C
    1, 0, 0,  # D
]
flags_concentrated = [
    1, 1, 0,  # A
    0, 0, 0,  # B
    1, 1, 0,  # C
    0, 0, 0,  # D
]

summaries = {}
for scenario, flags in {
    "dispersed": flags_dispersed,
    "concentrated": flags_concentrated,
}.items():
    scenario_events = events.assign(fp_flag=flags)
    result = audit(
        scenario_events,
        entity_col="entity_id",
        event_id_col="event_id",
        final_stratum_col="final_stratum",
        legitimate_col="legitimate",
        flag_col="fp_flag",
        exposure=ExposureSpec(3, 3),
        permutation=PermutationSpec(500, 2026091601),
        metadata={
            "entity_semantics": "payment instrument",
            "context_definition": "prespecified example strata",
        },
    )
    summaries[scenario] = {
        "total_false_positive_flags": int(scenario_events["fp_flag"].sum()),
        "flags_by_final_stratum": {
            str(stratum): int(count)
            for stratum, count in scenario_events.groupby(
                "final_stratum", sort=True
            )["fp_flag"].sum().items()
        },
        "eligible_entity_count": result.population.eligible_entity_count,
        "observed_persistence": result.estimates.observed_repeat_fp_proportion,
        "reference_expectation": result.estimates.reference_expected_repeat_fp_proportion,
        "excess_persistence_percentage_points": result.estimates.excess_repeat_fp_percentage_points,
    }

print(json.dumps(summaries, indent=2, sort_keys=True))
```

With the documented seed and settings, this produces:

```json
{
  "concentrated": {
    "eligible_entity_count": 4,
    "excess_persistence_percentage_points": 31.0,
    "flags_by_final_stratum": {"day-1": 2, "day-2": 2},
    "observed_persistence": 0.5,
    "reference_expectation": 0.19,
    "total_false_positive_flags": 4
  },
  "dispersed": {
    "eligible_entity_count": 4,
    "excess_persistence_percentage_points": -19.0,
    "flags_by_final_stratum": {"day-1": 2, "day-2": 2},
    "observed_persistence": 0.0,
    "reference_expectation": 0.19,
    "total_false_positive_flags": 4
  }
}
```

The dispersed scenario gives each entity one flag, so no entity has a repeated false positive. The concentrated scenario places repeated flags on A and C, producing observed persistence of `0.5`. Because both scenarios retain four total flags and exactly two flags in both `day-1` and `day-2`, the contrast demonstrates entity-level concentration rather than a change in overall flag burden or context-level flag counts.

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

Run the public known-answer regression directly with:

```bash
pytest tests/test_known_answer_reference.py -v
```

No claim of external adoption or independent validation is made.

## Version status

Version `0.1.1` is the current generic audit release. Paper-reproduction and RQ2 workflows are not part of this release.

The archived v0.1.0 release is available at [https://doi.org/10.5281/zenodo.23288629](https://doi.org/10.5281/zenodo.23288629).
