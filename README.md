# fp-persistence-audit

[![DOI](https://zenodo.org/badge/1413689835.svg)](https://doi.org/10.5281/zenodo.23288628)

`fp-persistence-audit` audits whether false-positive decisions repeatedly concentrate on the same entities beyond a fixed-count, context-conditioned reference.

Two fraud systems may have the same total false-positive rate while one repeatedly false-positives the same legitimate entities. This package measures that repeated entity-level concentration relative to a fixed-count, context-conditioned reference.

`entity` is the repeated unit being audited, such as a customer, account, card, or payment instrument.

It is a retrospective, model-agnostic auditing tool. It accepts either externally supplied binary false-positive flags or externally generated scores converted to an explicit exact number of flags. It is not a fraud model.

## What it does not do

This package does not:

- train a fraud model or improve detection automatically;
- choose an optimal production threshold;
- establish causality;
- establish fairness or discrimination;
- prove consumer harm.

## Installation

Install the release from PyPI:

```bash
python -m pip install fp-persistence-audit
```

From a local source checkout:

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e '.[test]'
```

Python `3.10` through `3.13` is the declared compatibility range.

## Legitimate audit population

> **Important:** If `legitimate_col` is omitted, every input row is treated as legitimate. For datasets that contain fraudulent or otherwise positive events, users should normally provide a strict binary legitimate indicator or pre-filter the data to legitimate audit events. The package does not infer ground truth, and a false-positive flag is meaningful only on the legitimate audit population.

## Five-minute quick start

Each row is an event. The two scenarios below use the same 12 legitimate events, four entities, four total false-positive flags, and two flags inside each final context stratum. Only the entity-level concentration of those flags changes.

`final_stratum` is a caller-supplied, prespecified resolved context bucket. The generic package does not construct the Persistence paper's context hierarchy.

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

The dispersed scenario gives each entity one flag, so no entity has a repeated false positive. The concentrated scenario places repeated flags on A and C, producing observed persistence of `0.5`. Because both scenarios retain four total flags and exactly two flags in both `day-1` and `day-2`, the contrast illustrates the mechanics of entity-level concentration rather than a change in overall flag burden or context-level flag counts.

The exact conditional reference expectation for this toy fixture is `0.20`; the displayed `0.19` is the documented 500-replicate Monte Carlo estimate. The documented run's conditional reference interval is `[0, 0.5]`. This toy example is pedagogical and is not evidence of statistical significance.

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

`n_select` is always explicit and is the exact number selected among the legitimate rows passed to `flags_from_scores()`. It is not automatically the production system's total alert count or alert rate. The helper never infers an alert fraction or threshold and does not automatically impose the Persistence paper's 2% operating point. To reproduce actual system alerts or declines across all transactions, generate those system decision flags upstream and audit them with `flag_col`.

## Choosing final context strata

`final_stratum` should be a prespecified context bucket whose event-level conditions should be held fixed by the reference procedure. A good example is calendar period crossed with coarse amount bands, channel, region, or other prespecified event-level context variables. Entity ID itself—or customer ID, card ID, or another almost-unique bucket tied to the entity—is a bad final stratum because it conditions away the entity-level concentration being audited.

Overly fine strata can mechanically constrain the fixed-count permutation reference and suppress apparent excess persistence. Define and document the context specification before interpreting the audit.

## Interpreting the output

- **Observed persistence** is the proportion of exposure-eligible entities with at least two false-positive events.
- **Reference expectation** is the mean persistence after repeatedly permuting the fixed realized flag count within each supplied final context stratum.
- **Excess persistence** is observed persistence minus the reference expectation, reported in percentage points.
- **Observed/expected ratio** compares observed persistence with the reference expectation when that ratio is defined.
- **Reference interval** is the configured quantile interval of the conditional permutation reference distribution.
- **Eligible entity count** is the denominator defined by the supplied legitimate-exposure bounds.
- **Context diagnostics** report the final-stratum count and lossless stratum-size structure used by the reference procedure.

> **Excess persistence is conditional on the supplied entity definition and context stratification. Report and interpret it together with those definitions and structural diagnostics.**

Report observed persistence, reference expectation, excess persistence in percentage points, the observed/expected ratio when defined, and the reference interval together with the entity definition, exposure bounds, and final-stratum definition. The reference interval is a conditional reference-distribution interval—not a confidence interval for excess persistence and not a p-value. Excess persistence in percentage points can also depend on the underlying false-positive operating point or base rate, so comparisons across very different operating points should be made cautiously.

An estimable zero is a completed audit whose defined persistence value is zero. `NOT_ESTIMABLE` instead means the completed, well-formed audit had zero exposure-eligible entities, so the scientific estimand is undefined. Validation failures and interrupted executions produce no completed result.

## Supported input domain

The public API deliberately supports a finite, deterministic input domain; arbitrary hashable Python objects are not accepted. Event IDs and entity IDs may be `str`, non-Boolean `int`, `bytes`, `UUID`, `date`, timezone-naive `datetime`/`pandas.Timestamp`, UTC or `datetime.timezone` fixed-offset `datetime`/`pandas.Timestamp`, finite `float`, finite `Decimal`, losslessly normalized NumPy scalar equivalents, or recursively valid tuples made only from those types. Dynamic timezone-rule values (including `zoneinfo`, pytz regional zones, dateutil regional zones, and custom `tzinfo` implementations) are intentionally unsupported in v0.1.2; normalize them to UTC, a `datetime.timezone` fixed offset, or a stable string before calling the audit. Values within each event-ID or entity-ID column must use one consistent representation. Event IDs must also be nonmissing, unique, and totally ordered; dataframe index and incoming row order are never identifiers or tie breakers.

Final-stratum labels use the same supported value types and datetime timezone restriction, but may mix supported representations when their equality is deterministic; equality-equivalent numeric or supported datetime labels form one stratum. Unsupported containers (`list`, `dict`, `set`, `frozenset`), nonfinite numbers, Booleans used as identifiers, dynamic/custom timezone values, and arbitrary custom objects are rejected with `AuditValidationError`.

Scores may be finite Python or NumPy integers/floats, or finite `Decimal` values. They are ranked without lossy float coercion, and mixed numeric ordering is independent of the caller's mutable `Decimal` precision and `FloatOperation` trap settings. Complex values, Booleans, nonfinite values, strings, and custom numeric-like objects are rejected. Flag and legitimate-indicator columns accept only Python/NumPy Booleans or integer `0`/`1`; generic truthiness and floating `0.0`/`1.0` are not accepted.

## Reproducibility

The public API requires deterministic event IDs and an explicit random seed. Permutations use canonical event ordering, so dataframe row order alone cannot change scientific outputs. The implementation is covered by synthetic tests, dataframe row-shuffle invariance tests, and an exact known-answer regression fixture.

From a source checkout, run the public known-answer regression directly with:

```bash
pytest tests/test_known_answer_reference.py -v
```

No claim of external adoption or independent validation is made.

## Citation

Use the repository's `CITATION.cff` metadata when citing this software. The DOI badge at the top of this README links to the archived software/version series.

## Version status

Version `0.1.2` is the current generic audit release. Paper-reproduction and RQ2 workflows are not part of this release.

The archived v0.1.0 release is available at [https://doi.org/10.5281/zenodo.23288629](https://doi.org/10.5281/zenodo.23288629).
