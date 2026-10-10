"""Known-answer regression ported from authoritative Simulation Validation Stage A."""

from collections import Counter
from itertools import combinations, product
import math

import numpy as np
import pandas as pd
import pytest

from fp_persistence import ExposureSpec, PermutationSpec, audit


STAGE_A_FIXTURE_ID = "stage_a_primary_three_strata_unequal_exposure_v1"
STAGE_A_SEED = 2026091601
STAGE_A_REPLICATES = 2000


def _authoritative_stage_a_fixture_adapter() -> pd.DataFrame:
    entity_layout = (
        ("2026-01-01", ("A", "A", "B", "B", "C", "C")),
        ("2026-01-02", ("A", "A", "B", "C", "C", "C")),
        ("2026-01-03", ("A", "B", "B", "B", "C", "C")),
    )
    realized_flag_positions = {
        "2026-01-01": {0, 1},
        "2026-01-02": {0, 2},
        "2026-01-03": {4, 5},
    }
    rows = []
    event_number = 0
    for day, entities in entity_layout:
        for within_day, entity in enumerate(entities):
            rows.append(
                {
                    "transaction_id": f"P{event_number:03d}",
                    "entity": entity,
                    "legitimate": True,
                    "false_positive": int(within_day in realized_flag_positions[day]),
                    # Stage A's frozen residual builder resolves these rows to
                    # three day-level final strata; the public core accepts the
                    # already resolved partition directly.
                    "final_stratum": day,
                }
            )
            event_number += 1
    return pd.DataFrame(rows)


def _exact_stage_a_oracle(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    entity_values = frame["entity"].to_numpy()
    groups = tuple(
        indices.to_numpy(dtype=int)
        for _, indices in frame.groupby("final_stratum", sort=True).groups.items()
    )
    flags_per_group = tuple(int(frame.loc[indices, "false_positive"].sum()) for indices in groups)
    choices = tuple(
        tuple(combinations(indices.tolist(), flag_count))
        for indices, flag_count in zip(groups, flags_per_group)
    )
    values = []
    for selected_by_group in product(*choices):
        flags = np.zeros(len(frame), dtype=np.int8)
        selected = [position for group_selection in selected_by_group for position in group_selection]
        flags[selected] = 1
        repeated = sum(
            int(flags[entity_values == entity].sum()) >= 2
            for entity in ("A", "B", "C")
        )
        values.append(repeated / 3)
    return np.asarray(values, dtype=float), math.prod(len(group) for group in choices)


def test_public_generic_audit_matches_authoritative_stage_a_known_answer():
    frame = _authoritative_stage_a_fixture_adapter()
    exact_values, allocation_count = _exact_stage_a_oracle(frame)
    result = audit(
        frame,
        entity_col="entity",
        event_id_col="transaction_id",
        final_stratum_col="final_stratum",
        exposure=ExposureSpec(5, 19),
        permutation=PermutationSpec(STAGE_A_REPLICATES, STAGE_A_SEED),
        flag_col="false_positive",
        legitimate_col="legitimate",
        metadata={"known_answer_fixture": STAGE_A_FIXTURE_ID},
    )

    exact_counts = Counter(exact_values)
    exact_mean = float(exact_values.mean())
    exact_interval = tuple(
        float(value) for value in np.quantile(exact_values, [0.025, 0.975], method="linear")
    )
    exact_mcse = float(exact_values.std(ddof=0) / math.sqrt(STAGE_A_REPLICATES))
    z_mc = (
        result.estimates.reference_expected_repeat_fp_proportion - exact_mean
    ) / exact_mcse

    assert allocation_count == 3375
    assert exact_counts == Counter({1 / 3: 424, 2 / 3: 2422, 1.0: 529})
    assert exact_mean == pytest.approx(0.677037037037037)
    assert exact_interval == (1 / 3, 1.0)
    assert result.population.eligible_entity_count == 3
    assert result.diagnostics.entity_legitimate_exposure_distribution == {5: 1, 6: 1, 7: 1}
    assert result.counts.repeated_fp_entity_count == 2
    assert result.estimates.observed_repeat_fp_proportion == pytest.approx(2 / 3)
    assert result.reference.replicates == STAGE_A_REPLICATES
    assert result.reference.seed == STAGE_A_SEED
    assert result.estimates.reference_expected_repeat_fp_proportion == pytest.approx(
        0.6686666666666666
    )
    assert (result.reference.interval_lower, result.reference.interval_upper) == exact_interval
    assert abs(z_mc) <= 3.0
