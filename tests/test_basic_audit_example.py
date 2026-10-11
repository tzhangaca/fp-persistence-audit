from itertools import combinations, product
from pathlib import Path
import runpy

import numpy as np
import pytest

from fp_persistence import AuditStatus


def test_dispersed_and_concentrated_example_controls_flag_burden():
    namespace = runpy.run_path(
        Path(__file__).resolve().parents[1] / "examples" / "basic_audit.py"
    )
    events = namespace["events"]
    flags_dispersed = namespace["flags_dispersed"]
    flags_concentrated = namespace["flags_concentrated"]
    results = namespace["audit_results"]

    assert sum(flags_dispersed) == sum(flags_concentrated) == 4
    counts_by_stratum = []
    for flags in (flags_dispersed, flags_concentrated):
        frame = events.assign(fp_flag=flags)
        counts_by_stratum.append(
            frame.groupby("final_stratum", sort=True)["fp_flag"].sum().to_dict()
        )
    assert counts_by_stratum == [
        {"day-1": 2, "day-2": 2},
        {"day-1": 2, "day-2": 2},
    ]

    dispersed = results["dispersed"]
    concentrated = results["concentrated"]
    assert dispersed.status is AuditStatus.ESTIMABLE
    assert concentrated.status is AuditStatus.ESTIMABLE
    assert dispersed.estimates.observed_repeat_fp_proportion == pytest.approx(0.0)
    assert concentrated.estimates.observed_repeat_fp_proportion == pytest.approx(0.5)
    assert (
        concentrated.estimates.excess_repeat_fp_percentage_points
        > dispersed.estimates.excess_repeat_fp_percentage_points
    )

    stratum_choices = []
    for _, group in events.groupby("final_stratum", sort=True):
        stratum_choices.append(list(combinations(group.index.tolist(), 2)))
    exact_values = []
    for allocation in product(*stratum_choices):
        selected = {index for stratum_selection in allocation for index in stratum_selection}
        flags = events.index.to_series().map(lambda index: int(index in selected))
        repeated = (
            flags.groupby(events["entity_id"], sort=False).sum().ge(2).sum()
        )
        exact_values.append(repeated / 4)
    assert float(np.mean(exact_values)) == pytest.approx(0.20)
    assert dispersed.estimates.reference_expected_repeat_fp_proportion == pytest.approx(0.19)
    assert concentrated.estimates.reference_expected_repeat_fp_proportion == pytest.approx(0.19)
