"""Entity exposure and repeated false-positive summaries."""

import pandas as pd

from .schema import PersistenceSummary


def compute_repeat_fp_by_entity(
    frame: pd.DataFrame, *, entity_column: str, legitimate_column: str,
    false_positive_column: str, exposure_min: int, exposure_max: int | None,
    minimum_false_positives: int = 2,
) -> pd.DataFrame:
    """Calculate eligibility from legitimate exposure and repeat-FP status."""
    legitimate = frame.loc[frame[legitimate_column].astype(bool)]
    result = legitimate.groupby(entity_column, dropna=False, sort=False)[false_positive_column].agg(
        legitimate_exposure="size", false_positive_count="sum"
    )
    upper = True if exposure_max is None else result["legitimate_exposure"] <= exposure_max
    result["eligible"] = (result["legitimate_exposure"] >= exposure_min) & upper
    result["repeat_fp"] = result["false_positive_count"] >= minimum_false_positives
    return result.reset_index()


def compute_persistence_summary(entity_frame: pd.DataFrame) -> PersistenceSummary:
    """Summarize repeated false positives among eligible entities."""
    eligible = entity_frame.loc[entity_frame["eligible"].astype(bool)]
    denominator = len(eligible)
    numerator = int(eligible["repeat_fp"].sum())
    return PersistenceSummary(numerator, denominator, numerator / denominator if denominator else float("nan"))
