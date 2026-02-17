"""Validate all JSON scale definitions and sanity-check scoring with Hypothesis."""

import json
from pathlib import Path

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from opencdt.models import ClinicalScale

SCALES_DIR = Path(__file__).resolve().parent.parent / "scales"
SCALE_PATHS = sorted(SCALES_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# 1. Every JSON in scales/ must parse into a valid ClinicalScale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", SCALE_PATHS, ids=lambda p: p.stem)
def test_scale_json_is_valid(path: Path):
    data = json.loads(path.read_text())
    scale = ClinicalScale(**data)
    scale.validate_complete()


# ---------------------------------------------------------------------------
# 2. Hypothesis: random valid inputs always produce a score within bounds
# ---------------------------------------------------------------------------


def _load_all_scales() -> list[ClinicalScale]:
    scales = []
    for path in SCALE_PATHS:
        try:
            data = json.loads(path.read_text())
            scale = ClinicalScale(**data)
            scale.validate_complete()
            scales.append(scale)
        except Exception:
            pass  # test_scale_json_is_valid will catch these
    return scales


ALL_SCALES = _load_all_scales()


def _build_values_strategy(scale: ClinicalScale) -> st.SearchStrategy[dict[str, float]]:
    """Build a Hypothesis strategy that produces a valid item_values dict.

    For each item, randomly pick one of its allowed point values.
    Then fix combinations that violate mutual-exclusivity constraints.
    """
    item_strategies = {
        item.label: st.sampled_from(sorted(set(item.points.values())))
        for item in scale.items
    }

    @st.composite
    def strategy(draw):
        values = {label: draw(s) for label, s in item_strategies.items()}
        # Respect mutual-exclusivity constraints
        for constraint in scale.constraints:
            non_zero = [label for label in constraint.item_labels if values.get(label, 0) != 0]
            if len(non_zero) > 1:
                # Keep only the first non-zero, zero out the rest
                for label in non_zero[1:]:
                    values[label] = 0
        return values

    return strategy()  # type: ignore[call-arg]


@pytest.mark.parametrize("scale", ALL_SCALES, ids=lambda s: s.name)
@given(data=st.data())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_score_within_declared_bounds(scale: ClinicalScale, data):
    values = data.draw(_build_values_strategy(scale))
    score = scale.calculate_score(values)
    assert scale.min_score is not None and scale.max_score is not None
    assert scale.min_score <= score <= scale.max_score, (
        f"{scale.name}: score {score} outside [{scale.min_score}, {scale.max_score}] "
        f"with values {values}"
    )


@pytest.mark.parametrize("scale", ALL_SCALES, ids=lambda s: s.name)
@given(data=st.data())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    derandomize=True
)
def test_score_has_matching_interpretation(scale: ClinicalScale, data):
    """Every valid score should fall into at least one interpretation range."""
    if not scale.interpretation:
        pytest.skip("No interpretation defined")
    values = data.draw(_build_values_strategy(scale))
    score = scale.calculate_score(values)
    interpretation = scale.interpret_score(score)
    assert interpretation is not None, (
        f"{scale.name}: score {score} has no matching interpretation "
        f"with values {values}"
    )
