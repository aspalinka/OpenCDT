import json

import pytest

from opencdt.models import ItemInput
from opencdt.server import calculate_score, list_scales, get_scale, search_scales

# Scale item labels
CHA2DS2_LABELS = [
    "chf", "hypertension", "age_75", "diabetes",
    "stroke_tia", "vascular_disease", "age_65_74", "female",
]

WELLS_DVT_LABELS = [
    "active_cancer", "paralysis", "bedridden", "localized_tenderness",
    "leg_swelling", "calf_swelling", "pitting_edema", "collateral_veins",
    "previous_dvt", "alternative_dx",
]


def _make_inputs(labels: list[str], values: dict[str, float | None] | None = None) -> list[ItemInput]:
    """Build a full list of ItemInput for a scale. Items not in values get value=None."""
    values = values or {}
    inputs = []
    for label in labels:
        v = values.get(label)
        if v is not None:
            inputs.append(ItemInput(name=label, value=v, reasoning=f"{label} evidence"))
        else:
            inputs.append(ItemInput(name=label, value=None))
    return inputs


class TestListScales:
    def test_returns_list(self):
        result = json.loads(list_scales())
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_contains_expected_fields(self):
        result = json.loads(list_scales())
        for entry in result:
            assert "name" in entry
            assert "description" in entry


class TestGetScale:
    def test_found(self):
        result = json.loads(get_scale("CHA2DS2-VASc"))
        assert result["name"] == "CHA2DS2-VASc"
        assert "constraints" in result

    def test_case_insensitive(self):
        result = json.loads(get_scale("cha2ds2-vasc"))
        assert result["name"] == "CHA2DS2-VASc"

    def test_not_found(self):
        result = json.loads(get_scale("NonExistentScale"))
        assert "error" in result


class TestSearchScales:
    def test_search_by_keyword(self):
        result = json.loads(search_scales("stroke"))
        assert any(r["name"] == "CHA2DS2-VASc" for r in result)

    def test_search_no_results(self):
        result = json.loads(search_scales("xyznonexistent"))
        assert "message" in result

    def test_search_with_tags(self):
        result = json.loads(search_scales("DVT", tags=["DVT"]))
        assert any(r["name"] == "Wells DVT" for r in result)


class TestCalculateScoreUnified:
    """Tests for the unified calculate_score MCP tool with ItemInput."""

    ALL_ZERO_CHA2DS2 = {k: 0 for k in CHA2DS2_LABELS}

    def test_exact_result_all_items(self):
        inputs = _make_inputs(CHA2DS2_LABELS, self.ALL_ZERO_CHA2DS2)
        result = json.loads(calculate_score("CHA2DS2-VASc", inputs))
        assert result["result_type"] == "exact"
        assert result["score"] == 0
        assert "interpretation" in result

    def test_exact_result_known_score(self):
        vals = {**self.ALL_ZERO_CHA2DS2, "chf": 1, "age_75": 2}
        inputs = _make_inputs(CHA2DS2_LABELS, vals)
        result = json.loads(calculate_score("CHA2DS2-VASc", inputs))
        assert result["result_type"] == "exact"
        assert result["score"] == 3  # 1 + 2

    def test_range_result_partial(self):
        inputs = _make_inputs(CHA2DS2_LABELS, {"chf": 1})
        result = json.loads(calculate_score("CHA2DS2-VASc", inputs))
        assert result["result_type"] == "range"
        assert result["min_score"] <= result["max_score"]
        assert len(result["missing_items"]) > 0
        assert "possible_interpretations" in result

    def test_range_result_all_null(self):
        inputs = _make_inputs(CHA2DS2_LABELS)
        result = json.loads(calculate_score("CHA2DS2-VASc", inputs))
        assert result["result_type"] == "range"
        assert result["min_score"] == 0
        assert result["max_score"] == 9

    def test_constraint_violation_returns_error(self):
        vals = {**self.ALL_ZERO_CHA2DS2, "age_75": 2, "age_65_74": 1}
        inputs = _make_inputs(CHA2DS2_LABELS, vals)
        result = json.loads(calculate_score("CHA2DS2-VASc", inputs))
        assert "error" in result
        assert "Constraint violated" in result["error"]

    def test_auto_fill_in_range(self):
        inputs = _make_inputs(CHA2DS2_LABELS, {"age_75": 2, "chf": 1})
        result = json.loads(calculate_score("CHA2DS2-VASc", inputs))
        assert result["result_type"] == "range"
        assert result["auto_filled"] == {"age_65_74": 0}
        labels = [m["label"] for m in result["missing_items"]]
        assert "age_65_74" not in labels

    def test_auto_fill_yields_exact(self):
        vals = {
            "chf": 0, "hypertension": 1, "age_75": 2, "diabetes": 0,
            "stroke_tia": 0, "vascular_disease": 0, "female": 0,
        }
        inputs = _make_inputs(CHA2DS2_LABELS, vals)
        result = json.loads(calculate_score("CHA2DS2-VASc", inputs))
        assert result["result_type"] == "exact"
        assert result["score"] == 3
        assert result["auto_filled"] == {"age_65_74": 0}

    def test_no_auto_filled_key_when_empty(self):
        inputs = _make_inputs(CHA2DS2_LABELS, self.ALL_ZERO_CHA2DS2)
        result = json.loads(calculate_score("CHA2DS2-VASc", inputs))
        assert "auto_filled" not in result

    def test_scale_not_found(self):
        result = json.loads(calculate_score("DoesNotExist", []))
        assert "error" in result

    def test_missing_items_include_options(self):
        inputs = _make_inputs(CHA2DS2_LABELS, {"chf": 0})
        result = json.loads(calculate_score("CHA2DS2-VASc", inputs))
        for item in result["missing_items"]:
            assert "label" in item
            assert "description" in item
            assert "options" in item

    # --- Input validation errors ---

    def test_missing_item_from_list_returns_error(self):
        inputs = [ItemInput(name="chf", value=None)]  # missing 7 items
        result = json.loads(calculate_score("CHA2DS2-VASc", inputs))
        assert "error" in result
        assert "Missing items" in result["error"]

    def test_value_without_reasoning_raises(self):
        with pytest.raises(ValueError, match="reasoning/evidence"):
            ItemInput(name="chf", value=1)  # no reasoning

    def test_invalid_value_returns_error(self):
        inputs = _make_inputs(CHA2DS2_LABELS)
        inputs[0] = ItemInput(name="chf", value=5, reasoning="invalid")
        result = json.loads(calculate_score("CHA2DS2-VASc", inputs))
        assert "error" in result
        assert "Invalid value" in result["error"]

    # --- Wells DVT (negative contributions, no constraints) ---

    def test_wells_dvt_exact(self):
        all_zero = {k: 0 for k in WELLS_DVT_LABELS}
        inputs = _make_inputs(WELLS_DVT_LABELS, all_zero)
        result = json.loads(calculate_score("Wells DVT", inputs))
        assert result["result_type"] == "exact"
        assert result["score"] == 0

    def test_wells_dvt_partial(self):
        inputs = _make_inputs(WELLS_DVT_LABELS, {"active_cancer": 1})
        result = json.loads(calculate_score("Wells DVT", inputs))
        assert result["result_type"] == "range"
        assert result["min_score"] == -1
        assert result["max_score"] == 9

    def test_wells_dvt_negative_score(self):
        all_zero = {k: 0 for k in WELLS_DVT_LABELS}
        all_zero["alternative_dx"] = -2
        inputs = _make_inputs(WELLS_DVT_LABELS, all_zero)
        result = json.loads(calculate_score("Wells DVT", inputs))
        assert result["result_type"] == "exact"
        assert result["score"] == -2
