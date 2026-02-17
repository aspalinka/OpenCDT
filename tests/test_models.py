import pytest

from opencdt.models import (
    ClinicalScale,
    ItemInput,
    MutualExclusivityConstraint,
    ScoreItem,
    ScoreInterpretation,
)


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

class TestModelValidation:
    def test_duplicate_item_ids_rejected(self):
        with pytest.raises(ValueError, match="Duplicate item IDs"):
            ClinicalScale(
                name="Bad",
                description="d",
                purpose="p",
                when_to_use=["x"],
                items=[
                    ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1}),
                    ScoreItem(label="a", description="a2", points={"No": 0, "Yes": 1}),
                ],
                formula="a",
                min_score=0,
                max_score=1,
                interpretation=[ScoreInterpretation(min_score=0, max_score=1, description="ok")],
            )

    def test_formula_referencing_undefined_item_rejected(self):
        with pytest.raises(ValueError, match="Formula references undefined items"):
            ClinicalScale(
                name="Bad",
                description="d",
                purpose="p",
                when_to_use=["x"],
                items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
                formula="a + nonexistent",
                min_score=0,
                max_score=1,
                interpretation=[ScoreInterpretation(min_score=0, max_score=1, description="ok")],
            )

    def test_constraint_referencing_undefined_item_rejected(self):
        with pytest.raises(ValueError, match="Constraint references undefined items"):
            ClinicalScale(
                name="Bad",
                description="d",
                purpose="p",
                when_to_use=["x"],
                items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
                constraints=[
                    MutualExclusivityConstraint(
                        item_labels=["a", "ghost"],
                        description="bad ref",
                    )
                ],
                formula="a",
                min_score=0,
                max_score=1,
                interpretation=[ScoreInterpretation(min_score=0, max_score=1, description="ok")],
            )

    def test_valid_constraint_accepted(self, cha2ds2_vasc):
        assert len(cha2ds2_vasc.constraints) == 1
        assert set(cha2ds2_vasc.constraints[0].item_labels) == {"age_75", "age_65_74"}

    def test_no_constraints_defaults_empty(self, wells_dvt):
        assert wells_dvt.constraints == []


# ---------------------------------------------------------------------------
# validate_constraints
# ---------------------------------------------------------------------------

class TestValidateConstraints:
    def test_no_violation_when_one_nonzero(self, cha2ds2_vasc):
        violations = cha2ds2_vasc.validate_constraints({"age_75": 1, "age_65_74": 0})
        assert violations == []

    def test_no_violation_when_both_zero(self, cha2ds2_vasc):
        violations = cha2ds2_vasc.validate_constraints({"age_75": 0, "age_65_74": 0})
        assert violations == []

    def test_no_violation_when_only_one_provided(self, cha2ds2_vasc):
        violations = cha2ds2_vasc.validate_constraints({"age_75": 1})
        assert violations == []

    def test_violation_when_both_nonzero(self, cha2ds2_vasc):
        violations = cha2ds2_vasc.validate_constraints({"age_75": 1, "age_65_74": 1})
        assert len(violations) == 1
        assert "age_75" in violations[0]
        assert "age_65_74" in violations[0]

    def test_no_constraints_no_violations(self, wells_dvt):
        violations = wells_dvt.validate_constraints({"active_cancer": 1, "paralysis": 1})
        assert violations == []


# ---------------------------------------------------------------------------
# calculate_score (exact)
# ---------------------------------------------------------------------------

class TestCalculateScore:
    ALL_ZERO = {
        "chf": 0, "hypertension": 0, "age_75": 0, "diabetes": 0,
        "stroke_tia": 0, "vascular_disease": 0, "age_65_74": 0, "female": 0,
    }

    def test_all_zero(self, cha2ds2_vasc):
        assert cha2ds2_vasc.calculate_score(self.ALL_ZERO) == 0

    def test_known_score(self, cha2ds2_vasc):
        vals = {**self.ALL_ZERO, "chf": 1, "hypertension": 1, "age_75": 1}
        # 1 + 1 + 2 = 4
        assert cha2ds2_vasc.calculate_score(vals) == 4

    def test_max_score_age_75(self, cha2ds2_vasc):
        vals = {k: 1 for k in self.ALL_ZERO}
        vals["age_65_74"] = 0  # can't have both
        # 1+1+2+1+2+1+0+1 = 9
        assert cha2ds2_vasc.calculate_score(vals) == 9

    def test_missing_item_raises(self, cha2ds2_vasc):
        with pytest.raises(ValueError, match="Missing values"):
            cha2ds2_vasc.calculate_score({"chf": 0})

    def test_constraint_violation_raises(self, cha2ds2_vasc):
        vals = {k: 1 for k in self.ALL_ZERO}
        with pytest.raises(ValueError, match="Constraint violated"):
            cha2ds2_vasc.calculate_score(vals)

    def test_negative_contribution(self, wells_dvt):
        all_zero = {item.label: 0 for item in wells_dvt.items}
        vals = {**all_zero, "alternative_dx": 1}
        assert wells_dvt.calculate_score(vals) == -2

    def test_wells_dvt_full_positive(self, wells_dvt):
        all_one = {item.label: 1 for item in wells_dvt.items}
        all_one["alternative_dx"] = 0
        assert wells_dvt.calculate_score(all_one) == 9


# ---------------------------------------------------------------------------
# calculate_score_range
# ---------------------------------------------------------------------------

class TestCalculateScoreRange:
    def test_all_items_provided_returns_exact(self, simple_no_constraints):
        min_s, max_s, missing, auto = simple_no_constraints.calculate_score_range(
            {"a": 1, "b": 0}
        )
        assert min_s == max_s == 1
        assert missing == []
        assert auto == {}

    def test_empty_input_full_range(self, simple_no_constraints):
        min_s, max_s, missing, auto = simple_no_constraints.calculate_score_range({})
        assert min_s == 0
        assert max_s == 2
        assert set(missing) == {"a", "b"}

    def test_partial_input(self, simple_no_constraints):
        min_s, max_s, missing, auto = simple_no_constraints.calculate_score_range({"a": 1})
        assert min_s == 1
        assert max_s == 2
        assert missing == ["b"]

    # --- Constraint auto-fill ---

    def test_autofill_when_nonzero_provided(self, cha2ds2_vasc):
        min_s, max_s, missing, auto = cha2ds2_vasc.calculate_score_range(
            {"age_75": 1, "chf": 1}
        )
        assert auto == {"age_65_74": 0}
        assert "age_65_74" not in missing

    def test_no_autofill_when_zero_provided(self, cha2ds2_vasc):
        """Providing age_75=0 should NOT auto-fill age_65_74 (patient might be 65-74)."""
        _, _, missing, auto = cha2ds2_vasc.calculate_score_range({"age_75": 0})
        assert "age_65_74" not in auto
        assert "age_65_74" in missing

    def test_autofill_yields_exact_when_all_resolved(self, cha2ds2_vasc):
        vals = {
            "chf": 0, "hypertension": 1, "age_75": 1, "diabetes": 0,
            "stroke_tia": 0, "vascular_disease": 0, "female": 0,
        }
        # age_65_74 is missing but should be auto-filled to 0
        min_s, max_s, missing, auto = cha2ds2_vasc.calculate_score_range(vals)
        assert min_s == max_s == 3  # hypertension(1) + age_75(2)
        assert missing == []
        assert auto == {"age_65_74": 0}

    def test_constraint_violation_in_range_raises(self, cha2ds2_vasc):
        with pytest.raises(ValueError, match="Constraint violated"):
            cha2ds2_vasc.calculate_score_range({"age_75": 1, "age_65_74": 1})

    # --- Constraint-aware range bounds ---

    def test_empty_input_respects_constraint_max(self, cha2ds2_vasc):
        """Max score with all items missing must be 9, not 10 (constraint)."""
        _, max_s, _, _ = cha2ds2_vasc.calculate_score_range({})
        assert max_s == 9

    def test_empty_input_min_is_zero(self, cha2ds2_vasc):
        min_s, _, _, _ = cha2ds2_vasc.calculate_score_range({})
        assert min_s == 0

    # --- Negative contribution (Wells DVT) ---

    def test_negative_item_range(self, wells_dvt):
        """alternative_dx subtracts 2; with only active_cancer=1 known, range is [-1, 9]."""
        min_s, max_s, missing, auto = wells_dvt.calculate_score_range(
            {"active_cancer": 1}
        )
        assert min_s == -1
        assert max_s == 9
        assert "alternative_dx" in missing

    def test_wells_dvt_all_missing(self, wells_dvt):
        min_s, max_s, _, _ = wells_dvt.calculate_score_range({})
        assert min_s == -2
        assert max_s == 9

    # --- Non-linear formula: brute-force required for correct bounds ---

    def test_nonlinear_formula_finds_true_min(self, nonlinear_product):
        """
        Formula (a-1)*(b-1), items in {0,1,2}, no constraints.

        True min=-1 at (0,2) or (2,0). The direction-detection heuristic
        fails: from baseline (0,0) score=1, raising either a or b lowers
        the score, so the heuristic assigns max-values to both items for
        minimisation and evaluates (2,2)=1 — the wrong answer.
        """
        min_s, max_s, missing, auto = nonlinear_product.calculate_score_range({})
        assert min_s == -1
        assert max_s == 1
        assert set(missing) == {"a", "b"}
        assert auto == {}

    def test_nonlinear_formula_partial_known(self, nonlinear_product):
        """
        With a=0 known, only b is missing; brute-force tries b in {0,1,2}
        and finds scores {1, 0, -1} → min=-1, max=1.
        """
        min_s, max_s, missing, auto = nonlinear_product.calculate_score_range({"a": 0})
        assert min_s == -1
        assert max_s == 1
        assert missing == ["b"]
        assert auto == {}

    # --- Multi-valued constrained items (brute-force enumeration) ---

    def test_constrained_multivalued_full_range(self, constrained_multivalued):
        """All missing: valid combos are (0,0),(1,0),(2,0),(0,1),(0,2) → [0, 2]."""
        min_s, max_s, missing, auto = constrained_multivalued.calculate_score_range({})
        assert min_s == 0
        assert max_s == 2
        assert set(missing) == {"a", "b"}
        assert auto == {}

    def test_constrained_multivalued_partial(self, constrained_multivalued):
        """b=1 known → a auto-filled to 0; exact score = 1."""
        min_s, max_s, missing, auto = constrained_multivalued.calculate_score_range({"b": 1})
        assert min_s == max_s == 1
        assert auto == {"a": 0}
        assert missing == []

    # --- Return tuple structure ---

    def test_returns_four_tuple(self, simple_no_constraints):
        result = simple_no_constraints.calculate_score_range({})
        assert len(result) == 4


# ---------------------------------------------------------------------------
# interpret_score / interpret_score_range
# ---------------------------------------------------------------------------

class TestInterpretation:
    def test_interpret_exact(self, cha2ds2_vasc):
        assert cha2ds2_vasc.interpret_score(0) == "Low risk"
        assert cha2ds2_vasc.interpret_score(1) == "Low-moderate risk"
        assert cha2ds2_vasc.interpret_score(5) == "Moderate-high risk"

    def test_interpret_score_out_of_range(self, cha2ds2_vasc):
        assert cha2ds2_vasc.interpret_score(100) is None

    def test_interpret_range(self, cha2ds2_vasc):
        interps = cha2ds2_vasc.interpret_score_range(0, 3)
        assert "Low risk" in interps
        assert "Moderate risk" in interps
        assert len(interps) == 3  # Low, Low-moderate, Moderate

    def test_interpret_range_single_bucket(self, cha2ds2_vasc):
        interps = cha2ds2_vasc.interpret_score_range(0, 0)
        assert interps == ["Low risk"]


# ---------------------------------------------------------------------------
# ItemInput validation
# ---------------------------------------------------------------------------

class TestItemInput:
    def test_value_with_reasoning_accepted(self):
        iv = ItemInput(name="chf", value=1, reasoning="Patient has CHF")
        assert iv.value == 1
        assert iv.reasoning == "Patient has CHF"

    def test_null_value_no_reasoning_accepted(self):
        iv = ItemInput(name="chf", value=None)
        assert iv.value is None
        assert iv.reasoning is None

    def test_null_value_with_reasoning_accepted(self):
        iv = ItemInput(name="chf", value=None, reasoning="Not sure yet")
        assert iv.reasoning == "Not sure yet"

    def test_value_without_reasoning_rejected(self):
        with pytest.raises(ValueError, match="reasoning/evidence must be provided"):
            ItemInput(name="chf", value=1)

    def test_zero_value_requires_reasoning(self):
        with pytest.raises(ValueError, match="reasoning/evidence must be provided"):
            ItemInput(name="chf", value=0)


# ---------------------------------------------------------------------------
# parse_item_inputs
# ---------------------------------------------------------------------------

class TestParseItemInputs:
    def _all_null(self, scale):
        return [ItemInput(name=item.label, value=None) for item in scale.items]

    def _all_with_values(self, scale, value_map):
        inputs = []
        for item in scale.items:
            v = value_map.get(item.label)
            if v is not None:
                inputs.append(ItemInput(name=item.label, value=v, reasoning=f"{item.label} reason"))
            else:
                inputs.append(ItemInput(name=item.label, value=None))
        return inputs

    def test_all_null_returns_empty_dict(self, simple_no_constraints):
        result = simple_no_constraints.parse_item_inputs(self._all_null(simple_no_constraints))
        assert result == {}

    def test_all_values_returns_full_dict(self, simple_no_constraints):
        inputs = self._all_with_values(simple_no_constraints, {"a": 1, "b": 0})
        result = simple_no_constraints.parse_item_inputs(inputs)
        assert result == {"a": 1, "b": 0}

    def test_partial_values(self, simple_no_constraints):
        inputs = self._all_with_values(simple_no_constraints, {"a": 1})
        result = simple_no_constraints.parse_item_inputs(inputs)
        assert result == {"a": 1}

    def test_missing_item_raises(self, simple_no_constraints):
        with pytest.raises(ValueError, match="Missing items"):
            simple_no_constraints.parse_item_inputs([ItemInput(name="a", value=None)])

    def test_unknown_item_raises(self, simple_no_constraints):
        inputs = self._all_null(simple_no_constraints) + [
            ItemInput(name="unknown", value=None)
        ]
        with pytest.raises(ValueError, match="Unknown items"):
            simple_no_constraints.parse_item_inputs(inputs)

    def test_duplicate_item_name_raises(self, simple_no_constraints):
        inputs = [
            ItemInput(name="a", value=None),
            ItemInput(name="a", value=None),
            ItemInput(name="b", value=None),
        ]
        with pytest.raises(ValueError, match="Duplicate item name"):
            simple_no_constraints.parse_item_inputs(inputs)

    def test_invalid_value_rejected(self, simple_no_constraints):
        inputs = [
            ItemInput(name="a", value=3, reasoning="bad value"),
            ItemInput(name="b", value=None),
        ]
        with pytest.raises(ValueError, match="Invalid value"):
            simple_no_constraints.parse_item_inputs(inputs)

    def test_valid_value_accepted(self, simple_no_constraints):
        inputs = [
            ItemInput(name="a", value=1, reasoning="Yes"),
            ItemInput(name="b", value=0, reasoning="No"),
        ]
        result = simple_no_constraints.parse_item_inputs(inputs)
        assert result == {"a": 1, "b": 0}

    def test_cha2ds2_vasc_full(self, cha2ds2_vasc):
        vals = {
            "chf": 0, "hypertension": 1, "age_75": 1, "diabetes": 0,
            "stroke_tia": 0, "vascular_disease": 0, "age_65_74": 0, "female": 0,
        }
        inputs = self._all_with_values(cha2ds2_vasc, vals)
        result = cha2ds2_vasc.parse_item_inputs(inputs)
        assert result == vals


# ---------------------------------------------------------------------------
# Dynamic input model
# ---------------------------------------------------------------------------

class TestDynamicInputModel:
    def test_input_model_cached(self, simple_no_constraints):
        assert simple_no_constraints._input_model is not None
        assert simple_no_constraints._input_model.__name__ == "SimpleInput"

    def test_input_model_has_all_fields(self, cha2ds2_vasc):
        model = cha2ds2_vasc._input_model
        expected = {"chf", "hypertension", "age_75", "diabetes",
                    "stroke_tia", "vascular_disease", "age_65_74", "female"}
        assert set(model.model_fields.keys()) == expected


# ---------------------------------------------------------------------------
# Optional fields and guard clauses
# ---------------------------------------------------------------------------

class TestOptionalFields:
    """Test that ClinicalScale accepts null for LLM-generated fields."""

    def test_accepts_null_formula(self):
        scale = ClinicalScale(
            name="Test",
            items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
        )
        assert scale.formula is None

    def test_accepts_null_min_max_score(self):
        scale = ClinicalScale(
            name="Test",
            items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
        )
        assert scale.min_score is None
        assert scale.max_score is None

    def test_accepts_null_metadata(self):
        scale = ClinicalScale(
            name="Test",
            items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
        )
        assert scale.description is None
        assert scale.purpose is None
        assert scale.when_to_use is None

    def test_formula_validator_skips_when_none(self):
        """Formula validation should not run when formula is None."""
        scale = ClinicalScale(
            name="Test",
            items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
            formula=None,
        )
        assert scale.formula is None

    def test_calculate_score_raises_when_no_formula(self):
        scale = ClinicalScale(
            name="Test",
            items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
        )
        with pytest.raises(ValueError, match="formula not defined"):
            scale.calculate_score({"a": 1})

    def test_calculate_score_range_raises_when_no_formula(self):
        scale = ClinicalScale(
            name="Test",
            items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
        )
        with pytest.raises(ValueError, match="formula not defined"):
            scale.calculate_score_range({"a": 1})

    def test_interpret_score_returns_none_when_no_interpretation(self):
        scale = ClinicalScale(
            name="Test",
            items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
            formula="a",
            min_score=0,
            max_score=1,
        )
        assert scale.interpret_score(0) is None

    def test_interpret_score_range_returns_empty_when_no_interpretation(self):
        scale = ClinicalScale(
            name="Test",
            items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
            formula="a",
            min_score=0,
            max_score=1,
        )
        assert scale.interpret_score_range(0, 1) == []


class TestValidateComplete:
    def test_raises_when_formula_missing(self):
        scale = ClinicalScale(
            name="Test",
            items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
        )
        with pytest.raises(ValueError, match="formula"):
            scale.validate_complete()

    def test_raises_when_min_score_missing(self):
        scale = ClinicalScale(
            name="Test",
            items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
            formula="a",
        )
        with pytest.raises(ValueError, match="min_score"):
            scale.validate_complete()

    def test_raises_lists_all_missing(self):
        scale = ClinicalScale(
            name="Test",
            items=[ScoreItem(label="a", description="a", points={"No": 0, "Yes": 1})],
        )
        with pytest.raises(ValueError, match="formula.*min_score.*max_score"):
            scale.validate_complete()

    def test_passes_when_complete(self, simple_no_constraints):
        simple_no_constraints.validate_complete()  # should not raise
