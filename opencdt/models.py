from typing import Any, List, Optional, Dict, Tuple, Type
import logging
import re
from itertools import product as itertools_product

import numexpr as ne
from pydantic import BaseModel, Field, create_model, field_validator, model_validator

logger = logging.getLogger(__name__)


class Reference(BaseModel):
    """Citation for the scale."""
    citation: str
    pmid: Optional[str] = None
    url: Optional[str] = None


class ScoreItem(BaseModel):
    """Single element of the scale."""
    label: str = Field(description="Snake_case label")
    description: str = Field(description="Clinical definition/criteria")
    points: Dict[str, float] = Field(
        description="Mapping of choices to point values"
    )

    @classmethod
    @field_validator('label')
    def validate_snake_case(cls, v: str) -> str:
        """Ensure label is valid snake_case (safe for numexpr)."""
        if not re.match(r'^[a-z][a-z0-9_]*$', v):
            raise ValueError(
                f"Item id must be snake_case (lowercase, underscores, no spaces): {v}"
            )
        return v


class ScoreInterpretation(BaseModel):
    """Score range mapped to clinical interpretation."""
    min_score: Optional[float]
    max_score: Optional[float]
    description: str


class MutualExclusivityConstraint(BaseModel):
    """Declares that at most one item in the group can have a non-zero value."""
    item_labels: List[str]
    description: str


class ItemInput(BaseModel):
    """Structured input for a single scale item from an LLM."""
    name: str = Field(description="Item label (must match a ScoreItem.label)")
    value: Optional[float] = Field(None, description="Point value (None if unknown)")
    reasoning: Optional[str] = Field(
        None, description="Clinical reasoning/evidence for the chosen value"
    )

    @model_validator(mode='after')
    def reasoning_required_with_value(self):
        if self.value is not None and self.reasoning is None:
            raise ValueError(
                f"reasoning/evidence must be provided when value is set "
                f"(item '{self.name}', value={self.value})"
            )
        return self


class ClinicalScale(BaseModel):
    """A clinical scoring scale."""

    name: str = Field(description="Official name")
    full_name: Optional[str] = Field(None, description="Expanded name if acronym")

    description: Optional[str] = Field(None, description="What the scale measures")
    purpose: Optional[str] = Field(None, description="Clinical purpose")
    when_to_use: Optional[List[str]] = Field(None, description="Indications for use")
    when_not_to_use: Optional[List[str]] = Field(None, description="Contraindications")

    tags: List[str] = Field(default_factory=list, description="Searchable clinical keywords")
    category: Optional[str] = Field(None, description="Clinical category")

    items: List[ScoreItem] = Field(description="Scoring elements")
    constraints: List[MutualExclusivityConstraint] = Field(
        default_factory=list,
        description="Mutual exclusivity constraints between items",
    )

    formula: Optional[str] = Field(
        None,
        description="Python expression using item IDs (evaluated with numexpr)",
        examples=[
            "chf + hypertension + age_75*2 + diabetes + stroke*2",
            "active_cancer + paralysis - alternative_dx*2"
        ]
    )

    min_score: Optional[float] = Field(None, description="Minimum possible score")
    max_score: Optional[float] = Field(None, description="Maximum possible score")

    interpretation: Optional[List[ScoreInterpretation]] = Field(
        None, description="Score ranges mapped to clinical interpretation"
    )

    references: List[Reference] = Field(default_factory=list)
    notes: Optional[str] = None

    # Cached dynamic input model (set in model_post_init)
    _input_model: Type[BaseModel] = None  # ty: ignore[invalid-assignment]

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode='after')
    def validate_unique_item_ids(self):
        """Ensure all item IDs are unique."""
        item_ids = [item.label for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            duplicates = [id for id in item_ids if item_ids.count(id) > 1]
            raise ValueError(f"Duplicate item IDs found: {duplicates}")
        return self

    @model_validator(mode='after')
    def validate_formula_references(self):
        """Check that formula only references defined item IDs."""
        if self.formula is None:
            return self
        item_ids = {item.label for item in self.items}

        referenced = set(re.findall(r'\b[a-z][a-z0-9_]*\b', self.formula))

        allowed_builtins = {'sum', 'min', 'max', 'abs', 'int', 'float'}

        undefined = referenced - item_ids - allowed_builtins
        if undefined:
            raise ValueError(
                f"Formula references undefined items: {undefined}. "
                f"Available items: {item_ids}"
            )
        return self

    @model_validator(mode='after')
    def validate_constraint_references(self):
        """Check that constraint item_labels reference defined items."""
        item_ids = {item.label for item in self.items}
        for constraint in self.constraints:
            undefined = set(constraint.item_labels) - item_ids
            if undefined:
                raise ValueError(
                    f"Constraint references undefined items: {undefined}. "
                    f"Available items: {item_ids}"
                )
        return self

    def model_post_init(self, __context: Any) -> None:
        """Build and cache the dynamic input model after initialization."""
        self._input_model = self._build_input_model()

    def _build_input_model(self) -> Type[BaseModel]:
        """Create a dynamic Pydantic model that validates input completeness and values."""
        # Build fields: one required ItemInput per scale item
        fields: Dict[str, Any] = {}
        for item in self.items:
            fields[item.label] = (ItemInput, ...)

        return create_model(
            f"{self.name.replace(' ', '')}Input",
            **fields,
        )

    def parse_item_inputs(self, item_values: List[ItemInput]) -> Dict[str, float]:
        """Validate a list of ItemInput against this scale and extract known values.

        Checks:
        - All scale items are present (by name)
        - No unknown item names
        - Values are in allowed ScoreItem.points
        - Reasoning is provided when value is set

        Returns dict of {label: value} for items with non-None values.
        """
        by_name: Dict[str, ItemInput] = {}
        for iv in item_values:
            if iv.name in by_name:
                raise ValueError(f"Duplicate item name: '{iv.name}'")
            by_name[iv.name] = iv

        required_labels = {item.label for item in self.items}
        provided_labels = set(by_name.keys())
        missing = required_labels - provided_labels
        if missing:
            raise ValueError(f"Missing items: {sorted(missing)}")
        unknown = provided_labels - required_labels
        if unknown:
            raise ValueError(f"Unknown items: {sorted(unknown)}")

        self._input_model(**by_name)

        items_by_label = {item.label: item for item in self.items}
        for name, iv in by_name.items():
            if iv.value is not None:
                allowed = set(items_by_label[name].points.values())
                if iv.value not in allowed:
                    raise ValueError(
                        f"Invalid value {iv.value} for item '{name}'. "
                        f"Allowed values: {sorted(allowed)}"
                    )

        return {name: iv.value for name, iv in by_name.items() if iv.value is not None}

    def validate_constraints(self, item_values: Dict[str, float]) -> List[str]:
        """Check mutual exclusivity constraints on provided values.

        Returns list of violation messages (empty = valid).
        """
        violations = []
        for constraint in self.constraints:
            non_zero = [
                label for label in constraint.item_labels
                if label in item_values and item_values[label] != 0
            ]
            if len(non_zero) > 1:
                violations.append(
                    f"Constraint violated: {constraint.description}. "
                    f"Multiple items have non-zero values: {non_zero}"
                )
        return violations

    def _evaluate_formula(self, item_values: Dict[str, float]) -> float:
        """Evaluate the formula without constraint checks (internal use)."""
        if self.formula is None:
            raise ValueError("Cannot evaluate: formula not defined")
        result = ne.evaluate(self.formula, local_dict=item_values)
        return float(result)

    def validate_complete(self) -> None:
        """Raise ValueError if scale is missing fields required for scoring."""
        missing = []
        if self.formula is None:
            missing.append("formula")
        if self.min_score is None:
            missing.append("min_score")
        if self.max_score is None:
            missing.append("max_score")
        if missing:
            raise ValueError(
                f"Scale '{self.name}' is incomplete, missing: {', '.join(missing)}"
            )

    def calculate_score(self, item_values: Dict[str, float]) -> float:
        """Calculate the total score given item values."""
        if self.formula is None:
            raise ValueError("Cannot calculate score: formula not defined")
        required_ids = {item.label for item in self.items}
        provided_ids = set(item_values.keys())

        missing = required_ids - provided_ids
        if missing:
            raise ValueError(f"Missing values for items: {missing}")

        violations = self.validate_constraints(item_values)
        if violations:
            raise ValueError("; ".join(violations))

        result = ne.evaluate(self.formula, local_dict=item_values)
        return float(result)

    def calculate_score_range(
            self,
            known_values: Dict[str, float]
    ) -> Tuple[float, float, List[str], Dict[str, float]]:
        """
        Calculate possible score range given partial information.

        Returns (min_score, max_score, missing_ids, auto_filled) where
        auto_filled contains items whose values were determined by constraints.

        For each missing item, tests whether increasing the item's value
        increases or decreases the total score, then assigns min/max
        point values accordingly. Correct for all linear formulas,
        including those with subtraction.
        """
        if self.formula is None:
            raise ValueError("Cannot calculate score range: formula not defined")
        violations = self.validate_constraints(known_values)
        if violations:
            raise ValueError("; ".join(violations))

        all_item_ids = {item.label for item in self.items}
        known_ids = set(known_values.keys())
        missing_ids = list(all_item_ids - known_ids)

        # Auto-fill from constraints: if any constraint member has a non-zero
        # value, set all other members of that constraint to 0.
        auto_filled: Dict[str, float] = {}
        for constraint in self.constraints:
            has_non_zero = any(
                label in known_values and known_values[label] != 0
                for label in constraint.item_labels
            )
            if has_non_zero:
                for label in constraint.item_labels:
                    if label not in known_values and label in all_item_ids:
                        auto_filled[label] = 0

        # Merge auto-filled into known values
        effective_values = {**known_values, **auto_filled}
        missing_ids = list(all_item_ids - set(effective_values.keys()))

        if not missing_ids:
            score = self.calculate_score(effective_values)
            return (score, score, [], auto_filled)

        # Build per-item value lists for enumeration (deduplicate point values)
        items_by_label = {item.label: item for item in self.items}
        value_options = [sorted(set(items_by_label[label].points.values())) for label in missing_ids]

        # Safety cap: if too many combinations, fall back to independent min/max heuristic
        total_combinations = 1
        for opts in value_options:
            total_combinations *= len(opts)

        if total_combinations > 1_000_000:
            logger.warning(
                "Score range for '%s': %d combinations exceed limit; falling back to "
                "independent per-item min/max bounds (approximate for constrained groups).",
                self.name, total_combinations,
            )
            baseline = {**effective_values, **{label: value_options[i][0] for i, label in enumerate(missing_ids)}}
            baseline_score = self._evaluate_formula(baseline)
            min_values = baseline.copy()
            max_values = baseline.copy()
            for i, label in enumerate(missing_ids):
                pt_min = value_options[i][0]
                pt_max = value_options[i][-1]
                test = baseline.copy()
                test[label] = pt_max
                test_score = self._evaluate_formula(test)
                if test_score >= baseline_score:
                    max_values[label] = pt_max
                else:
                    min_values[label] = pt_max
                    max_values[label] = pt_min
            return (
                self._evaluate_formula(min_values),
                self._evaluate_formula(max_values),
                missing_ids,
                auto_filled,
            )

        min_score = float('inf')
        max_score = float('-inf')

        for combo in itertools_product(*value_options):
            assignment = {**effective_values, **dict(zip(missing_ids, combo))}
            if self.validate_constraints(assignment):  # non-empty = violation
                continue
            score = self._evaluate_formula(assignment)
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score

        return (min_score, max_score, missing_ids, auto_filled)

    def interpret_score(self, score: float) -> Optional[str]:
        """Find the interpretation matching an exact score."""
        if not self.interpretation:
            return None
        for interp in self.interpretation:
            if interp.min_score is not None and interp.max_score is not None:
                if interp.min_score <= score <= interp.max_score:
                    return interp.description
        return None

    def interpret_score_range(self, min_score: float, max_score: float) -> List[str]:
        """Find all interpretations overlapping with a score range."""
        if not self.interpretation:
            return []
        results = []
        for interp in self.interpretation:
            if interp.min_score is not None and interp.max_score is not None:
                if interp.min_score <= max_score and interp.max_score >= min_score:
                    results.append(interp.description)
        return results
