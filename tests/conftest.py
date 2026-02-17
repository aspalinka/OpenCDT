import pytest

from opencdt.models import (
    ClinicalScale,
    MutualExclusivityConstraint,
    ScoreItem,
    ScoreInterpretation,
)


def _binary_item(label: str) -> ScoreItem:
    return ScoreItem(label=label, description=f"{label} item", points={"No": 0, "Yes": 1})


@pytest.fixture
def cha2ds2_vasc() -> ClinicalScale:
    """Minimal CHA2DS2-VASc-like scale with age constraint."""
    return ClinicalScale(
        name="CHA2DS2-VASc",
        description="Stroke risk in AF",
        purpose="Guide anticoagulation",
        when_to_use=["AF patients"],
        items=[
            _binary_item("chf"),
            _binary_item("hypertension"),
            _binary_item("age_75"),
            _binary_item("diabetes"),
            _binary_item("stroke_tia"),
            _binary_item("vascular_disease"),
            _binary_item("age_65_74"),
            _binary_item("female"),
        ],
        constraints=[
            MutualExclusivityConstraint(
                item_labels=["age_75", "age_65_74"],
                description="Patient can only be in one age group",
            )
        ],
        formula="chf + hypertension + age_75*2 + diabetes + stroke_tia*2 + vascular_disease + age_65_74 + female",
        min_score=0,
        max_score=9,
        interpretation=[
            ScoreInterpretation(min_score=0, max_score=0, description="Low risk"),
            ScoreInterpretation(min_score=1, max_score=1, description="Low-moderate risk"),
            ScoreInterpretation(min_score=2, max_score=3, description="Moderate risk"),
            ScoreInterpretation(min_score=4, max_score=5, description="Moderate-high risk"),
            ScoreInterpretation(min_score=6, max_score=9, description="High risk"),
        ],
    )


@pytest.fixture
def wells_dvt() -> ClinicalScale:
    """Minimal Wells DVT scale (has negative contribution, no constraints)."""
    return ClinicalScale(
        name="Wells DVT",
        description="DVT probability",
        purpose="Risk-stratify DVT",
        when_to_use=["Suspected DVT"],
        items=[
            _binary_item("active_cancer"),
            _binary_item("paralysis"),
            _binary_item("bedridden"),
            _binary_item("localized_tenderness"),
            _binary_item("leg_swelling"),
            _binary_item("calf_swelling"),
            _binary_item("pitting_edema"),
            _binary_item("collateral_veins"),
            _binary_item("previous_dvt"),
            _binary_item("alternative_dx"),
        ],
        formula="active_cancer + paralysis + bedridden + localized_tenderness + leg_swelling + calf_swelling + pitting_edema + collateral_veins + previous_dvt - alternative_dx*2",
        min_score=-2,
        max_score=9,
        interpretation=[
            ScoreInterpretation(min_score=-2, max_score=0, description="Low probability"),
            ScoreInterpretation(min_score=1, max_score=2, description="Moderate probability"),
            ScoreInterpretation(min_score=3, max_score=9, description="High probability"),
        ],
    )


@pytest.fixture
def nonlinear_product() -> ClinicalScale:
    """Scale with non-linear formula (a-1)*(b-1) and items in {0, 1, 2}.

    Valid (a, b) → score:
        (0,0)=1, (0,1)=0, (0,2)=-1
        (1,0)=0, (1,1)=0, (1,2)=0
        (2,0)=-1,(2,1)=0, (2,2)=1

    True min=-1 (at (0,2) or (2,0)), true max=1 (at (0,0) or (2,2)).

    The direction-detection heuristic fails here: from baseline (0,0) score=1,
    raising either item decreases the score, so it assigns max to both items
    for minimisation → evaluates (2,2)=1, wrong answer.
    """
    return ClinicalScale(
        name="Nonlinear",
        description="Test",
        purpose="Testing",
        when_to_use=["Tests"],
        items=[
            ScoreItem(label="a", description="item a", points={"Low": 0, "Mid": 1, "High": 2}),
            ScoreItem(label="b", description="item b", points={"Low": 0, "Mid": 1, "High": 2}),
        ],
        formula="(a - 1) * (b - 1)",
        min_score=-1,
        max_score=1,
        interpretation=[
            ScoreInterpretation(min_score=-1, max_score=0, description="Negative"),
            ScoreInterpretation(min_score=1, max_score=1, description="Positive"),
        ],
    )


@pytest.fixture
def constrained_multivalued() -> ClinicalScale:
    """Scale with constrained multi-valued items (points: 0/1/2 each)."""
    return ClinicalScale(
        name="Multivalued",
        description="Test",
        purpose="Testing",
        when_to_use=["Tests"],
        items=[
            ScoreItem(label="a", description="item a", points={"None": 0, "Mild": 1, "Severe": 2}),
            ScoreItem(label="b", description="item b", points={"None": 0, "Mild": 1, "Severe": 2}),
        ],
        constraints=[
            MutualExclusivityConstraint(item_labels=["a", "b"], description="Only one active")
        ],
        formula="a + b",
        min_score=0,
        max_score=2,
        interpretation=[
            ScoreInterpretation(min_score=0, max_score=0, description="None"),
            ScoreInterpretation(min_score=1, max_score=2, description="Present"),
        ],
    )


@pytest.fixture
def simple_no_constraints() -> ClinicalScale:
    """Trivial 2-item scale with no constraints."""
    return ClinicalScale(
        name="Simple",
        description="Test scale",
        purpose="Testing",
        when_to_use=["Tests"],
        items=[_binary_item("a"), _binary_item("b")],
        formula="a + b",
        min_score=0,
        max_score=2,
        interpretation=[
            ScoreInterpretation(min_score=0, max_score=0, description="None"),
            ScoreInterpretation(min_score=1, max_score=2, description="Present"),
        ],
    )
