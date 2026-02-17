import json

import pytest

from opencdt.scale_store import ScaleStore


@pytest.fixture
def store() -> ScaleStore:
    return ScaleStore()


class TestScaleStore:
    def test_loads_both_scales(self, store):
        scales = store.list_scales()
        names = {s["name"] for s in scales}
        assert "CHA2DS2-VASc" in names
        assert "Wells DVT" in names

    def test_get_scale_case_insensitive(self, store):
        assert store.get_scale("cha2ds2-vasc") is not None
        assert store.get_scale("CHA2DS2-VASc") is not None

    def test_get_scale_not_found(self, store):
        assert store.get_scale("nonexistent") is None

    def test_loaded_scale_has_constraints(self, store):
        scale = store.get_scale("CHA2DS2-VASc")
        assert len(scale.constraints) == 1

    def test_wells_dvt_no_constraints(self, store):
        scale = store.get_scale("Wells DVT")
        assert scale.constraints == []

    def test_search_by_query(self, store):
        results = store.search_scales("stroke")
        assert any(r["name"] == "CHA2DS2-VASc" for r in results)

    def test_search_by_tag(self, store):
        results = store.search_scales("", tags=["DVT"])
        # query is empty string but tags filter applies
        assert any(r["name"] == "Wells DVT" for r in results)

    def test_nonexistent_dir_logs_warning(self, tmp_path):
        store = ScaleStore(scales_dir=tmp_path / "nope")
        assert store.list_scales() == []

    def test_custom_dir_with_valid_json(self, tmp_path):
        scale_data = {
            "name": "Test Scale",
            "description": "A test",
            "purpose": "Testing",
            "when_to_use": ["Tests"],
            "items": [{"label": "x", "description": "x item", "points": {"No": 0, "Yes": 1}}],
            "formula": "x",
            "min_score": 0,
            "max_score": 1,
            "interpretation": [{"min_score": 0, "max_score": 1, "description": "ok"}],
        }
        (tmp_path / "test.json").write_text(json.dumps(scale_data))
        store = ScaleStore(scales_dir=tmp_path)
        assert len(store.list_scales()) == 1
        assert store.get_scale("Test Scale") is not None
