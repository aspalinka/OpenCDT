import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from opencdt.models import ItemInput
from opencdt.scale_store import ScaleStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Clinical Scale MCP")
store = ScaleStore()


@mcp.tool()
def list_scales() -> str:
    """List all available clinical scoring scales with brief info.

    Use search_scales to find a scale by keyword, or this tool to browse all.
    Then call get_scale -> calculate_score.
    """
    return json.dumps(store.list_scales(), indent=2)


@mcp.tool()
def get_scale(name: str) -> str:
    """Get full details of a clinical scale by name (case-insensitive).

    Step 2 in the workflow (after search_scales, before calculate_score).
    IMPORTANT: Always call this before calculate_score. The response contains
    the exact item labels, descriptions, and allowed values you must use.
    """
    scale = store.get_scale(name)
    if scale is None:
        return json.dumps({"error": f"Scale '{name}' not found"})
    return scale.model_dump_json(indent=2)


@mcp.tool()
def calculate_score(scale_name: str, item_values: list[ItemInput]) -> str:
    """Calculate score for a clinical scale.

    Step 3 in the workflow: search_scales -> get_scale -> calculate_score.
    IMPORTANT: Call get_scale first to retrieve the exact item labels.
    Item names must exactly match the 'label' field from the scale definition
    (e.g. use "urea", not "urea >7 mmol/L"; use "age_65", not "age >=65").

    ALL items from the scale must be included. Set value to null for unknown items.
    Each item with a value must include reasoning/evidence for that value.

    If all values are provided, returns an exact score with interpretation.
    If some values are null, returns a possible score range with the
    missing items (and their options) so you know what to ask the patient.

    Mutual exclusivity constraints are enforced automatically. For example,
    if age_75=1 is provided for CHA2DS2-VASc, age_65_74 is auto-filled to 0.

    Args:
        scale_name: Name of the scale (case-insensitive).
        item_values: List of all scale items. Each item has:
            - name: item label from get_scale output (snake_case, e.g. "confusion", "urea")
            - value: numeric point value, or null if unknown
            - reasoning: clinical reasoning/evidence (required when value is set)
    """
    scale = store.get_scale(scale_name)
    if scale is None:
        return json.dumps({"error": f"Scale '{scale_name}' not found"})

    try:
        known_values = scale.parse_item_inputs(item_values)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    try:
        min_score, max_score, missing_ids, auto_filled = (
            scale.calculate_score_range(known_values)
        )
    except ValueError as e:
        return json.dumps({"error": str(e)})

    if not missing_ids and min_score == max_score:
        # Exact result
        interpretation = scale.interpret_score(min_score)
        result = {
            "result_type": "exact",
            "score": min_score,
            "interpretation": interpretation,
        }
        if auto_filled:
            result["auto_filled"] = auto_filled
        return json.dumps(result, indent=2)

    # Range result
    items_by_label = {item.label: item for item in scale.items}
    missing_items = [
        {
            "label": label,
            "description": items_by_label[label].description,
            "options": items_by_label[label].points,
        }
        for label in missing_ids
    ]

    interpretations = scale.interpret_score_range(min_score, max_score)

    result = {
        "result_type": "range",
        "min_score": min_score,
        "max_score": max_score,
        "missing_items": missing_items,
        "possible_interpretations": interpretations,
    }
    if auto_filled:
        result["auto_filled"] = auto_filled
    return json.dumps(result, indent=2)


@mcp.tool()
def search_scales(query: str, tags: Optional[list[str]] = None) -> str:
    """Search for clinical scales by keyword and optional tag filter.

    This is typically the first step in the workflow:
    1. search_scales — find the right scale by keyword/tags
    2. get_scale — retrieve exact item labels and allowed values
    3. calculate_score — submit item values using labels from step 2

    Args:
        query: Search term matched against scale name, description, purpose, tags, etc.
        tags: Optional list of tags to filter results (must overlap with scale tags).
    """
    results = store.search_scales(query, tags)
    if not results:
        return json.dumps({"message": "No scales found matching the query"})
    return json.dumps(results, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
