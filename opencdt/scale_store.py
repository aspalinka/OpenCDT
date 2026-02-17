import json
import logging
from pathlib import Path
from typing import Optional

from opencdt.models import ClinicalScale

logger = logging.getLogger(__name__)


class ScaleStore:
    """Loads and indexes clinical scales from JSON files."""

    def __init__(self, scales_dir: Optional[Path] = None):
        if scales_dir is None:
            scales_dir = Path(__file__).resolve().parent.parent / "scales"
        self._scales: dict[str, ClinicalScale] = {}
        self._load_all(scales_dir)

    def _load_all(self, scales_dir: Path) -> None:
        if not scales_dir.is_dir():
            logger.warning("Scales directory not found: %s", scales_dir)
            return
        for path in sorted(scales_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                scale = ClinicalScale(**data)
                scale.validate_complete()
                self._scales[scale.name.lower()] = scale
                logger.info("Loaded scale: %s", scale.name)
            except Exception:
                logger.exception("Failed to load scale from %s", path)

    def list_scales(self) -> list[dict]:
        """Return brief info for all scales."""
        return [
            {
                "name": s.name,
                "full_name": s.full_name,
                "description": s.description,
                "category": s.category,
                "tags": s.tags,
            }
            for s in self._scales.values()
        ]

    def get_scale(self, name: str) -> ClinicalScale | None:
        """Case-insensitive lookup by name."""
        return self._scales.get(name.lower())

    def search_scales(self, query: str, tags: list[str] | None = None) -> list[dict]:
        """Substring search across scale metadata. Optional tag filter."""
        query_lower = query.lower()
        results = []
        for scale in self._scales.values():
            searchable = " ".join([
                scale.name,
                scale.full_name or "",
                scale.description or "",
                scale.purpose or "",
                " ".join(scale.when_to_use or []),
                " ".join(scale.tags),
                scale.category or "",
            ]).lower()

            if query_lower not in searchable:
                continue

            if tags and not set(t.lower() for t in tags) & set(t.lower() for t in scale.tags):
                continue

            results.append({
                "name": scale.name,
                "full_name": scale.full_name,
                "description": scale.description,
                "category": scale.category,
                "tags": scale.tags,
            })
        return results
