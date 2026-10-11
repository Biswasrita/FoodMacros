"""Food density lookup for the experimental volume x density path.

Densities live in ``configs/food_densities.json`` and are explicitly marked
*approximate*. A label with no entry returns ``None`` so the estimator reports
"unavailable" instead of inventing a density.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..config import PROJECT_ROOT

DEFAULT_DENSITY_FILE = PROJECT_ROOT / "configs" / "food_densities.json"


@dataclass
class DensityEntry:
    density_g_cm3: float
    approximate: bool = True
    note: str = ""


class DensityTable:
    """Case-insensitive, label-keyed density table."""

    def __init__(self, entries: Optional[Dict[str, DensityEntry]] = None) -> None:
        self._entries: Dict[str, DensityEntry] = {
            str(k).strip().lower(): v for k, v in (entries or {}).items()
        }

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "DensityTable":
        path = Path(path) if path is not None else DEFAULT_DENSITY_FILE
        if not path.exists():
            return cls({})
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("densities", {})
        entries: Dict[str, DensityEntry] = {}
        for label, value in raw.items():
            if "density_g_cm3" not in value:
                raise ValueError(
                    f"Density entry for '{label}' is missing 'density_g_cm3'."
                )
            entries[label] = DensityEntry(
                density_g_cm3=float(value["density_g_cm3"]),
                approximate=bool(value.get("approximate", True)),
                note=str(value.get("note", "")),
            )
        return cls(entries)

    def get(self, label) -> Optional[DensityEntry]:
        return self._entries.get(str(label).strip().lower())

    def labels(self):
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
