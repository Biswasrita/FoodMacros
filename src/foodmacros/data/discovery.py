"""Runtime discovery of input datasets.

Nothing here assumes a fixed machine path. Given a list of search roots
(``/kaggle/input`` and a local ``data`` folder by default) it locates:

* image classification roots (directories whose children are class folders),
* the nutrition CSV (validated by its header columns),
* a Nutrition5k root (by its known markers), and
* food-detection weight files (``*.pt``).

When something is missing, the result carries an explicit "attach this
dataset" message instead of failing silently or inventing a path.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from ..config import AppConfig, PROJECT_ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}

# Required columns for a usable Food-101 nutrition CSV.
NUTRITION_REQUIRED_COLUMNS = {
    "label",
    "weight",
    "calories",
    "protein",
    "carbohydrates",
    "fats",
}
NUTRITION_OPTIONAL_COLUMNS = {"fiber", "sugars", "sodium"}

NUTRITION5K_MARKERS = {"realsense_overhead", "rgb", "depth_raw"}


def default_search_roots() -> List[Path]:
    """Standard locations to look for attached/local data."""
    roots = [Path("/kaggle/input"), PROJECT_ROOT / "data"]
    return [r for r in roots]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _dir_has_images(path: Path) -> bool:
    try:
        with os.scandir(path) as it:
            return any(e.is_file() and _is_image(Path(e.name)) for e in it)
    except (PermissionError, FileNotFoundError, NotADirectoryError):
        return False


def _class_subdirs(root: Path) -> List[Path]:
    subs: List[Path] = []
    try:
        with os.scandir(root) as it:
            for e in it:
                if e.is_dir() and _dir_has_images(Path(e.path)):
                    subs.append(Path(e.path))
    except (PermissionError, FileNotFoundError, NotADirectoryError):
        return []
    return sorted(subs, key=lambda p: p.name)


def _count_images(class_dirs: Sequence[Path]) -> int:
    total = 0
    for d in class_dirs:
        try:
            with os.scandir(d) as it:
                total += sum(1 for e in it if e.is_file() and _is_image(Path(e.name)))
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
    return total


def _iter_dirs(roots: Iterable[Path], max_depth: int) -> Iterable[Path]:
    """Yield every directory under ``roots`` up to ``max_depth`` (root = 0)."""
    for root in roots:
        root = Path(root)
        if not root.exists() or not root.is_dir():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            rel = Path(dirpath).relative_to(root)
            if len(rel.parts) > max_depth:
                dirnames[:] = []
                continue
            yield Path(dirpath)


def find_classification_roots(
    search_roots: Sequence[Path], max_depth: int = 3, min_classes: int = 2
) -> List[dict]:
    """Find directories that look like classification roots (class subfolders)."""
    found: List[dict] = []

    def recurse(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        classes = _class_subdirs(directory)
        if len(classes) >= min_classes:
            found.append(
                {
                    "root": str(directory),
                    "num_classes": len(classes),
                    "num_images": _count_images(classes),
                    "class_sample": [c.name for c in classes[:5]],
                }
            )
            return  # do not descend into a valid classification root
        try:
            with os.scandir(directory) as it:
                subdirs = [Path(e.path) for e in it if e.is_dir()]
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            return
        for sub in subdirs:
            recurse(sub, depth + 1)

    for root in search_roots:
        root = Path(root)
        if root.exists() and root.is_dir():
            recurse(root, 0)
    return found


def inspect_csv_columns(path: Path) -> List[str]:
    """Read only the header row of a CSV."""
    try:
        with Path(path).open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except (OSError, StopIteration):
        return []
    if not header:
        return []
    return [h.strip() for h in header]


def discover_nutrition_csvs(
    search_roots: Sequence[Path],
    required: set = NUTRITION_REQUIRED_COLUMNS,
    max_depth: int = 4,
) -> List[dict]:
    results: List[dict] = []
    seen: set = set()
    for directory in _iter_dirs(search_roots, max_depth):
        try:
            with os.scandir(directory) as it:
                csv_files = [
                    Path(e.path) for e in it if e.is_file() and e.name.lower().endswith(".csv")
                ]
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        for csv_file in csv_files:
            key = str(csv_file.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            columns = inspect_csv_columns(csv_file)
            lower = {c.lower() for c in columns}
            if required.issubset(lower):
                results.append(
                    {
                        "path": str(csv_file),
                        "columns": columns,
                        "missing_optional": sorted(NUTRITION_OPTIONAL_COLUMNS - lower),
                    }
                )
    return results


def discover_detection_weights(
    search_roots: Sequence[Path], max_depth: int = 4
) -> List[str]:
    results: List[str] = []
    seen: set = set()
    for directory in _iter_dirs(search_roots, max_depth):
        try:
            with os.scandir(directory) as it:
                pts = [
                    Path(e.path)
                    for e in it
                    if e.is_file() and e.name.lower().endswith(".pt")
                ]
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        for pt in pts:
            key = str(pt.resolve()).lower()
            if key not in seen:
                seen.add(key)
                results.append(str(pt))
    return results


def discover_nutrition5k_roots(
    search_roots: Sequence[Path], max_depth: int = 4
) -> List[str]:
    results: List[str] = []
    for directory in _iter_dirs(search_roots, max_depth):
        try:
            with os.scandir(directory) as it:
                entries = list(it)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        names = {e.name.lower() for e in entries}
        has_metadata = any(n.startswith("dish_metadata") and n.endswith(".csv") for n in names)
        has_marker = bool(names & NUTRITION5K_MARKERS)
        if has_metadata or has_marker:
            results.append(str(directory))
    return results


@dataclass
class ResolvedInputs:
    image_root: str | None = None
    validation_root: str | None = None
    nutrition_csv: str | None = None
    nutrition5k_root: str | None = None
    detection_weights: str | None = None
    candidates: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    @property
    def has_classification(self) -> bool:
        return self.image_root is not None

    def summary(self) -> str:
        lines = [
            f"image_root        : {self.image_root}",
            f"validation_root   : {self.validation_root}",
            f"nutrition_csv     : {self.nutrition_csv}",
            f"nutrition5k_root  : {self.nutrition5k_root}",
            f"detection_weights : {self.detection_weights}",
        ]
        if self.notes:
            lines += ["notes:"] + [f"  - {n}" for n in self.notes]
        if self.missing:
            lines += ["MISSING:"] + [f"  - {m}" for m in self.missing]
        return "\n".join(lines)


def _pick_train_validation(class_roots: List[dict]) -> tuple[str | None, str | None]:
    """Choose a train root and, if present, a sibling validation root."""
    train = None
    for cand in class_roots:
        name = Path(cand["root"]).name.lower()
        if "train" in name:
            train = cand["root"]
            break
    if train is None and class_roots:
        train = class_roots[0]["root"]

    validation = None
    if train is not None:
        parent = Path(train).parent
        for cand in class_roots:
            p = Path(cand["root"])
            if p == Path(train):
                continue
            if "val" in p.name.lower() and p.parent == parent:
                validation = str(p)
                break
    return train, validation


def resolve_inputs(
    config: AppConfig | None = None, search_roots: Sequence[Path] | None = None
) -> ResolvedInputs:
    """Locate all inputs, honouring explicit config overrides first."""
    config = config or AppConfig()
    roots = list(search_roots) if search_roots is not None else default_search_roots()
    resolved = ResolvedInputs()

    class_roots = find_classification_roots(roots)
    resolved.candidates["classification"] = class_roots

    csvs = discover_nutrition_csvs(roots)
    resolved.candidates["nutrition_csv"] = csvs

    n5k = discover_nutrition5k_roots(roots)
    resolved.candidates["nutrition5k"] = n5k

    pts = discover_detection_weights(roots)
    resolved.candidates["detection_weights"] = pts

    # Explicit overrides win.
    if config.data.image_root:
        resolved.image_root = config.data.image_root
    else:
        resolved.image_root, auto_val = _pick_train_validation(class_roots)
        if auto_val:
            resolved.validation_root = auto_val
    if config.data.validation_root:
        resolved.validation_root = config.data.validation_root
    if config.data.nutrition_csv:
        resolved.nutrition_csv = config.data.nutrition_csv
    elif csvs:
        resolved.nutrition_csv = csvs[0]["path"]
    if config.data.nutrition5k_root:
        resolved.nutrition5k_root = config.data.nutrition5k_root
    elif n5k:
        resolved.nutrition5k_root = n5k[0]
    if config.data.detection_weights:
        resolved.detection_weights = config.data.detection_weights
    elif pts:
        resolved.detection_weights = pts[0]

    # Honest, actionable missing-resource messages.
    if not resolved.image_root:
        resolved.missing.append(
            "Food image dataset not found. Attach Kaggle dataset "
            "'hari31416/food-101' (train/ + validation/) or set config.data.image_root."
        )
    if not resolved.nutrition_csv:
        resolved.missing.append(
            "Nutrition CSV not found. Attach Kaggle dataset "
            "'sanadalali/food-101-nutritional-information' (nutrition.csv) or set "
            "config.data.nutrition_csv."
        )
    if not resolved.nutrition5k_root:
        resolved.notes.append(
            "Nutrition5k not found: portion-weight regression will report "
            "'unavailable' until it is attached."
        )
    if not resolved.detection_weights:
        resolved.notes.append(
            "No YOLO food-detection weights (*.pt) found: detection will report "
            "'unavailable' until compatible weights are provided."
        )
    if resolved.image_root and csvs:
        resolved.notes.append(
            f"Discovered {len(csvs)} candidate nutrition CSV file(s); "
            f"using {resolved.nutrition_csv}."
        )
    return resolved
