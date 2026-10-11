"""Deterministic, leak-free dataset splitting.

A single ordered ``class_names`` list is produced and reused everywhere so
that a label index always refers to the same class. Splits are:
  * train/validation: carved from the training root (per-class, seeded),
  * test: the held-out official validation root when available.

Overlaps are removed by absolute path, and the result is asserted leak-free.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .discovery import IMAGE_EXTS, _class_subdirs


def list_classes(root: str | Path) -> List[str]:
    """Return sorted class names (direct subdirectories containing images)."""
    return [p.name for p in _class_subdirs(Path(root))]


def build_class_names(
    train_root: str | Path,
    validation_root: str | Path | None = None,
    class_names_file: str | Path | None = None,
) -> List[str]:
    """Resolve the canonical, ordered class list.

    Precedence: explicit file > training root. When a validation root exists,
    its classes must be a subset of the chosen list (a warning note is not
    enough here, so mismatches raise).
    """
    if class_names_file and Path(class_names_file).exists():
        names = json.loads(Path(class_names_file).read_text(encoding="utf-8"))
        if not isinstance(names, list) or not names:
            raise ValueError(f"Invalid class_names file: {class_names_file}")
        names = [str(n) for n in names]
    else:
        names = list_classes(train_root)
        if not names:
            raise ValueError(f"No class subdirectories with images found under {train_root}")

    if validation_root is not None and Path(validation_root).exists():
        val_classes = set(list_classes(validation_root))
        extra = val_classes - set(names)
        if extra:
            raise ValueError(
                "Validation root contains classes absent from the training class list: "
                f"{sorted(extra)}. Refusing to build inconsistent label indices."
            )
    return names


def save_class_names(path: str | Path, class_names: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(class_names), indent=2), encoding="utf-8")
    return path


def load_class_names(path: str | Path) -> List[str]:
    return [str(n) for n in json.loads(Path(path).read_text(encoding="utf-8"))]


def _index_root(root: Path) -> Dict[str, List[str]]:
    """Map class name -> sorted absolute image paths."""
    index: Dict[str, List[str]] = {}
    for class_dir in _class_subdirs(root):
        files = [
            str(p.resolve())
            for p in sorted(class_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        index[class_dir.name] = files
    return index


def _split_class_files(
    files: Sequence[str], val_fraction: float, rng: random.Random
) -> Tuple[List[str], List[str]]:
    """Deterministically split one class's files into (train, val)."""
    shuffled = list(files)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    n_val = min(n_val, len(shuffled) - 1)  # never empty the train side
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


@dataclass
class SplitResult:
    class_names: List[str]
    train: List[Tuple[str, int]] = field(default_factory=list)
    val: List[Tuple[str, int]] = field(default_factory=list)
    test: List[Tuple[str, int]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"classes : {len(self.class_names)}",
            f"train   : {len(self.train)}",
            f"val     : {len(self.val)}",
            f"test    : {len(self.test)}",
        ]
        if self.notes:
            lines += ["notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


def make_splits(
    train_root: str | Path,
    class_names: Sequence[str],
    validation_root: str | Path | None = None,
    val_fraction: float = 0.10,
    seed: int = 42,
    test_source: str = "validation",
) -> SplitResult:
    """Build deterministic train/val/test splits.

    ``validation_root`` (the official held-out folder) is used as the TEST set
    when available; otherwise the test set is carved from the training data so
    a held-out evaluation still exists. Validation is always carved from train.
    """
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1).")

    name_to_index = {name: i for i, name in enumerate(class_names)}
    rng = random.Random(seed)
    result = SplitResult(class_names=list(class_names))

    train_index = _index_root(Path(train_root))
    seen_paths: set = set()

    for class_name in class_names:
        files = train_index.get(class_name, [])
        train_files, val_files = _split_class_files(files, val_fraction, rng)
        idx = name_to_index[class_name]
        for p in train_files:
            if p not in seen_paths:
                seen_paths.add(p)
                result.train.append((p, idx))
        for p in val_files:
            if p not in seen_paths:
                seen_paths.add(p)
                result.val.append((p, idx))

    use_official_val = (
        validation_root is not None
        and Path(validation_root).exists()
        and test_source == "validation"
    )
    if use_official_val:
        val_index = _index_root(Path(validation_root))
        for class_name in class_names:
            idx = name_to_index[class_name]
            for p in val_index.get(class_name, []):
                if p not in seen_paths:
                    seen_paths.add(p)
                    result.test.append((p, idx))
        result.notes.append("test set = official validation root")
    else:
        # Carve a deterministic test set from the training files.
        carve_rng = random.Random(seed + 1)
        test_frac = 0.10
        new_train: List[Tuple[str, int]] = []
        for class_name in class_names:
            idx = name_to_index[class_name]
            class_train = [p for p, i in result.train if i == idx]
            shuffled = sorted(class_train)
            carve_rng.shuffle(shuffled)
            if len(shuffled) <= 1:
                new_train.extend((p, idx) for p in shuffled)
                continue
            n_test = max(1, int(round(len(shuffled) * test_frac)))
            n_test = min(n_test, len(shuffled) - 1)
            for p in shuffled[:n_test]:
                result.test.append((p, idx))
            for p in shuffled[n_test:]:
                new_train.append((p, idx))
        result.train = new_train
        result.notes.append("test set carved from training data (no official validation root)")

    _assert_no_overlap(result)
    return result


def _assert_no_overlap(result: SplitResult) -> None:
    train_paths = {p for p, _ in result.train}
    val_paths = {p for p, _ in result.val}
    test_paths = {p for p, _ in result.test}
    if train_paths & val_paths or train_paths & test_paths or val_paths & test_paths:
        raise ValueError("Data leak detected: splits share image paths.")
