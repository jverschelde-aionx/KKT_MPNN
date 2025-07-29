from pathlib import Path
from typing import Tuple


def ensure_dirs(root: Path, problem: str, size: int, split: str) -> Tuple[Path, Path]:
    inst = root / problem / "instance" / split / f"{size}"
    bg = root / problem / "BG" / split / f"{size}"
    inst.mkdir(parents=True, exist_ok=True)
    bg.mkdir(parents=True, exist_ok=True)
    return inst, bg
