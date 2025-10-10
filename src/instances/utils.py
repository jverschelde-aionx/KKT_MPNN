from pathlib import Path
from typing import Tuple


def ensure_dirs(root: Path, problem: str, size: int, split: str) -> Tuple[Path, Path]:
    inst = root / problem / "instance" / split / f"{size}"
    bg = root / problem / "BG" / split / f"{size}"
    inst.mkdir(parents=True, exist_ok=True)
    bg.mkdir(parents=True, exist_ok=True)
    return inst, bg


def lp_path_from_bg(bg_path: str) -> Path:
    """Map .../BG/<size>/<name>.lp.bg  ->  .../instance/<size>/<name>.lp"""
    p = Path(bg_path)
    parts = list(p.parts)
    try:
        idx = parts.index("BG")
    except ValueError:
        return Path()  # not a standard BG path
    parts[idx] = "instance"
    # replace .bg -> .sol
    stem = p.name[:-3] if p.name.endswith(".bg") else p.name
    sol_name = stem
    return Path(*parts[:-1]) / sol_name


def sol_path_from_bg(bg_path: str) -> Path:
    """Map .../BG/<size>/<name>.lp.bg  ->  .../solution/<size>/<name>.lp.sol"""
    p = Path(bg_path)
    parts = list(p.parts)
    try:
        idx = parts.index("BG")
    except ValueError:
        return Path()  # not a standard BG path
    parts[idx] = "solution"
    # replace .bg -> .sol
    stem = p.name[:-3] if p.name.endswith(".bg") else p.name
    sol_name = stem + ".sol"
    return Path(*parts[:-1]) / sol_name
