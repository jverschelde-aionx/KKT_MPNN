from pathlib import Path
from typing import Optional, Tuple

from torch_geometric.data import Batch


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


def infer_sol_path(instance_path: str) -> Optional[Path]:
    p = Path(instance_path)
    if p.suffix == ".bg":
        sol = sol_path_from_bg(p)  # your existing helper
        return sol if sol and sol.exists() else None
    elif p.suffix == ".lp":
        # If .sol sits next to .lp with same stem
        sol = p.with_suffix(".lp.sol")
        sol_str = str(sol).replace("/instance/", "/solution/")
        sol = Path(sol_str)
        return sol if sol.exists() else None
    else:
        return None


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


def batch_is_bipartite(batch: tuple) -> bool:
    return isinstance(batch[0], Batch)
