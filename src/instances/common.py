import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Settings:
    problems: Tuple[str, ...]  # ('IS','CA')
    is_sizes: Tuple[int, ...]
    ca_sizes: Tuple[int, ...]
    n_instances: int = 1000
    data_root: Path = Path("../data")
    test_split: float = 0.15
    val_split: float = 0.15
    # Independent‑Set
    edge_probability: float = 0.25
    # Combinatorial‑Auction
    ca_bid_factor: int = 5
    # Gurobi
    solve: bool = False
    gurobi_threads: int = 1
    gurobi_max_time: int = 3600
    gurobi_max_pool: int = 500
    gurobi_pool_mode: int = 2
    # system
    n_jobs: int = max(mp.cpu_count() // 2, 1)


INDEPENDANT_SET = "IS"
COMBINATORIAL_AUCTION = "CA"
