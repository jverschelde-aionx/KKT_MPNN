import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Settings:
    problems: Tuple[str, ...]  # ('IS','CA', 'SC', 'CFL','RND')
    is_sizes: Tuple[int, ...]
    ca_sizes: Tuple[int, ...]
    sc_sizes: Tuple[int, ...]
    cfl_sizes: Tuple[int, ...]
    rnd_sizes: Tuple[int, ...]
    n_instances: int = 1000
    data_root: Path = Path("../data")
    test_split: float = 0.15
    val_split: float = 0.15
    add_positional_features: bool = True
    normalize_positional_features: bool = False
    normalize_features: bool = True
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


class ProblemClass:
    INDEPENDANT_SET = "IS"
    COMBINATORIAL_AUCTION = "CA"
    SET_COVER = "SC"
    CAPACITATED_FACILITY_LOCATION = "CFL"
    RANDOM_LP = "RND"


class VariableFeature:
    OBJ_COEF = 0
    AVG_COEF = 1
    DEGREE = 2
    MAX_COEF = 3
    MIN_COEF = 4
    IS_INTEGER = 5


class ConstraintFeature:
    AVG_COEF = 0
    DEGREE = 1
    RHS = 2
    SENSE = 3


class EdgeFeature:
    COEF = 0


SCIP_INF = 1e20
VARS_PAD = 18
CONS_PAD = 4
