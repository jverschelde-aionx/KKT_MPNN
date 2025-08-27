import math
import multiprocessing as mp
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import gurobipy as gp
import numpy as np
import pyscipopt as scp
import torch
import tqdm
from ecole.instance import CombinatorialAuctionGenerator, IndependentSetGenerator
from loguru import logger

from instances.common import COMBINATORIAL_AUCTION, INDEPENDANT_SET, Settings
from instances.utils import ensure_dirs


def _detect_obj_sense(model: scp.Model, lp_path: Path) -> str:
    """Return 'min' or 'max'."""
    # 1) Try PySCIPOpt API (version-dependent)
    for attr in ("getObjectiveSense", "getObjSense", "getObjsense", "getObjSense"):
        if hasattr(model, attr):
            try:
                sense = getattr(model, attr)()
                # handle strings/enums
                if isinstance(sense, str):
                    s = sense.lower()
                    if s.startswith("max"):
                        return "max"
                    if s.startswith("min"):
                        return "min"
                else:
                    # Fallback for enum-like returns (SCIP_OBJSENSE.MAXIMIZE/MINIMIZE)
                    # Compare string name if available
                    s = str(sense).lower()
                    if "max" in s:
                        return "max"
                    if "min" in s:
                        return "min"
            except Exception:
                pass
    # 2) Fallback: scan LP header
    try:
        with open(lp_path, "r", errors="ignore") as f:
            for line in f:
                s = line.strip().lower()
                if s.startswith("maximize") or s.startswith("maximise"):
                    return "max"
                if s.startswith("minimize") or s.startswith("minimise"):
                    return "min"
    except Exception:
        pass
    raise ValueError(f"Could not detect objective sense in lp file: {lp_path}")


def _finite_lb(lb: float) -> bool:
    return lb > -1e20


def _finite_ub(ub: float) -> bool:
    return ub < 1e20


def _get_bounds(model: scp.Model, var: scp.Variable) -> Tuple[float, float]:
    """Best-effort: global bounds if available, else local, else +/-inf."""
    # Try several APIs to be robust across PySCIPOpt versions
    for fn in ("getVarLbGlobal", "getVarLb", "getVarLowerBound"):
        if hasattr(model, fn):
            try:
                lb = float(getattr(model, fn)(var))
                break
            except Exception:
                lb = -math.inf
    else:
        try:
            lb = float(var.getLb())  # may exist in some versions
        except Exception:
            lb = -math.inf

    for fn in ("getVarUbGlobal", "getVarUb", "getVarUpperBound"):
        if hasattr(model, fn):
            try:
                ub = float(getattr(model, fn)(var))
                break
            except Exception:
                ub = math.inf
    else:
        try:
            ub = float(var.getUb())
        except Exception:
            ub = math.inf

    # Normalize extreme infinities to large sentinels
    if not np.isfinite(lb):
        lb = -1e50
    if not np.isfinite(ub):
        ub = 1e50
    return lb, ub


def generate_IS_instances(
    settings: Settings, split: str, n_instances: int, lp_paths: List[Path]
) -> None:
    """Generate Independent Set instances."""

    for size in settings.is_sizes:
        inst_dir, _ = ensure_dirs(settings.data_root, INDEPENDANT_SET, size, split)
        for i in tqdm.trange(
            n_instances, desc=f"Generating {size} Independent Set instances"
        ):
            name = f"{INDEPENDANT_SET}-{size}-{i:04}.lp"
            lp_path = inst_dir / name
            if lp_path.exists():
                lp_paths.append(lp_path)
                continue

            gen = IndependentSetGenerator(
                n_nodes=size,
                edge_probability=settings.edge_probability,
                graph_type="barabasi_albert",
            )
            next(gen).write_problem(str(lp_path))
            lp_paths.append(lp_path)


def generate_CA_instances(
    settings: Settings, split: str, n_instances: int, lp_paths: List[Path]
) -> None:
    """Generate Combinatorial Auction instances."""
    for size in settings.ca_sizes:
        inst_dir, _ = ensure_dirs(
            settings.data_root, COMBINATORIAL_AUCTION, size, split
        )
        for i in tqdm.trange(
            n_instances, desc=f"Generating {size} Combinatorial Auction instances"
        ):
            name = f"{COMBINATORIAL_AUCTION}-{size}-{i:04}.lp"
            lp_path = inst_dir / name
            if lp_path.exists():
                lp_paths.append(lp_path)
                continue

            gen = CombinatorialAuctionGenerator(
                n_items=size, n_bids=size * settings.ca_bid_factor
            )

            next(gen).write_problem(str(lp_path))
            lp_paths.append(lp_path)


def solve_instance(
    settings: Settings, lp_path: Path, solution_path: Path, log_dir: Path
) -> None:
    gp.setParam("LogToConsole", 0)
    m = gp.read(str(lp_path))
    m.Params.Threads = settings.gurobi_threads
    m.Params.PoolSolutions = settings.gurobi_max_pool
    m.Params.PoolSearchMode = settings.gurobi_pool_mode
    m.Params.TimeLimit = settings.gurobi_max_time
    m.Params.LogFile = str(log_dir / (lp_path.name + ".log"))
    m.optimize()

    sols, objs = [], []
    var_names = [v.VarName for v in m.getVars()]
    for sn in range(m.SolCount):
        m.Params.SolutionNumber = sn
        sols.append(np.array(m.Xn, dtype=np.float32))
        objs.append(float(m.PoolObjVal))
    sol_data = {
        "var_names": var_names,
        "sols": np.vstack(sols),
        "objs": np.array(objs),
    }

    with open(solution_path, "wb") as f:
        pickle.dump(sol_data, f)
    m.dispose()


def generate_instances(settings: Settings) -> None:
    for problem in settings.problems:
        logger.info("Generating instances for problem type: {}", problem)

        test_n = round(settings.n_instances * settings.test_split)
        val_n = round(settings.n_instances * settings.val_split)
        train_n = settings.n_instances - test_n - val_n

        for split, n_instances in (
            ("train", train_n),
            ("val", val_n),
            ("test", test_n),
        ):
            lp_paths: List[Path] = []

            # generate the .lp files for this split only
            if problem == INDEPENDANT_SET:
                generate_IS_instances(settings, split, n_instances, lp_paths)
            elif problem == COMBINATORIAL_AUCTION:
                generate_CA_instances(settings, split, n_instances, lp_paths)
            else:
                raise ValueError(f"Unknown problem type: {problem}")

            logger.info("[{}] {} .lp files to process", split, len(lp_paths))

            # process those files
            for lp_path in tqdm.tqdm(lp_paths, desc=f"Processing .lp files ({split})"):
                bg_dir = (
                    settings.data_root / problem / "BG" / split / lp_path.parent.name
                )
                sol_dir = (
                    settings.data_root
                    / problem
                    / "solution"
                    / split
                    / lp_path.parent.name
                )
                log_dir = (
                    settings.data_root / problem / "logs" / split / lp_path.parent.name
                )
                for d in (bg_dir, sol_dir, log_dir):
                    d.mkdir(parents=True, exist_ok=True)

                solution_path = sol_dir / (lp_path.name + ".sol")
                if settings.solve and not solution_path.exists():
                    solve_instance(settings, lp_path, solution_path, log_dir)

                bg_path = bg_dir / (lp_path.name + ".bg")
                if not bg_path.exists():
                    try:
                        graph_data = get_bipartite_graph(lp_path)
                    except Exception as e:
                        logger.error("Failed on {} – {}", lp_path, e)
                        continue

                    with open(bg_path, "wb") as f:
                        pickle.dump(graph_data, f)

        logger.success(
            "All graphs for {} written to {}",
            problem,
            settings.data_root / problem / "BG",
        )


def position_get_ordered_flt(variable_features: torch.Tensor) -> torch.Tensor:
    """Encode relative variable order as a 20‑bit binary fraction."""
    length = variable_features.shape[0]
    feat_w = 20
    sorter = variable_features[:, 1]
    position = torch.argsort(sorter).float() / float(length)

    pos_feat = torch.zeros(length, feat_w)
    for row in range(length):
        flt = position[row].item()
        divider = 0.5
        for k in range(feat_w):
            if flt >= divider:
                pos_feat[row, k] = 1.0
                flt -= divider
        divider *= 0.5
    return torch.cat([variable_features, pos_feat], dim=1)


def get_bipartite_graph(lp_path: Path):
    """
    Parse an LP and return a 7‑tuple:
      (A, v_map, v_nodes, c_nodes, b_vars, b_vec, c_vec)

    - A: sparse_coo (m, n) with REAL coefficients
    - v_nodes: (n, 6) encoder features ∈ [0,1] (not used by KKT math)
    - c_nodes: (m, 4) encoder features ∈ [0,1]
               [avg_coef_per_row, degree, rhs, is_bound]
    - b_vars: indices of binary vars (for reference)
    - b_vec: (m,) RHS for Ax ≤ b
    - c_vec: (n,) objective for MINIMIZATION (we flip if original LP is 'maximize')
    """
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(lp_path)

    # ----- variables --------------------------------------------------------
    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)
    nvars = len(mvars)
    v_map: Dict[scp.Variable, int] = {v: i for i, v in enumerate(mvars)}

    ORI_FEATS = 6  # [obj_coef, avg_coef, degree, max_coef, min_coef, is_binary]
    v_nodes = np.zeros((nvars, ORI_FEATS), dtype=np.float32)
    b_vars: List[int] = []
    for i, v in enumerate(mvars):
        try:
            vtype = v.vtype() if hasattr(v, "vtype") else v.getVType()
            if str(vtype).upper().startswith("BINARY") or str(vtype).upper() == "B":
                v_nodes[i, 5] = 1.0
                b_vars.append(i)
        except Exception:
            pass  # Non-MIP LPs may not expose a vtype API; ignore

    # ----- objective (detect sense, flip to min if needed) ------------------
    obj = m.getObjective()
    c_raw = np.zeros(nvars, dtype=np.float32)
    for term in obj:
        var = term.vartuple[0]
        j = v_map[var]
        coef = float(obj[term])
        c_raw[j] += coef
        v_nodes[j, 0] = coef  # keep raw objective coef as a feature

    sense = _detect_obj_sense(m, lp_path)  # 'min' or 'max'
    c_vec = torch.tensor(c_raw if sense == "min" else -c_raw, dtype=torch.float32)

    # ----- constraints -> Ax ≤ b (also add bounds as rows) ------------------
    conss = [c for c in m.getConss() if len(m.getValsLinear(c)) > 0]

    ind_rows: List[int] = []
    ind_cols: List[int] = []
    vals: List[float] = []
    b_vec_list: List[float] = []
    c_feat_rows: List[List[float]] = []  # [avg_coef, degree, rhs, is_bound]

    def add_leq_row(coeffs_dict, rhs_val, is_bound: float):
        row_idx = len(b_vec_list)
        deg = 0
        coef_sum = 0.0
        for var_obj, coef in coeffs_dict.items():
            j = v_map[var_obj]
            coef = float(coef)
            if coef != 0.0:
                ind_rows.append(row_idx)
                ind_cols.append(j)
                vals.append(coef)
                deg += 1
                coef_sum += coef
                # per-variable stats for encoder
                v_nodes[j, 2] += 1.0  # degree
                v_nodes[j, 1] += coef  # sum of coefs (avg later)
                v_nodes[j, 3] = max(v_nodes[j, 3], coef)
                # initialize min with large pos first time
                v_nodes[j, 4] = (
                    coef if v_nodes[j, 4] == 0.0 else min(v_nodes[j, 4], coef)
                )
        b_vec_list.append(float(rhs_val))
        avg = coef_sum / max(deg, 1)
        c_feat_rows.append([avg, float(deg), float(rhs_val), is_bound])

    # Linear rows (turn => and == into ≤)
    for c in sorted(conss, key=lambda x: (len(m.getValsLinear(x)), str(x))):
        coeffs = m.getValsLinear(c)  # dict[var_obj] -> coef
        rhs = float(m.getRhs(c))
        lhs = float(m.getLhs(c))
        if rhs < 1e20:  # expr ≤ rhs
            add_leq_row(coeffs, rhs, is_bound=0.0)
        if lhs > -1e20:  # expr ≥ lhs  →  -expr ≤ -lhs
            neg_coeffs = {v: -coef for v, coef in coeffs.items()}
            add_leq_row(neg_coeffs, -lhs, is_bound=0.0)

    # Variable bounds (two rows per finite bound)
    for v in mvars:
        j = v_map[v.name]
        lb, ub = _get_bounds(m, v)  # may be +/-1e50 for “infinite”
        if _finite_ub(ub):
            # x_j ≤ ub
            add_leq_row({v: +1.0}, ub, is_bound=1.0)
        if _finite_lb(lb):
            # -x_j ≤ -lb
            add_leq_row({v: -1.0}, -lb, is_bound=1.0)

    # Build sparse A and vectors
    m_rows = len(b_vec_list)
    A = torch.sparse_coo_tensor(
        indices=torch.tensor([ind_rows, ind_cols], dtype=torch.long),
        values=torch.tensor(vals, dtype=torch.float32),
        size=(m_rows, nvars),
    ).coalesce()
    b_vec = torch.tensor(b_vec_list, dtype=torch.float32)

    # ----- normalize encoder features (NOT used by KKT math) ----------------
    v_nodes_t = torch.tensor(v_nodes, dtype=torch.float32)
    # avg coef per var = sum / degree
    deg = v_nodes_t[:, 2].clamp(min=1.0)
    v_nodes_t[:, 1] = v_nodes_t[:, 1] / deg

    # min–max per column to [0,1]
    def _minmax01(x: torch.Tensor) -> torch.Tensor:
        mn, mx = x.min(dim=0).values, x.max(dim=0).values
        return (x - mn) / (mx - mn + 1e-9)

    v_nodes_t = _minmax01(v_nodes_t).clamp_(0.0, 1.0)

    c_nodes_t = (
        torch.tensor(c_feat_rows, dtype=torch.float32)
        if m_rows > 0
        else torch.zeros(0, 4)
    )
    if c_nodes_t.numel() > 0:
        c_nodes_t = _minmax01(c_nodes_t).clamp_(0.0, 1.0)

    return (
        A,
        v_map,
        # {v.name: i for v, i in v_map.items()},  # return a name→index view for reference
        v_nodes_t,
        c_nodes_t,
        torch.tensor(b_vars, dtype=torch.int32),
        b_vec,
        c_vec,
    )
