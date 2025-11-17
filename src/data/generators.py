import multiprocessing as mp
import pickle
from dataclasses import asdict, dataclass
from math import pow
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
import gurobipy as gp
import numpy as np
import pyscipopt as scp
import torch
import tqdm
from ecole.instance import (
    CapacitatedFacilityLocationGenerator,
    CombinatorialAuctionGenerator,
    IndependentSetGenerator,
    SetCoverGenerator,
)
from gurobipy import GRB
from loguru import logger

from data.common import (
    ConstraintFeature,
    EdgeFeature,
    ProblemClass,
    Settings,
    VariableFeature,
)
from data.utils import ensure_dirs


@dataclass
class CFLSize:
    n_customers: int
    n_facilities: int
    variables: int
    constraints: int


def factor_pairs_for_vars(V: int) -> List[Tuple[int, int]]:
    """
    Return all (n_facilities, n_customers) pairs s.t.
    n_facilities * (n_customers + 1) = V
    """
    pairs = []
    for d in range(1, V + 1):
        if V % d == 0:
            n_fac = d
            n_cust_plus_1 = V // d
            n_cust = n_cust_plus_1 - 1
            if n_cust >= 1:  # need at least 1 customer
                pairs.append((n_fac, n_cust))
    return pairs


def pick_cfl_shape(
    var_target: int,
    cons_target: Optional[int] = None,
    min_facilities: int = 2,  # avoid trivial single-facility by default
    max_facilities: Optional[int] = None,
) -> CFLSize:
    """
    Pick (n_facilities, n_customers) to match a desired total variable count.
    If cons_target is given, pick the pair that gets constraints (n_f + n_c)
    closest to it.
    """
    candidates = [
        (nf, nc)
        for (nf, nc) in factor_pairs_for_vars(var_target)
        if nf >= min_facilities and (max_facilities is None or nf <= max_facilities)
    ] or factor_pairs_for_vars(var_target)  # fallback if filter too strict

    if not candidates:
        raise ValueError("No valid (n_facilities, n_customers) pairs found.")

    if cons_target is None:
        # heuristic: prefer more than one facility, and keep counts balanced-ish
        # score = penalize |nf - nc|
        def score(pair):
            nf, nc = pair
            return abs(nf - nc)

        nf, nc = min(candidates, key=score)
    else:
        # pick closest to desired constraints
        def score(pair):
            nf, nc = pair
            return abs((nf + nc) - cons_target)

        nf, nc = min(candidates, key=score)

    return CFLSize(
        n_customers=nc, n_facilities=nf, variables=nf * (nc + 1), constraints=nf + nc
    )


def _scaled_density(n_rows: int, n_cols: int, k: int = 7, eps: float = None) -> float:
    if eps is not None:
        d1 = 1 - pow(eps, 1.0 / n_cols)
        d2 = 1 - pow(eps, 1.0 / n_rows)
        d = max(d1, d2)
    else:
        d = max(k / n_cols, k / n_rows)
    return min(max(d, 0.01), 0.95)


# instance-wise min–max normalization for features
def _minmax_normalization(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    mn = x.min(dim=0).values
    mx = x.max(dim=0).values
    return (x - mn) / (mx - mn + 1e-9)


# helper to encode constraint sense
def _constraint_sense_code(problem, constraint) -> Tuple[int, float]:
    rhs = float(problem.getRhs(constraint))
    lhs = float(problem.getLhs(constraint))
    if abs(rhs - lhs) < 1e-12:
        return 2, rhs  # equality
    if rhs < 1e20:
        return 0, rhs  # <=
    # else treat as >= with lhs finite
    return 1, lhs  # >=


def relax_to_lp_and_minimize_inplace(lp_path: Path) -> None:
    """
    Ensure the on-disk model is a *linear minimization* problem.

    Steps:
    - Read original model (.lp may contain MIP/SOS/indicator).
    - Build LP relaxation with m.relax() (removes integrality & SOS/indicator constraints).
    - If it was a maximization, negate objective and switch to minimization.
    - Overwrite the same .lp file with the relaxed/minimized model.
    """
    m = gp.read(str(lp_path))
    mr = None
    try:
        # LP relaxation removes integrality/SOS/indicator automatically
        mr = m.relax()
        mr.update()

        # normalize objective direction to MIN
        if mr.ModelSense == -1:  # -1 = maximize
            for v in mr.getVars():
                v.Obj = -v.Obj
            mr.ModelSense = 1  # +1 = minimize
            mr.update()

        # overwrite original path with the relaxed model
        mr.write(str(lp_path))
        logger.info(f"[relaxed->LP & MIN] Rewrote {lp_path.name}")
    finally:
        try:
            if mr is not None:
                mr.dispose()
        except Exception:
            pass
        m.dispose()


def generate_IS_instances(
    settings: Settings, split: str, n_instances: int, lp_paths: List[Path]
) -> None:
    """Generate Independent Set instances."""

    for size in settings.is_sizes:
        inst_dir, _ = ensure_dirs(
            settings.data_root, ProblemClass.INDEPENDANT_SET, size, split
        )
        for i in tqdm.trange(
            n_instances, desc=f"Generating {size} Independent Set instances"
        ):
            name = f"{ProblemClass.INDEPENDANT_SET}-{size}-{i:04}.lp"
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
            relax_to_lp_and_minimize_inplace(lp_path)
            lp_paths.append(lp_path)


def generate_CA_instances(
    settings: Settings, split: str, n_instances: int, lp_paths: List[Path]
) -> None:
    """Generate Combinatorial Auction instances."""
    for size in settings.ca_sizes:
        inst_dir, _ = ensure_dirs(
            settings.data_root, ProblemClass.COMBINATORIAL_AUCTION, size, split
        )
        for i in tqdm.trange(
            n_instances, desc=f"Generating {size} Combinatorial Auction instances"
        ):
            name = f"{ProblemClass.COMBINATORIAL_AUCTION}-{size}-{i:04}.lp"
            lp_path = inst_dir / name
            if lp_path.exists():
                lp_paths.append(lp_path)
                continue

            gen = CombinatorialAuctionGenerator(
                n_items=size, n_bids=size * settings.ca_bid_factor
            )

            next(gen).write_problem(str(lp_path))
            relax_to_lp_and_minimize_inplace(lp_path)
            lp_paths.append(lp_path)


def generate_SC_instances(
    settings: Settings, split: str, n_instances: int, lp_paths: List[Path]
) -> None:
    """Generate Set Cover instances."""
    for size in settings.sc_sizes:
        inst_dir, _ = ensure_dirs(
            settings.data_root, ProblemClass.SET_COVER, size, split
        )
        density = _scaled_density(size, size, k=7)
        for i in tqdm.trange(
            n_instances, desc=f"Generating {size} Set Cover instances"
        ):
            name = f"{ProblemClass.SET_COVER}-{size}-{i:04}.lp"
            lp_path = inst_dir / name
            if lp_path.exists():
                lp_paths.append(lp_path)
                continue

            gen = SetCoverGenerator(
                n_rows=size,
                n_cols=size,
                density=density,
                max_coef=100,
            )

            next(gen).write_problem(str(lp_path))
            relax_to_lp_and_minimize_inplace(lp_path)
            lp_paths.append(lp_path)


def generate_CFL_instances(
    settings: Settings, split: str, n_instances: int, lp_paths: List[Path]
) -> None:
    """Generate Capacitated Facility Location instances."""
    for size in settings.cfl_sizes:
        inst_dir, _ = ensure_dirs(
            settings.data_root,
            ProblemClass.CAPACITATED_FACILITY_LOCATION,
            size,
            split,
        )

        cfg = pick_cfl_shape(
            var_target=size, cons_target=int(size / 2), min_facilities=2
        )

        for i in tqdm.trange(
            n_instances,
            desc=f"Generating {size} Capacitated Facility Location instances",
        ):
            name = f"{ProblemClass.CAPACITATED_FACILITY_LOCATION}-{size}-{i:04}.lp"
            lp_path = inst_dir / name
            if lp_path.exists():
                lp_paths.append(lp_path)
                continue

            gen = CapacitatedFacilityLocationGenerator(
                n_customers=cfg.n_customers,
                n_facilities=cfg.n_facilities,
                continuous_assignment=False,
                ratio=1.2,
            )

            next(gen).write_problem(str(lp_path))
            relax_to_lp_and_minimize_inplace(lp_path)
            lp_paths.append(lp_path)


def generate_random_instances(
    settings: Settings, split: str, n_instances: int, lp_paths: List[Path]
) -> None:
    for size in settings.rnd_sizes:
        inst_dir, _ = ensure_dirs(
            settings.data_root,
            ProblemClass.RANDOM_LP,
            size,
            split,
        )

        optimal_count = 0
        logger.info(f"Generating {n_instances} random LP instances of size {size}")
        while optimal_count < n_instances:
            name = f"{ProblemClass.RANDOM_LP}-{size}-{optimal_count:04}.lp"
            lp_path = inst_dir / name
            print(f"generating lp: {lp_path}")
            if lp_path.exists():
                lp_paths.append(lp_path)
                optimal_count += 1
                continue
            c1 = np.random.uniform(-3, 3, size)
            A1 = np.random.uniform(-3, 3, (size, size))
            b1 = np.random.uniform(-3, 3, size)

            max_value = max(np.max(np.abs(c1)), np.max(np.abs(A1)), np.max(np.abs(b1)))

            # Normalize A, b, and c by dividing by the max value
            c = c1 / max_value
            A = A1 / max_value
            b = b1 / max_value

            x = cp.Variable(size)

            # Define the LP problem
            objective = cp.Minimize(c.T @ x)
            constraints = [A @ x <= b]

            # Create and solve the problem
            problem = cp.Problem(objective, constraints)

            try:
                # Solve the problem
                problem.solve()
                # Check if the problem is solvable and optimal
                if problem.status == cp.OPTIMAL:
                    # write the problem to lp file
                    problem.solve(solver=cp.GUROBI, save_file=str(lp_path))
                    optimal_count += 1

                    if optimal_count % 100 == 0:
                        logger.info(f"Generated {optimal_count} optimal instances")

                    relax_to_lp_and_minimize_inplace(lp_path)
                    lp_paths.append(lp_path)
                    logger.info(f"Generated optimal instance: {lp_path.name}")

            except cp.SolverError:
                logger.info("Problem is not solvable, discarding.")


def solve_instance(
    settings: Settings, lp_path: Path, solution_path: Path, log_dir: Path
) -> None:
    gp.setParam("LogToConsole", 0)
    m = gp.read(str(lp_path))

    m.Params.Threads = settings.gurobi_threads
    m.Params.TimeLimit = settings.gurobi_max_time
    m.Params.LogFile = str(log_dir / (lp_path.name + ".log"))

    try:
        m.optimize()
    except gp.GurobiError as e:
        logger.error("Gurobi failed on {}: {}", lp_path, e)
        m.dispose()
        return

    status = m.Status
    if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        logger.error("LP {} infeasible or unbounded (status={})", lp_path, status)
        m.dispose()
        return

    vars_ = m.getVars()
    var_names = [v.VarName for v in vars_]

    # For LPs, read the primal solution from X (no pool)
    try:
        x = m.getAttr(gp.GRB.Attr.X, vars_)
        obj = float(m.ObjVal)
    except gp.GurobiError as e:
        logger.error("No primal solution available for {}: {}", lp_path, e)
        m.dispose()
        return

    sol_data = {
        "var_names": var_names,
        "sols": np.asarray([x], dtype=np.float32),  # (1, nvars)
        "objs": np.asarray([obj], dtype=np.float64),  # (1,)
        "status": status,
        "is_lp": True,
    }

    solution_path.parent.mkdir(parents=True, exist_ok=True)
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
            if problem == ProblemClass.INDEPENDANT_SET:
                generate_IS_instances(settings, split, n_instances, lp_paths)
            elif problem == ProblemClass.COMBINATORIAL_AUCTION:
                generate_CA_instances(settings, split, n_instances, lp_paths)
            elif problem == ProblemClass.SET_COVER:
                generate_SC_instances(settings, split, n_instances, lp_paths)
            elif problem == ProblemClass.CAPACITATED_FACILITY_LOCATION:
                generate_CFL_instances(settings, split, n_instances, lp_paths)
            elif problem == ProblemClass.RANDOM_LP:
                generate_random_instances(settings, split, n_instances, lp_paths)
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
                if (
                    settings.solve
                    and not solution_path.exists()
                    and split in ["val", "test"]
                ):
                    try:
                        solve_instance(settings, lp_path, solution_path, log_dir)
                    except Exception as e:
                        logger.error("Failed to solve {} – {}", lp_path, e)
                        continue

                bg_path = bg_dir / (lp_path.name + ".bg")
                if not bg_path.exists():
                    try:
                        graph_data = get_bipartite_graph(
                            lp_path,
                            settings.add_positional_features,
                            settings.normalize_positional_features,
                            settings.normalize_features,
                        )
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


def rank_variables_by_features(
    v_feats: torch.Tensor,
    cols: Sequence[int] = (0, 1, 2, 3, 4, 5),
    descending: bool = True,
) -> torch.Tensor:
    """
    Stable lexicographic ranks from multiple feature columns.
    Returns ranks[j] ∈ {0..n-1} (0 = top).
    """
    n = v_feats.size(0)
    order = torch.arange(n, device=v_feats.device)
    for col in reversed(cols):  # least -> most important
        vals = v_feats[:, col]
        idx = torch.argsort(vals[order], descending=descending, stable=True)
        order = order[idx]
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(n, device=v_feats.device)
    return ranks


def encode_ranks_as_bits(
    v_feats: torch.Tensor, ranks: torch.Tensor, feat_w: int = 20
) -> torch.Tensor:
    """
    Append 'feat_w' binary-fraction bits that encode the quantile of each rank.
    """
    n = v_feats.size(0)
    dev, dtype = v_feats.device, v_feats.dtype
    denom = float(max(n, 1))  # -> [0,1)
    q = ranks.to(torch.float32) / denom

    pos_bits = torch.zeros((n, feat_w), dtype=dtype, device=dev)
    for i in range(n):
        frac = float(q[i].item())
        div = 0.5
        for k in range(feat_w):
            if frac >= div:
                pos_bits[i, k] = 1.0
                frac -= div
            div *= 0.5
    return torch.cat([v_feats, pos_bits], dim=1)


def get_bipartite_graph(
    lp_path: Path,
    add_pos_feat: bool = True,
    normalize_pos_feat: bool = False,
    normalize_features: bool = True,
) -> Tuple:
    # load model
    problem = scp.Model()
    problem.hideOutput(True)
    problem.readProblem(str(lp_path))

    # variables
    scip_variables = problem.getVars()
    scip_variables.sort(key=lambda v: v.name)
    n_variables = len(scip_variables)
    v_map: Dict[str, int] = {v.name: i for i, v in enumerate(scip_variables)}

    # base variable features + 12-bit positional embedding
    N_FEATURES = 6
    v_feats = np.zeros((n_variables, N_FEATURES), dtype=np.float32)
    # init sentinels for min/max
    v_feats[:, VariableFeature.MAX_COEF] = -np.inf  # max_coef
    v_feats[:, VariableFeature.MIN_COEF] = np.inf  # min_coef
    b_idx: List[int] = []
    # binary flags
    for constraint_idx, var in enumerate(scip_variables):
        vtype = getattr(var, "vtype", None)
        try:
            vtype = vtype() if callable(vtype) else var.getVType()
        except Exception:
            pass
        s = str(vtype).upper()
        if s.startswith("B") or s.startswith("BIN"):
            v_feats[constraint_idx, VariableFeature.IS_INTEGER] = 1.0
            b_idx.append(constraint_idx)

    c_vec = np.zeros(n_variables, dtype=np.float32)
    obj = problem.getObjective()
    for term in obj:
        var = term.vartuple[0]
        variable_idx = v_map[var.name]
        coef = float(obj[term])
        c_vec[variable_idx] += coef
        v_feats[variable_idx, VariableFeature.OBJ_COEF] = coef

    # original constraints
    conss_all = [c for c in problem.getConss() if len(problem.getValsLinear(c)) > 0]
    # stable order: by nnz then name
    conss_all.sort(key=lambda c: (len(problem.getValsLinear(c)), str(c)))
    m_orig = len(conss_all)

    # edge_index/attr and constraint features
    gi_rows: List[int] = []
    gi_cols: List[int] = []
    edge_attr_vals: List[float] = []
    c_rows_feats: List[List[float]] = []  # [avg_row_coef, degree, rhs, sense_code]
    graph_rhs: List[float] = []
    graph_sense: List[int] = []

    for constraint_idx, constraint in enumerate(conss_all):
        coeffs = problem.getValsLinear(constraint)  # dict[var] -> coef
        sense_code, rhs_or_lhs = _constraint_sense_code(problem, constraint)
        deg = 0
        ssum = 0.0
        for var_obj, coef in coeffs.items():
            coef = float(coef)
            if coef == 0.0:
                continue
            variable_idx = v_map[var_obj]
            gi_rows.append(constraint_idx)
            gi_cols.append(variable_idx)
            edge_attr_vals.append(coef)  # TRUE coefficient on the edge
            deg += 1
            ssum += coef
            # update variable-side statistics
            v_feats[variable_idx, VariableFeature.DEGREE] += 1.0  # degree
            v_feats[variable_idx, VariableFeature.AVG_COEF] += (
                coef  # sum of coefs (avg later)
            )
            v_feats[variable_idx, VariableFeature.MAX_COEF] = max(
                v_feats[variable_idx, VariableFeature.MAX_COEF], coef
            )
            v_feats[variable_idx, VariableFeature.MIN_COEF] = min(
                v_feats[variable_idx, VariableFeature.MIN_COEF], coef
            )
        avg_row = ssum / max(deg, 1)
        c_rows_feats.append([avg_row, float(deg), float(rhs_or_lhs), float(sense_code)])
        graph_rhs.append(float(rhs_or_lhs))
        graph_sense.append(int(sense_code))

    # Build paper-view tensors
    if gi_rows:
        edge_index = torch.tensor([gi_rows, gi_cols], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float32).unsqueeze(1)
        # binary incidence adjacency
        incidence = torch.sparse_coo_tensor(
            indices=edge_index,
            values=torch.ones(len(edge_attr_vals), dtype=torch.float32),
            size=(m_orig, n_variables),
        ).coalesce()
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1, dtype=torch.float32)
        incidence = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=(m_orig, n_variables),
        ).coalesce()

    c_nodes = (
        torch.tensor(c_rows_feats, dtype=torch.float32)
        if c_rows_feats
        else torch.zeros((0, 4), dtype=torch.float32)
    )
    graph_b = (
        torch.tensor(graph_rhs, dtype=torch.float32)
        if graph_rhs
        else torch.zeros((0,), dtype=torch.float32)
    )
    graph_s = (
        torch.tensor(graph_sense, dtype=torch.int32)
        if graph_sense
        else torch.zeros((0,), dtype=torch.int32)
    )

    # finalize variable avg coef = sum/degree
    v_nodes = torch.tensor(v_feats, dtype=torch.float32)
    deg = v_nodes[:, VariableFeature.DEGREE].clamp(min=1.0)
    v_nodes[:, 1] = v_nodes[:, 1] / deg
    # clean infs for vars with zero degree
    v_nodes[:, VariableFeature.MAX_COEF] = torch.where(
        torch.isfinite(v_nodes[:, VariableFeature.MAX_COEF]),
        v_nodes[:, VariableFeature.MAX_COEF],
        torch.zeros_like(v_nodes[:, VariableFeature.MAX_COEF]),
    )
    v_nodes[:, VariableFeature.MIN_COEF] = torch.where(
        torch.isfinite(v_nodes[:, VariableFeature.MIN_COEF]),
        v_nodes[:, VariableFeature.MIN_COEF],
        torch.zeros_like(v_nodes[:, VariableFeature.MIN_COEF]),
    )

    if add_pos_feat:
        ranks = rank_variables_by_features(
            v_nodes,
            cols=(
                VariableFeature.OBJ_COEF,
                VariableFeature.AVG_COEF,
                VariableFeature.DEGREE,
                VariableFeature.MAX_COEF,
                VariableFeature.MIN_COEF,
                VariableFeature.IS_INTEGER,
            ),
            descending=True,
        )
        pos_bits = encode_ranks_as_bits(v_nodes, ranks, feat_w=12)[:, -12:]
        v_nodes = torch.cat([v_nodes, pos_bits], dim=1)

    # Add original coefficients needed for KKT loss calculation (<= rows incl. senses & bounds)
    ind_rows: List[int] = []
    ind_cols: List[int] = []
    vals: List[float] = []
    b_leq: List[float] = []

    def _add_leq_row(coeffs_dict, rhs_val: float):
        row = len(b_leq)
        for var_obj, coef in coeffs_dict.items():
            j = v_map[var_obj]
            cval = float(coef)
            if cval != 0.0:
                ind_rows.append(row)
                ind_cols.append(j)
                vals.append(cval)
        b_leq.append(float(rhs_val))

    # fold each original row into <= form
    for constraint in conss_all:
        coeffs = problem.getValsLinear(constraint)  # dict[var]->coef
        rhs = float(problem.getRhs(constraint))
        lhs = float(problem.getLhs(constraint))
        if abs(rhs - lhs) < 1e-12:
            # equality -> two rows
            _add_leq_row(coeffs, rhs)
            _add_leq_row({v: -coef for v, coef in coeffs.items()}, -lhs)
        elif rhs < 1e20:
            # <= rhs
            _add_leq_row(coeffs, rhs)
        else:
            # >= lhs -> -expr <= -lhs
            _add_leq_row({v: -coef for v, coef in coeffs.items()}, -lhs)

    # add finite variable bounds as <= rows
    INF = 1e40
    for var in scip_variables:
        variable_idx = v_map[var.name]
        try:
            lb = float(var.getLb())
            ub = float(var.getUb())
        except Exception:
            lb, ub = -INF, INF
        if ub < INF:
            _add_leq_row({var: +1.0}, ub)  # x_j <= ub
        if lb > -INF:
            _add_leq_row({var: -1.0}, -lb)  # -x_j <= -lb

    A = torch.sparse_coo_tensor(
        indices=torch.tensor([ind_rows, ind_cols], dtype=torch.long),
        values=torch.tensor(vals, dtype=torch.float32),
        size=(len(b_leq), n_variables),
    ).coalesce()
    b_vec = torch.tensor(b_leq, dtype=torch.float32)

    if normalize_features:
        # Apply min-max normalization to features
        if add_pos_feat and not normalize_pos_feat:
            # normalize only first 6 numeric features, keep 20 bits crisp {0,1}
            v_num, v_bits = v_nodes[:, :6], v_nodes[:, 6:]
            v_num = _minmax_normalization(v_num).clamp_(1e-5, 1.0)
            v_nodes = torch.cat([v_num, v_bits], dim=1)
        else:
            # normalize everything (numeric + bits, or numeric only if no pos bits)
            v_nodes = _minmax_normalization(v_nodes).clamp_(1e-5, 1.0)

        c_nodes = (
            _minmax_normalization(c_nodes).clamp_(1e-5, 1.0)
            if c_nodes.numel() > 0
            else c_nodes
        )
    else:
        # Normalization disabled - using raw features
        logger.info("Normalization disabled - using raw features")

    # attach artifacts to A
    # (So they can be accessed later without changing the return signature.)
    A.edge_index = edge_index  # [2, nnz] over ORIGINAL constraints
    A.edge_attr = edge_attr  # [nnz, 1] true A_ij
    A.incidence = incidence  # (m_orig, n) binary adjacency
    A.graph_b = graph_b  # [m_orig] rhs
    A.graph_sense = graph_s  # [m_orig] sense codes

    # tensors to return
    b_vars = torch.tensor(b_idx, dtype=torch.int32)

    return (
        A,
        v_map,  # name -> column index
        v_nodes,  # variable node features [n, 6+12] in [0,1]
        c_nodes,  # constraint node features [m_orig, 4] in [0,1]
        b_vars,  # indices of binary variables
        b_vec,  # RHS for <= rows (KKT)
        c_vec,  # objective for MIN
    )
