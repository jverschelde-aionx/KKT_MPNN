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


# ---------------------------------------------------------------------------
# Row-type codes for standardized <= rows (used in constraint node features)
# ---------------------------------------------------------------------------
ROW_TYPE_LEQ = 0  # original <= row
ROW_TYPE_GEQ_FLIP = 1  # original >= row, sign-flipped to <=
ROW_TYPE_EQ_POS = 2  # positive half of split equality
ROW_TYPE_EQ_NEG = 3  # negative half of split equality
ROW_TYPE_UB = 4  # variable upper bound: x_j <= ub
ROW_TYPE_LB = 5  # variable lower bound: -x_j <= -lb
N_ROW_TYPES = 6


@dataclass
class StandardizedRow:
    """One row of the standardized all-<= system used for the KKT loss."""

    coeffs: Dict  # {scip_var_obj: float} — signed coefficients
    rhs: float
    row_type: int  # one of ROW_TYPE_*


def _standardize_rows(
    problem,
    conss_all: list,
    scip_variables: list,
    v_map: Dict[str, int],
) -> List[StandardizedRow]:
    """Convert all original constraints + variable bounds into <= rows.

    Returns one StandardizedRow per KKT row. The graph, edge_index/edge_attr,
    c_nodes, A, and b_vec should all be built from this list so they are
    guaranteed to be consistent.
    """
    rows: List[StandardizedRow] = []

    # 1) Original linear constraints
    for constraint in conss_all:
        coeffs = problem.getValsLinear(constraint)  # dict[var]->coef
        rhs = float(problem.getRhs(constraint))
        lhs = float(problem.getLhs(constraint))

        if abs(rhs - lhs) < 1e-12:
            # equality -> two <= rows
            rows.append(StandardizedRow(
                coeffs=dict(coeffs), rhs=rhs, row_type=ROW_TYPE_EQ_POS,
            ))
            rows.append(StandardizedRow(
                coeffs={v: -c for v, c in coeffs.items()},
                rhs=-lhs,
                row_type=ROW_TYPE_EQ_NEG,
            ))
        elif rhs < 1e20:
            # already <=
            rows.append(StandardizedRow(
                coeffs=dict(coeffs), rhs=rhs, row_type=ROW_TYPE_LEQ,
            ))
        else:
            # >= lhs -> flip to <=
            rows.append(StandardizedRow(
                coeffs={v: -c for v, c in coeffs.items()},
                rhs=-lhs,
                row_type=ROW_TYPE_GEQ_FLIP,
            ))

    # 2) Finite variable bounds (use var.name as key since SCIP Variable is unhashable)
    for var in scip_variables:
        ub = float(var.getUbOriginal())
        lb = float(var.getLbOriginal())
        if ub < 1e20:
            rows.append(StandardizedRow(
                coeffs={var.name: 1.0}, rhs=ub, row_type=ROW_TYPE_UB,
            ))
        if lb > -1e20:
            rows.append(StandardizedRow(
                coeffs={var.name: -1.0}, rhs=-lb, row_type=ROW_TYPE_LB,
            ))

    return rows


def relax_problem(lp_path: Path) -> None:
    """
    Relax the on-disk model to a linear program.

    Reads the original model (.lp may contain MIP/SOS/indicator) and builds
    the LP relaxation with m.relax(), which removes integrality, SOS, and
    indicator constraints.  Overwrites the file in place.
    """
    m = gp.read(str(lp_path))
    mr = None
    try:
        mr = m.relax()
        mr.update()
        mr.write(str(lp_path))
        logger.info(f"[relaxed->LP] Rewrote {lp_path.name}")
    finally:
        try:
            if mr is not None:
                mr.dispose()
        except Exception:
            pass
        m.dispose()


def convert_to_minimize_problem(lp_path: Path) -> None:
    """
    Convert the on-disk model to a minimization problem.

    If the model is a maximization problem, negates every objective
    coefficient and switches the sense to minimization.  Overwrites the
    file in place.
    """
    m = gp.read(str(lp_path))
    try:
        if m.ModelSense == -1:  # -1 = maximize
            for v in m.getVars():
                v.Obj = -v.Obj
            m.ModelSense = 1  # +1 = minimize
            m.update()
            m.write(str(lp_path))
            logger.info(f"[->MIN] Rewrote {lp_path.name}")
    finally:
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
            if settings.relax:
                relax_problem(lp_path)
            convert_to_minimize_problem(lp_path)
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
            if settings.relax:
                relax_problem(lp_path)
            convert_to_minimize_problem(lp_path)
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
            if settings.relax:
                relax_problem(lp_path)
            convert_to_minimize_problem(lp_path)
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
            if settings.relax:
                relax_problem(lp_path)
            convert_to_minimize_problem(lp_path)
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

                    if settings.relax:
                        relax_problem(lp_path)
                    convert_to_minimize_problem(lp_path)
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

    # Build the unified standardized-row system.
    # Graph edges, constraint node features, A, and b_vec are ALL derived
    # from these rows so they are guaranteed consistent.
    std_rows = _standardize_rows(problem, conss_all, scip_variables, v_map)
    m_std = len(std_rows)

    gi_rows: List[int] = []
    gi_cols: List[int] = []
    edge_attr_vals: List[float] = []
    c_rows_feats: List[List[float]] = []  # [avg_coef, degree, rhs, row_type_0..5]
    b_leq: List[float] = []

    for row_idx, srow in enumerate(std_rows):
        deg = 0
        ssum = 0.0
        for var_obj, coef in srow.coeffs.items():
            coef = float(coef)
            if coef == 0.0:
                continue
            variable_idx = v_map[var_obj.name if hasattr(var_obj, "name") else str(var_obj)]
            gi_rows.append(row_idx)
            gi_cols.append(variable_idx)
            edge_attr_vals.append(coef)
            deg += 1
            ssum += coef
            # update variable-side statistics
            v_feats[variable_idx, VariableFeature.DEGREE] += 1.0
            v_feats[variable_idx, VariableFeature.AVG_COEF] += coef
            v_feats[variable_idx, VariableFeature.MAX_COEF] = max(
                v_feats[variable_idx, VariableFeature.MAX_COEF], coef
            )
            v_feats[variable_idx, VariableFeature.MIN_COEF] = min(
                v_feats[variable_idx, VariableFeature.MIN_COEF], coef
            )

        avg_row = ssum / max(deg, 1)
        # Constraint node features: [avg_coef, degree, rhs, one-hot row type (6)]
        one_hot = [0.0] * N_ROW_TYPES
        one_hot[srow.row_type] = 1.0
        c_rows_feats.append([avg_row, float(deg), srow.rhs] + one_hot)
        b_leq.append(srow.rhs)

    # Build tensors — graph edges and KKT A are identical by construction
    edge_index = torch.tensor([gi_rows, gi_cols], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float32).unsqueeze(1)

    c_nodes = (
        torch.tensor(c_rows_feats, dtype=torch.float32)
        if c_rows_feats
        else torch.zeros((0, 3 + N_ROW_TYPES), dtype=torch.float32)
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

    # KKT A and b_vec — built from the SAME edge data (guaranteed consistent)
    A = torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.tensor(edge_attr_vals, dtype=torch.float32),
        size=(len(b_leq), n_variables),
    ).coalesce()
    b_vec = torch.tensor(b_leq, dtype=torch.float32)

    # Hard alignment assertions
    assert c_nodes.size(0) == b_vec.numel(), (
        f"c_nodes ({c_nodes.size(0)}) != b_vec ({b_vec.numel()})"
    )
    assert A.shape[0] == b_vec.numel(), (
        f"A rows ({A.shape[0]}) != b_vec ({b_vec.numel()})"
    )

    if normalize_features:
        # Apply min-max normalization to features
        if add_pos_feat and not normalize_pos_feat:
            # normalize only first 6 numeric features, keep 12 bits crisp {0,1}
            v_num, v_bits = v_nodes[:, :6], v_nodes[:, 6:]
            v_num = _minmax_normalization(v_num).clamp_(1e-5, 1.0)
            v_nodes = torch.cat([v_num, v_bits], dim=1)
        else:
            v_nodes = _minmax_normalization(v_nodes).clamp_(1e-5, 1.0)

        # Normalize only the 3 scalar features, keep one-hot row-type crisp
        if c_nodes.numel() > 0:
            c_scalar, c_onehot = c_nodes[:, :3], c_nodes[:, 3:]
            c_scalar = _minmax_normalization(c_scalar).clamp_(1e-5, 1.0)
            c_nodes = torch.cat([c_scalar, c_onehot], dim=1)
    else:
        logger.info("Normalization disabled - using raw features")

    # Attach graph-edge artifacts to A for backward compatibility.
    # edge_index and edge_attr now correspond to the standardized rows
    # (same as A), so graph edges == KKT matrix entries.
    A.edge_index = edge_index
    A.edge_attr = edge_attr

    b_vars = torch.tensor(b_idx, dtype=torch.int32)

    return (
        A,
        v_map,  # name -> column index
        v_nodes,  # variable node features [n, 6+12] in [0,1]
        c_nodes,  # constraint node features [m_std, 3+6] in [0,1]
        b_vars,  # indices of binary variables
        b_vec,  # RHS for standardized <= rows
        c_vec,  # objective for MIN
    )
