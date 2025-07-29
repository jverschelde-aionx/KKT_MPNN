import multiprocessing as mp
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import gurobipy as gp
import numpy as np
import pyscipopt as scp
import torch
import tqdm
from ecole.instance import CombinatorialAuctionGenerator, IndependentSetGenerator
from loguru import logger

from instances.common import COMBINATORIAL_AUCTION, INDEPENDANT_SET, Settings
from instances.utils import ensure_dirs


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


# Instance generation
def generate_instances(settings: Settings) -> None:
    lp_paths: List[Path] = []

    for problem in settings.problems:
        logger.info("Generating instances for problem type: {}", problem)
        test_instances = round(settings.n_instances * settings.test_split)
        val_instances = round(settings.n_instances * settings.val_split)
        train_instances = round(settings.n_instances - test_instances - val_instances)

        sets = (
            ("train", train_instances),
            ("val", val_instances),
            ("test", test_instances),
        )

        for split, n_instances in sets:
            if problem == INDEPENDANT_SET:
                generate_IS_instances(settings, split, n_instances, lp_paths)
            elif problem == COMBINATORIAL_AUCTION:
                generate_CA_instances(settings, split, n_instances, lp_paths)
            else:
                raise ValueError(f"Unknown problem type: {problem}")

        logger.info("Total .lp files to process: {}", len(lp_paths))
        for lp_path in tqdm.tqdm(lp_paths, desc="Processing .lp files: "):
            bg_dir = (
                settings.data_root / problem / "BG" / split / f"{lp_path.parent.name}"
            )
            sol_dir = (
                settings.data_root
                / problem
                / "solution"
                / split
                / f"{lp_path.parent.name}"
            )
            log_dir = (
                settings.data_root / problem / "logs" / split / f"{lp_path.parent.name}"
            )
            for d in (bg_dir, sol_dir, log_dir):
                d.mkdir(parents=True, exist_ok=True)

            solution_path = sol_dir / (lp_path.name + ".sol")
            # Solve with Gurobi
            if settings.solve and not solution_path.exists():
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

            bg_path = bg_dir / (lp_path.name + ".bg")

            if not bg_path.exists():
                # Extract bipartite graph
                graph_data = get_biparite_graph(str(lp_path))
                with open(bg_path, "wb") as f:
                    print(bg_path)
                    pickle.dump(graph_data, f)
        logger.success(
            "All graphs for problem type {} written to {}",
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
        divider = 1.0
        for k in range(feat_w):
            if divider <= flt:
                pos_feat[row, k] = 1
                flt -= divider
            divider /= 2
    return torch.cat([variable_features, pos_feat], dim=1)


def get_biparite_graph(lp_path: str):
    """
    Convert a .lp file into the 5‑tuple required by GraphDataset:
       A (sparse [m+1, n]), v_map, v_nodes, c_nodes, b_vars
    Reproduces the logic in your helper.py but without GPU side‑effects.
    """
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(lp_path)

    ncons = m.getNConss()
    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)
    nvars = len(mvars)

    # Node feature placeholders
    ORI_FEATS = 6  # obj coef, avg coef, degree, max, min, binary‑flag
    v_nodes = [[0.0] * ORI_FEATS for _ in range(nvars)]
    b_vars = []

    v_map: Dict[str, int] = {v.name: i for i, v in enumerate(mvars)}
    for i, v in enumerate(mvars):
        if v.vtype() == "BINARY":
            v_nodes[i][-1] = 1
            b_vars.append(i)

    # Objective
    obj = m.getObjective()
    obj_node = [0.0, 0.0, 0.0, 0.0]
    indices = [[], []]
    values = []
    for term in obj:
        v_idx = v_map[term.vartuple[0].name]
        coef = obj[term]
        v_nodes[v_idx][0] = coef
        indices[0].append(0)  # dummy constraint row for objective
        indices[1].append(v_idx)
        values.append(1.0)  # mark presence not magnitude
        obj_node[0] += coef
        obj_node[1] += 1
    obj_node[0] /= max(obj_node[1], 1)

    # Constraints
    cons = [c for c in m.getConss() if len(m.getValsLinear(c)) > 0]
    ncons = len(cons)
    c_nodes = []

    for c_idx, c in enumerate(
        sorted(cons, key=lambda x: (len(m.getValsLinear(x)), str(x)))
    ):
        coeffs = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        if rhs == lhs:
            sense = 2  # equality
        elif rhs >= 1e20:
            sense = 1  # >=
            rhs = lhs
        else:
            sense = 0  # <=

        nz_sum = 0.0
        for var, coef in coeffs.items():
            v_idx = v_map[var]
            if coef != 0:
                indices[0].append(c_idx)
                indices[1].append(v_idx)
                values.append(1.0)
            v_nodes[v_idx][2] += 1
            v_nodes[v_idx][1] += coef / ncons
            v_nodes[v_idx][3] = max(v_nodes[v_idx][3], coef)
            v_nodes[v_idx][4] = min(v_nodes[v_idx][4], coef)
            nz_sum += coef

        deg = max(len(coeffs), 1)
        c_nodes.append([nz_sum / deg, deg, rhs, sense])

    c_nodes.append(obj_node)

    # Normalise node features
    v_nodes = torch.tensor(v_nodes, dtype=torch.float32)
    c_nodes = torch.tensor(c_nodes, dtype=torch.float32)

    v_nodes = (v_nodes - v_nodes.min(dim=0).values) / (
        v_nodes.max(dim=0).values - v_nodes.min(dim=0).values + 1e-9
    )
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)
    v_nodes = position_get_ordered_flt(v_nodes)

    c_nodes = (c_nodes - c_nodes.min(dim=0).values) / (
        c_nodes.max(dim=0).values - c_nodes.min(dim=0).values + 1e-9
    )
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    A = torch.sparse_coo_tensor(indices, values, (ncons + 1, nvars)).coalesce()

    return A, v_map, v_nodes, c_nodes, torch.tensor(b_vars, dtype=torch.int32)
