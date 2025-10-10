import hashlib
import pickle
from typing import Optional, Tuple

import numpy as np
import torch

from instances.utils import sol_path_from_bg


def get_prediction_signature(v):
    return hashlib.sha1(np.round(v, 3).tobytes()).hexdigest()


def get_dual_feasibility_violation(
    A_i: torch.Tensor, lambda_i: torch.Tensor, c_i: torch.Tensor
) -> (float, float):
    lam_plus = torch.relu(lambda_i)
    if A_i.is_sparse:
        v = torch.sparse.mm(A_i.transpose(0, 1), lam_plus.unsqueeze(1)).squeeze(1) - c_i
    else:
        v = A_i.transpose(0, 1).matmul(lam_plus) - c_i
    v_pos = v.clamp_min(0.0)
    return float(v_pos.pow(2).mean().sqrt().item()), float(v_pos.abs().max().item())


def get_optimality_gap(
    x_i: torch.Tensor, lambda_i: torch.Tensor, c_i: torch.Tensor, b_i: torch.Tensor
) -> float:
    primal = (x_i @ c_i).item()
    dual = (lambda_i @ b_i).item()
    # optimality gap (primal vs dual)
    optimality_gap = (2.0 * abs(primal - dual)) / (abs(primal) + abs(dual) + 1e-6)
    return optimality_gap


def get_objective_gap(
    predicted_obj: float,
    optimal_obj: float,
) -> float:
    """
    Compute relative objective gap for predicted objective vs known optimal objective.
    """
    objective_gap = (predicted_obj - optimal_obj) / (abs(optimal_obj) + 1e-6)
    return objective_gap


def get_optimal_solution(
    bg_path: str,
    x_i: torch.Tensor,
    c_i: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    optimal_pool = load_optimal_solutions(bg_path=bg_path)

    optimal_objectives = (optimal_pool.float() @ c_i).numpy()

    predicted_objective = (c_i @ x_i).item()

    objective_gaps = np.array(
        [
            get_objective_gap(predicted_objective, optimal_objective)
            for optimal_objective in optimal_objectives
        ],
        dtype=np.float64,
    )

    smallest_gap = float(objective_gaps.min())

    tolerance = max(1e-12, 1e-9 * (1.0 + abs(smallest_gap)))
    tie_idx = np.flatnonzero(np.abs(objective_gaps - smallest_gap) <= tolerance)

    if tie_idx.size == 1:
        chosen_idx: int = int(tie_idx[0])
    else:
        # Tie-break by smallest L2 distance to x_pred
        opt_torch = optimal_pool[tie_idx].float()
        l2_distances = ((opt_torch - x_i) ** 2).mean(dim=1)
        chosen_idx = int(tie_idx[int(torch.argmin(l2_distances).item())])

    chosen_solution = optimal_pool[chosen_idx].float()
    objective_gap = objective_gaps[chosen_idx]

    return chosen_solution, objective_gap


def load_optimal_solutions(bg_path: str) -> torch.Tensor:
    """
    Load the solution pool
    """
    sol_path = sol_path_from_bg(bg_path)
    if (not sol_path) or (not sol_path.exists()):
        print(f"[warn] No .sol file found for {bg_path} -> {sol_path}")
        return None

    try:
        with open(sol_path, "rb") as f:
            sol_data = pickle.load(f)
        var_names = sol_data["var_names"]
        sols = np.asarray(sol_data["sols"])
    except Exception as e:
        print(f"[warn] Failed to read solution pool at {sol_path}: {e}")
        return None

    meta = load_bg_meta(bg_path)
    if meta is None:
        print(f"[warn] BG meta not found for {bg_path}")
        return None
    v_map, _b_vars = meta

    sols_model = reorder_solutions_to_model(v_map, var_names, sols)
    if sols_model is None:
        print("[warn] Could not reorder solution pool to model order.")
        return None

    min_obj_value = sol_data["objs"].min()
    best_idx = np.flatnonzero(sol_data["objs"] == min_obj_value)
    best_sols = sols_model[best_idx]
    return torch.from_numpy(best_sols)


def load_bg_meta(bg_path: str) -> Optional[Tuple[dict, np.ndarray]]:
    """
    Returns (v_map: name->idx, b_vars: np.int64[nb]) or None if failed.
    """
    try:
        with open(bg_path, "rb") as f:
            bg = pickle.load(f)
            A, v_map, v_nodes, c_nodes, b_vars, b_vec, c_vec = bg

        # v_map is name->index view saved by your generator
        b_vars = np.asarray(b_vars, dtype=np.int64)
        return v_map, b_vars
    except Exception:
        return None


def reorder_solutions_to_model(
    v_map: dict, var_names: list[str], sols: np.ndarray
) -> Optional[np.ndarray]:
    """
    Reorder solution pool columns to match model (BG) order using v_map.
    """
    try:
        perm = np.asarray([v_map[name] for name in var_names], dtype=np.int64)
    except KeyError:
        return None
    out = np.empty_like(sols)
    out[:, perm] = sols
    return out
