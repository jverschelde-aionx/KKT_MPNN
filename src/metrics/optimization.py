import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from data.utils import infer_sol_path


def get_primal_feasibility(
    x_pred: torch.Tensor, A: torch.Tensor, b: torch.Tensor, mask_m: torch.Tensor
) -> torch.Tensor:
    # Primal feasibility: relu(Ax - b)
    x_col = x_pred.unsqueeze(-1)  # [B, n, 1]
    Ax = torch.bmm(A, x_col).squeeze(-1)  # [B, m]
    primal_res = torch.relu(Ax - b)  # [B, m]
    # Mask constraints
    primal_res = primal_res * mask_m
    primal_feasibility = (primal_res**2).sum(dim=1) / (
        mask_m.sum(dim=1).clamp_min(1.0)
    )  # mean over real m
    return primal_feasibility


def get_dual_feasibility(
    lambda_pred: torch.Tensor, mask_m: torch.Tensor
) -> torch.Tensor:
    dual_feasibility = torch.relu(-lambda_pred)
    dual_feasibility = ((dual_feasibility**2) * mask_m).sum(dim=1) / mask_m.sum(
        dim=1
    ).clamp_min(1.0)
    return dual_feasibility


def get_complementary_slackness(
    x_pred: torch.Tensor,
    lambda_pred: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    mask_m: torch.Tensor,
):
    x_col = x_pred.unsqueeze(-1)  # [B, n, 1]
    Ax = torch.bmm(A, x_col).squeeze(-1)  # [B, m]
    complementary_slackness = lambda_pred * (Ax - b)  # [B, m]
    complementary_slackness = ((complementary_slackness**2) * mask_m).sum(
        dim=1
    ) / mask_m.sum(dim=1).clamp_min(1.0)
    return complementary_slackness


def get_stationarity(
    lambda_pred: torch.Tensor,
    A: torch.Tensor,
    c: torch.Tensor,
    mask_n: torch.Tensor,
):
    # Stationarity: c + A^T * lambda = 0
    lam_col = lambda_pred.unsqueeze(-1)  # [B, m, 1]
    ATlam = torch.bmm(A.transpose(1, 2), lam_col).squeeze(-1)  # [B, n]
    stationarity = c + ATlam  # [B, n]
    stationarity = ((stationarity**2) * mask_n).sum(dim=1) / mask_n.sum(
        dim=1
    ).clamp_min(1.0)
    return stationarity


def kkt(
    y_pred: torch.Tensor,  # [B, n+m]
    A: torch.Tensor,  # [B, m, n]
    b: torch.Tensor,  # [B, m]
    c: torch.Tensor,  # [B, n]
    mask_m: torch.Tensor,
    mask_n: torch.Tensor,
    primal_weight: float,
    dual_weight: float,
    stationarity_weight: float,
    complementary_slackness_weight: float,
):
    """
    Computes weighted KKT residuals. Supports padded batches via masks.
    """
    B, m, n = A.shape

    x_pred = y_pred[:, :n]  # [B, n]
    lambda_pred = y_pred[:, n:]  # [B, m]

    primal_feasibility = get_primal_feasibility(x_pred, A, b, mask_m)
    dual_feasibility = get_dual_feasibility(lambda_pred, mask_m)
    stationarity = get_stationarity(lambda_pred, A, c, mask_n)
    complementary_slackness = get_complementary_slackness(
        x_pred, lambda_pred, A, b, mask_m
    )

    weighted_primal = primal_weight * primal_feasibility
    weighted_dual = dual_weight * dual_feasibility
    weighted_stat = stationarity_weight * stationarity
    weighted_comp = complementary_slackness_weight * complementary_slackness

    loss = (weighted_primal + weighted_dual + weighted_stat + weighted_comp).mean()

    return loss, {
        "kkt_loss": loss.item(),
        "primal_feasibility": weighted_primal.mean(),
        "dual_feasibility": weighted_dual.mean(),
        "stationarity": weighted_stat.mean(),
        "complementary_slackness": weighted_comp.mean(),
    }


def get_dual_feasibility_violation(
    A_i: torch.Tensor, lambda_i: torch.Tensor, c_i: torch.Tensor
) -> Tuple[float, float]:
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
    dual = -(lambda_i @ b_i).item()
    return (2.0 * abs(primal - dual)) / (abs(primal) + abs(dual) + 1e-9)


def get_objective_gap(
    predicted_obj: float,
    optimal_obj: float,
) -> float:
    """Relative suboptimality (minimization). Non-negative by definition."""
    denominator = max(abs(optimal_obj), 1.0)
    return max(0.0, (predicted_obj - optimal_obj) / (denominator + 1e-9))


def get_optimal_solution(
    input_path: str,
    x_i: torch.Tensor,
    c_i: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Move inputs to CPU for computation with loaded solutions
    x_i_cpu = x_i.detach().cpu()
    c_i_cpu = c_i.detach().cpu()

    optimal_pool = load_optimal_solutions(instance_path=input_path)

    optimal_objectives = (optimal_pool.float() @ c_i_cpu).numpy()

    predicted_objective = (c_i_cpu @ x_i_cpu).item()

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
        l2_distances = ((opt_torch - x_i_cpu) ** 2).mean(dim=1)
        chosen_idx = int(tie_idx[int(torch.argmin(l2_distances).item())])

    chosen_solution = optimal_pool[chosen_idx].float()
    objective_gap = objective_gaps[chosen_idx]

    return chosen_solution, objective_gap


def load_optimal_solutions(instance_path: str) -> torch.Tensor:
    """
    Load the solution pool
    """
    sol_path = infer_sol_path(instance_path)
    if (not sol_path) or (not sol_path.exists()):
        print(f"[warn] No .sol file found for {instance_path} -> {sol_path}")
        return None

    try:
        with open(sol_path, "rb") as f:
            sol_data = pickle.load(f)
        var_names = sol_data["var_names"]
        sols = np.asarray(sol_data["sols"])
    except Exception as e:
        print(f"[warn] Failed to read solution pool at {sol_path}: {e}")
        return None

    if Path(instance_path).suffix == ".bg":
        meta = _load_bg_meta(instance_path)
        if meta is None:
            print(f"[warn] BG meta not found for {instance_path}")
            return None
        v_map, _b_vars = meta

        sols = _reorder_solutions_to_model(v_map, var_names, sols)
        if sols is None:
            print("[warn] Could not reorder solution pool to model order.")
            return None

    min_obj_value = sol_data["objs"].min()
    best_idx = np.flatnonzero(sol_data["objs"] == min_obj_value)
    best_sols = sols[best_idx]
    return torch.from_numpy(best_sols)


def _load_bg_meta(bg_path: str) -> Optional[Tuple[dict, np.ndarray]]:
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


def _reorder_solutions_to_model(
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
