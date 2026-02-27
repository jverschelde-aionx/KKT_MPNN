import pickle
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

import numpy as np
import torch

from data.utils import sol_path_from_bg
from models.policy_encoder import BipartiteNodeData


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


class MetricsLogger(Protocol):
    """Minimal protocol for loggers like wandb."""

    def log(self, data: Mapping[str, float], step: Optional[int] = None) -> None: ...


class NodesTap:
    """
    Registers a forward-pre hook on a module and captures the first input tensor
    of the forward call (detached, float32, moved to CPU).

    Use as a context manager to ensure the hook is removed.
    """

    def __init__(self, prefix, module: torch.nn.Module) -> None:
        if not isinstance(module, torch.nn.Module):
            raise TypeError("FeatureTap requires a torch.nn.Module.")
        self.module: torch.nn.Module = module
        self.forward_hook = module.register_forward_hook(self._forward_hook_handler)
        self.last_input: Optional[torch.Tensor] = None
        self.last_output: Optional[torch.Tensor] = None
        self.prefix = prefix

    def _to_tensor(self, x: Any) -> torch.Tensor:
        """Conversion of the input/output into a Tensor."""
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, BipartiteNodeData):
            return x.to_node_tensor()
        elif self.prefix.lower() == "xhat/":
            return x[0]

        raise TypeError(
            "Expected object to be a torch.Tensor or an object "
            "exposing .to_node_tensor()."
        )

    def _forward_hook_handler(
        self, module: torch.nn.Module, inputs: Tuple, output: Any
    ) -> None:
        if not inputs:
            raise ValueError("Forward hook received an empty inputs tuple.")

        x = inputs[0]

        self.last_input = (
            self._to_tensor(x).detach().to("cpu", non_blocking=False).float()
        )
        self.last_output = (
            self._to_tensor(output).detach().to("cpu", non_blocking=False).float()
        )

    def close(self) -> None:
        self.forward_hook.remove()


def compute_node_stats(
    nodes: torch.Tensor,
    prefix: str,
    round_decimals: int = 3,
    sample_size: int = 256,
) -> Dict[str, float]:
    if not isinstance(nodes, torch.Tensor):
        raise TypeError("nodes must be a torch.Tensor.")

    if nodes.ndim == 2:
        n_nodes, embedding_size = int(nodes.shape[0]), int(nodes.shape[1])
    else:
        n_nodes, embedding_size = int(nodes.shape[0]), 1

    with torch.inference_mode():
        stats = {
            f"{prefix}nodes": float(n_nodes),
            f"{prefix}embedding_size": float(embedding_size),
            f"{prefix}mean": float(nodes.mean()),
            f"{prefix}std": float(nodes.std()),
            f"{prefix}min": float(nodes.min()),
            f"{prefix}max": float(nodes.max()),
        }

        rounded_nodes = torch.round(nodes, decimals=round_decimals)
        unique_rows = torch.unique(rounded_nodes, dim=0)
        stats[f"{prefix}node_diversity_score"] = float(unique_rows.shape[0]) / max(
            1.0, float(n_nodes)
        )

        # Median cosine similarity between random sample of node embeddings
        sample_n = min(sample_size, n_nodes)
        if sample_n > 1:
            idx = torch.randperm(n_nodes)[:sample_n]
            sample_nodes = nodes[idx]
            sample_nodes = (
                sample_nodes.unsqueeze(1) if sample_nodes.ndim == 1 else sample_nodes
            )
            sample_nodes = torch.nn.functional.normalize(sample_nodes, dim=1)
            sim = (
                sample_nodes @ sample_nodes.T
            )  # cosine similarity matrix between i,j in node
            # remove diagonal for median-of-off-diagonal
            mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
            median_cos = float(sim[mask].median())
            stats[f"{prefix}node_cosine_similarity"] = median_cos

        return stats


class NodeStatsLogger:
    """
    Optional 'set-and-forget' hook that:
      1) taps nodes before a target module runs,
      2) computes stats,
      3) logs to a provided logger (e.g., wandb),
      4) respects a user-provided step function and log interval.

    This avoids calling summarize+log separately inside the train loop.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        logger: MetricsLogger,
        get_step: Callable[[], int],
        prefix: str,
        log_every: int = 100,
        round_decimals: int = 3,
        sample_size: int = 256,
    ) -> None:
        if log_every <= 0:
            raise ValueError("log_every must be a positive integer.")

        self._logger: MetricsLogger = logger
        self._get_step: Callable[[], int] = get_step
        self._prefix: str = prefix
        self._log_every: int = log_every
        self._round_decimals: int = round_decimals
        self._sample_size: int = sample_size

        self._tap = NodesTap(prefix, module)
        self._hook = module.register_forward_hook(self._log_hook)

    def _log_hook(self, _module: torch.nn.Module, _inputs: tuple, output: Any) -> None:
        """
        Runs before the target module's forward. If it's the right step, compute & log.
        """
        step = self._get_step()
        if step % self._log_every != 0:
            return

        input_nodes = self._tap.last_input
        output_nodes = self._tap.last_output

        if input_nodes is None:
            raise ValueError("No inpuy nodes captured by NodesTap.")
        if output_nodes is None:
            raise ValueError("No output nodes captured by NodesTap.")

        input_stats = compute_node_stats(
            nodes=input_nodes,
            prefix=f"{self._prefix}input/",
            round_decimals=self._round_decimals,
            sample_size=self._sample_size,
        )

        output_stats = compute_node_stats(
            nodes=output_nodes,
            prefix=f"{self._prefix}output/",
            round_decimals=self._round_decimals,
            sample_size=self._sample_size,
        )

        # log stats
        self._logger.log(dict(input_stats), step=step)
        self._logger.log(dict(output_stats), step=step)

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None  # type: ignore[assignment]
        self._tap.close()

    def __enter__(self) -> "NodeStatsLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
