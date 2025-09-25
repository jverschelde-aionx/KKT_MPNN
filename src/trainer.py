from __future__ import annotations

import math
import os
import pickle
import random

# --------------------------- Diagnostics / Logging ---------------------------
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import wandb
from instances.common import ProblemClass
from models.gnn_transformer import GNNTransformer
from models.losses import KKTLoss, kkt_metrics
from models.policy_encoder import (
    GNNPolicy,
    GraphDataset,
    PolicyEncoder,
    collate,
)

MAX_HISTOGRAM_SAMPLES = 200_000  # cap histogram payload size


def _is_better(val, best, min_delta=0.0):
    return (best - val) > min_delta


def _apply_overrides(args, overrides: Mapping) -> None:
    """
    Apply hyperparameter sweep overrides to argparse namespace in-place.
    We accept {training: {...}, model: {...}, transformer: {...}} and/or flat keys.
    """
    if not overrides:
        return

    # 1) flat keys (top-level)
    for k, v in overrides.items():
        if k in ("training", "model", "transformer"):
            continue
        if hasattr(args, k):
            setattr(args, k, v)

    # 2) nested sections
    for section in ("model", "training", "transformer"):
        block = overrides.get(section, {})
        if not isinstance(block, Mapping):
            continue
        for k, v in block.items():
            # map nested key → argparse attribute (last token)
            attr = k
            if hasattr(args, attr):
                setattr(args, attr, v)


def compute_percentiles(
    tensor: torch.Tensor,
    percentiles=(0.0, 0.5, 0.9, 0.99, 1.0),
) -> Dict[str, float]:
    """Return percentiles p0/p50/p90/p99/p100 for a tensor (ignores non‑finite values)."""
    if tensor.numel() == 0:
        return {f"p{int(100 * p)}": float("nan") for p in percentiles}
    finite_mask = torch.isfinite(tensor)
    if not finite_mask.any():
        return {f"p{int(100 * p)}": float("nan") for p in percentiles}
    finite = tensor[finite_mask].float()
    q = (
        torch.quantile(
            finite,
            torch.tensor(percentiles, device=finite.device),
        )
        .cpu()
        .tolist()
    )
    return {f"p{int(100 * p)}": float(v) for p, v in zip(percentiles, q)}


def summarize_tensor_distribution(tensor: torch.Tensor) -> Dict[str, float]:
    """
    Robust distribution summary that works even if the whole tensor is non‑finite.
    Emits: finite_fraction, has_nan, has_inf, min, max, mean, abs_min, abs_max, and percentiles.
    """
    det = tensor.detach()
    num_elements = det.numel()
    finite_mask = torch.isfinite(det)

    summary: Dict[str, float] = {
        "finite_fraction": (
            finite_mask.float().mean().item() if num_elements > 0 else 1.0
        ),
        "has_nan": torch.isnan(det).any().item(),
        "has_inf": torch.isinf(det).any().item(),
    }

    if finite_mask.any():
        finite = det[finite_mask].float()
        summary.update(
            {
                "min": float(finite.min().item()),
                "max": float(finite.max().item()),
                "mean": float(finite.mean().item()),
                "abs_min": float(finite.abs().min().item()),
                "abs_max": float(finite.abs().max().item()),
            }
        )
        pct = compute_percentiles(finite)
        summary.update({f"percentile/{k}": v for k, v in pct.items()})
    else:
        # keep keys stable
        summary.update(
            {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "abs_min": float("nan"),
                "abs_max": float("nan"),
            }
        )
        summary.update({f"percentile/p{k}": float("nan") for k in [0, 50, 90, 99, 100]})

    return summary


def count_nonzero_entries_in_sparse_matrices(
    sparse_matrices: List[torch.Tensor],
) -> int:
    """Total number of non‑zero entries across a list of coalesced sparse COO tensors."""
    return sum(int(matrix._nnz()) for matrix in sparse_matrices)


def log_model_parameter_and_gradient_stats(model: nn.Module, step: int, topk: int = 8):
    """
    Log global parameter/gradient L2 norms and a top‑K table of layers by gradient norm.
    """
    global_param_l2_sq = 0.0
    global_grad_l2_sq = 0.0
    rows: List[Tuple[float, float, str]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_l2 = float(param.detach().norm(p=2).item())
        grad_l2 = (
            float(param.grad.detach().norm(p=2).item())
            if (param.grad is not None)
            else 0.0
        )
        global_param_l2_sq += param_l2 * param_l2
        global_grad_l2_sq += grad_l2 * grad_l2
        rows.append((grad_l2, param_l2, name))

    global_param_l2 = global_param_l2_sq**0.5
    global_grad_l2 = global_grad_l2_sq**0.5
    rows.sort(reverse=True, key=lambda row: row[0])

    wandb.log(
        {
            "parameters/global_l2_norm": global_param_l2,
            "gradients/global_l2_norm": global_grad_l2,
            "gradients/topk_by_l2_norm": wandb.Table(
                data=[
                    [name, grad_l2, param_l2]
                    for (grad_l2, param_l2, name) in rows[:topk]
                ],
                columns=["parameter_name", "gradient_l2", "parameter_l2"],
            ),
        },
        step=step,
    )


def log_optimizer_adam_moment_stats(optimizer: optim.Optimizer, step: int):
    """
    Summarize Adam/AdamW moments to catch runaway second moments (a common precursor to NaNs).
    """
    num_tracked = 0
    exp_avg_abs_sum = 0.0
    exp_avg_sq_mean_sum = 0.0
    max_rms_second_moment = 0.0

    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param, None)
            if not state or "exp_avg_sq" not in state:
                continue
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_abs_sum += float(exp_avg.detach().abs().mean().item())
            exp_avg_sq_mean_sum += float(exp_avg_sq.detach().mean().item())
            max_rms_second_moment = max(
                max_rms_second_moment, float(exp_avg_sq.detach().mean().sqrt().item())
            )
            num_tracked += 1

    if num_tracked:
        wandb.log(
            {
                "optimizer/adam_exp_avg_abs_mean": exp_avg_abs_sum / num_tracked,
                "optimizer/adam_exp_avg_sq_mean": exp_avg_sq_mean_sum / num_tracked,
                "optimizer/adam_max_rms_second_moment": max_rms_second_moment,
            },
            step=step,
        )


def log_input_batch_statistics(
    batch_graph,
    sparse_A_matrices: List[torch.Tensor],
    b_padded: torch.Tensor,
    c_padded: torch.Tensor,
    b_mask: torch.Tensor,
    c_mask: torch.Tensor,
    step: int,
    histogram_every: int,
):
    """
    Log input feature distributions, sentinel fractions, and A (or edge) density.
    """
    constraint_features = batch_graph.constraint_features
    variable_features = batch_graph.variable_features
    edge_coefficients = batch_graph.edge_attr.squeeze(-1)

    # Sentinel fractions from your nan_to_num (±1e6)
    sentinel_logs = {
        "inputs/sentinel_fraction/constraint_features/+1e6": (
            constraint_features == 1e6
        )
        .float()
        .mean()
        .item()
        if constraint_features.numel()
        else 0.0,
        "inputs/sentinel_fraction/constraint_features/-1e6": (
            constraint_features == -1e6
        )
        .float()
        .mean()
        .item()
        if constraint_features.numel()
        else 0.0,
        "inputs/sentinel_fraction/variable_features/+1e6": (variable_features == 1e6)
        .float()
        .mean()
        .item()
        if variable_features.numel()
        else 0.0,
        "inputs/sentinel_fraction/variable_features/-1e6": (variable_features == -1e6)
        .float()
        .mean()
        .item()
        if variable_features.numel()
        else 0.0,
        "inputs/sentinel_fraction/b_vector/+1e6": (b_padded[b_mask] == 1e6)
        .float()
        .mean()
        .item()
        if b_mask.any()
        else 0.0,
        "inputs/sentinel_fraction/b_vector/-1e6": (b_padded[b_mask] == -1e6)
        .float()
        .mean()
        .item()
        if b_mask.any()
        else 0.0,
        "inputs/sentinel_fraction/c_vector/+1e6": (c_padded[c_mask] == 1e6)
        .float()
        .mean()
        .item()
        if c_mask.any()
        else 0.0,
        "inputs/sentinel_fraction/c_vector/-1e6": (c_padded[c_mask] == -1e6)
        .float()
        .mean()
        .item()
        if c_mask.any()
        else 0.0,
    }

    # Count edges / non‑zeros and density
    if len(sparse_A_matrices) > 0:
        num_nonzero_entries = count_nonzero_entries_in_sparse_matrices(
            sparse_A_matrices
        )
        num_constraints_in_batch = int(b_mask.sum().item())
        num_variables_in_batch = int(c_mask.sum().item())
    else:
        num_nonzero_entries = int(batch_graph.edge_index.size(1))
        num_constraints_in_batch = int(b_mask.sum().item())
        num_variables_in_batch = int(c_mask.sum().item())

    density = float(num_nonzero_entries) / max(
        1, num_constraints_in_batch * num_variables_in_batch
    )

    distribution_logs = {
        **{
            f"inputs/constraint_features/{k}": v
            for k, v in summarize_tensor_distribution(constraint_features).items()
        },
        **{
            f"inputs/variable_features/{k}": v
            for k, v in summarize_tensor_distribution(variable_features).items()
        },
        **{
            f"inputs/edge_coefficients/{k}": v
            for k, v in summarize_tensor_distribution(edge_coefficients).items()
        },
        "structure/num_nonzero_entries": num_nonzero_entries,
        "structure/num_constraints_in_batch": num_constraints_in_batch,
        "structure/num_variables_in_batch": num_variables_in_batch,
        "structure/density": density,
    }
    wandb.log({**sentinel_logs, **distribution_logs}, step=step)

    # Optional histograms (downsampled)
    if histogram_every and (step % histogram_every == 0):

        def log_histogram(t: torch.Tensor, key: str):
            values = t.detach().float().flatten()
            if values.numel() > MAX_HISTOGRAM_SAMPLES:
                idx = torch.randint(0, values.numel(), (MAX_HISTOGRAM_SAMPLES,))
                values = values[idx]
            wandb.log({key: wandb.Histogram(values.cpu().numpy())}, step=step)

        log_histogram(constraint_features, "inputs/constraint_features/histogram")
        log_histogram(variable_features, "inputs/variable_features/histogram")
        log_histogram(edge_coefficients, "inputs/edge_coefficients/histogram")


class NonFiniteValueDetector:
    """
    Module‑hook based detector that logs the first module that produces non‑finite values
    either in forward outputs or backward gradients.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._installed = False
        self._has_fired = False
        self._step = 0

    def set_step(self, step: int):
        self._step = step

    def install(self):
        if self._installed:
            return
        for module_name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                module.register_forward_hook(self._forward_hook(module_name))
                module.register_full_backward_hook(self._backward_hook(module_name))
        self._installed = True

    def _log_once(self, where: str, module_name: str):
        if self._has_fired:
            return
        self._has_fired = True
        wandb.log(
            {f"alerts/first_nonfinite_{where}_module": module_name}, step=self._step
        )

    def _forward_hook(self, module_name: str):
        def hook(_module, _inputs, outputs):
            def is_bad(x):
                return torch.is_tensor(x) and not torch.isfinite(x).all()

            bad = False
            if torch.is_tensor(outputs):
                bad = is_bad(outputs)
            elif isinstance(outputs, (list, tuple)):
                bad = any(is_bad(t) for t in outputs)
            if bad:
                self._log_once("forward", module_name)

        return hook

    def _backward_hook(self, module_name: str):
        def hook(_module, grad_inputs, _grad_outputs):
            def is_bad(x):
                return (
                    torch.is_tensor(x)
                    and (x is not None)
                    and not torch.isfinite(x).all()
                )

            bad = False
            if isinstance(grad_inputs, (list, tuple)):
                bad = any(is_bad(t) for t in grad_inputs)
            else:
                bad = is_bad(grad_inputs)
            if bad:
                self._log_once("backward", module_name)

        return hook


def save_nonfinite_debug_artifact(
    save_directory: Path,
    step: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    batch_tuple,
):
    """
    Save a compact artifact with the exact batch and states that triggered a non‑finite loss.
    """
    filename = save_directory / f"nonfinite_step_{step}.pt"
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": getattr(scaler, "state_dict", lambda: {})(),
        "batch": {
            "A_list": [A.cpu().coalesce() for A in batch_tuple[1]],
            "b_pad": batch_tuple[2].cpu(),
            "c_pad": batch_tuple[3].cpu(),
            "b_mask": batch_tuple[4].cpu(),
            "c_mask": batch_tuple[5].cpu(),
            "m_sizes": batch_tuple[6],
            "n_sizes": batch_tuple[7],
            "sources": batch_tuple[8],
        },
    }
    torch.save(payload, filename)
    artifact = wandb.Artifact(f"nonfinite_step_{step}", type="debug-batch")
    artifact.add_file(str(filename))
    wandb.log_artifact(artifact)
    wandb.log({"alerts/nonfinite_artifact_saved": 1.0}, step=step)


def _sol_path_from_bg(bg_path: str) -> Path:
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


@lru_cache(maxsize=20000)
def _load_bg_meta(bg_path: str) -> Optional[Tuple[dict, np.ndarray]]:
    """
    Returns (v_map: name->idx, b_vars: np.int64[nb]) or None if failed.
    """
    try:
        with open(bg_path, "rb") as f:
            A, v_map, v_nodes, c_nodes, b_vars, b_vec, c_vec = pickle.load(f)
        # v_map is name->index view saved by your generator
        b_vars = np.asarray(b_vars, dtype=np.int64)
        return v_map, b_vars
    except Exception:
        return None


def _reorder_pool_to_model(
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


def _finite_stats(t: torch.Tensor) -> dict:
    # Safely compute stats even if all values are non‑finite
    t_det = t.detach()
    finite = torch.isfinite(t_det)
    numel = t_det.numel()

    out = {
        "finite_frac": (finite.float().mean().item() if numel > 0 else 1.0),
        "has_nan": torch.isnan(t_det).any().item(),
        "has_inf": torch.isinf(t_det).any().item(),
    }
    if finite.any():
        t_f = t_det[finite]
        out.update(
            {
                "min": t_f.min().item(),
                "max": t_f.max().item(),
                "mean": t_f.mean().item(),
                "abs_max": t_f.abs().max().item(),
            }
        )
    else:
        out.update(
            {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "abs_max": float("nan"),
            }
        )
    return out


def check_grads(model):
    bad = []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            bad.append(n)
    return bad


def log_tensor_stats(prefix: str, t: torch.Tensor, step: int, hist_every: int = 0):
    d = {f"{prefix}/{k}": v for k, v in _finite_stats(t).items()}
    wandb.log(d, step=step)
    if hist_every and (step % hist_every == 0):
        wandb.log(
            {f"{prefix}/hist": wandb.Histogram(t.detach().float().cpu().numpy())},
            step=step,
        )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: KKTLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float | None = None,
    start_step: int = 0,
    log_every: int = 50,
    histogram_every: int = 2000,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
    save_dir: Path | None = None,
    nonfinite_detector: "NonFiniteValueDetector" | None = None,
    parameter_stats_every: int = 500,
    save_nonfinite_artifacts: bool = False,
    stop_on_nonfinite: bool = False,
) -> tuple[float, int]:
    model.train()
    total_loss, n_batches = 0.0, 0
    step = start_step

    for batch in loader:
        step_start_time = time.time()
        (
            batch_graph,
            sparse_A_matrices,
            b_padded,
            c_padded,
            b_mask,
            c_mask,
            num_constraints_per_graph,
            num_variables_per_graph,
            source_paths,
        ) = batch

        step += 1

        wandb.log({"training/step": step}, step=step)
        if nonfinite_detector is not None:
            nonfinite_detector.set_step(step)

        # ----- Batch structure (counts) -----
        num_graphs_in_batch = len(num_constraints_per_graph)
        nodes_per_graph = [
            int(m) + int(n)
            for m, n in zip(num_constraints_per_graph, num_variables_per_graph)
        ]
        edges_in_batch = (
            count_nonzero_entries_in_sparse_matrices(sparse_A_matrices)
            if len(sparse_A_matrices) > 0
            else int(batch_graph.edge_index.size(1))
        )
        wandb.log(
            {
                "structure/num_graphs_in_batch": num_graphs_in_batch,
                "structure/max_sequence_length": max(nodes_per_graph),
                "structure/mean_sequence_length": sum(nodes_per_graph)
                / max(num_graphs_in_batch, 1),
                "structure/edges_in_batch": edges_in_batch,
            },
            step=step,
        )

        batch_graph = batch_graph.to(device)
        b_padded = b_padded.to(device)
        c_padded = c_padded.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler.is_enabled()):
            x_hat, lambda_hat = model(batch_graph)

        # --- DATA STATS (valid entries only) ---
        # Masked valid b, c
        device = x_hat.device

        # Ensure each sparse matrix lives on the right device (no-op if already there).
        sparse_A_matrices = [
            A if A.device == device else A.to(device, non_blocking=True)
            for A in sparse_A_matrices
        ]

        b_valid = b_padded[b_mask].detach()
        c_valid = c_padded[c_mask].detach()

        if (step % log_every) == 0:
            # Rich input stats + density + sentinels
            log_input_batch_statistics(
                batch_graph=batch_graph,
                sparse_A_matrices=sparse_A_matrices,
                b_padded=b_padded,
                c_padded=c_padded,
                b_mask=b_mask,
                c_mask=c_mask,
                step=step,
                histogram_every=histogram_every,
            )
            # Also log distributions of b and c for convenience
            log_tensor_stats("inputs/b_vector", b_valid, step)
            log_tensor_stats("inputs/c_vector", c_valid, step)
            if len(sparse_A_matrices) > 0:
                A_values = torch.cat([A.values() for A in sparse_A_matrices]).to(
                    device=x_hat.device
                )
                log_tensor_stats("inputs/A_values", A_values, step)

        # --- MODEL OUTPUT STATS ---
        # ----- Predictions -----
        log_tensor_stats("predictions/x_hat", x_hat, step, hist_every=histogram_every)
        log_tensor_stats(
            "predictions/lambda_hat", lambda_hat, step, hist_every=histogram_every
        )
        if (step % log_every) == 0:
            lambda_negative_fraction = (lambda_hat < 0).float().mean().item()
            wandb.log(
                {"predictions/lambda_negative_fraction": lambda_negative_fraction},
                step=step,
            )

        with autocast(enabled=scaler.is_enabled()):
            # matrix loss expects A_list; keep them on the right device
            sparse_A_matrices = [
                A if A.device == x_hat.device else A.to(x_hat.device, non_blocking=True)
                for A in sparse_A_matrices
            ]
            loss = criterion(
                x_hat,
                lambda_hat,
                sparse_A_matrices,
                b_padded,
                c_padded,
                num_constraints_per_graph,
                num_variables_per_graph,
            )

        # Term breakdown (extremely useful to see what diverges)
        if (step % log_every) == 0:
            kkt = kkt_metrics(
                x_hat,
                lambda_hat,
                sparse_A_matrices,
                b_padded,
                c_padded,
                num_constraints_per_graph,
                num_variables_per_graph,
            )
            wandb.log(
                {f"train_kkt/{name}": value for name, value in kkt.items()}, step=step
            )
            wandb.log({"train/loss": float(loss)}, step=step)

        # shape invariants
        assert x_hat.numel() == sum(num_variables_per_graph), (
            x_hat.shape,
            num_variables_per_graph,
        )
        assert lambda_hat.numel() == sum(num_constraints_per_graph), (
            lambda_hat.shape,
            num_constraints_per_graph,
        )

        # If the loss is non‑finite, log diagnostics and skip the step
        if not torch.isfinite(loss):
            wandb.log({"alerts/nonfinite_loss": 1.0}, step=step)
            x_finite = torch.isfinite(x_hat).all().item()
            lambda_finite = torch.isfinite(lambda_hat).all().item()
            b_finite = torch.isfinite(b_valid).all().item()
            c_finite = torch.isfinite(c_valid).all().item()
            wandb.log(
                {
                    "alerts/x_hat_all_finite": float(x_finite),
                    "alerts/lambda_hat_all_finite": float(lambda_finite),
                    "alerts/b_vector_all_finite": float(b_finite),
                    "alerts/c_vector_all_finite": float(c_finite),
                    "alerts/source_files": wandb.Table(
                        data=[[p] for p in source_paths], columns=["file"]
                    ),
                },
                step=step,
            )
            if save_nonfinite_artifacts and save_dir is not None:
                save_nonfinite_debug_artifact(
                    save_directory=save_dir,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    batch_tuple=(
                        batch_graph,
                        sparse_A_matrices,
                        b_padded,
                        c_padded,
                        b_mask,
                        c_mask,
                        num_constraints_per_graph,
                        num_variables_per_graph,
                        source_paths,
                    ),
                )
            if stop_on_nonfinite:
                raise RuntimeError(f"Non‑finite loss at step {step}")
            continue

        # ----- Backward -----
        scaler.scale(loss).backward()

        # Gradients (after unscale) + clipping
        scaler.unscale_(optimizer)
        total_gradient_l2 = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=math.inf
        )
        gradient_was_clipped = 0.0
        if grad_clip:
            gradient_was_clipped = float(total_gradient_l2 > grad_clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        wandb.log(
            {
                "gradients/global_l2_norm": float(total_gradient_l2),
                "gradients/clip_threshold": float(grad_clip if grad_clip else 0.0),
                "gradients/was_clipped": gradient_was_clipped,
            },
            step=step,
        )

        # Periodic deep stats
        if (step % parameter_stats_every) == 0:
            log_model_parameter_and_gradient_stats(model, step)
            log_optimizer_adam_moment_stats(optimizer, step)

        # Optimizer + AMP
        previous_amp_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        # Batch‑stepped schedulers
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()

        new_amp_scale = scaler.get_scale()
        log_dict = {}
        if new_amp_scale < previous_amp_scale:
            log_dict.update(
                {
                    "amp/scale": float(new_amp_scale),
                    "amp/scale_drop_ratio": float(
                        previous_amp_scale / max(new_amp_scale, 1e-12)
                    ),
                }
            )
        elif (step % log_every) == 0:
            log_dict.update({"amp/scale": float(new_amp_scale)})

        # Lightweight perf + memory every log_every
        if (step % log_every) == 0:
            log_dict.update(
                {
                    "optimizer/lr": optimizer.param_groups[0]["lr"],
                    "performance/step_time_ms": (time.time() - step_start_time)
                    * 1000.0,
                }
            )
            if torch.cuda.is_available():
                log_dict.update(
                    {
                        "cuda/memory_allocated_mb": torch.cuda.memory_allocated()
                        / (1024**2),
                        "cuda/memory_reserved_mb": torch.cuda.memory_reserved()
                        / (1024**2),
                        "cuda/max_memory_allocated_mb": torch.cuda.max_memory_allocated()
                        / (1024**2),
                    }
                )
        if log_dict:
            wandb.log(log_dict, step=step)

        total_loss += float(loss)
        n_batches += 1

    return total_loss / max(n_batches, 1), step


@torch.no_grad()
def eval_epoch(
    model, loader, criterion, device, amp=False
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    term_sums = {"primal": 0.0, "dual": 0.0, "stationarity": 0.0, "compl_slack": 0.0}

    # closeness accumulators
    close = {
        "obj_gap_best": 0.0,
        "l2_best": 0.0,
        "cos_best": 0.0,
        "hamm_bin_best": 0.0,
        "num_evaluated": 0,
        "num_total": 0,
    }
    eps = 1e-8

    for (
        batch_graph,
        A_list,
        b_pad,
        c_pad,
        b_mask,
        c_mask,
        m_sizes,
        n_sizes,
        sources,
    ) in loader:
        batch_graph = batch_graph.to(device)
        b_pad, c_pad = b_pad.to(device), c_pad.to(device)

        A_list = [A.to(device) if A.device != device else A for A in A_list]

        with autocast(enabled=amp):
            x_hat, lam_hat = model(batch_graph)
            A_list = [
                A if A.device == x_hat.device else A.to(x_hat.device, non_blocking=True)
                for A in A_list
            ]
            loss = criterion(
                x_hat,
                lam_hat,
                A_list,
                b_pad,
                c_pad,
                m_sizes,
                n_sizes,
            )
            metrics = kkt_metrics(
                x_hat,
                lam_hat,
                A_list,
                b_pad,
                c_pad,
                m_sizes,
                n_sizes,
            )
        total_loss += loss.item()
        for k in term_sums:
            term_sums[k] += metrics[k]
        n_batches += 1

        # ---------- Closeness to solver ----------
        # Slice per-instance views from concatenated x_hat / lam_hat
        off_x = 0
        off_l = 0
        B = len(m_sizes)
        close["num_total"] += B
        x_hat_f = x_hat.detach().float().cpu()
        for i in range(B):
            m_i, n_i = int(m_sizes[i]), int(n_sizes[i])
            x_i = x_hat_f[off_x : off_x + n_i]  # (n_i,)
            off_x += n_i
            off_l += m_i

            # Build paths & load pool
            bg_path = sources[i]
            sol_path = _sol_path_from_bg(bg_path)
            if not sol_path or not sol_path.exists():
                continue
            try:
                with open(sol_path, "rb") as f:
                    sol_data = pickle.load(f)
                var_names = sol_data["var_names"]
                sols = sol_data["sols"]  # shape (S, n), Gurobi order
            except Exception:
                continue

            meta = _load_bg_meta(bg_path)
            if meta is None:
                continue
            v_map, b_vars = meta

            sols_model = _reorder_pool_to_model(v_map, var_names, sols)
            if sols_model is None:
                continue

            # Objective (min orientation): use c from the batch (already flipped to min)
            c_i = c_pad[i, :n_i].detach().float().cpu().numpy()  # (n_i,)
            pool_obj_min = (sols_model @ c_i).min()  # best pool value (min)
            x_obj = float((x_i * c_pad[i, :n_i].detach().float().cpu()).sum().item())
            obj_gap = (x_obj - float(pool_obj_min)) / (abs(float(pool_obj_min)) + eps)

            # Distances to *best-objective* pool solution
            best_idx = int((sols_model @ c_i).argmin())
            best_sol = torch.tensor(sols_model[best_idx], dtype=torch.float32)
            # L2 (MSE) and cosine
            l2 = torch.mean((x_i - best_sol).pow(2)).item()
            denom = (x_i.norm(p=2) * best_sol.norm(p=2)).item()
            cos = float((x_i @ best_sol).item() / (denom + eps))

            # Hamming on binary vars, if any
            hamm = 0.0
            if b_vars.size > 0:
                b = torch.from_numpy(b_vars)
                x_bin = (x_i[b] >= 0.5).float()
                sol_bin = best_sol[b]
                hamm = torch.mean(torch.abs(x_bin - sol_bin)).item()

            # accumulate
            close["obj_gap_best"] += float(obj_gap)
            close["l2_best"] += float(l2)
            close["cos_best"] += float(cos)
            close["hamm_bin_best"] += float(hamm)
            close["num_evaluated"] += 1

    avg_metrics = {k: term_sums[k] / n_batches for k in term_sums}
    if close["num_evaluated"] > 0:
        Z = float(close["num_evaluated"])
        avg_metrics.update(
            {
                "close/num_evaluated": float(close["num_evaluated"]),
                "close/coverage": float(close["num_evaluated"])
                / float(close["num_total"]),
                "close/obj_gap_best": close["obj_gap_best"] / Z,
                "close/l2_best": close["l2_best"] / Z,
                "close/cos_best": close["cos_best"] / Z,
                "close/hamm_bin_best": close["hamm_bin_best"] / Z,
            }
        )
    else:
        avg_metrics.update(
            {
                "close/num_evaluated": 0.0,
                "close/coverage": 0.0,
            }
        )
    return total_loss / n_batches, avg_metrics


def train(overrides: Optional[Mapping] = None):
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        description="[KKT] Train GNN-Transformer",
        default_config_files=["config.yml"],
    )

    # Add GNNPolicy specific arguments
    GNNPolicy.add_args(parser)

    # Add GNNTransformer specific arguments
    GNNTransformer.add_args(parser)

    # Training
    t = parser.add_argument_group("training")
    t.add_argument("--devices", type=str, default="0")
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--num_workers", type=int, default=0)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument(
        "--scheduler", choices=["plateau", "cosine", "onecycle", "none"], default="none"
    )
    t.add_argument("--pct_start", type=float, default=0.3)  # OneCycle
    t.add_argument("--grad_clip", type=float, default=1.0)
    t.add_argument(
        "--amp", action="store_true", help="Use automatic mixed precision (fp16/bf16)"
    )

    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--kkt_w_primal", type=float, default=0.1)
    t.add_argument("--kkt_w_dual", type=float, default=0.1)
    t.add_argument("--kkt_w_station", type=float, default=0.6)
    t.add_argument("--kkt_w_comp", type=float, default=0.2)
    t.add_argument("--max_lr", type=float, default=0.001)
    t.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Log lightweight scalars every N steps.",
    )
    t.add_argument(
        "--histogram_every",
        type=int,
        default=2000,
        help="Log histograms every N steps (set 0 to disable).",
    )
    t.add_argument(
        "--parameter_stats_every",
        type=int,
        default=500,
        help="Log parameter/gradient top‑K and optimizer moments every N steps.",
    )
    t.add_argument(
        "--save_nonfinite_artifacts",
        action="store_true",
        help="When loss becomes non‑finite, save a debug artifact with batch + states.",
    )
    t.add_argument(
        "--stop_on_nonfinite",
        action="store_true",
        help="Raise immediately if a non‑finite loss is observed.",
    )
    t.add_argument(
        "--early_stop",
        action="store_true",
        help="Enable epoch-level early stopping on valid/loss",
    )
    t.add_argument(
        "--es_patience",
        type=int,
        default=4,
        help="#epochs without improvement before stopping",
    )
    t.add_argument(
        "--es_min_delta",
        type=float,
        default=6,
        help="Minimum absolute improvement to reset patience",
    )
    t.add_argument(
        "--es_min_epochs",
        type=int,
        default=3,
        help="Do not consider early stop before this epoch",
    )
    t.add_argument(
        "--es_cooldown",
        type=int,
        default=1,
        help="#epochs to wait after an improvement before counting patience",
    )
    t.add_argument(
        "--es_restore_best",
        action="store_true",
        help="Load best.pt weights into the model on early stop",
    )

    # Data
    d = parser.add_argument_group("data")
    d.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=[ProblemClass.INDEPENDANT_SET, ProblemClass.COMBINATORIAL_AUCTION],
        help="Problem type",
    )
    d.add_argument(
        "--is_sizes",
        type=int,
        nargs="+",
        default=[10, 50, 100, 200, 500, 1000],
    )
    d.add_argument(
        "--ca_sizes",
        type=int,
        nargs="+",
        default=[10, 50, 100, 200, 500, 1000],
    )
    d.add_argument(
        "--n_instances", type=int, default=1000, help="Number of instances per size"
    )
    d.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Fraction of instances for the test set",
    )
    d.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of instances for the validation set",
    )
    d.add_argument("--data_root", type=str, default="../data")
    d.add_argument(
        "--solve", action="store_true", help="Run Gurobi to collect solution pools"
    )
    d.add_argument(
        "--gurobi_threads", type=int, default=4, help="Number of threads for Gurobi"
    )
    d.add_argument(
        "--n_jobs",
        type=int,
        default=32,
        help="Number of parallel jobs for data generation",
    )

    args, unkown = parser.parse_known_args()

    if overrides is not None:
        _apply_overrides(args, overrides)

    # Ensure reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and args.devices
        else torch.device("cpu")
    )

    # Initialize wandb
    run_name = (
        "kkt_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + GNNTransformer._get_name(args)
        + GNNPolicy._get_name(args)
    )

    if wandb.run is None:
        wandb.init(project="kkt_transformer", name=run_name, config=vars(args))

    wandb.define_metric("training/step")
    wandb.define_metric("*", step_metric="training/step")

    # Setup data
    train_files, valid_files = [], []

    for problem in args.problems:
        dir_bg = Path(args.data_root) / problem / "BG"
        for dir in (dir_bg / "train").iterdir():
            train_files.extend([str(dir / f) for f in os.listdir(dir)])
        for dir in (dir_bg / "val").iterdir():
            valid_files.extend([str(dir / f) for f in os.listdir(dir)])

    train_files = sorted(train_files)
    valid_files = sorted(valid_files)

    train_data = GraphDataset(train_files)
    valid_data = GraphDataset(valid_files)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    # Retrieve problem instance dimensions (m,n) from the first sample
    (batch_graph, A_list, _b, _c, _bm, _cm, m_sizes, n_sizes, sources) = next(
        iter(valid_loader)
    )
    m, n = m_sizes[0], n_sizes[0]

    # Model, loss, optimiser, scheduler
    node_encoder = PolicyEncoder(args)
    model = GNNTransformer(
        args=args,
        gnn_node=node_encoder,
    ).to(device)

    nonfinite_detector = NonFiniteValueDetector(model)
    nonfinite_detector.install()

    logger.info("Model parameters: {}", model.count_parameters())
    wandb.watch(model, log="gradients", log_graph=False)

    loss_fn = KKTLoss(
        m,
        n,
        w_primal=args.kkt_w_primal,
        w_dual=args.kkt_w_dual,
        w_stat=args.kkt_w_station,
        w_comp=args.kkt_w_comp,
    )

    criterion = loss_fn.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=args.amp)

    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=(args.max_lr or args.lr),
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=args.pct_start,
        )
    else:
        scheduler = None

    # Create save directory
    save_dir = Path("exps") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
    cooldown_left = 0

    logger.info(f"train size: {len(train_data)} | valid size: {len(valid_data)}")
    wandb.log({"data/train_size": len(train_data), "data/valid_size": len(valid_data)})

    # Train loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.epoch_callback(epoch)

        train_loss, global_step = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            start_step=global_step,
            log_every=args.log_every,
            histogram_every=args.histogram_every,
            scheduler=scheduler,
            save_dir=save_dir,
            nonfinite_detector=nonfinite_detector,
            parameter_stats_every=args.parameter_stats_every,
            save_nonfinite_artifacts=args.save_nonfinite_artifacts,
            stop_on_nonfinite=args.stop_on_nonfinite,
        )

        # Epoch-level schedulers
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        val_loss, val_terms = eval_epoch(
            model, valid_loader, criterion, device, args.amp
        )

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        # logging
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"valid/{k}": v for k, v in val_terms.items()},
            },
            step=global_step,
        )

        # Model checkpointing
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")

        # track best
        improved = _is_better(val_loss, best_val, args.es_min_delta)
        if improved:
            best_val = float(val_loss)
            best_epoch = epoch
            epochs_since_improve = 0
            cooldown_left = args.es_cooldown
            torch.save(ckpt, save_dir / "best.pt")
            wandb.run.summary["best_val_loss"] = best_val
            wandb.run.summary["best_epoch"] = best_epoch
        else:
            if cooldown_left > 0:
                cooldown_left -= 1
            else:
                epochs_since_improve += 1

        # if val_loss < best_val:
        #     best_val = val_loss
        #     torch.save(ckpt, save_dir / "best.pt")
        #     wandb.run.summary["best_val_loss"] = best_val

        logger.info(
            "Epoch {:03d} | train {:.4f} | valid {:.4f} (best {:.4f})",
            epoch,
            train_loss,
            val_loss,
            best_val,
        )
        # ---- early stopping check ----
        if (
            args.early_stop
            and epoch >= args.es_min_epochs
            and epochs_since_improve >= args.es_patience
        ):
            logger.info(
                "Early stopping at epoch {:03d} (no improvement for {} epochs). Best {:.4f} @ {:03d}",
                epoch,
                epochs_since_improve,
                best_val,
                best_epoch,
            )
            wandb.run.summary["early_stopped"] = True
            wandb.run.summary["early_stop_epoch"] = epoch

            if args.es_restore_best and (save_dir / "best.pt").exists():
                state = torch.load(save_dir / "best.pt", map_location=device)
                model.load_state_dict(state["model"])
                logger.info("Restored model weights from best.pt")

            break  # exit training loop
    logger.info("Finished training. Best validation loss = {:.4f}", best_val)


if __name__ == "__main__":
    train()
