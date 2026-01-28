"""
Warm-start experiment

Goal
----
Test whether the learned solver's prediction can be used as a warm-start for a
(classical) LP solver. For each LP instance:
  1) Solve with solver default initialization (baseline)
  2) Predict a primal solution x_hat (and optionally dual multipliers lambda_hat)
  3) Solve again, but warm-started from (x_hat, lambda_hat)

We report median (P50) and P90 solve time + iteration count, and the median speedup.

Notes
-----
- This script is *backend-agnostic*:
    * If gurobipy is installed, it will warm-start Gurobi (best option)
    * Else if highspy is installed, it will warm-start HiGHS
    * Else it falls back to an internal primal-dual interior-point solver (PDIPM),
      which DOES support warm-start from (x_hat, lambda_hat) and gives meaningful
      time/iteration comparisons without extra dependencies.

- The PDIPM backend assumes LP in standard form:
      minimize   c^T x
      subject to A x = b
                 x >= 0
  which matches many KKT-learning pipelines (x + equality multipliers lambda).
  If your data is Ax <= b, either your preprocessing likely already introduced
  slacks, or you should adapt the backend to inequality form.

Usage
-----
Run once per method/checkpoint/config and collect the printed P50/P90 and speedup:
  - Solver default init
  - Warm-start (your method tag)

Then fill your paper table across methods (MLP-BL / GNN-BL / GNN-FF / GNN-FE).

Example:
  python warm_start_eval.py --config ../configs/...yml --encoder_path ... --method_tag GNN-FE --warmstart_dual

"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import configargparse
import numpy as np
import torch
from loguru import logger
from tqdm.notebook import tqdm

from data.common import ProblemClass
from data.utils import lp_path_from_bg
from jobs.utils import build_dataloaders, device_from_args, pack_by_sizes, set_all_seeds
from models.gnn import GNNPolicy
from models.mlp import KKTNetMLP

# ---------------------------------------------------------------------
# Optional: robust check for PyG batch
# ---------------------------------------------------------------------
try:
    from torch_geometric.data import Batch as PyGBatch
except Exception:
    PyGBatch = None


# ---------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------
@dataclass
class SolveStats:
    ok: bool
    status: str
    time_s: float
    iterations: int
    obj: Optional[float] = None


def p50(x: List[float]) -> float:
    return float(np.median(np.asarray(x, dtype=float)))


def p90(x: List[float]) -> float:
    return float(np.percentile(np.asarray(x, dtype=float), 90))


# ---------------------------------------------------------------------
# PDIPM backend (infeasible-start Mehrotra predictor-corrector)
# Standard form: min c^T x s.t. A x = b, x >= 0
# ---------------------------------------------------------------------
def _max_step(z: np.ndarray, dz: np.ndarray, eta: float) -> float:
    """Max step in (0,1] to keep z + alpha*dz >= 0 (with safety eta)."""
    neg = dz < 0
    if not np.any(neg):
        return 1.0
    return float(min(1.0, eta * np.min(-z[neg] / dz[neg])))


def _newton_step_eq(
    A: np.ndarray,
    x: np.ndarray,
    lam: np.ndarray,
    s: np.ndarray,
    r_p: np.ndarray,
    r_d: np.ndarray,
    r_c: np.ndarray,
    reg: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Newton system for equality-form LP:

        A dx             = r_p
        A^T dlam + ds    = r_d
        S dx + X ds      = r_c

    using elimination.

    Returns (dx, dlam, ds).
    """
    # W^{-1} Z where W=S and Z=X
    # D_j = x_j / s_j
    D = x / s  # shape (n,)

    # t = S^{-1}(r_c - X r_d) = (r_c - x*r_d)/s
    t = (r_c - x * r_d) / s  # shape (n,)

    # Solve (A * diag(D) * A^T) dlam = r_p - A t
    rhs = r_p - A @ t  # shape (m,)

    # M = A diag(D) A^T
    # Implement as: A @ (diag(D) @ A.T) = A @ ((D[:,None] * A.T))
    M = A @ (D[:, None] * A.T)  # shape (m,m)
    if reg > 0:
        M.flat[:: M.shape[0] + 1] += reg

    try:
        dlam = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        dlam = np.linalg.lstsq(M, rhs, rcond=None)[0]

    At_dlam = A.T @ dlam  # shape (n,)
    dx = (r_c - x * r_d + x * At_dlam) / s
    ds = r_d - At_dlam
    return dx, dlam, ds


def solve_lp_pdipm(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    x0: Optional[np.ndarray] = None,
    lam0: Optional[np.ndarray] = None,
    max_iter: int = 80,
    tol: float = 1e-8,
    eta: float = 0.995,
    reg: float = 1e-9,
    eps: float = 1e-8,
) -> SolveStats:
    """
    Solve LP (standard equality form) with Mehrotra predictor-corrector PDIPM.

    Warm-start:
      - If x0 is provided, initializes x from x0 (clipped to eps)
      - If lam0 is provided, initializes lambda from lam0, otherwise zeros
      - s is initialized from c - A^T lam (clipped to eps). If that is too small,
        we still clip to eps to keep interior feasibility.
    """
    t0 = time.perf_counter()

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    m, n = A.shape
    if b.shape != (m,):
        b = b.reshape(m)
    if c.shape != (n,):
        c = c.reshape(n)

    # Init
    if x0 is None:
        x = np.ones(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(n).copy()
    x = np.maximum(x, eps)

    if lam0 is None:
        lam = np.zeros(m, dtype=float)
    else:
        lam = np.asarray(lam0, dtype=float).reshape(m).copy()

    # Dual slack s = c - A^T lam  (clipped to interior)
    s = c - A.T @ lam
    s = np.maximum(s, eps)

    # If user didn't warm-start anything, keep a very stable default dual slack
    if x0 is None and lam0 is None:
        s = np.ones(n, dtype=float)

    # Main loop
    iters = 0
    ok = False
    status = "max_iter"

    # Scaling for stopping criteria
    b_scale = 1.0 + np.linalg.norm(b, ord=np.inf)
    c_scale = 1.0 + np.linalg.norm(c, ord=np.inf)

    for k in range(1, max_iter + 1):
        iters = k

        # Residuals
        r_p = b - A @ x  # primal feasibility
        r_d = c - A.T @ lam - s  # dual feasibility

        mu = float(x @ s) / n

        # Stop: scaled infinity norms + complementarity
        rp_norm = float(np.linalg.norm(r_p, ord=np.inf)) / b_scale
        rd_norm = float(np.linalg.norm(r_d, ord=np.inf)) / c_scale
        if max(rp_norm, rd_norm, mu) < tol:
            ok = True
            status = "optimal"
            break

        # Affine-scaling (sigma=0) predictor
        r_c_aff = -x * s
        dx_aff, dlam_aff, ds_aff = _newton_step_eq(
            A=A, x=x, lam=lam, s=s, r_p=r_p, r_d=r_d, r_c=r_c_aff, reg=reg
        )

        alpha_aff_pri = _max_step(x, dx_aff, eta)
        alpha_aff_dual = _max_step(s, ds_aff, eta)

        x_aff = x + alpha_aff_pri * dx_aff
        s_aff = s + alpha_aff_dual * ds_aff
        mu_aff = (
            float(x_aff @ s_aff) / n if np.all(x_aff > 0) and np.all(s_aff > 0) else mu
        )

        # Centering parameter (Mehrotra)
        sigma = float((mu_aff / mu) ** 3) if mu > 0 else 0.0
        sigma = min(max(sigma, 0.0), 1.0)

        # Corrector RHS
        # r_c = sigma*mu*e - XSe - (dx_aff * ds_aff)
        r_c = sigma * mu * np.ones(n, dtype=float) - x * s - dx_aff * ds_aff

        # Combined direction
        dx, dlam, ds = _newton_step_eq(
            A=A, x=x, lam=lam, s=s, r_p=r_p, r_d=r_d, r_c=r_c, reg=reg
        )

        alpha_pri = _max_step(x, dx, eta)
        alpha_dual = _max_step(s, ds, eta)

        # Update
        x = x + alpha_pri * dx
        lam = lam + alpha_dual * dlam
        s = s + alpha_dual * ds

        # Safeguard
        x = np.maximum(x, eps)
        s = np.maximum(s, eps)

    dt = time.perf_counter() - t0
    obj = float(c @ x)
    return SolveStats(ok=ok, status=status, time_s=dt, iterations=iters, obj=obj)


# ---------------------------------------------------------------------
# External solver backends (optional)
# ---------------------------------------------------------------------
class LPSolver:
    """
    Uniform interface for the warm-start experiment.

    - If backend is gurobi/highs: solves from .lp file path (true "classical solver" run)
    - If backend is pdipm/scipy: solves from arrays (fallback)

    We expose solve_pair(...) so the evaluation loop can do baseline + warm for the same instance.
    """

    def __init__(
        self,
        backend: str = "auto",
        warmstart_dual: bool = False,
        max_iter: int = 80,
        tol: float = 1e-8,
        quiet: bool = True,
    ) -> None:
        self.backend = backend
        self.warmstart_dual = warmstart_dual
        self.max_iter = max_iter
        self.tol = tol
        self.quiet = quiet

        self._resolved = self._resolve_backend(backend)

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend

        # Prefer real solvers if available
        try:
            import gurobipy  # noqa: F401

            return "gurobi"
        except Exception:
            pass

        try:
            import highspy  # noqa: F401

            return "highs"
        except Exception:
            pass

        return "pdipm"

    @property
    def supports_file(self) -> bool:
        return self._resolved in {"gurobi", "highs"}

    # ----------------------------
    # Public entry: baseline+warm
    # ----------------------------
    def solve_pair(
        self,
        *,
        lp_path: Optional[str] = None,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
        lam0: Optional[np.ndarray] = None,
    ) -> Tuple[SolveStats, SolveStats]:
        if self.supports_file:
            if lp_path is None:
                raise ValueError("lp_path must be provided for file-based solvers.")
            return self._solve_pair_file(lp_path=lp_path, x0=x0, lam0=lam0)

        # Array-based fallback
        if A is None or b is None or c is None:
            raise ValueError("A,b,c must be provided for array-based solvers.")
        base = self._solve_arrays(A=A, b=b, c=c, x0=None, lam0=None)
        warm = self._solve_arrays(
            A=A, b=b, c=c, x0=x0, lam0=lam0 if self.warmstart_dual else None
        )
        return base, warm

    # ----------------------------
    # File-based implementations
    # ----------------------------
    def _solve_pair_file(
        self,
        *,
        lp_path: str,
        x0: Optional[np.ndarray],
        lam0: Optional[np.ndarray],
    ) -> Tuple[SolveStats, SolveStats]:
        if self._resolved == "gurobi":
            return self._solve_pair_file_gurobi(lp_path=lp_path, x0=x0, lam0=lam0)
        if self._resolved == "highs":
            return self._solve_pair_file_highs(lp_path=lp_path, x0=x0, lam0=lam0)
        raise ValueError(f"File-based solve not supported for backend={self._resolved}")

    def _solve_pair_file_gurobi(
        self,
        *,
        lp_path: str,
        x0: Optional[np.ndarray],
        lam0: Optional[np.ndarray],
    ) -> Tuple[SolveStats, SolveStats]:
        import gurobipy as gp
        from gurobipy import GRB

        # Read models (read time excluded from solve time; both baseline/warm do it)
        base_model = gp.read(lp_path)
        warm_model = gp.read(lp_path)

        for m in (base_model, warm_model):
            if self.quiet:
                m.Params.OutputFlag = 0

            # You can comment these out if you truly want method auto-selection,
            # but barrier makes "iterations" comparable and uses PStart/DStart.
            m.Params.Method = 2  # barrier
            m.Params.Crossover = 0  # avoid crossover noise

        # ---- baseline solve (no warm-start) ----
        t0 = time.perf_counter()
        base_model.optimize()
        dt_base = time.perf_counter() - t0

        bar_iter_base = int(getattr(base_model, "BarIterCount", 0) or 0)
        iters_base = (
            bar_iter_base
            if bar_iter_base > 0
            else int(getattr(base_model, "IterCount", 0) or 0)
        )
        ok_base = base_model.Status == GRB.OPTIMAL

        stats_base = SolveStats(
            ok=ok_base,
            status=str(base_model.Status),
            time_s=float(getattr(base_model, "Runtime", dt_base)),
            iterations=int(iters_base),
            obj=float(base_model.ObjVal) if ok_base else None,
        )

        # ---- warm-start solve ----
        if x0 is not None:
            x0 = np.asarray(x0, dtype=float).reshape(-1)
            vars_ = warm_model.getVars()
            n_lp = len(vars_)
            n_set = min(n_lp, x0.shape[0])
            if x0.shape[0] != n_lp:
                logger.warning(
                    f"[warmstart] Var count mismatch: pred n={x0.shape[0]} vs lp n={n_lp} for {lp_path}"
                )

            for j in range(n_set):
                v = vars_[j]
                val = float(x0[j])

                # Clip to bounds
                lb = float(v.LB)
                ub = float(v.UB)
                if val < lb:
                    val = lb
                if ub < GRB.INFINITY and val > ub:
                    val = ub

                # Best-effort: set both Start and barrier PStart
                try:
                    v.Start = val
                except Exception:
                    pass
                try:
                    v.PStart = val
                except Exception:
                    pass

        if self.warmstart_dual and lam0 is not None:
            lam0 = np.asarray(lam0, dtype=float).reshape(-1)
            constrs = warm_model.getConstrs()
            m_lp = len(constrs)
            m_set = min(m_lp, lam0.shape[0])
            if lam0.shape[0] != m_lp:
                logger.warning(
                    f"[warmstart] Constr count mismatch: pred m={lam0.shape[0]} vs lp m={m_lp} for {lp_path}"
                )

            for i in range(m_set):
                try:
                    constrs[i].DStart = float(lam0[i])
                except Exception:
                    pass

        t1 = time.perf_counter()
        warm_model.optimize()
        dt_warm = time.perf_counter() - t1

        bar_iter_warm = int(getattr(warm_model, "BarIterCount", 0) or 0)
        iters_warm = (
            bar_iter_warm
            if bar_iter_warm > 0
            else int(getattr(warm_model, "IterCount", 0) or 0)
        )
        ok_warm = warm_model.Status == GRB.OPTIMAL

        stats_warm = SolveStats(
            ok=ok_warm,
            status=str(warm_model.Status),
            time_s=float(getattr(warm_model, "Runtime", dt_warm)),
            iterations=int(iters_warm),
            obj=float(warm_model.ObjVal) if ok_warm else None,
        )

        return stats_base, stats_warm

    def _solve_pair_file_highs(
        self,
        *,
        lp_path: str,
        x0: Optional[np.ndarray],
        lam0: Optional[np.ndarray],
    ) -> Tuple[SolveStats, SolveStats]:
        import highspy

        def run_one(warm: bool) -> SolveStats:
            highs = highspy.Highs()
            try:
                highs.setOptionValue("output_flag", False if self.quiet else True)
            except Exception:
                pass

            # Prefer IPM if available (iterations metric)
            try:
                highs.setOptionValue("solver", "ipm")
            except Exception:
                pass

            t_read0 = time.perf_counter()
            highs.readModel(lp_path)
            _ = (
                time.perf_counter() - t_read0
            )  # read time excluded from solve timing below

            if warm and x0 is not None:
                sol = highspy.HighsSolution()
                x_arr = np.asarray(x0, dtype=float).reshape(-1)
                sol.col_value = x_arr.tolist()

                if self.warmstart_dual and lam0 is not None:
                    sol.row_dual = np.asarray(lam0, dtype=float).reshape(-1).tolist()

                try:
                    highs.setSolution(sol)
                except Exception:
                    pass

            t0 = time.perf_counter()
            highs.run()
            dt = time.perf_counter() - t0

            info = highs.getInfo()
            ipm_it = int(getattr(info, "ipm_iteration_count", 0) or 0)
            smp_it = int(getattr(info, "simplex_iteration_count", 0) or 0)
            iters = ipm_it if ipm_it > 0 else smp_it

            status = str(highs.getModelStatus())
            ok = "Optimal" in status or "kOptimal" in status

            obj = None
            try:
                obj = float(highs.getObjectiveValue())
            except Exception:
                pass

            return SolveStats(
                ok=ok, status=status, time_s=dt, iterations=int(iters), obj=obj
            )

        base = run_one(warm=False)
        warm = run_one(warm=True)
        return base, warm

    # ----------------------------
    # Array-based fallback
    # ----------------------------
    def _solve_arrays(
        self,
        *,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        x0: Optional[np.ndarray],
        lam0: Optional[np.ndarray],
    ) -> SolveStats:
        if self._resolved == "pdipm":
            return solve_lp_pdipm(
                A=A,
                b=b,
                c=c,
                x0=x0,
                lam0=lam0 if self.warmstart_dual else None,
                max_iter=self.max_iter,
                tol=self.tol,
            )

        if self._resolved == "scipy":
            from scipy.optimize import linprog

            t0 = time.perf_counter()
            res = linprog(c=c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
            dt = time.perf_counter() - t0
            iters = int(getattr(res, "nit", 0) or 0)
            return SolveStats(
                ok=bool(res.success),
                status=str(res.message),
                time_s=dt,
                iterations=iters,
                obj=float(res.fun) if res.success else None,
            )

        raise ValueError(
            f"Array-based solve not implemented for backend={self._resolved}"
        )


# ---------------------------------------------------------------------
# Data -> instance slicing
# ---------------------------------------------------------------------
def _infer_sizes_from_masks(
    mask_m: torch.Tensor, mask_n: torch.Tensor, i: int
) -> Tuple[int, int]:
    # mask_* are typically [B, m_max] / [B, n_max] booleans or {0,1}
    m_i = int(mask_m[i].detach().sum().item())
    n_i = int(mask_n[i].detach().sum().item())
    return m_i, n_i


def _slice_instance(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    i: int,
    m_i: int,
    n_i: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_i = A[i, :m_i, :n_i].detach().cpu().numpy()
    b_i = b[i, :m_i].detach().cpu().numpy()
    c_i = c[i, :n_i].detach().cpu().numpy()
    return A_i, b_i, c_i


def _slice_prediction(
    x_pred: torch.Tensor,
    lam_pred: torch.Tensor,
    i: int,
    m_i: int,
    n_i: int,
    clip_primal_nonneg: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    x0 = x_pred[i, :n_i].detach().cpu().numpy()
    lam0 = lam_pred[i, :m_i].detach().cpu().numpy()
    if clip_primal_nonneg:
        x0 = np.maximum(x0, 0.0)
    return x0, lam0


def resolve_lp_instance_path(sample_path: Any, is_bipartite: bool) -> str:
    """
    sample_path:
      - bipartite/GNN: path to *.lp.bg (BG file)
      - non-bipartite: path to *.lp
    Returns a string path to the *.lp file.
    """
    sp = str(sample_path)
    if is_bipartite or sp.endswith(".bg"):
        lp = lp_path_from_bg(sp)
        lp_str = str(lp) if lp is not None else ""
        # lp_path_from_bg returns Path() on failure -> str(Path()) == '.'
        if lp_str and lp_str != ".":
            return lp_str
        # fallback: return original if mapping fails
        return sp
    return sp


# ---------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------
@torch.no_grad()
def run_warm_start_eval(
    model: torch.nn.Module,
    loader: Union[PyGDataLoader, DataLoader],
    solver: LPSolver,
    *,
    max_instances: Optional[int] = None,
) -> Dict[str, Any]:
    times_base: List[float] = []
    iters_base: List[int] = []
    times_warm: List[float] = []
    iters_warm: List[int] = []
    statuses: List[str] = []

    n_seen = 0
    n_ok = 0

    for batch in tqdm(loader):
        # -----------------------------------------------------------------
        # Unpack batch
        # -----------------------------------------------------------------
        if PyGBatch is not None and isinstance(batch[0], PyGBatch):
            # Bipartite GNN batch
            batch_graph, A, b, c, mask_m, mask_n, m_sizes, n_sizes, sample_paths = batch
            print(sample_paths[0])

            batch_graph = batch_graph.to(
                next(model.parameters()).device, non_blocking=True
            )
            A = A.to(next(model.parameters()).device, non_blocking=True)
            b = b.to(next(model.parameters()).device, non_blocking=True)
            c = c.to(next(model.parameters()).device, non_blocking=True)
            mask_m = mask_m.to(next(model.parameters()).device, non_blocking=True)
            mask_n = mask_n.to(next(model.parameters()).device, non_blocking=True)

            # Predict (packed flat -> padded per instance)
            x_all, lam_all = model(
                batch_graph.constraint_features,
                batch_graph.edge_index,
                batch_graph.edge_attr,
                batch_graph.variable_features,
            )

            n_max = int(c.shape[1])
            m_max = int(b.shape[1])

            # m_sizes/n_sizes may be tensors or lists
            if torch.is_tensor(n_sizes):
                n_sizes_list = [int(v) for v in n_sizes.detach().cpu().tolist()]
            else:
                n_sizes_list = [int(v) for v in n_sizes]
            if torch.is_tensor(m_sizes):
                m_sizes_list = [int(v) for v in m_sizes.detach().cpu().tolist()]
            else:
                m_sizes_list = [int(v) for v in m_sizes]

            x_pred = pack_by_sizes(x_all, n_sizes_list, n_max)  # [B, n_max]
            lam_pred = pack_by_sizes(lam_all, m_sizes_list, m_max)  # [B, m_max]
            B = int(A.size(0))

            sizes_list = list(zip(m_sizes_list, n_sizes_list))

        else:
            # Non-graph (MLP) batch
            model_input, A, b, c, mask_m, mask_n, sample_paths = batch

            device = next(model.parameters()).device
            model_input = model_input.to(device, non_blocking=True)
            A = A.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            mask_m = mask_m.to(device, non_blocking=True)
            mask_n = mask_n.to(device, non_blocking=True)

            y_pred = model(model_input)  # expected [B, n_max+m_max]
            B = int(A.size(0))
            n_max = int(c.shape[1])
            m_max = int(b.shape[1])

            x_pred = y_pred[:, :n_max]
            lam_pred = y_pred[:, n_max : n_max + m_max]

            sizes_list = []
            for i in range(B):
                m_i, n_i = _infer_sizes_from_masks(mask_m, mask_n, i)
                sizes_list.append((m_i, n_i))

        # -----------------------------------------------------------------
        # Per-instance solving
        # -----------------------------------------------------------------
        for i in range(B):
            m_i, n_i = sizes_list[i]
            if m_i <= 0 or n_i <= 0:
                continue

            A_i, b_i, c_i = _slice_instance(A, b, c, i, m_i, n_i)
            x0_i, lam0_i = _slice_prediction(x_pred, lam_pred, i, m_i, n_i)

            # Resolve the .lp path from sample_paths
            is_bip = PyGBatch is not None and isinstance(batch[0], PyGBatch)
            lp_inst_path = resolve_lp_instance_path(
                sample_paths[i], is_bipartite=is_bip
            )

            # If solver supports file runs, solve from .lp file; else fallback to arrays
            stats_base, stats_warm = solver.solve_pair(
                lp_path=lp_inst_path,
                A=A_i,
                b=b_i,
                c=c_i,  # only used by array fallback backends
                x0=x0_i,
                lam0=lam0_i,
            )

            # Record
            n_seen += 1
            statuses.append(f"base:{stats_base.status}|warm:{stats_warm.status}")

            # Skip instances where baseline failed (fair comparison)
            if not stats_base.ok:
                if n_seen % 200 == 0:
                    logger.warning(
                        f"Skipping instance {n_seen}: baseline not ok ({stats_base.status})"
                    )
            else:
                times_base.append(stats_base.time_s)
                iters_base.append(int(stats_base.iterations))

                # Keep warm-start even if it fails, but you may choose to skip it too
                if stats_warm.ok:
                    times_warm.append(stats_warm.time_s)
                    iters_warm.append(int(stats_warm.iterations))
                    n_ok += 1
                else:
                    # If warm-start failed, still record it as max_iter-ish so it doesn't look "free"
                    times_warm.append(stats_warm.time_s)
                    iters_warm.append(int(stats_warm.iterations))

            if max_instances is not None and n_seen >= max_instances:
                break

        if max_instances is not None and n_seen >= max_instances:
            break

    if len(times_base) == 0 or len(times_warm) == 0:
        raise RuntimeError(
            "No successful instances recorded. Check solver backend / data formatting."
        )

    base_p50_t, base_p90_t = p50(times_base), p90(times_base)
    warm_p50_t, warm_p90_t = p50(times_warm), p90(times_warm)
    base_p50_it, base_p90_it = p50(iters_base), p90(iters_base)
    warm_p50_it, warm_p90_it = p50(iters_warm), p90(iters_warm)

    speedup = base_p50_t / max(warm_p50_t, 1e-12)

    return {
        "n_seen": n_seen,
        "n_ok_warm": n_ok,
        "baseline": {
            "time_p50": base_p50_t,
            "time_p90": base_p90_t,
            "iters_p50": base_p50_it,
            "iters_p90": base_p90_it,
        },
        "warm": {
            "time_p50": warm_p50_t,
            "time_p90": warm_p90_t,
            "iters_p50": warm_p50_it,
            "iters_p90": warm_p90_it,
        },
        "speedup_p50": speedup,
    }


# ---------------------------------------------------------------------
# Script entry
# ---------------------------------------------------------------------
def main() -> None:
    out_dir = Path("./results/warm_start")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_file = Path(
        "../configs/finetune/finetune_ALL_200/finetune_ALL_200_mlp_baseline.yml"
    )
    assert config_file.exists(), "Config file should exist."
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        default_config_files=[config_file],
    )

    parser.add_argument("--encoder_path", type=str, help="Path to encoder checkpoint.")
    parser.add_argument(
        "--finetune_mode",
        type=str,
        choices=["full", "heads"],
        default="full",
        help="'full' updates encoder+heads; 'heads' freezes encoder and trains heads only.",
    )
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--max_eval_instances", type=int, default=1000)
    parser.add_argument(
        "--method_tag", type=str, default="MLP", help="Label for outputs/rows."
    )

    # Warm-start options
    parser.add_argument(
        "--warmstart_dual",
        action="store_true",
        help="Also warm-start dual multipliers.",
    )
    parser.add_argument(
        "--solver_backend",
        type=str,
        default="auto",
        choices=["auto", "gurobi", "highs", "pdipm", "scipy"],
        help="LP solver backend. 'auto' picks gurobi->highs->pdipm.",
    )
    parser.add_argument("--solver_max_iter", type=int, default=80)
    parser.add_argument("--solver_tol", type=float, default=1e-8)

    # Data args (matching your project)
    d = parser.add_argument_group("data")
    d.add_argument(
        "--use_bipartite_graphs", action="store_true", help="Must be set for GNNPolicy."
    )
    d.add_argument(
        "--problems", type=str, nargs="+", default=[ProblemClass.COMBINATORIAL_AUCTION]
    )
    d.add_argument("--is_sizes", type=int, nargs="+", default=[200])
    d.add_argument("--ca_sizes", type=int, nargs="+", default=[100])
    d.add_argument("--sc_sizes", type=int, nargs="+", default=[200])
    d.add_argument("--cfl_sizes", type=int, nargs="+", default=[200])
    d.add_argument("--rnd_sizes", type=int, nargs="+", default=[200])
    d.add_argument("--n_instances", type=int, default=35000)
    d.add_argument("--data_root", type=str, default="../data/instances")
    d.add_argument("--val_split", type=float, default=0.15)

    # Model args
    GNNPolicy.add_args(parser)
    KKTNetMLP.add_args(parser)

    args, _ = parser.parse_known_args()

    set_all_seeds(args.seed)
    device = device_from_args(args)
    logger.info(f"Device: {device}")

    # Data loaders
    train_loader, valid_loader, test_loader, N_max, M_max = build_dataloaders(
        args, None, None, for_pretraining=False
    )
    loader = valid_loader if args.split == "val" else test_loader

    # Model (this script assumes the config selects the right architecture)
    model = (
        GNNPolicy(args).to(device)
        if args.use_bipartite_graphs
        else KKTNetMLP(args, M_max, N_max).to(device)
    )
    experiment_dir = Path(
        "/home/joachim-verschelde/Repos/KKT_MPNN/src/experiments/kkt_gnn_node_finetuning_experiments/"
    )

    size = 200
    dataset = "all"
    scenario = "mlp"
    size_dir = experiment_dir / Path(str(size))
    model_dir = list(size_dir.glob(f"*-{dataset}-{scenario}"))[0]
    encoder_path = model_dir / f"epoch_{50:03d}.pt"

    args.encoder_path = str(encoder_path)
    assert encoder_path.exists(), "Encoder path should exist for this experiment"

    pkg = torch.load(encoder_path, map_location="cpu")
    if isinstance(pkg, dict):
        if "model" in pkg:
            model.load_state_dict(pkg["model"], strict=True)
        elif "state_dict" in pkg:
            model.load_state_dict(pkg["state_dict"], strict=True)
    else:
        # last resort
        model.load_state_dict(pkg, strict=True)

    model.eval()

    # Solver
    solver = LPSolver(
        backend=args.solver_backend,
        warmstart_dual=bool(args.warmstart_dual),
        max_iter=int(args.solver_max_iter),
        tol=float(args.solver_tol),
        quiet=True,
    )
    logger.info(
        f"Solver backend resolved to: {solver._resolved} (warmstart_dual={solver.warmstart_dual})"
    )

    # Run
    results = run_warm_start_eval(
        model=model,
        loader=loader,
        solver=solver,
        max_instances=int(args.max_eval_instances)
        if args.max_eval_instances is not None
        else None,
    )

    # Print summary
    base = results["baseline"]
    warm = results["warm"]
    speedup = results["speedup_p50"]

    logger.info(
        f"[Baseline]   time P50={base['time_p50']:.4f}s (P90={base['time_p90']:.4f}), "
        f"iters P50={base['iters_p50']:.0f} (P90={base['iters_p90']:.0f})"
    )
    logger.info(
        f"[Warm-start] time P50={warm['time_p50']:.4f}s (P90={warm['time_p90']:.4f}), "
        f"iters P50={warm['iters_p50']:.0f} (P90={warm['iters_p90']:.0f}), "
        f"speedup(P50)={speedup:.2f}x"
    )

    # Save JSON + LaTeX row snippet
    out_json = out_dir / f"warmstart_{args.method_tag}_{args.split}.json"
    out_json.write_text(json_dumps_pretty(results))
    logger.info(f"Saved: {out_json}")

    latex_row = (
        f"Solver default init & {base['time_p50']:.3f} & {base['iters_p50']:.0f} & -- \\\\\n"
        f"Warm-start ({args.method_tag}) & {warm['time_p50']:.3f} & {warm['iters_p50']:.0f} & {speedup:.2f} \\\\\n"
    )
    out_tex = out_dir / f"warmstart_{args.method_tag}_{args.split}.tex"
    out_tex.write_text(latex_row)
    logger.info(f"Saved: {out_tex}")


def json_dumps_pretty(x: Any) -> str:
    import json

    return json.dumps(x, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
