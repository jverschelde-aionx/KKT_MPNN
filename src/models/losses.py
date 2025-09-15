import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class KKTLoss(nn.Module):
    r"""
    KKT loss for LPs in minimization form:
        min c^T x  s.t.  A x <= b,  x >= 0
    Includes lower-bound multipliers μ for -x <= 0 and supports Gurobi Pi sign.

    Parameters
    ----------
    m : int
        Number of constraints.
    n : int
        Number of variables.
    w_primal, w_dual, w_stat, w_comp : float
        Positive coefficients for each term (default 0.1, 0.1, 0.6, 0.2).
    reduction : {'mean', 'sum', 'none'}
        Aggregation over batch.
    lam_is_gurobi_pi : bool
        If True, lam_hat is solver Pi for min with <= (non-positive); we flip its sign.
    include_lb : bool
        If True, enforce the x >= 0 bound via μ and add corresponding residuals.
    """

    def __init__(
        self,
        m: int,
        n: int,
        *,
        w_primal: float = 0.1,
        w_dual: float = 0.1,
        w_stat: float = 0.6,
        w_comp: float = 0.2,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.m, self.n = m, n
        self.w_primal, self.w_dual = w_primal, w_dual
        self.w_stat, self.w_comp = w_stat, w_comp
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(
        self,
        x_hat: torch.Tensor,  # (Σ nᵢ,)
        lam_hat: torch.Tensor,  # (Σ mᵢ,)
        A_list: list,  # list[torch.sparse_coo_tensor] with shapes (mᵢ, nᵢ)
        b_pad: torch.Tensor,  # (B, max_m)
        c_pad: torch.Tensor,  # (B, max_n)
        m_sizes: list,  # [m₀, …, m_{B‑1}]
        n_sizes: list,  # [n₀, …, n_{B‑1}]
    ) -> torch.Tensor:
        """
        Compute weighted KKT residual across a batch (float32 under AMP).
        """
        with autocast(enabled=False):
            B = len(A_list)
            device = x_hat.device

            # Ensure each sparse matrix is on the right device.
            A_list = [
                A if A.device == device else A.to(device, non_blocking=True)
                for A in A_list
            ]

            x_hat_f = x_hat.float()
            lam_hat_f = lam_hat.float()
            b_pad_f = b_pad.to(device=device, dtype=torch.float32)
            c_pad_f = c_pad.to(device=device, dtype=torch.float32)

            # Accumulate as 0-dim tensors (on-graph).
            loss_primal = x_hat_f.new_zeros(())
            loss_dual = x_hat_f.new_zeros(())
            loss_stat = x_hat_f.new_zeros(())
            loss_comp = x_hat_f.new_zeros(())

            total_n = sum(n_sizes)
            total_m = sum(m_sizes)
            if x_hat.numel() != total_n or lam_hat.numel() != total_m:
                raise RuntimeError(
                    f"Length mismatch: x_hat={x_hat.numel()} vs sum(n)={total_n}, "
                    f"lam_hat={lam_hat.numel()} vs sum(m)={total_m}"
                )

            off_x = 0
            off_l = 0
            for i, A_i in enumerate(A_list):
                m = int(m_sizes[i])
                n = int(n_sizes[i])

                x_i = x_hat_f[off_x : off_x + n]  # (n,)
                lam_i = lam_hat_f[off_l : off_l + m]  # (m,)

                if x_i.numel() != n or lam_i.numel() != m:
                    raise RuntimeError(
                        f"Empty/misaligned slice at i={i}: "
                        f"[off_x={off_x}, n={n}, xi.numel()={x_i.numel()}] "
                        f"[off_l={off_l}, m={m}, li.numel()={lam_i.numel()}]"
                    )
                if A_i.size() != (m, n):
                    raise RuntimeError(
                        f"A[{i}].size={tuple(A_i.size())} != (m={m}, n={n})"
                    )

                off_x += n
                off_l += m

                # Sparse matmuls in FP32
                Ax_minus_b = (
                    _spmm_fp32(A_i, x_i.unsqueeze(-1)).squeeze(-1) - b_pad_f[i, :m]
                )  # (m,)
                At_lambda = _spmm_fp32(A_i.t(), lam_i.unsqueeze(-1)).squeeze(-1)  # (n,)
                c_i = c_pad_f[i, :n]  # (n,)

                # Lower-bound multipliers μ := max(0, c + A^T λ)
                mu_i = torch.relu(c_i + At_lambda)  # (n,)

                # Denominators
                den_n = float(max(n, 1))
                den_tot = float(max(m + (n), 1))

                # ---- Primal feasibility: g_ineq = Ax-b <= 0, g_lb = -x <= 0
                p_ineq = torch.relu(Ax_minus_b).square().sum()
                p_lb = torch.relu(-x_i).square().sum()
                primal = (p_ineq + p_lb) / den_tot

                # ---- Dual feasibility: λ >= 0, μ >= 0
                d_lam = torch.relu(-lam_i).square().sum()
                d_mu = torch.relu(-mu_i).square().sum()  # =0 when μ is built via relu
                dual = (d_lam + d_mu) / den_tot

                # ---- Stationarity: c + A^T λ - μ = 0
                station = (At_lambda + c_i - mu_i).square().sum() / den_n

                # ---- Complementary slackness: λ ⊙ (Ax-b) = 0, μ ⊙ x = 0
                c_ineq = (lam_i * Ax_minus_b).square().sum()
                c_lb = (mu_i * x_i).square().sum()
                compl = (c_ineq + c_lb) / den_tot

                loss_primal = loss_primal + primal
                loss_dual = loss_dual + dual
                loss_stat = loss_stat + station
                loss_comp = loss_comp + compl

            inv_B = 1.0 / float(B)
            total = (
                self.w_primal * loss_primal * inv_B
                + self.w_dual * loss_dual * inv_B
                + self.w_stat * loss_stat * inv_B
                + self.w_comp * loss_comp * inv_B
            )

            if self.reduction == "sum":
                return total * B
            if self.reduction == "none":
                return (
                    torch.stack([loss_primal, loss_dual, loss_stat, loss_comp]) * inv_B
                )
            return total


@torch.no_grad()
def kkt_metrics(
    x_hat: torch.Tensor,
    lam_hat: torch.Tensor,
    A_list: list,
    b_pad: torch.Tensor,
    c_pad: torch.Tensor,
    m_sizes: list,
    n_sizes: list,
) -> dict[str, float]:
    B = len(A_list)
    device = x_hat.device
    A_list = [
        A if A.device == device else A.to(device, non_blocking=True) for A in A_list
    ]

    sum_primal = 0.0
    sum_dual = 0.0
    sum_station = 0.0
    sum_comp = 0.0

    off_x = 0
    off_l = 0
    with autocast(enabled=False):
        x_hat_f = x_hat.float()
        lam_hat_f = lam_hat.float()
        b_pad_f = b_pad.to(device=device, dtype=torch.float32)
        c_pad_f = c_pad.to(device=device, dtype=torch.float32)

        for i, A_i in enumerate(A_list):
            m = int(m_sizes[i])
            n = int(n_sizes[i])
            x_i = x_hat_f[off_x : off_x + n]
            l_i = lam_hat_f[off_l : off_l + m]
            off_x += n
            off_l += m

            Ax_minus_b = _spmm_fp32(A_i, x_i.unsqueeze(-1)).squeeze(-1) - b_pad_f[i, :m]
            At_lambda = _spmm_fp32(A_i.t(), l_i.unsqueeze(-1)).squeeze(-1)
            c_i = c_pad_f[i, :n]
            mu_i = torch.relu(c_i + At_lambda)

            den_n = float(max(n, 1))
            den_tot = float(max(m + (n), 1))

            p_ineq = torch.relu(Ax_minus_b).square().sum().item()
            p_lb = torch.relu(-x_i).square().sum().item()
            sum_primal += (p_ineq + p_lb) / den_tot

            d_lam = torch.relu(-l_i).square().sum().item()
            d_mu = torch.relu(-mu_i).square().sum().item()
            sum_dual += (d_lam + d_mu) / den_tot

            sum_station += (At_lambda + c_i - mu_i).square().sum().item() / den_n

            c_ineq = (l_i * Ax_minus_b).square().sum().item()
            c_lb = (mu_i * x_i).square().sum().item()
            sum_comp += (c_ineq + c_lb) / den_tot

    inv_B = 1.0 / float(B)
    return {
        "primal": sum_primal * inv_B,
        "dual": sum_dual * inv_B,
        "stationarity": sum_station * inv_B,
        "compl_slack": sum_comp * inv_B,
    }


def _spmm_fp32(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes A @ x in float32 because CUDA sparse kernels
    do not support fp16/bf16 yet.  The result is cast back
    to x.dtype so the surrounding computation stays in AMP.
    """
    out = torch.sparse.mm(
        A.float(),  # (m × n)  FP32
        x.float(),  # (n × k)  FP32
    )
    return out.to(x.dtype)  # back to fp16/bf16 if needed
