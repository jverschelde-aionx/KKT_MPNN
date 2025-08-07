import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class KKTLoss(nn.Module):
    r"""
    KKT loss for primal-dual optimization problems.
    Parameters
    m : int
        Number of constraints.
    n : int
        Number of variables.
    w_primal, w_dual, w_station, w_comp : float
        Positive coefficients for each term (default 0.1, 0.1, 0.6, 0.2).
    reduction : {'mean', 'sum', 'none'}
        How to aggregate the loss over the batch.
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
        x_hat: torch.Tensor,
        lam_hat: torch.Tensor,
        A_list: list,
        b_pad: torch.Tensor,
        c_pad: torch.Tensor,
        b_mask: torch.Tensor,
        c_mask: torch.Tensor,
        m_sizes: list,
        n_sizes: list,
    ) -> torch.Tensor:
        B = len(A_list)
        device = x_hat.device

        # One‑time, lazy device move for the whole batch
        A_list = [
            A if A.device == device else A.to(device, non_blocking=True) for A in A_list
        ]

        # Accumulators kept as tensors on the same device so the final
        # weighted sum stays on‑graph for autograd.
        loss_primal = x_hat.new_zeros(())
        loss_dual = x_hat.new_zeros(())
        loss_stat = x_hat.new_zeros(())
        loss_comp = x_hat.new_zeros(())

        off_x = off_l = 0
        for i, A_i in enumerate(A_list):
            m, n = m_sizes[i], n_sizes[i]

            x_i = x_hat[off_x : off_x + n]
            lam_i = lam_hat[off_l : off_l + m]
            off_x += n
            off_l += m

            with autocast(enabled=False):  # disable AMP just for the spmm’s
                Ax_minus_b = _spmm_fp32(A_i, x_i.unsqueeze(-1)).squeeze(-1) - b_pad[
                    i, :m
                ].to(device, dtype=x_i.dtype)

                At_lambda = _spmm_fp32(A_i.t(), lam_i.unsqueeze(-1)).squeeze(-1)

            primal = torch.relu(Ax_minus_b).square().mean()
            dual = torch.relu(-lam_i).square().mean()
            station = (At_lambda + c_pad[i, :n].to(device)).square().mean()
            compl = (lam_i * Ax_minus_b).square().mean()

            loss_primal += primal
            loss_dual += dual
            loss_stat += station
            loss_comp += compl

        inv_B = 1.0 / B
        total = (
            self.w_primal * loss_primal * inv_B
            + self.w_dual * loss_dual * inv_B
            + self.w_stat * loss_stat * inv_B
            + self.w_comp * loss_comp * inv_B
        )

        # honour reduction flag
        if self.reduction == "sum":
            return total * B
        if self.reduction == "none":
            # return per‑instance tensor list
            return torch.stack([loss_primal, loss_dual, loss_stat, loss_comp]) * inv_B
        return total  # default 'mean'


@torch.no_grad()
def kkt_metrics(
    x_hat: torch.Tensor,  # (Σ nᵢ,)
    lam_hat: torch.Tensor,  # (Σ mᵢ,)
    A_list: list,  # list[torch.sparse_coo_tensor]
    b_pad: torch.Tensor,  # (B, max_m)
    c_pad: torch.Tensor,  # (B, max_n)
    b_mask: torch.Tensor,  # (B, max_m)
    c_mask: torch.Tensor,  # (B, max_n)
    m_sizes: list,  # [m₀, …, m_{B‑1}]
    n_sizes: list,  # [n₀, …, n_{B‑1}]
) -> dict[str, float]:
    """
    Computes per‑instance KKT residuals and returns batch averages
    **without allocating new CUDA memory in the inner loop**.
    """
    B = len(A_list)
    device = x_hat.device  # everything should live here

    # Move every sparse matrix to the correct device exactly ONCE.
    # .to(device) is a no‑op when already resident, so this is cheap.
    A_list = [
        A if A.device == device else A.to(device, non_blocking=True) for A in A_list
    ]

    sum_primal = sum_dual = sum_station = sum_comp = 0.0
    off_x = off_l = 0

    for i, A_i in enumerate(A_list):
        m, n = m_sizes[i], n_sizes[i]

        x_i = x_hat[off_x : off_x + n]
        lam_i = lam_hat[off_l : off_l + m]
        off_x += n
        off_l += m

        with autocast(enabled=False):
            Ax_minus_b = _spmm_fp32(A_i, x_i.unsqueeze(-1)).squeeze(-1) - b_pad[
                i, :m
            ].to(device, dtype=x_i.dtype)

            At_lambda = _spmm_fp32(A_i.t(), lam_i.unsqueeze(-1)).squeeze(-1)

        primal = torch.relu(Ax_minus_b).square().mean()
        dual = torch.relu(-lam_i).square().mean()
        station = (At_lambda + c_pad[i, :n]).square().mean()
        compl = (lam_i * Ax_minus_b).square().mean()

        sum_primal += primal.item()
        sum_dual += dual.item()
        sum_station += station.item()
        sum_comp += compl.item()

    inv_B = 1.0 / B
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
