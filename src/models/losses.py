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
        x_hat: torch.Tensor,  # (Σ nᵢ,)
        lam_hat: torch.Tensor,  # (Σ mᵢ,)
        A_list: list,  # list[torch.sparse_coo_tensor] with shapes (mᵢ, nᵢ)
        b_pad: torch.Tensor,  # (B, max_m)
        c_pad: torch.Tensor,  # (B, max_n)
        b_mask: torch.Tensor,  # (B, max_m)  (unused; we slice by mᵢ)
        c_mask: torch.Tensor,  # (B, max_n)  (unused; we slice by nᵢ)
        m_sizes: list,  # [m₀, …, m_{B‑1}]
        n_sizes: list,  # [n₀, …, n_{B‑1}]
    ) -> torch.Tensor:
        """
        Compute the weighted KKT loss across a batch of independent problems.
        The computation is done in float32 for numerical stability under AMP.
        """
        B = len(A_list)
        device = x_hat.device

        # Ensure each sparse matrix lives on the right device (no-op if already there).
        A_list = [
            A if A.device == device else A.to(device, non_blocking=True) for A in A_list
        ]

        # Compute the loss in FP32 to avoid overflow/underflow with AMP.
        with autocast(enabled=False):
            x_hat_f = x_hat.float()
            lam_hat_f = lam_hat.float()
            b_pad_f = b_pad.to(device=device, dtype=torch.float32)
            c_pad_f = c_pad.to(device=device, dtype=torch.float32)

            # Accumulators kept as 0-dim tensors so the result stays on-graph.
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

                # Sparse matmuls in FP32; helper returns in x_i.dtype (FP32 here).
                Ax_minus_b = (
                    _spmm_fp32(A_i, x_i.unsqueeze(-1)).squeeze(-1) - b_pad_f[i, :m]
                )
                At_lambda = _spmm_fp32(A_i.t(), lam_i.unsqueeze(-1)).squeeze(-1)

                # KKT residual terms
                primal = torch.relu(Ax_minus_b).square().mean()
                dual = torch.relu(-lam_i).square().mean()
                station = (At_lambda + c_pad_f[i, :n]).square().mean()
                compl = (lam_i * Ax_minus_b).square().mean()

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

        # Honour reduction flag
        if self.reduction == "sum":
            return total * B
        if self.reduction == "none":
            # Returns per-term batch averages (same semantics as your original code).
            return torch.stack([loss_primal, loss_dual, loss_stat, loss_comp]) * (
                1.0 / float(B)
            )
        return total  # default: 'mean'


@torch.no_grad()
def kkt_metrics(
    x_hat: torch.Tensor,  # (Σ nᵢ,)
    lam_hat: torch.Tensor,  # (Σ mᵢ,)
    A_list: list,  # list[torch.sparse_coo_tensor] with shapes (mᵢ, nᵢ)
    b_pad: torch.Tensor,  # (B, max_m)
    c_pad: torch.Tensor,  # (B, max_n)
    b_mask: torch.Tensor,  # (B, max_m)  (unused; we slice by mᵢ)
    c_mask: torch.Tensor,  # (B, max_n)  (unused; we slice by nᵢ)
    m_sizes: list,  # [m₀, …, m_{B‑1}]
    n_sizes: list,  # [n₀, …, n_{B‑1}]
) -> dict[str, float]:
    """
    Compute average KKT residual terms over the batch in FP32 (AMP‑safe),
    while avoiding unnecessary device transfers or allocations.
    """
    B = len(A_list)
    device = x_hat.device

    # Ensure all sparse matrices live on the right device (no-op if already there).
    A_list = [
        A if A.device == device else A.to(device, non_blocking=True) for A in A_list
    ]

    sum_primal = 0.0
    sum_dual = 0.0
    sum_station = 0.0
    sum_comp = 0.0

    off_x = 0
    off_l = 0

    # Do the whole computation in FP32 to be consistent with the loss.
    with autocast(enabled=False):
        x_hat_f = x_hat.float()
        lam_hat_f = lam_hat.float()
        b_pad_f = b_pad.to(device=device, dtype=torch.float32)
        c_pad_f = c_pad.to(device=device, dtype=torch.float32)

        for i, A_i in enumerate(A_list):
            m = int(m_sizes[i])
            n = int(n_sizes[i])

            x_i = x_hat_f[off_x : off_x + n]  # (n,)
            lam_i = lam_hat_f[off_l : off_l + m]  # (m,)
            off_x += n
            off_l += m

            # Sparse matmuls in FP32.
            Ax_minus_b = _spmm_fp32(A_i, x_i.unsqueeze(-1)).squeeze(-1) - b_pad_f[i, :m]
            At_lambda = _spmm_fp32(A_i.t(), lam_i.unsqueeze(-1)).squeeze(-1)

            # Per‑instance KKT residuals (as scalars)
            primal = torch.relu(Ax_minus_b).square().mean().item()
            dual = torch.relu(-lam_i).square().mean().item()
            station = (At_lambda + c_pad_f[i, :n]).square().mean().item()
            compl = (lam_i * Ax_minus_b).square().mean().item()

            sum_primal += primal
            sum_dual += dual
            sum_station += station
            sum_comp += compl

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
