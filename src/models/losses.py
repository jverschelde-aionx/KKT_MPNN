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
        with autocast(enabled=False):
            B = len(A_list)
            device = x_hat.device

            # Ensure each sparse matrix lives on the right device (no-op if already there).
            A_list = [
                A if A.device == device else A.to(device, non_blocking=True)
                for A in A_list
            ]

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
                )  # (m,)
                At_lambda = _spmm_fp32(A_i.t(), lam_i.unsqueeze(-1)).squeeze(-1)  # (n,)

                # Use sums with denominators max(m,1)/max(n,1) to avoid NaNs on empty segments.
                den_m = float(max(m, 1))
                den_n = float(max(n, 1))

                primal = torch.relu(Ax_minus_b).square().sum() / den_m  # (0,) if m==0
                dual = torch.relu(-lam_i).square().sum() / den_m  # (0,) if m==0
                station = (
                    At_lambda + c_pad_f[i, :n]
                ).square().sum() / den_n  # (0,) if n==0
                compl = (lam_i * Ax_minus_b).square().sum() / den_m  # (0,) if m==0

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

            den_m = float(max(m, 1))
            den_n = float(max(n, 1))

            sum_primal += torch.relu(Ax_minus_b).square().sum().item() / den_m
            sum_dual += torch.relu(-l_i).square().sum().item() / den_m
            sum_station += (At_lambda + c_pad_f[i, :n]).square().sum().item() / den_n
            sum_comp += (l_i * Ax_minus_b).square().sum().item() / den_m

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


class KKTLossGraph(nn.Module):
    """
    Graph-native KKT loss (minimization form).
    Uses edge_index/edge_attr to compute Ax and A^T λ via scatter-add,
    then averages per-instance (not per-element) like the matrix-based loss.
    """

    def __init__(
        self,
        *,
        w_primal: float = 0.1,
        w_dual: float = 0.1,
        w_stat: float = 0.6,
        w_comp: float = 0.2,
        reduction: str = "mean",
    ):
        super().__init__()
        assert reduction in {"mean", "sum", "none"}
        self.w_primal, self.w_dual = w_primal, w_dual
        self.w_stat, self.w_comp = w_stat, w_comp
        self.reduction = reduction

    def forward(
        self,
        x_hat: torch.Tensor,  # (Σ n_i,)
        lam_hat: torch.Tensor,  # (Σ m_i,)
        batch_graph,  # PyG Batch with edge_index (2,E), edge_attr (E,1)
        b_pad: torch.Tensor,  # (B, max_m)
        c_pad: torch.Tensor,  # (B, max_n)
        b_mask: torch.Tensor,  # (B, max_m)
        c_mask: torch.Tensor,  # (B, max_n)
        m_sizes: list[int],  # [m_0, ..., m_{B-1}]
        n_sizes: list[int],  # [n_0, ..., n_{B-1}]
    ) -> torch.Tensor:
        B = len(m_sizes)
        device = x_hat.device
        sum_m = int(sum(m_sizes))
        sum_n = int(sum(n_sizes))

        # Basic shape invariants
        assert x_hat.numel() == sum_n, f"x_hat {x_hat.numel()} vs Σn {sum_n}"
        assert lam_hat.numel() == sum_m, f"lam_hat {lam_hat.numel()} vs Σm {sum_m}"

        # Flatten true b and c in the SAME per-graph order as edge_index uses
        b_cat = b_pad[b_mask].to(device=device, dtype=torch.float32)  # (Σ m_i,)
        c_cat = c_pad[c_mask].to(device=device, dtype=torch.float32)  # (Σ n_i,)

        # Edges
        rows = batch_graph.edge_index[0].to(
            device
        )  # (E,) constraint row ids in [0, Σm)
        cols = batch_graph.edge_index[1].to(device)  # (E,) variable col ids in [0, Σn)
        a = batch_graph.edge_attr.squeeze(-1).to(device)  # (E,) coefficients

        with autocast(enabled=False):  # do the loss math in float32
            x = x_hat.float()
            l = lam_hat.float()
            a = a.float()

            # Ax and A^T λ via scatter-add (index_add_)
            Ax = x.new_zeros(sum_m, dtype=torch.float32)
            Ax.index_add_(0, rows, a * x[cols])  # (Σm,)

            ATlam = l.new_zeros(sum_n, dtype=torch.float32)
            ATlam.index_add_(0, cols, a * l[rows])  # (Σn,)

            Ax_minus_b = Ax - b_cat  # (Σm,)
            primal_vec = torch.relu(Ax_minus_b)  # (Σm,)
            dual_vec = torch.relu(-l)  # (Σm,)
            stat_vec = ATlam + c_cat  # (Σn,)
            compl_vec = l * Ax_minus_b  # (Σm,)

            # Per-instance means (same weighting as your matrix-based version)
            # Build batch ids for rows (constraints) and cols (variables)
            row_batch = torch.repeat_interleave(
                torch.arange(B, device=device, dtype=torch.long),
                torch.tensor(m_sizes, device=device),
            )
            col_batch = torch.repeat_interleave(
                torch.arange(B, device=device, dtype=torch.long),
                torch.tensor(n_sizes, device=device),
            )

            def seg_mean_sq(
                vec: torch.Tensor, seg: torch.Tensor, counts: list[int]
            ) -> torch.Tensor:
                # mean of squares per segment; returns (B,)
                sums = torch.zeros(B, dtype=torch.float32, device=device).index_add_(
                    0, seg, vec.square()
                )
                denom = torch.tensor(
                    counts, dtype=torch.float32, device=device
                ).clamp_min(1.0)
                return sums / denom

            loss_pr = seg_mean_sq(
                primal_vec, row_batch, m_sizes
            ).sum()  # sum over graphs of mean over rows
            loss_du = seg_mean_sq(dual_vec, row_batch, m_sizes).sum()
            loss_st = seg_mean_sq(stat_vec, col_batch, n_sizes).sum()
            loss_cs = seg_mean_sq(compl_vec, row_batch, m_sizes).sum()

            total = (
                self.w_primal * loss_pr
                + self.w_dual * loss_du
                + self.w_stat * loss_st
                + self.w_comp * loss_cs
            ) / float(B)

        if self.reduction == "sum":
            return total * B
        if self.reduction == "none":
            # return per-term tensor for debugging
            return torch.stack([loss_pr, loss_du, loss_st, loss_cs]) / float(B)
        return total


@torch.no_grad()
def kkt_metrics_graph(
    x_hat, lam_hat, batch_graph, b_pad, c_pad, b_mask, c_mask, m_sizes, n_sizes
):
    B = len(m_sizes)
    device = x_hat.device
    sum_m = int(sum(m_sizes))
    sum_n = int(sum(n_sizes))

    b_cat = b_pad[b_mask].to(device=device, dtype=torch.float32)
    c_cat = c_pad[c_mask].to(device=device, dtype=torch.float32)

    rows = batch_graph.edge_index[0].to(device)
    cols = batch_graph.edge_index[1].to(device)
    a = batch_graph.edge_attr.squeeze(-1).to(device).float()

    x = x_hat.float()
    l = lam_hat.float()

    Ax = x.new_zeros(sum_m, dtype=torch.float32)
    Ax.index_add_(0, rows, a * x[cols])

    ATlam = l.new_zeros(sum_n, dtype=torch.float32)
    ATlam.index_add_(0, cols, a * l[rows])

    Ax_minus_b = Ax - b_cat
    primal_vec = torch.relu(Ax_minus_b)
    dual_vec = torch.relu(-l)
    stat_vec = ATlam + c_cat
    compl_vec = l * Ax_minus_b

    row_batch = torch.repeat_interleave(
        torch.arange(B, device=device), torch.tensor(m_sizes, device=device)
    )
    col_batch = torch.repeat_interleave(
        torch.arange(B, device=device), torch.tensor(n_sizes, device=device)
    )

    def seg_mean(vec, seg, counts):
        sums = torch.zeros(B, dtype=torch.float32, device=device).index_add_(
            0, seg, vec.square()
        )
        denom = torch.tensor(counts, dtype=torch.float32, device=device).clamp_min(1.0)
        return (sums / denom).mean().item()

    return {
        "primal": seg_mean(primal_vec, row_batch, m_sizes),
        "dual": seg_mean(dual_vec, row_batch, m_sizes),
        "stationarity": seg_mean(stat_vec, col_batch, n_sizes),
        "compl_slack": seg_mean(compl_vec, row_batch, m_sizes),
    }
