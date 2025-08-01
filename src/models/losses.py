import torch
import torch.nn as nn
import torch.nn.functional as F


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
        w_station: float = 0.6,
        w_comp: float = 0.2,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.m, self.n = m, n
        self.w_primal, self.w_dual = w_primal, w_dual
        self.w_station, self.w_comp = w_station, w_comp
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(
        self, x_hat, lam_hat, A_list, b_pad, c_pad, b_mask, c_mask, m_sizes, n_sizes
    ):
        """
        Computes four KKT terms for every instance *i*, then averages.

        • Ax – b   : use torch.sparse.mm
        • stationarity: Aᵀ λ + c
        • primal/dual feasibility & complementarity
        """
        B = len(A_list)
        loss_primal = loss_dual = loss_stat = loss_comp = 0.0

        offset_x = 0
        offset_l = 0
        for i in range(B):
            m, n = m_sizes[i], n_sizes[i]

            x_i = x_hat[offset_x : offset_x + n]
            lam_i = lam_hat[offset_l : offset_l + m]
            A_i = A_list[i].to(x_i.device)

            offset_x += n
            offset_l += m

            # Ax - b
            Ax_minus_b = (A_i @ x_i.unsqueeze(-1)).squeeze(-1) - b_pad[i, :m]

            primal = torch.relu(Ax_minus_b).pow(2).mean()
            dual = torch.relu(-lam_i).pow(2).mean()

            At_lambda = (A_i.t() @ lam_i.unsqueeze(-1)).squeeze(-1)
            station = (c_pad[i, :n] + At_lambda).pow(2).mean()
            compl = (lam_i * Ax_minus_b).pow(2).mean()

            loss_primal += primal
            loss_dual += dual
            loss_stat += station
            loss_comp += compl

        loss_primal /= B
        loss_dual /= B
        loss_stat /= B
        loss_comp /= B

        total = (
            self.w_primal * loss_primal
            + self.w_dual * loss_dual
            + self.w_stat * loss_stat
            + self.w_comp * loss_comp
        )
        return total


# class AdaptiveKKTLoss(KKTLoss):
#     def __init__(self, m: int, n: int, init: float = 0.1):
#         super().__init__(m, n, w_primal=init, w_dual=init, w_station=init, w_comp=init)
#         # register as parameters so they get optimised
#         self.w_primal = nn.Parameter(torch.tensor(init))
#         self.w_dual = nn.Parameter(torch.tensor(init))
#         self.w_station = nn.Parameter(torch.tensor(init))
#         self.w_comp = nn.Parameter(torch.tensor(init))
