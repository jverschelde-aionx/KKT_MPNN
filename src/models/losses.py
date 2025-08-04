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

            Ax_minus_b = torch.sparse.mm(A_i, x_i.unsqueeze(-1)).squeeze(-1) - b_pad[
                i, :m
            ].to(device)

            primal = torch.relu(Ax_minus_b).square().mean()
            dual = torch.relu(-lam_i).square().mean()
            station = (
                (
                    torch.sparse.mm(A_i.t(), lam_i.unsqueeze(-1)).squeeze(-1)
                    + c_pad[i, :n].to(device)
                )
                .square()
                .mean()
            )
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
