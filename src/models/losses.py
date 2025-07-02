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
        self,
        y_pred: torch.Tensor,  # (B, n+m) = [x̂ ‖ λ̂]
        A: torch.Tensor,  # (B, m, n)
        b: torch.Tensor,  # (B, m)
        c: torch.Tensor,  # (n,) or (B, n)
    ) -> torch.Tensor:
        B = y_pred.size(0)
        if y_pred.size(1) != self.n + self.m:
            raise RuntimeError(
                f"y_pred second dim={y_pred.size(1)} but expected {self.n + self.m}"
            )

        # split prediction
        x_hat: torch.Tensor = y_pred[:, : self.n]  # (B, n)
        lam_hat: torch.Tensor = y_pred[:, self.n :]  # (B, m)

        # broadcast / reshape helpers
        Ax: torch.Tensor = torch.bmm(A, x_hat.unsqueeze(-1)).squeeze(-1)  # (B, m)
        c_exp: torch.Tensor = c if c.dim() == 2 else c.unsqueeze(0).expand(B, -1)

        # primal feasibility
        primal_res = Ax - b  # (B, m)
        primal_term = F.relu(primal_res).pow(2).mean(dim=1)  # (B,)

        # dual feasibility
        dual_term = F.relu(-lam_hat).pow(2).mean(dim=1)  # (B,)

        # stationarity
        At_lambda = torch.bmm(A.transpose(1, 2), lam_hat.unsqueeze(-1)).squeeze(-1)
        station_term = (c_exp + At_lambda).pow(2).mean(dim=1)  # (B,)

        # complementary slackness
        comp_term = (lam_hat * primal_res).pow(2).mean(dim=1)  # (B,)

        # weighted sum of terms
        loss_vec = (
            self.w_primal * primal_term
            + self.w_dual * dual_term
            + self.w_station * station_term
            + self.w_comp * comp_term
        )  # (B,)

        if self.reduction == "mean":
            return loss_vec.mean()
        if self.reduction == "sum":
            return loss_vec.sum()
        return loss_vec  # 'none'


class AdaptiveKKTLoss(KKTLoss):
    def __init__(self, m: int, n: int, init: float = 0.1):
        super().__init__(m, n, w_primal=init, w_dual=init, w_station=init, w_comp=init)
        # register as parameters so they get optimised
        self.w_primal = nn.Parameter(torch.tensor(init))
        self.w_dual = nn.Parameter(torch.tensor(init))
        self.w_station = nn.Parameter(torch.tensor(init))
        self.w_comp = nn.Parameter(torch.tensor(init))
