import torch


def get_primal_feasibility(
    x_pred: torch.Tensor, A: torch.Tensor, b: torch.Tensor, mask_m: torch.Tensor
) -> torch.Tensor:
    # Primal feasibility: relu(Ax - b)
    x_col = x_pred.unsqueeze(-1)  # [B, n, 1]
    Ax = torch.bmm(A, x_col).squeeze(-1)  # [B, m]
    primal_res = torch.relu(Ax - b)  # [B, m]
    # Mask constraints
    primal_res = primal_res * mask_m
    primal_feasibility = (primal_res**2).sum(dim=1) / (
        mask_m.sum(dim=1).clamp_min(1.0)
    )  # mean over real m
    return primal_feasibility


def get_dual_feasibility(
    lambda_pred: torch.Tensor, mask_m: torch.Tensor
) -> torch.Tensor:
    dual_feasibility = torch.relu(-lambda_pred)
    dual_feasibility = ((dual_feasibility**2) * mask_m).sum(dim=1) / mask_m.sum(
        dim=1
    ).clamp_min(1.0)
    return dual_feasibility


def get_complementary_slackness(
    x_pred: torch.Tensor,
    lambda_pred: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    mask_m: torch.Tensor,
):
    x_col = x_pred.unsqueeze(-1)  # [B, n, 1]
    Ax = torch.bmm(A, x_col).squeeze(-1)  # [B, m]
    complementary_slackness = lambda_pred * (Ax - b)  # [B, m]
    complementary_slackness = ((complementary_slackness**2) * mask_m).sum(
        dim=1
    ) / mask_m.sum(dim=1).clamp_min(1.0)


def get_stationarity(
    lambda_pred: torch.Tensor,
    A: torch.Tensor,
    c: torch.Tensor,
    mask_n: torch.Tensor,
):
    # Stationarity: c + A^T * lambda = 0
    lam_col = lambda_pred.unsqueeze(-1)  # [B, m, 1]
    ATlam = torch.bmm(A.transpose(1, 2), lam_col).squeeze(-1)  # [B, n]
    stationarity = c + ATlam  # [B, n]
    stationarity = ((stationarity**2) * mask_n).sum(dim=1) / mask_n.sum(
        dim=1
    ).clamp_min(1.0)


def kkt_loss(
    y_pred: torch.Tensor,  # [B, n+m]
    A: torch.Tensor,  # [B, m, n]
    b: torch.Tensor,  # [B, m]
    c: torch.Tensor,  # [B, n]
    mask_m: torch.Tensor,
    mask_n: torch.Tensor,
    primal_weight: float,
    dual_weight: float,
    stationarity_weight: float,
    complementary_slackness_weight: float,
):
    """
    Computes weighted KKT residuals. Supports padded batches via masks.
    """
    B, m, n = A.shape

    x_pred = y_pred[:, :n]  # [B, n]
    lambda_pred = y_pred[:, n:]  # [B, m]

    primal_feasibility = get_primal_feasibility(x_pred, A, b, mask_m)
    dual_feasibility = get_dual_feasibility(lambda_pred, mask_m)
    stationarity = get_stationarity(lambda_pred, A, c, mask_m)
    complementary_slackness = get_complementary_slackness(
        x_pred, lambda_pred, A, b, mask_m
    )

    kkt = (
        primal_weight * primal_feasibility
        + dual_weight * dual_feasibility
        + stationarity_weight * stationarity
        + complementary_slackness_weight * complementary_slackness
    ).mean()

    return kkt
