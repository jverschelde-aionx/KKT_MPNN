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
    return complementary_slackness


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
    return stationarity


def surrogate_loss(
    x_pred: torch.Tensor,  # [B, n] — relaxed selection probabilities p
    A: torch.Tensor,       # [B, m, n]
    b: torch.Tensor,       # [B, m]
    c: torch.Tensor,       # [B, n]
    mask_m: torch.Tensor,  # [B, m]
    mask_n: torch.Tensor,  # [B, n]
    alpha: float = 10.0,
    delta: float = 1.0,
    beta: float = 0.0,
    maximize: bool = False,
):
    """
    Differentiable surrogate loss for training scores without discrete decisions.

    Components:
        (a) L_viol = ||max(0, Ap - b)||^2  — constraint violation penalty
        (b) L_obj  = c^T p (or -c^T p if maximize)  — objective term
        (c) L_int  = mean(p_j * (1 - p_j))  — integrality pressure (pushes away from 0.5)

    Total: L = alpha * L_viol + delta * L_obj + beta * L_int

    Args:
        alpha: Weight for feasibility (constraint violation). Default 10.0.
        delta: Weight for objective term. Default 1.0.
        beta: Weight for integrality pressure. Should be ramped from 0 during training.
        maximize: If True, negate the objective term (for maximization problems).
    """
    # (a) Constraint violation: v = max(0, Ap - b), L_viol = ||v||^2
    p_col = x_pred.unsqueeze(-1)       # [B, n, 1]
    Ap = torch.bmm(A, p_col).squeeze(-1)  # [B, m]
    violation = torch.relu(Ap - b)     # [B, m]
    violation = violation * mask_m     # zero out padding
    L_viol = (violation ** 2).sum(dim=1) / mask_m.sum(dim=1).clamp_min(1.0)  # [B]

    # (b) Objective term: L_obj = c^T p  (masked)
    obj = (c * x_pred * mask_n).sum(dim=1) / mask_n.sum(dim=1).clamp_min(1.0)  # [B]
    if maximize:
        obj = -obj
    L_obj = obj

    # (c) Integrality pressure: L_int = (1/n) sum_j p_j (1 - p_j)
    integrality = x_pred * (1.0 - x_pred) * mask_n  # [B, n]
    L_int = integrality.sum(dim=1) / mask_n.sum(dim=1).clamp_min(1.0)  # [B]

    # Total surrogate loss
    loss = (alpha * L_viol + delta * L_obj + beta * L_int).mean()

    return loss, {
        "surrogate_viol": (alpha * L_viol).mean(),
        "surrogate_obj": (delta * L_obj).mean(),
        "surrogate_int": (beta * L_int).mean(),
    }


def get_surrogate_beta(epoch: int, total_epochs: int, beta_final: float, warmup_frac: float = 0.3) -> float:
    """
    Compute ramped beta for integrality pressure.

    Schedule:
        epochs 0..warmup_end: beta = 0
        epochs warmup_end..total_epochs: linearly ramp to beta_final

    Args:
        epoch: Current epoch (0-indexed).
        total_epochs: Total number of training epochs.
        beta_final: Target beta value at end of training.
        warmup_frac: Fraction of epochs with beta=0 (default 0.3 = 30%).
    """
    warmup_end = int(total_epochs * warmup_frac)
    if epoch < warmup_end:
        return 0.0
    ramp_epochs = total_epochs - warmup_end
    if ramp_epochs <= 0:
        return beta_final
    progress = (epoch - warmup_end) / ramp_epochs
    return beta_final * min(progress, 1.0)


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
    stationarity = get_stationarity(lambda_pred, A, c, mask_n)
    complementary_slackness = get_complementary_slackness(
        x_pred, lambda_pred, A, b, mask_m
    )

    weighted_primal = primal_weight * primal_feasibility
    weighted_dual = dual_weight * dual_feasibility
    weighted_stat = stationarity_weight * stationarity
    weighted_comp = complementary_slackness_weight * complementary_slackness

    loss = (weighted_primal + weighted_dual + weighted_stat + weighted_comp).mean()

    return loss, {
        "primal_feasibility": weighted_primal.mean(),
        "dual_feasibility": weighted_dual.mean(),
        "stationarity": weighted_stat.mean(),
        "complementary_slackness": weighted_comp.mean(),
    }
