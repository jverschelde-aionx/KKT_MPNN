import torch

from models.models import GNNPolicy, KKTNetMLP


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


def lejepa_pred_loss(emb_all, emb_global):
    # emb_*: list of [B, D] tensors
    centers = torch.stack(emb_global, 0).mean(0)  # [B, D]
    return torch.stack([((z - centers) ** 2).mean() for z in emb_all]).mean()


def lejepa_loss_mlp(model: KKTNetMLP, x_globals, x_all, sigreg, lambd=0.05):
    """
    LeJEPA loss for MLP architecture.

    Args:
        model: KKTNetMLP with LeJEPA components
        x_globals: List of global views (lightly masked)
        x_all: List of all views (globals + locals)
        sigreg: SIGReg regularizer
        lambd: Weight for regularization term (default: 0.05)

    Returns:
        Scalar loss: (1-λ) * L_pred + λ * L_reg
    """
    g = [model.lejepa_embed(x) for x in x_globals]  # lists of [B,D], no normalization
    a = [model.lejepa_embed(x) for x in x_all]
    l_pred = lejepa_pred_loss(a, g)
    l_reg = torch.stack([sigreg(z) for z in a]).mean()
    return (1 - lambd) * l_pred + lambd * l_reg


def lejepa_loss_gnn(model: GNNPolicy, globals_, alls_, sigreg, lambd=0.05):
    """
    LeJEPA loss for GNN architecture.

    Args:
        model: GNNPolicy with LeJEPA components
        globals_: List of global views (lightly masked graphs)
        alls_: List of all views (globals + locals)
        sigreg: SIGReg regularizer
        lambd: Weight for regularization term (default: 0.05)

    Returns:
        Scalar loss: (1-λ) * L_pred + λ * L_reg
    """
    # globals_/alls_ are lists of PyG Batch objects (same batch order/size)
    g = [
        model.lejepa_embed_graph(
            g.constraint_features, g.edge_index, g.edge_attr, g.variable_features
        )
        for g in globals_
    ]  # each [B, D] after pooling per graph
    a = [
        model.lejepa_embed_graph(
            g.constraint_features, g.edge_index, g.edge_attr, g.variable_features
        )
        for g in alls_
    ]
    # stack to [B, D] for each view
    g = [x.squeeze(1) if x.dim() == 3 else x for x in g]
    a = [x.squeeze(1) if x.dim() == 3 else x for x in a]
    l_pred = lejepa_pred_loss(a, g)
    l_reg = torch.stack([sigreg(z) for z in a]).mean()
    return (1 - lambd) * l_pred + lambd * l_reg
