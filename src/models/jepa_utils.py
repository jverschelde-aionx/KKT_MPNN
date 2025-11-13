"""
JEPA (Joint-Embedding Predictive Architecture) utility functions for self-supervised learning.

This module implements core JEPA functionality including:
- EMA (Exponential Moving Average) parameter updates
- Cosine similarity loss for prediction
- LP-aware masking strategies for MLP models (structure-preserving masking)
- Node-level masking for GNN models
- JEPA loss computation for both architectures

References:
- I-JEPA: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (Assran et al., CVPR 2023)
- BYOL: "Bootstrap Your Own Latent" (Grill et al., NeurIPS 2020)
- SimSiam: "Exploring Simple Siamese Representation Learning" (Chen & He, CVPR 2021)
"""

from typing import Tuple

import torch
import torch.nn.functional as F
import torch_geometric


def ema_update(target_model: torch.nn.Module, online_model: torch.nn.Module, m: float = 0.996):
    """
    Update target model parameters using Exponential Moving Average (EMA).

    Target parameters are updated as: θ_target ← m * θ_target + (1 - m) * θ_online

    Args:
        target_model: The target/teacher model (updated via EMA)
        online_model: The online/student model (updated via backprop)
        m: EMA momentum coefficient (default: 0.996)
           Higher values → slower updates, more stable target
           Lower values → faster updates, less stable

    Note: target_model parameters should have requires_grad=False
    """
    with torch.no_grad():
        for param_target, param_online in zip(target_model.parameters(), online_model.parameters()):
            param_target.data.mul_(m).add_(param_online.data, alpha=1.0 - m)


def cosine_pred_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity loss for JEPA prediction.

    Loss = 2 - 2 * cos(pred, target)

    This formulation:
    - Returns 0 when pred == target (perfect alignment)
    - Returns 4 when pred == -target (worst case, opposite directions)
    - Is smooth and differentiable everywhere

    Args:
        pred: Predicted embeddings from online path (after predictor), shape [B, D]
        target: Target embeddings from target path (after projector), shape [B, D]
               Should be detached (stop-gradient)

    Returns:
        Scalar loss averaged over batch

    Note: Both pred and target should be L2-normalized before calling this function
    """
    # Cosine similarity: -1 (opposite) to +1 (aligned)
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    # Transform to loss: 0 (perfect) to 4 (worst)
    loss = 2.0 - 2.0 * cos_sim
    return loss.mean()


def make_lp_jepa_views(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    mask_m: torch.Tensor,
    mask_n: torch.Tensor,
    r_entry_on: float = 0.40,
    r_row_on: float = 0.20,
    r_col_on: float = 0.20,
    r_entry_tg: float = 0.10,
    r_row_tg: float = 0.05,
    r_col_tg: float = 0.05,
    noisy_mask: bool = False,
    row_scaling: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create two asymmetric views of LP instances with LP-aware structured masking.

    This function respects the structure of Linear Programming problems by tying masks
    to semantic units (constraints and variables):
    - Row masking: Masks entire constraint (A[i,:] and b[i])
    - Column masking: Masks entire variable (A[:,j] and c[j])
    - Entry masking: Masks individual A[i,j] coefficients

    The masking strategy forces the model to learn structural patterns:
    - Infer constraint relationships from visible constraints
    - Infer variable relationships from visible variables
    - Handle sparse/incomplete problem representations

    Args:
        A: Constraint matrix [B, M, N]
        b: RHS vector [B, M]
        c: Objective coefficients [B, N]
        mask_m: Real number of constraints per sample [B] (for padding safety)
        mask_n: Real number of variables per sample [B] (for padding safety)
        r_entry_on: Online view - fraction of A entries to mask (default: 0.40)
        r_row_on: Online view - fraction of constraint rows to mask (default: 0.20)
        r_col_on: Online view - fraction of variable columns to mask (default: 0.20)
        r_entry_tg: Target view - fraction of A entries to mask (default: 0.10, or 0 for clean)
        r_row_tg: Target view - fraction of constraint rows to mask (default: 0.05, or 0 for clean)
        r_col_tg: Target view - fraction of variable columns to mask (default: 0.05, or 0 for clean)
        noisy_mask: If True, use Gaussian noise at masked positions; if False, use zeros (default: False)
        row_scaling: If True, apply row scaling augmentation s_i ~ LogUniform(0.5, 2.0) (default: False)

    Returns:
        (x_online, x_target): Tuple of flattened inputs [B, M*N+M+N]
        - x_online: Heavily masked view (context) for online encoder
        - x_target: Lightly masked or clean view for target encoder

    Masking composition: M_A = M_row ∨ M_col ∨ M_entry

    Safety guarantees:
    - Only masks within real region (respects mask_m, mask_n)
    - Always keeps ≥1 unmasked row AND ≥1 unmasked column
    - Maintains LP semantic coherence through tied masking

    Example:
        If row i is masked: A[i,:] = masked AND b[i] = masked (entire constraint hidden)
        If column j is masked: A[:,j] = masked AND c[j] = masked (entire variable hidden)
    """
    device = A.device
    B, M, N = A.shape

    def create_view(r_entry, r_row, r_col):
        """Create a single masked view with given masking ratios."""
        # Start with copies
        A_view = A.clone()
        b_view = b.clone()
        c_view = c.clone()

        # Apply row scaling augmentation if enabled (before masking)
        if row_scaling and r_row > 0:
            # Sample scaling factors s_i ~ LogUniform(0.5, 2.0)
            # LogUniform: log(s) ~ Uniform(log(0.5), log(2.0))
            log_scales = torch.rand(B, M, 1, device=device) * (torch.log(torch.tensor(2.0)) - torch.log(torch.tensor(0.5))) + torch.log(torch.tensor(0.5))
            scales = torch.exp(log_scales)  # [B, M, 1]
            A_view = A_view * scales  # Scale A rows
            b_view = b_view * scales.squeeze(-1)  # Scale b values

        # Process each sample in batch
        for i in range(B):
            m_real = int(mask_m[i].item())  # Real number of constraints
            n_real = int(mask_n[i].item())  # Real number of variables

            # Safety check: need at least 2 rows and 2 columns to guarantee context
            if m_real < 2 or n_real < 2:
                continue  # Skip masking for tiny problems

            # --- Row masking (constraints) ---
            n_rows_to_mask = max(0, min(int(r_row * m_real), m_real - 1))  # Keep at least 1 row
            if n_rows_to_mask > 0:
                row_indices = torch.randperm(m_real, device=device)[:n_rows_to_mask]
                # Mask entire constraint: A[i,:] and b[i]
                A_view[i, row_indices, :n_real] = 0.0
                b_view[i, row_indices] = 0.0

            # --- Column masking (variables) ---
            n_cols_to_mask = max(0, min(int(r_col * n_real), n_real - 1))  # Keep at least 1 column
            if n_cols_to_mask > 0:
                col_indices = torch.randperm(n_real, device=device)[:n_cols_to_mask]
                # Mask entire variable: A[:,j] and c[j]
                A_view[i, :m_real, col_indices] = 0.0
                c_view[i, col_indices] = 0.0

            # --- Entry masking (individual coefficients) ---
            # Only mask entries that are not already masked by row/col masking
            if r_entry > 0:
                # Create mask showing which entries are NOT already masked
                row_mask = torch.ones(m_real, device=device, dtype=torch.bool)
                col_mask = torch.ones(n_real, device=device, dtype=torch.bool)

                if n_rows_to_mask > 0:
                    row_mask[row_indices] = False
                if n_cols_to_mask > 0:
                    col_mask[col_indices] = False

                # Available positions: intersection of unmasked rows and unmasked columns
                available_mask = row_mask.unsqueeze(1) & col_mask.unsqueeze(0)  # [m_real, n_real]
                available_positions = available_mask.nonzero(as_tuple=False)  # [num_available, 2]

                if len(available_positions) > 0:
                    n_entries_to_mask = max(0, int(r_entry * available_positions.shape[0]))
                    if n_entries_to_mask > 0:
                        entry_perm = torch.randperm(available_positions.shape[0], device=device)[:n_entries_to_mask]
                        entry_indices = available_positions[entry_perm]
                        A_view[i, entry_indices[:, 0], entry_indices[:, 1]] = 0.0

            # --- Optional noisy masking ---
            if noisy_mask:
                # Add small Gaussian noise to masked positions instead of zeros
                # Noise scale: 1% of median absolute coefficient value
                A_nonzero = A[i, :m_real, :n_real][A[i, :m_real, :n_real] != 0]
                if len(A_nonzero) > 0:
                    noise_scale = 0.01 * torch.median(torch.abs(A_nonzero)).item()

                    # Find masked positions and add noise
                    A_masked = (A[i, :m_real, :n_real] != A_view[i, :m_real, :n_real])
                    if A_masked.any():
                        noise_A = torch.randn_like(A_view[i, :m_real, :n_real]) * noise_scale
                        A_view[i, :m_real, :n_real] = torch.where(
                            A_masked,
                            noise_A,
                            A_view[i, :m_real, :n_real]
                        )

                    # Add noise to masked b and c values
                    b_masked = (b[i, :m_real] != b_view[i, :m_real])
                    if b_masked.any():
                        noise_b = torch.randn_like(b_view[i, :m_real]) * noise_scale
                        b_view[i, :m_real] = torch.where(b_masked, noise_b, b_view[i, :m_real])

                    c_masked = (c[i, :n_real] != c_view[i, :n_real])
                    if c_masked.any():
                        noise_c = torch.randn_like(c_view[i, :n_real]) * noise_scale
                        c_view[i, :n_real] = torch.where(c_masked, noise_c, c_view[i, :n_real])

        # Flatten to model input: [vec(A), b, c]
        A_flat = A_view.flatten(start_dim=1)  # [B, M*N]
        x_flat = torch.cat([A_flat, b_view, c_view], dim=1)  # [B, M*N+M+N]
        return x_flat

    # Create asymmetric views
    x_online = create_view(r_entry_on, r_row_on, r_col_on)  # Heavier mask (context)
    x_target = create_view(r_entry_tg, r_row_tg, r_col_tg)  # Lighter/clean mask (target)

    return x_online, x_target


def make_gnn_views(
    batch_graph: torch_geometric.data.Batch,
    mask_ratio: float = 0.3,
) -> Tuple[torch_geometric.data.Batch, torch_geometric.data.Batch, torch.Tensor, torch.Tensor]:
    """
    Create context and target views for GNN with node-level masking.

    For bipartite graphs representing LP problems:
    - Constraint nodes: One per constraint (Ax ≤ b)
    - Variable nodes: One per decision variable (x_j)
    - Edges: Connect constraint i to variable j if A[i,j] != 0

    Masking strategy:
    - Randomly select mask_ratio of constraint nodes, zero out their features
    - Randomly select mask_ratio of variable nodes, zero out their features
    - Graph topology (edges, edge features) remains unchanged
    - Target view is the original graph (clean, no masking)

    Args:
        batch_graph: Batched bipartite graph from PyG dataloader
        mask_ratio: Fraction of nodes to mask in context view (default: 0.3)

    Returns:
        (ctx_graph, tgt_graph, mask_cons, mask_vars): Tuple of:
        - ctx_graph: Context view with masked node features
        - tgt_graph: Target view (original graph, clean)
        - mask_cons: Boolean mask showing which constraint nodes were masked [num_cons_nodes]
        - mask_vars: Boolean mask showing which variable nodes were masked [num_var_nodes]

    Note: The graph structure (edge_index, edge_attr) is identical in both views.
    Only node features (constraint_features, variable_features) differ.
    """
    # Clone the batch to create two views
    ctx_graph = batch_graph.clone()
    tgt_graph = batch_graph  # Target is original (clean)

    # Get number of constraint and variable nodes
    num_cons_nodes = ctx_graph.constraint_features.shape[0]
    num_var_nodes = ctx_graph.variable_features.shape[0]

    # Determine number of nodes to mask
    n_cons_to_mask = int(mask_ratio * num_cons_nodes)
    n_vars_to_mask = int(mask_ratio * num_var_nodes)

    # Create random masks
    cons_perm = torch.randperm(num_cons_nodes, device=ctx_graph.constraint_features.device)
    vars_perm = torch.randperm(num_var_nodes, device=ctx_graph.variable_features.device)

    mask_cons_indices = cons_perm[:n_cons_to_mask]
    mask_vars_indices = vars_perm[:n_vars_to_mask]

    # Create boolean masks for tracking
    mask_cons = torch.zeros(num_cons_nodes, dtype=torch.bool, device=ctx_graph.constraint_features.device)
    mask_vars = torch.zeros(num_var_nodes, dtype=torch.bool, device=ctx_graph.variable_features.device)
    mask_cons[mask_cons_indices] = True
    mask_vars[mask_vars_indices] = True

    # Zero out masked node features in context view
    ctx_graph.constraint_features = ctx_graph.constraint_features.clone()
    ctx_graph.variable_features = ctx_graph.variable_features.clone()
    ctx_graph.constraint_features[mask_cons_indices] = 0.0
    ctx_graph.variable_features[mask_vars_indices] = 0.0

    return ctx_graph, tgt_graph, mask_cons, mask_vars


def jepa_loss_mlp(
    online_model: torch.nn.Module,
    target_model: torch.nn.Module,
    x_online: torch.Tensor,
    x_target: torch.Tensor,
    mode: str = "ema",
) -> torch.Tensor:
    """
    Compute JEPA loss for MLP architecture.

    Forward pass:
    - Online path: x_on → encoder → proj → pred → p_online (L2-normalized)
    - Target path: x_tg → encoder → proj → z_target (L2-normalized, stop-grad)
    - Loss: cosine_pred_loss(p_online, z_target)

    Args:
        online_model: The online/student model (KKTNetMLP with JEPA components)
        target_model: The target/teacher model (same architecture, EMA-updated or shared)
        x_online: Context view (heavily masked) [B, D_in]
        x_target: Target view (lightly masked or clean) [B, D_in]
        mode: "ema" (use target_model) or "simsiam" (share encoder, stop-grad on target path)

    Returns:
        Scalar JEPA loss

    Note: Models must implement jepa_embed() and jepa_pred() methods.
    """
    # Online path: heavily masked → embed → predict
    z_online = online_model.jepa_embed(x_online)  # [B, D], L2-normalized
    p_online = online_model.jepa_pred(z_online)   # [B, D], L2-normalized

    # Target path: lightly masked/clean → embed
    if mode == "ema":
        with torch.no_grad():
            z_target = target_model.jepa_embed(x_target)  # [B, D], L2-normalized, EMA params
    else:  # simsiam
        with torch.no_grad():
            z_target = online_model.jepa_embed(x_target).detach()  # [B, D], stop-gradient

    # Compute cosine similarity loss
    loss = cosine_pred_loss(p_online, z_target)
    return loss


def jepa_loss_gnn(
    online_model: torch.nn.Module,
    target_model: torch.nn.Module,
    ctx_graph: torch_geometric.data.Batch,
    tgt_graph: torch_geometric.data.Batch,
    mask_cons: torch.Tensor,
    mask_vars: torch.Tensor,
    mode: str = "ema",
) -> torch.Tensor:
    """
    Compute JEPA loss for GNN architecture.

    Forward pass:
    - Online path: ctx_graph (masked) → encoder → proj → pred → p_online (L2-normalized, per node)
    - Target path: tgt_graph (clean) → encoder → proj → z_target (L2-normalized, stop-grad, per node)
    - Loss: average cosine_pred_loss over constraint and variable nodes

    Args:
        online_model: The online/student model (GNNPolicy with JEPA components)
        target_model: The target/teacher model (same architecture, EMA-updated or shared)
        ctx_graph: Context view with masked node features
        tgt_graph: Target view (original graph, clean)
        mask_cons: Boolean mask showing which constraint nodes were masked [num_cons_nodes]
        mask_vars: Boolean mask showing which variable nodes were masked [num_var_nodes]
        mode: "ema" (use target_model) or "simsiam" (share encoder, stop-grad on target path)

    Returns:
        Scalar JEPA loss averaged over all nodes

    Note: Models must implement jepa_embed_nodes() method.
    """
    # Extract graph components
    cons_feat_ctx = ctx_graph.constraint_features
    var_feat_ctx = ctx_graph.variable_features
    edge_index_ctx = ctx_graph.edge_index
    edge_attr_ctx = ctx_graph.edge_features

    cons_feat_tgt = tgt_graph.constraint_features
    var_feat_tgt = tgt_graph.variable_features
    edge_index_tgt = tgt_graph.edge_index
    edge_attr_tgt = tgt_graph.edge_features

    # Online path: context → embed (both node types) → project → predict
    cons_z_on, var_z_on = online_model.jepa_embed_nodes(
        cons_feat_ctx, edge_index_ctx, edge_attr_ctx, var_feat_ctx
    )  # Both [num_nodes, D], L2-normalized

    # Predictors (online only)
    cons_p_on = online_model.cons_jepa_pred(cons_z_on)  # [num_cons, D], L2-normalized
    var_p_on = online_model.var_jepa_pred(var_z_on)    # [num_vars, D], L2-normalized

    # Target path: clean → embed
    if mode == "ema":
        with torch.no_grad():
            cons_z_tg, var_z_tg = target_model.jepa_embed_nodes(
                cons_feat_tgt, edge_index_tgt, edge_attr_tgt, var_feat_tgt
            )  # EMA params
    else:  # simsiam
        with torch.no_grad():
            cons_z_tg, var_z_tg = online_model.jepa_embed_nodes(
                cons_feat_tgt, edge_index_tgt, edge_attr_tgt, var_feat_tgt
            )  # Stop-gradient
            cons_z_tg = cons_z_tg.detach()
            var_z_tg = var_z_tg.detach()

    # Compute loss on masked nodes (optional: focus loss on masked regions only)
    # For simplicity, we compute loss on all nodes (common in JEPA)
    loss_cons = cosine_pred_loss(cons_p_on, cons_z_tg)
    loss_vars = cosine_pred_loss(var_p_on, var_z_tg)

    # Average over node types
    loss = (loss_cons + loss_vars) / 2.0
    return loss
