"""
LeJEPA (Latent Embedding Joint-Embedding Predictive Architecture) utility functions.

This module implements LeJEPA functionality including:
- SIGReg: Sliced Gaussianity regularization (Epps-Pulley test)
- LP-aware masking strategies for MLP models (structure-preserving masking)
- Node-level masking for GNN models
- LeJEPA view creation for both architectures

Key differences from traditional JEPA:
- No EMA/teacher model (heuristics-free)
- No predictor networks
- Predict-to-center loss instead of asymmetric prediction
- SIGReg Gaussianity regularizer

References:
- LeJEPA: "Latent Embedding Joint-Embedding Predictive Architecture" (arXiv)
- Epps-Pulley: "A test for normality based on the empirical characteristic function"
"""

from typing import Tuple

import torch
import torch.distributed.nn
import torch_geometric
from torch import distributed as dist
from torch.distributed import ReduceOp
from torch.distributed.nn import all_reduce as functional_all_reduce


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


class UnivariateTest(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, sorted: bool = False):
        super().__init__()
        self.eps = eps
        self.sorted = sorted
        self.g = torch.distributions.normal.Normal(0, 1)

    def prepare_data(self, x):
        if self.sorted:
            s = x
        else:
            s = x.sort(descending=False, dim=-2)[0]
        return s

    def dist_mean(self, x):
        if is_dist_avail_and_initialized():
            torch.distributed.nn.functional.all_reduce(
                x, torch.distributed.ReduceOp.AVG
            )
        return x

    @property
    def world_size(self):
        if is_dist_avail_and_initialized():
            return dist.get_world_size()
        return 1


def all_reduce(x, op="AVG"):
    if dist.is_available() and dist.is_initialized():
        op = ReduceOp.__dict__[op.upper()]
        return functional_all_reduce(x, op)
    else:
        return x


class SlicingUnivariateTest(torch.nn.Module):
    """
    Multivariate distribution test using random slicing and univariate test statistics.
    This module extends univariate statistical tests to multivariate data by projecting
    samples onto random 1D directions (slices) and aggregating univariate test statistics
    across all projections. The approach is based on the sliced method for comparing
    high-dimensional distributions.
    The test projects multivariate samples x ∈ ℝᴰ onto random unit vectors:
        x_projected = x @ A
    where A ∈ ℝᴰˣᴷ contains K normalized random direction vectors. A univariate
    test is then applied to each of the K projected samples, and results are aggregated.
    Args:
        univariate_test (torch.nn.Module): A univariate test module that accepts
            (*, N, K) tensors and returns (*, K) test statistics, where N is the
            number of samples and K is the number of slices.
        num_slices (int): Number of random 1D projections (slices) to use. More
            slices increase test power but add computational cost.
        reduction (str, optional): How to aggregate statistics across slices:
            - 'mean': Return the average statistic across all slices
            - 'sum': Return the sum of statistics across all slices
            - None: Return individual statistics for each slice (*, num_slices)
            Default: 'mean'.
        sampler (str, optional): Random sampling method for projection directions:
            - 'gaussian': Sample from standard normal distribution (Gaussian projections)
            Default: 'gaussian'.
        clip_value (float, optional): Minimum threshold for test statistics. Values
            below this threshold are clipped to zero. Useful for reducing noise from
            negligible deviations. Default: None (no clipping).
    Attributes:
        global_step (torch.Tensor): Counter for deterministic random seed generation,
            synchronized across distributed processes to ensure consistent projections.
    Notes:
        - Projection directions are normalized to unit vectors (L2 norm = 1).
        - In distributed training, the random seed is synchronized across all ranks
          using all_reduce to ensure identical projections on all devices.
        - The generator is cached and reused across forward passes for efficiency.
        - The global step counter increments after each forward pass to ensure
          different random projections in successive calls.
    Shape:
        - Input: (*, N, D) where * is any number of batch dimensions, N is the
          number of samples, and D is the feature dimension.
        - Output:
            - Scalar if reduction='mean' or 'sum'
            - (*, num_slices) if reduction=None
    Example:
        >>> from your_module import FastEppsPulley, SlicingUnivariateTest
        >>>
        >>> # Create univariate test
        >>> univariate_test = FastEppsPulley(t_max=5.0, n_points=21)
        >>>
        >>> # Wrap with slicing for multivariate testing
        >>> test = SlicingUnivariateTest(
        ...     univariate_test=univariate_test,
        ...     num_slices=100,
        ...     reduction='mean',
        ...     sampler='gaussian',
        ...     clip_value=0.01
        ... )
        >>>
        >>> # Test multivariate samples
        >>> samples = torch.randn(1000, 50)  # 1000 samples, 50 dimensions
        >>> statistic = test(samples)
        >>> print(f"Test statistic: {statistic.item():.4f}")
        >>>
        >>> # Batch processing
        >>> batch_samples = torch.randn(32, 1000, 50)  # 32 batches
        >>> batch_stats = test(batch_samples)  # Returns scalar (averaged over slices)
    References:
        - Rabin, J., Peyré, G., Delon, J., & Bernot, M. (2012). Wasserstein
          barycenter and its application to texture mixing. In Scale Space and
          Variational Methods in Computer Vision (pp. 435-446).
        - Bonneel, N., Rabin, J., Peyré, G., & Pfister, H. (2015). Sliced and
          Radon Wasserstein barycenters of measures. Journal of Mathematical
          Imaging and Vision, 51(1), 22-45.
    """

    def __init__(
        self,
        univariate_test,
        num_slices: int,
        reduction: str = "mean",
        sampler: str = "gaussian",
        clip_value: float = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        # Generator reuse
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        """Get or create generator for given device and seed."""
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        """
        Apply sliced univariate test to multivariate samples.
        Args:
            x (torch.Tensor): Input samples of shape (*, N, D) where * represents
                any number of batch dimensions, N is the number of samples, and
                D is the feature dimension.
        Returns:
            torch.Tensor: Aggregated test statistic(s).
                - Scalar tensor if reduction='mean' or 'sum'
                - Shape (*, num_slices) if reduction=None
        """
        with torch.no_grad():
            # Synchronize global_step across all ranks
            global_step_sync = all_reduce(self.global_step.clone(), op="MAX")
            seed = global_step_sync.item()
            dev = dict(device=x.device)

            # Get reusable generator
            g = self._get_generator(x.device, seed)

            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, **dev, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats


class EppsPulley(UnivariateTest):
    """
    Fast Epps-Pulley two-sample test statistic for univariate distributions.

    This implementation uses numerical integration over the characteristic function
    to compute a goodness-of-fit test statistic. The test compares the empirical
    characteristic function against a standard normal distribution.

    The statistic is computed as:
        T = N * ∫ |φ_empirical(t) - φ_normal(t)|² w(t) dt

    where φ_empirical is the empirical characteristic function, φ_normal is the
    standard normal characteristic function, and w(t) is an integration weight.

    Args:
        t_max (float, optional): Maximum integration point for linear spacing methods.
            Only used for 'trapezoid' and 'simpson' integration. Default: 3.
        n_points (int, optional): Number of integration points. Must be odd for
            'simpson' integration. For 'gauss-hermite', this determines the number
            of positive nodes. Default: 17.
        integration (str, optional): Integration method to use. One of:
            - 'trapezoid': Trapezoidal rule with linear spacing over [0, t_max]
            Default: 'trapezoid'.

    Attributes:
        t (torch.Tensor): Integration points (positive half, including 0).
        weights (torch.Tensor): Precomputed integration weights incorporating
            symmetry and φ(t) = exp(-t²/2).
        phi (torch.Tensor): Precomputed φ(t) = exp(-t²/2) values.
        integration (str): Selected integration method.
        n_points (int): Number of integration points.

    Notes:
        - The implementation exploits symmetry: only t ≥ 0 are computed, and
          contributions from -t are implicitly added via doubled weights.
        - For 'gauss-hermite', nodes and weights are adapted from the standard
          Gauss-Hermite quadrature to integrate against exp(-t²).
        - Supports distributed training via all_reduce operations.

    Example:
        >>> test = EppsPulley(t_max=5.0, n_points=21, integration='simpson')
        >>> samples = torch.randn(1000)  # Standard normal samples
        >>> statistic = test(samples)
        >>> print(f"Test statistic: {statistic.item():.4f}")
    """

    def __init__(
        self, t_max: float = 3, n_points: int = 17, integration: str = "trapezoid"
    ):
        super().__init__()
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points
        # Precompute phi

        # Linearly spaced positive points (including 0)
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Half-weight at t=0
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x):
        N = x.size(-2)
        # Compute cos/sin only for t >= 0
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # Mean across batch
        cos_mean = cos_vals.mean(-3)  # (*, n_points)
        sin_mean = sin_vals.mean(-3)  # (*, n_points)

        # DDP reduction
        cos_mean = all_reduce(cos_mean)
        sin_mean = all_reduce(sin_mean)

        # Compute error (symmetry already in weights)
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted integration
        return (err @ self.weights) * N * self.world_size


class SigRegWrapper(torch.nn.Module):
    def __init__(self, num_slices=1024, num_points=17):
        super().__init__()
        self.loss = SlicingUnivariateTest(
            EppsPulley(n_points=num_points), num_slices=num_slices
        )

    def forward(self, z):  # z: [N, D]
        # Make sure internal test lives on the same device as z
        self.loss = self.loss.to(z.device)
        return self.loss(z)


def make_gnn_lejepa_views(batch_graph, Vg=2, Vl=2, light_ratio=0.05, heavy_ratio=0.30):
    """
    Create multiple masked views for LeJEPA GNN training.

    Args:
        batch_graph: PyG Batch object with bipartite graph data
        Vg: Number of global views (lightly masked)
        Vl: Number of local views (heavily masked)
        light_ratio: Masking ratio for global views (default: 0.05)
        heavy_ratio: Masking ratio for local views (default: 0.30)

    Returns:
        (globals_, all_): Tuple of:
        - globals_: List of Vg lightly masked graphs
        - all_: List of all views (globals + locals)
    """

    def mask_graph(bg, ratio):
        """Create a masked copy of the graph."""
        # Clone the batch
        ctx_graph = bg.clone()

        # Get number of nodes
        num_cons = ctx_graph.constraint_features.shape[0]
        num_vars = ctx_graph.variable_features.shape[0]

        # Determine number to mask
        n_cons_mask = int(ratio * num_cons)
        n_vars_mask = int(ratio * num_vars)

        # Random permutation
        cons_perm = torch.randperm(
            num_cons, device=ctx_graph.constraint_features.device
        )
        vars_perm = torch.randperm(num_vars, device=ctx_graph.variable_features.device)

        # Zero out masked features
        if n_cons_mask > 0:
            ctx_graph.constraint_features = ctx_graph.constraint_features.clone()
            ctx_graph.constraint_features[cons_perm[:n_cons_mask]] = 0.0
        if n_vars_mask > 0:
            ctx_graph.variable_features = ctx_graph.variable_features.clone()
            ctx_graph.variable_features[vars_perm[:n_vars_mask]] = 0.0

        return ctx_graph

    globals_ = [mask_graph(batch_graph.clone(), light_ratio) for _ in range(Vg)]
    locals_ = [mask_graph(batch_graph.clone(), heavy_ratio) for _ in range(Vl)]
    all_ = globals_ + locals_
    return globals_, all_


def make_lp_lejepa_views(
    A,
    b,
    c,
    mask_m,
    mask_n,
    Vg: int = 2,
    Vl: int = 2,
    light=(0.10, 0.05, 0.05),  # (entry,row,col) for globals
    heavy=(0.40, 0.20, 0.20),  # for locals
    noisy_mask=False,
    row_scaling=False,
):
    x_globals = []
    for _ in range(Vg):
        xg, _ = make_lp_jepa_views(
            A,
            b,
            c,
            mask_m,
            mask_n,
            r_entry_on=light[0],
            r_row_on=light[1],
            r_col_on=light[2],
            r_entry_tg=0.0,
            r_row_tg=0.0,
            r_col_tg=0.0,  # unused
            noisy_mask=noisy_mask,
            row_scaling=row_scaling,
        )
        x_globals.append(xg)

    x_all = list(x_globals)
    for _ in range(Vl):
        xl, _ = make_lp_jepa_views(
            A,
            b,
            c,
            mask_m,
            mask_n,
            r_entry_on=heavy[0],
            r_row_on=heavy[1],
            r_col_on=heavy[2],
            r_entry_tg=0.0,
            r_row_tg=0.0,
            r_col_tg=0.0,
            noisy_mask=noisy_mask,
            row_scaling=row_scaling,
        )
        x_all.append(xl)
    return x_globals, x_all


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
        mask_m: Binary mask for constraints [B, M] - 1.0 = real, 0.0 = padding
        mask_n: Binary mask for variables [B, N] - 1.0 = real, 0.0 = padding
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
            log_scales = torch.rand(B, M, 1, device=device) * (
                torch.log(torch.tensor(2.0)) - torch.log(torch.tensor(0.5))
            ) + torch.log(torch.tensor(0.5))
            scales = torch.exp(log_scales)  # [B, M, 1]
            A_view = A_view * scales  # Scale A rows
            b_view = b_view * scales.squeeze(-1)  # Scale b values

        # Process each sample in batch
        for i in range(B):
            # mask_m and mask_n are 2D binary masks [B, M/N] from data pipeline
            # Count number of real (non-padded) constraints/variables
            m_real = int(mask_m[i].sum().item())  # Count 1s in binary mask
            n_real = int(mask_n[i].sum().item())  # Count 1s in binary mask

            # Safety check: need at least 2 rows and 2 columns to guarantee context
            if m_real < 2 or n_real < 2:
                continue  # Skip masking for tiny problems

            # --- Row masking (constraints) ---
            n_rows_to_mask = max(
                0, min(int(r_row * m_real), m_real - 1)
            )  # Keep at least 1 row
            if n_rows_to_mask > 0:
                row_indices = torch.randperm(m_real, device=device)[:n_rows_to_mask]
                # Mask entire constraint: A[i,:] and b[i]
                A_view[i, row_indices, :n_real] = 0.0
                b_view[i, row_indices] = 0.0

            # --- Column masking (variables) ---
            n_cols_to_mask = max(
                0, min(int(r_col * n_real), n_real - 1)
            )  # Keep at least 1 column
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
                available_mask = row_mask.unsqueeze(1) & col_mask.unsqueeze(
                    0
                )  # [m_real, n_real]
                available_positions = available_mask.nonzero(
                    as_tuple=False
                )  # [num_available, 2]

                if len(available_positions) > 0:
                    n_entries_to_mask = max(
                        0, int(r_entry * available_positions.shape[0])
                    )
                    if n_entries_to_mask > 0:
                        entry_perm = torch.randperm(
                            available_positions.shape[0], device=device
                        )[:n_entries_to_mask]
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
                    A_masked = A[i, :m_real, :n_real] != A_view[i, :m_real, :n_real]
                    if A_masked.any():
                        noise_A = (
                            torch.randn_like(A_view[i, :m_real, :n_real]) * noise_scale
                        )
                        A_view[i, :m_real, :n_real] = torch.where(
                            A_masked, noise_A, A_view[i, :m_real, :n_real]
                        )

                    # Add noise to masked b and c values
                    b_masked = b[i, :m_real] != b_view[i, :m_real]
                    if b_masked.any():
                        noise_b = torch.randn_like(b_view[i, :m_real]) * noise_scale
                        b_view[i, :m_real] = torch.where(
                            b_masked, noise_b, b_view[i, :m_real]
                        )

                    c_masked = c[i, :n_real] != c_view[i, :n_real]
                    if c_masked.any():
                        noise_c = torch.randn_like(c_view[i, :n_real]) * noise_scale
                        c_view[i, :n_real] = torch.where(
                            c_masked, noise_c, c_view[i, :n_real]
                        )

        # Flatten to model input: [vec(A), b, c]
        A_flat = A_view.flatten(start_dim=1)  # [B, M*N]
        x_flat = torch.cat([A_flat, b_view, c_view], dim=1)  # [B, M*N+M+N]
        return x_flat

    # Create asymmetric views
    x_online = create_view(r_entry_on, r_row_on, r_col_on)  # Heavier mask (context)
    x_target = create_view(
        r_entry_tg, r_row_tg, r_col_tg
    )  # Lighter/clean mask (target)

    return x_online, x_target
