from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import torch
import torch.distributed.nn
from configargparse import Namespace
from torch import Tensor, nn
from torch import distributed as dist
from torch.distributed import ReduceOp
from torch.distributed.nn import all_reduce as functional_all_reduce


class EncoderModule(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def encoder(self) -> nn.Module:
        pass

    @encoder.setter
    @abstractmethod
    def encoder(self, module: nn.Module):
        pass

    def freeze_encoder(self) -> None:
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def encoder_parameters(self):
        return self.encoder.parameters()

    def save_encoder(self, path: str) -> None:
        torch.save({"encoder": self.encoder.state_dict()}, path)

    def load_encoder(self, path: str, strict: bool = True) -> None:
        pkg = torch.load(path, map_location="cpu")
        if "encoder" not in pkg:
            raise RuntimeError(f"Missing 'encoder' key in {path}")
        self.encoder.load_state_dict(pkg["encoder"], strict=strict)


class LeJepaEncoderModule(EncoderModule, ABC):
    def load_model_and_encoder(self, args: Namespace, logger: logging.Logger) -> None:
        if args.encoder_path:
            # Load encoder-only weights
            self.load_encoder(args.encoder_path, strict=True)

        if args.finetune_mode == "heads":
            self.freeze_encoder()
            logger.info("Encoder frozen. Training heads only.")
        else:
            self.unfreeze_encoder()
            logger.info("Encoder unfrozen. Training encoder + heads.")

        # Show param counts
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        heads = total - enc
        logger.info(
            f"Trainable parameters: total={total:,} | encoder={enc:,} | heads={heads:,}"
        )

    def __init__(self, sigreg_slices: int, sigreg_points: int) -> None:
        super().__init__()
        assert sigreg_slices > 0
        assert sigreg_points > 0
        self.sigreg = SigRegWrapper(num_slices=sigreg_slices, num_points=sigreg_points)

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("model")
        group.add_argument(
            "--lejepa_n_global_views",
            type=int,
            default=2,
            help="Number of global views",
        )
        group.add_argument(
            "--lejepa_n_local_views",
            type=int,
            default=2,
            help="Number of local views (heavily masked)",
        )
        group.add_argument(
            "--sigreg_slices",
            type=int,
            default=1024,
            help="Number of slices for SIGReg regularizer",
        )
        group.add_argument(
            "--sigreg_points",
            type=int,
            default=17,
            help="Number of points for Epps-Pulley test in SIGReg",
        )
        group.add_argument(
            "--lejepa_lambda",
            type=float,
            default=0.05,
            help="Lejepa regularization weight",
        )
        group.add_argument(
            "--lejepa_std_loss_weight",
            type=float,
            default=0.00,
            help="Lejepa std loss weight",
        )

    @staticmethod
    def name(args):
        name = f"-gv={args.lejepa_n_global_views}"
        name += f"-lv={args.lejepa_n_local_views}"
        name += f"-slice={args.sigreg_slices}"
        name += f"-point={args.sigreg_points}"
        name += f"-lmbd={args.lejepa_lambda}"
        name += f"-std={args.lejepa_std_loss_weight}"

        return name

    @abstractmethod
    def make_lejepa_views(
        self,
        input,
    ) -> Tuple[List[Any], List[Any]]:
        pass

    @abstractmethod
    def embed(self, inputs) -> torch.Tensor:
        pass

    def _full_mask(self, view) -> torch.Tensor:
        cm = getattr(view, "cons_mask", None)
        vm = getattr(view, "var_mask", None)
        if cm is None:
            cm = torch.zeros(
                view.constraint_features.size(0),
                dtype=torch.bool,
                device=view.constraint_features.device,
            )
        if vm is None:
            vm = torch.zeros(
                view.variable_features.size(0),
                dtype=torch.bool,
                device=view.variable_features.device,
            )
        return torch.cat([cm, vm], dim=0)

    def lejepa_pred_loss(self, all_embeddings, global_embeddings, all_views):
        centers = torch.stack(global_embeddings, 0).mean(0)  # [N, D]

        masked_squared_error_sum = 0.0
        masked_count = 0

        total_squared_error_sum = 0.0
        total_count = 0

        for embedding, view in zip(all_embeddings, all_views):
            const_mask = getattr(view, "cons_mask", None)
            variable_mask = getattr(view, "var_mask", None)
            if const_mask is None:
                const_mask = torch.zeros(
                    view.constraint_features.size(0),
                    dtype=torch.bool,
                    device=embedding.device,
                )
            if variable_mask is None:
                variable_mask = torch.zeros(
                    view.variable_features.size(0),
                    dtype=torch.bool,
                    device=embedding.device,
                )
            full_mask = torch.cat(
                [const_mask, variable_mask], dim=0
            )  # [N] matches z concat order

            squared_error_per_node = (embedding - centers).pow(2).mean(dim=-1)  # [N]

            total_squared_error_sum += squared_error_per_node.sum()
            total_count += squared_error_per_node.numel()

            if full_mask.any():
                masked_squared_error_sum += squared_error_per_node[full_mask].sum()
                masked_count += int(full_mask.sum().item())

        pred_loss_total = total_squared_error_sum / max(1, total_count)
        pred_loss_masked = masked_squared_error_sum / max(1, masked_count)

        return pred_loss_total, pred_loss_masked

    def lejepa_loss(
        self,
        input,
        precomputed_views: Tuple[List[Any], List[Any]],
        lambd: float,
        std_loss_weight=0.0,
    ):
        global_views, all_views = precomputed_views

        global_embeddings = self.embed(global_views)
        all_embeddings = self.embed(all_views)

        if self.training:
            jitter = 1e-3  # small noise for stability
            embeddings_for_reg = [
                embedding + jitter * torch.randn_like(embedding)
                for embedding in all_embeddings
            ]
        else:
            embeddings_for_reg = all_embeddings

        # variance floor
        z_cat = torch.cat(all_embeddings, dim=0)  # [*, D]
        std = z_cat.std(dim=0, unbiased=False).clamp_min(1e-6)
        std_loss = torch.nn.functional.relu(1.0 - std).mean()

        pred_loss, pred_loss_masked = self.lejepa_pred_loss(
            all_embeddings, global_embeddings, all_views
        )

        sigreg_loss = torch.stack([self.sigreg(z) for z in embeddings_for_reg]).mean()

        loss = (
            (1 - lambd) * pred_loss + lambd * sigreg_loss + std_loss_weight * std_loss
        )
        return loss, pred_loss, pred_loss_masked, sigreg_loss


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
    def __init__(self, num_slices=1024, num_points=17, max_samples: int = 2048):
        super().__init__()
        self.loss = SlicingUnivariateTest(
            EppsPulley(n_points=num_points), num_slices=num_slices
        )

        self.max_samples = (
            None if (max_samples is None or max_samples <= 0) else int(max_samples)
        )

    def forward(self, z):  # z: [N, D]
        N_full = z.size(0)
        if self.max_samples is not None and N_full > self.max_samples:
            idx = torch.randperm(N_full, device=z.device)[: self.max_samples]
            z = z[idx]

        return self.loss(z)
