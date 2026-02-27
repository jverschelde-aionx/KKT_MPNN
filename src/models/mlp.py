from typing import List, Tuple

import torch
from configargparse import Namespace
from torch import nn

from models.base import LeJepaEncoderModule


class MLPEncoder(nn.Module):
    """
    Trunk + projector for LeJEPA. Projector (not normalized) is used for embeddings.
    Trunk activations are used by the KKT heads.
    """

    def __init__(self, m: int, n: int, hidden: int = 256, embed_dim: int = 128) -> None:
        super().__init__()
        D_in = m * n + m + n

        # trunk produces hidden features for heads
        self.trunk = nn.Sequential(
            nn.Linear(D_in, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
        )

        # projector produces LeJEPA embedding z (no predictor, no L2 norm)
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return trunk features h (used by KKT heads)."""
        return self.trunk(x)

    def embed_batch(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Return LeJEPA embedding z (project(trunk(x)))."""
        h = self.trunk(x)
        return self.projector(h)


class KKTNetMLP(LeJepaEncoderModule):
    def __init__(
        self,
        args: Namespace,
        m: int,
        n: int,
    ):
        super().__init__(
            sigreg_slices=args.sigreg_slices, sigreg_num_points=args.sigreg_points
        )
        self.m = m
        self.n = n
        self.encoder = MLPEncoder(m, n, args.hidden, args.embed_dim)

        self.head_x = nn.Sequential(
            nn.Linear(args.hidden, 64), nn.SELU(), nn.Linear(64, n)
        )
        self.head_lam = nn.Sequential(
            nn.Linear(args.hidden, 64), nn.SELU(), nn.Linear(64, m)
        )
        # lambdas must be >= 0 per dual feasibility; we'll ReLU at loss-time OR here:
        self.relu = nn.ReLU()

    @property
    def encoder(self) -> MLPEncoder:
        return self._encoder

    @encoder.setter
    def encoder(self, module):
        self._encoder = module

    @staticmethod
    def add_args(parser):
        mlp = parser.add_argument_group("mlp")
        mlp.add_argument("--hidden", type=int, default=256)
        mlp.add_argument("--embed_dim", type=int, default=128)

    @staticmethod
    def name(args):
        name = "mlp"
        name += LeJepaEncoderModule.name(args)
        name += f"-hidden={args.hidden}"
        name += f"-embed_dim={args.embed_dim}"
        name += f"-mask={args.lejepa_local_mask}"
        return name

    def forward(self, flat_input: torch.Tensor) -> torch.Tensor:
        h = self.encoder(flat_input)  # trunk features
        x = self.head_x(h)  # [B, n]
        lam = self.relu(self.head_lam(h))  # [B, m]  ≥ 0
        return torch.cat([x, lam], dim=-1)

    def make_lejepa_views(
        self,
        input: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            str,
        ],
        n_global_views: int = 2,
        n_local_views: int = 2,
        local_mask: List[float] = [0.40, 0.20, 0.20],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        (A, b, c, mask_m, mask_n, _) = input

        assert len(local_mask) == 3, "local_mask must have 3 elements"

        global_views = []

        for _ in range(n_global_views):
            global_view = self.create_view(
                A,
                b,
                c,
                mask_m,
                mask_n,
                r_entry=0,
                r_row=0,
                r_col=0,
            )
            global_views.append(global_view)

        all_views = list(global_views)
        for _ in range(n_local_views):
            local_view = self.create_view(
                A,
                b,
                c,
                mask_m,
                mask_n,
                r_entry=local_mask[0],
                r_row=local_mask[1],
                r_col=local_mask[2],
            )
            all_views.append(local_view)
        return global_views, all_views

    def create_view(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        mask_m: torch.Tensor,
        mask_n: torch.Tensor,
        r_entry: float = 0.40,
        r_row: float = 0.20,
        r_col: float = 0.20,
    ):
        """Create a single masked view with given masking ratios."""
        # Start with copies
        A_view = A.clone()
        b_view = b.clone()
        c_view = c.clone()

        B, M, N = A.shape
        device = A.device
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

        # Flatten to model input: [vec(A), b, c]
        A_flat = A_view.flatten(start_dim=1)  # [B, M*N]
        x_flat = torch.cat([A_flat, b_view, c_view], dim=1)  # [B, M*N+M+N]
        return x_flat

    def embed(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.encoder.embed_batch(batch)
