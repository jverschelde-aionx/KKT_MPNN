import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import configargparse
import numpy as np
import torch
import torch_geometric
from loguru import logger
from torch.utils.data import DataLoader

import wandb
from data.common import ProblemClass
from data.datasets import GraphDataset, LPDataset, make_pad_collate, pad_collate_graphs
from models.jepa_utils import ema_update, jepa_loss_gnn, jepa_loss_mlp, make_gnn_views, make_lp_jepa_views
from models.losses import kkt_loss
from models.models import GNNPolicy, KKTNetMLP


class TrainingState:
    def __init__(self, log_every: int):
        self.steps = 0
        self.trained_items = 0
        self.training_loss_sum = 0.0
        self.jepa_loss_sum = 0.0  # Track JEPA loss separately
        self.epoch = 0
        self.best_epoch = 0
        self.log_every = log_every

    def add_training_step(self, loss: float):
        self._increment_step()
        if (self.steps % self.log_every) == 0:
            wandb.log({"train/loss": float(loss)}, step=self.steps)

        self.training_loss_sum += loss

    def add_jepa_loss(self, loss: float):
        """Add JEPA loss to running sum for epoch averaging."""
        self.jepa_loss_sum += loss

    def finish_epoch(self) -> Tuple[float, Optional[float]]:
        """
        Finish the current epoch and return average losses.

        Returns:
            (training_loss, jepa_loss): Tuple of average losses
            jepa_loss is None if JEPA is not being used
        """
        training_loss = self.training_loss_sum / self.trained_items
        jepa_loss = (
            self.jepa_loss_sum / self.trained_items if self.jepa_loss_sum > 0 else None
        )
        self._reset_training_state()
        self._increment_epoch()
        return training_loss, jepa_loss

    def _reset_training_state(self):
        self.trained_items = 0
        self.training_loss_sum = 0.0
        self.jepa_loss_sum = 0.0

    def _increment_epoch(self):
        self.epoch += 1

    def _increment_step(self):
        self.trained_items += 1
        self.steps += 1
        wandb.log({"training/step": self.steps}, step=self.steps)

    def get_step(self) -> int:
        return self.steps

    def get_epoch(self) -> int:
        return self.epoch


def apply_overrides(args, overrides: Mapping) -> None:
    if not overrides:
        return

    def apply_block(block: Mapping):
        for k, v in block.items():
            if isinstance(v, Mapping):
                apply_block(v)
            else:
                if hasattr(args, k):
                    setattr(args, k, v)

    apply_block(overrides)


def train(overrides: Optional[Mapping] = None):
    try:
        parser = configargparse.ArgumentParser(
            allow_abbrev=False,
            default_config_files=["config.yml"],
        )

        # Training
        t = parser.add_argument_group("training")
        t.add_argument("--devices", type=str, default="0")
        t.add_argument("--batch_size", type=int, default=4)
        t.add_argument("--use_bipartite_graphs", action="store_true")
        t.add_argument("--epochs", type=int, default=30)
        t.add_argument("--lr", type=float, default=1e-3)
        t.add_argument("--num_workers", type=int, default=0)
        t.add_argument("--seed", type=int, default=0)
        t.add_argument("--primal_weight", type=float, default=0.1)
        t.add_argument("--dual_weight", type=float, default=0.1)
        t.add_argument("--stationarity_weight", type=float, default=0.6)
        t.add_argument("--complementary_slackness_weight", type=float, default=0.2)
        t.add_argument("--max_lr", type=float, default=0.001)
        t.add_argument(
            "--log_every",
            type=int,
            default=50,
            help="Log lightweight scalars every N steps.",
        )

        # JEPA (Joint-Embedding Predictive Architecture)
        t.add_argument("--use_jepa", action="store_true", help="Enable JEPA self-supervised training")
        t.add_argument(
            "--jepa_mode",
            choices=["ema", "simsiam"],
            default="ema",
            help="JEPA mode: EMA teacher (BYOL/I-JEPA) or SimSiam (no EMA)",
        )
        t.add_argument("--jepa_weight", type=float, default=0.2, help="Weight for JEPA loss (relative to KKT loss)")
        t.add_argument(
            "--jepa_pretrain_epochs",
            type=int,
            default=3,
            help="Number of JEPA-only pre-training epochs before joint KKT+JEPA training (0 for joint from start)",
        )

        # LP-aware masking for MLP (online/context view - heavier mask)
        t.add_argument(
            "--jepa_mask_entry_online",
            type=float,
            default=0.40,
            help="MLP online view: fraction of A entries masked",
        )
        t.add_argument(
            "--jepa_mask_row_online",
            type=float,
            default=0.20,
            help="MLP online view: fraction of constraint rows masked",
        )
        t.add_argument(
            "--jepa_mask_col_online",
            type=float,
            default=0.20,
            help="MLP online view: fraction of variable columns masked",
        )

        # LP-aware masking for MLP (target view - lighter or clean mask)
        t.add_argument(
            "--jepa_mask_entry_target",
            type=float,
            default=0.10,
            help="MLP target view: fraction of A entries masked (0 for clean target)",
        )
        t.add_argument(
            "--jepa_mask_row_target",
            type=float,
            default=0.05,
            help="MLP target view: fraction of constraint rows masked (0 for clean target)",
        )
        t.add_argument(
            "--jepa_mask_col_target",
            type=float,
            default=0.05,
            help="MLP target view: fraction of variable columns masked (0 for clean target)",
        )

        # GNN masking (node-level)
        t.add_argument(
            "--jepa_mask_ratio_nodes",
            type=float,
            default=0.3,
            help="GNN: fraction of nodes masked",
        )

        # Augmentation options
        t.add_argument(
            "--jepa_noisy_mask",
            action="store_true",
            help="Add Gaussian noise at masked positions (vs hard zero masking)",
        )
        t.add_argument(
            "--jepa_row_scaling",
            action="store_true",
            help="Apply row scaling augmentation: s_i ~ LogUniform(0.5, 2.0) to constraints",
        )

        # EMA momentum
        t.add_argument(
            "--ema_momentum",
            type=float,
            default=0.996,
            help="Momentum for EMA target encoder (only used in EMA mode)",
        )

        # Data
        d = parser.add_argument_group("data")
        d.add_argument(
            "--problems",
            type=str,
            nargs="+",
            default=[ProblemClass.INDEPENDANT_SET, ProblemClass.COMBINATORIAL_AUCTION],
            help="Problem type",
        )
        d.add_argument(
            "--is_sizes",
            type=int,
            nargs="+",
            default=[10, 50, 100, 200, 500, 1000],
        )
        d.add_argument(
            "--ca_sizes",
            type=int,
            nargs="+",
            default=[10, 50, 100, 200, 500, 1000],
        )
        d.add_argument(
            "--sc_sizes",
            type=int,
            nargs="+",
            default=[10, 50, 100, 200, 500, 1000],
        )
        d.add_argument(
            "--cfl_sizes",
            type=int,
            nargs="+",
            default=[10, 50, 100, 200, 500, 1000],
        )

        d.add_argument(
            "--rnd_sizes",
            type=int,
            nargs="+",
            default=[10, 50, 100, 200, 500, 1000],
        )

        d.add_argument("--data_root", type=str, default="../data")

        GNNPolicy.add_args(parser)

        args, _ = parser.parse_known_args()

        if overrides is not None:
            apply_overrides(args, overrides)

        print("args:", args)

        # Ensure reproducibility
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        device = (
            torch.device("cuda")
            if torch.cuda.is_available() and args.devices
            else torch.device("cpu")
        )

        # Initialize wandb
        run_name = "kkt_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        if wandb.run is None:
            wandb.init(project="kkt_nets", name=run_name, config=vars(args))

        wandb.define_metric("training/step")
        wandb.define_metric("*", step_metric="training/step")

        # Setup data
        size_cfg = {
            "IS": args.is_sizes,
            "CA": args.ca_sizes,
            "SC": args.sc_sizes,
            "CFL": args.cfl_sizes,
            "RND": args.rnd_sizes,
        }

        train_files, valid_files = [], []
        for problem in args.problems:
            problem_dir = (
                Path(args.data_root)
                / problem
                / ("BG" if args.use_bipartite_graphs else "instance")
            )
            sizes = size_cfg.get(problem, [])
            for split, files in (("train", train_files), ("val", valid_files)):
                split_dir = problem_dir / split
                if sizes:
                    for size in sizes:
                        size_dir = split_dir / str(size)
                        if not size_dir.exists():
                            raise ValueError(f"Missing size dir: {size_dir}")
                        files.extend(
                            [str(size_dir / file) for file in os.listdir(size_dir)]
                        )
                else:
                    raise ValueError(
                        f"Should contain at least one problem size for class {problem}"
                    )

        train_files = sorted(train_files)
        valid_files = sorted(valid_files)

        train_data = (
            GraphDataset(train_files)
            if args.use_bipartite_graphs
            else LPDataset(train_files)
        )
        valid_data = (
            GraphDataset(valid_files)
            if args.use_bipartite_graphs
            else LPDataset(valid_files)
        )
        if not args.use_bipartite_graphs:
            M_max = max(
                [m for m, _ in train_data.shapes] + [m for m, _ in valid_data.shapes]
            )
            N_max = max(
                [n for _, n in train_data.shapes] + [n for _, n in valid_data.shapes]
            )

        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=pad_collate_graphs
            if args.use_bipartite_graphs
            else make_pad_collate(M_fixed=M_max, N_fixed=N_max),
        )
        valid_loader = DataLoader(
            valid_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=pad_collate_graphs
            if args.use_bipartite_graphs
            else make_pad_collate(M_fixed=M_max, N_fixed=N_max),
        )

        model = (
            GNNPolicy(args).to(device)
            if args.use_bipartite_graphs
            else KKTNetMLP(M_max, N_max).to(device)
        )

        # Create optional EMA target model for JEPA training
        target_model = None
        if args.use_jepa and args.jepa_mode == "ema":
            target_model = deepcopy(model)
            for p in target_model.parameters():
                p.requires_grad_(False)
            logger.info("Created EMA target model for JEPA training")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        save_dir = Path("exps") / run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        training_state = TrainingState(log_every=args.log_every)

        wandb.watch(model, log="gradients", log_graph=False)

        logger.info(f"train size: {len(train_data)} | valid size: {len(valid_data)}")
        wandb.log(
            {"data/train_size": len(train_data), "data/valid_size": len(valid_data)}
        )

        best_val = np.inf

        # Train loop
        for epoch in range(1, args.epochs + 1):
            train_loss, train_jepa_loss = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                training_state=training_state,
                primal_weight=args.primal_weight,
                dual_weight=args.dual_weight,
                stationarity_weight=args.stationarity_weight,
                complementary_slackness_weight=args.complementary_slackness_weight,
                args=args,
                target_model=target_model,
            )

            val_loss, validation_metrics = eval_epoch(
                model,
                valid_loader,
                device,
                primal_weight=args.primal_weight,
                dual_weight=args.dual_weight,
                stationarity_weight=args.stationarity_weight,
                complementary_slackness_weight=args.complementary_slackness_weight,
            )

            # logging
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **validation_metrics,
            }
            if train_jepa_loss is not None:
                log_dict["train/loss_jepa_epoch"] = train_jepa_loss
            wandb.log(log_dict, step=training_state.get_step())

            # Model checkpointing
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            # Save target model state if using EMA mode
            if target_model is not None:
                ckpt["target_model"] = target_model.state_dict()
            torch.save(ckpt, save_dir / "last.pt")

            if val_loss < best_val:
                best_val = float(val_loss)
                best_epoch = epoch
                torch.save(ckpt, save_dir / "best.pt")
                wandb.run.summary["best_val_loss"] = best_val
                wandb.run.summary["best_epoch"] = best_epoch

            logger.info(
                "Epoch {:03d} | train {:.4f} | valid {:.4f} (best {:.4f})",
                epoch,
                train_loss,
                val_loss,
                best_val,
            )

        logger.info("Finished training. Best validation loss = {:.4f}", best_val)

    except Exception as e:
        logger.exception("Exception during training: {}", e)
        raise
    finally:
        wandb.finish()


def pack_by_sizes(flat: torch.Tensor, sizes: List[int], max_size: int) -> torch.Tensor:
    B = len(sizes)
    out = flat.new_zeros((B, max_size))
    cursor = 0
    for i, sz in enumerate(sizes):
        out[i, :sz] = flat[cursor : cursor + sz]
        cursor += sz
    return out


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    training_state: TrainingState,
    primal_weight: float,
    dual_weight: float,
    stationarity_weight: float,
    complementary_slackness_weight: float,
    args=None,
    target_model=None,
) -> Tuple[float, Optional[float]]:
    model.train()
    for batch in loader:
        if isinstance(batch[0], torch_geometric.data.Batch):
            # Graph path
            batch_graph, A, b, c, mask_m, mask_n, m_sizes, n_sizes = batch
            batch_graph = batch_graph.to(device)
            A, b, c = A.to(device), b.to(device), c.to(device)
            mask_m, mask_n = mask_m.to(device), mask_n.to(device)

            # GNN forward (flat sequences)
            x_all, lam_all = model(
                batch_graph.constraint_features,
                batch_graph.edge_index,
                batch_graph.edge_attr,
                batch_graph.variable_features,
            )
            # Pack back to [B, max_*]
            x_pred = pack_by_sizes(x_all, n_sizes, c.shape[1])
            lam_pred = pack_by_sizes(lam_all, m_sizes, b.shape[1])

            y_pred = torch.cat([x_pred, lam_pred], dim=1)
        else:
            # MLP path
            model_input, A, b, c, mask_m, mask_n = batch
            model_input = model_input.to(device)
            A, b, c = A.to(device), b.to(device), c.to(device)
            mask_m, mask_n = mask_m.to(device), mask_n.to(device)
            y_pred = model(model_input)
        loss_kkt, _ = kkt_loss(
            y_pred=y_pred,
            A=A,
            b=b,
            c=c,
            mask_m=mask_m,
            mask_n=mask_n,
            primal_weight=primal_weight,
            dual_weight=dual_weight,
            stationarity_weight=stationarity_weight,
            complementary_slackness_weight=complementary_slackness_weight,
        )

        # JEPA loss computation (if enabled)
        loss_jepa = None
        if args and args.use_jepa:
            current_epoch = training_state.get_epoch()
            jepa_only = current_epoch < args.jepa_pretrain_epochs

            if isinstance(batch[0], torch_geometric.data.Batch):
                # GNN path: create node-level masked views
                ctx_graph = make_gnn_views(
                    batch_graph, mask_ratio=args.jepa_mask_ratio_nodes
                )
                tgt_graph = make_gnn_views(
                    batch_graph, mask_ratio=0.0  # clean target
                )
                loss_jepa = jepa_loss_gnn(
                    online_model=model,
                    target_model=target_model,
                    ctx_graph=ctx_graph,
                    tgt_graph=tgt_graph,
                    mode=args.jepa_mode,
                )
            else:
                # MLP path: create LP-aware asymmetric views
                x_online, x_target = make_lp_jepa_views(
                    A=A,
                    b=b,
                    c=c,
                    mask_m=mask_m,
                    mask_n=mask_n,
                    r_entry_on=args.jepa_mask_entry_online,
                    r_row_on=args.jepa_mask_row_online,
                    r_col_on=args.jepa_mask_col_online,
                    r_entry_tg=args.jepa_mask_entry_target,
                    r_row_tg=args.jepa_mask_row_target,
                    r_col_tg=args.jepa_mask_col_target,
                    noisy_mask=args.jepa_noisy_mask,
                    row_scaling=args.jepa_row_scaling,
                )
                loss_jepa = jepa_loss_mlp(
                    online_model=model,
                    target_model=target_model,
                    x_online=x_online,
                    x_target=x_target,
                    mode=args.jepa_mode,
                )

            # Track JEPA loss separately
            training_state.add_jepa_loss(loss_jepa.item())

            # Log JEPA and KKT losses to WandB
            wandb.log(
                {
                    "train/loss_jepa": loss_jepa.item(),
                    "train/loss_kkt": loss_kkt.item(),
                },
                step=training_state.get_step(),
            )

            # Combine losses based on training schedule
            if jepa_only:
                # Pre-training: JEPA loss only
                loss = loss_jepa
            else:
                # Joint training: weighted combination
                loss = loss_kkt + args.jepa_weight * loss_jepa
        else:
            # No JEPA: use KKT loss only
            loss = loss_kkt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target encoder with EMA (if using EMA mode)
        if args and args.use_jepa and args.jepa_mode == "ema" and target_model is not None:
            ema_update(target_model, model, m=args.ema_momentum)

        training_state.add_training_step(loss)

    return training_state.finish_epoch()


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    primal_weight: float,
    dual_weight: float,
    stationarity_weight: float,
    complementary_slackness_weight: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    terms_sum: Dict[str, float] = {}
    validation_metrics: Dict[str, float] = {}

    for batch in loader:
        n_batches += 1
        if isinstance(batch[0], torch_geometric.data.Batch):
            batch_graph, A, b, c, mask_m, mask_n, m_sizes, n_sizes = batch
            batch_graph = batch_graph.to(device)
            A, b, c = A.to(device), b.to(device), c.to(device)
            mask_m, mask_n = mask_m.to(device), mask_n.to(device)
            x_all, lam_all = model(
                batch_graph.constraint_features,
                batch_graph.edge_index,
                batch_graph.edge_attr,
                batch_graph.variable_features,
            )
            x_pred = pack_by_sizes(x_all, n_sizes, c.shape[1])
            lam_pred = pack_by_sizes(lam_all, m_sizes, b.shape[1])
            y_pred = torch.cat([x_pred, lam_pred], dim=1)
        else:
            model_input, A, b, c, mask_m, mask_n = batch
            model_input = model_input.to(device)
            A, b, c = A.to(device), b.to(device), c.to(device)
            mask_m, mask_n = mask_m.to(device), mask_n.to(device)
            y_pred = model(model_input)

        loss, terms = kkt_loss(
            y_pred=y_pred,
            A=A,
            b=b,
            c=c,
            mask_m=mask_m,
            mask_n=mask_n,
            primal_weight=primal_weight,
            dual_weight=dual_weight,
            stationarity_weight=stationarity_weight,
            complementary_slackness_weight=complementary_slackness_weight,
        )
        total_loss += loss.item()
        for term_name, term_sum in terms.items():
            terms_sum[term_name] = terms_sum.get(term_name, 0.0) + term_sum.item()

    for term_name, term_sum in terms_sum.items():
        validation_metrics[f"valid/{term_name}"] = term_sum / n_batches

    return total_loss / n_batches, validation_metrics


if __name__ == "__main__":
    train()
