import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import configargparse
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

import wandb
from data.common import CONS_PAD, VARS_PAD, ProblemClass
from data.datasets import (
    GraphDataset,
    LPDataset,
    PadFeaturesTransform,
    make_pad_collate,
)
from metrics.isotropy import isotropy_metrics
from models.base import LeJepaEncoderModule
from models.gnn import GNNPolicy
from models.mlp import KKTNetMLP


class TrainingState:
    def __init__(self, log_every: int) -> None:
        self.log_every = log_every
        self.steps: int = 0
        self.trained_items: int = 0
        self.jepa_loss_sum: float = 0.0
        self.jepa_pred_loss_sum: float = 0.0
        self.jepa_sigreg_loss_sum: float = 0.0
        self.epoch: int = 0

    def get_step(self) -> int:
        return self.steps

    def step(self, num_items: int) -> None:
        self.steps += 1
        self.trained_items += int(num_items)
        wandb.log({"training/step": self.steps}, step=self.steps)

    @property
    def should_log(self) -> bool:
        return self.steps % max(1, self.log_every) == 0

    def add_training_step(
        self, loss: float, loss_pred: float, loss_sigreg: float
    ) -> None:
        self.jepa_loss_sum += loss
        self.jepa_pred_loss_sum += loss_pred
        self.jepa_sigreg_loss_sum += loss_sigreg

    def finish_epoch(self) -> Tuple[float, float, float]:
        denom = max(1, self.trained_items)
        out = (
            self.jepa_loss_sum / denom,
            self.jepa_pred_loss_sum / denom,
            self.jepa_sigreg_loss_sum / denom,
        )
        self.trained_items = 0
        self.jepa_loss_sum = self.jepa_pred_loss_sum = self.jepa_sigreg_loss_sum = 0.0
        self.epoch += 1
        return out


def apply_overrides(
    args: configargparse.Namespace, overrides: Optional[Mapping]
) -> None:
    if not overrides:
        return

    def apply_block(block: Mapping) -> None:
        for k, v in block.items():
            if isinstance(v, Mapping):
                apply_block(v)
            elif hasattr(args, k):
                setattr(args, k, v)

    apply_block(overrides)


def train(overrides: Optional[Mapping] = None):
    try:
        parser = configargparse.ArgumentParser(
            allow_abbrev=False,
            default_config_files=["configs/pretrain_gnn.yml"],
        )

        # Training
        t = parser.add_argument_group("training")
        t.add_argument("--devices", type=str, default="0")
        t.add_argument("--batch_size", type=int, default=8)
        t.add_argument("--use_bipartite_graphs", action="store_true")
        t.add_argument("--epochs", type=int, default=20)
        t.add_argument("--lr", type=float, default=1e-3)
        t.add_argument("--num_workers", type=int, default=0)
        t.add_argument("--seed", type=int, default=0)
        t.add_argument("--log_every", type=int, default=50)
        t.add_argument("--optimizer", type=str, default="adam")

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

        args, _ = parser.parse_known_args()

        GNNPolicy.add_args(parser) if args.use_bipartite_graphs else KKTNetMLP.add_args(
            parser
        )

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
        project_name = "kkt_lejepa_pretraining"
        run_name = "kkt_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        if wandb.run is None:
            wandb.init(project=project_name, name=run_name, config=vars(args))

        save_dir = Path("exps") / project_name / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

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
            GraphDataset(
                train_files, transform=PadFeaturesTransform(CONS_PAD, VARS_PAD)
            )
            if args.use_bipartite_graphs
            else LPDataset(train_files)
        )
        valid_data = (
            GraphDataset(
                valid_files, transform=PadFeaturesTransform(CONS_PAD, VARS_PAD)
            )
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

        if args.use_bipartite_graphs:
            train_loader = PyGDataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.num_workers),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                follow_batch=[
                    "constraint_features",
                    "variable_features",
                ],
            )
            valid_loader = PyGDataLoader(
                valid_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(1, args.num_workers),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                follow_batch=["constraint_features", "variable_features"],
            )
        else:
            train_loader = DataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.num_workers),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=make_pad_collate(M_fixed=M_max, N_fixed=N_max),
            )
            valid_loader = DataLoader(
                valid_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(1, args.num_workers),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=make_pad_collate(M_fixed=M_max, N_fixed=N_max),
            )

        model = (
            GNNPolicy(args).to(device)
            if args.use_bipartite_graphs
            else KKTNetMLP(args, M_max, N_max).to(device)
        )

        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        proj = sum(
            p.numel() for p in model.encoder.graph_proj.parameters() if p.requires_grad
        )
        logger.info(
            f"Trainable params: total={total}, encoder={enc}, graph_proj={proj}"
        )

        optimizer = (
            torch.optim.Adam(model.parameters(), lr=args.lr)
            if args.optimizer == "adam"
            else torch.optim.AdamW(model.parameters(), lr=args.lr)
        )

        training_state = TrainingState(log_every=args.log_every)

        logger.info(f"train size: {len(train_data)} | valid size: {len(valid_data)}")
        wandb.log(
            {"data/train_size": len(train_data), "data/valid_size": len(valid_data)}
        )

        best_val = np.inf

        # Train loop
        for epoch in range(1, args.epochs + 1):
            train_jepa_loss, train_jepa_pred_loss, train_jepa_sigreg_loss = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                training_state=training_state,
                n_global_views=args.lejepa_n_global_views,
                n_local_views=args.lejepa_n_local_views,
                local_mask=args.lejepa_local_mask,
                global_mask=args.lejepa_global_mask,
                lejepa_lambda=args.lejepa_lambda,
            )

            val_loss, val_metrics = eval_epoch(
                model=model,
                loader=valid_loader,
                device=device,
                n_global_views=args.lejepa_n_global_views,
                n_local_views=args.lejepa_n_local_views,
                local_mask=args.lejepa_local_mask,
                global_mask=args.lejepa_global_mask,
                lejepa_lambda=args.lejepa_lambda,
            )

            log_dict = {
                "epoch": epoch,
                "train/lejepa_loss": train_jepa_loss,
                "train/lejepa_pred_loss": train_jepa_pred_loss,
                "train/lejepa_sigreg_loss": train_jepa_sigreg_loss,
                **val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
            }

            wandb.log(log_dict, step=training_state.steps)

            # Save checkpoints
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "last.pt")
            # Save encoder-only artifact for downstream B1/B2
            model.save_encoder(str(save_dir / "last_encoder.pt"))

            if val_loss < best_val:
                best_val = float(val_loss)
                best_epoch = epoch
                torch.save(ckpt, save_dir / "best.pt")
                # Save encoder-only artifact for downstream B1/B2
                model.save_encoder(str(save_dir / "best_encoder.pt"))
                wandb.run.summary["best_val_loss"] = best_val
                wandb.run.summary["best_epoch"] = best_epoch

            logger.info(
                "Epoch {:03d} | train {:.4f} | valid {:.4f} (best {:.4f})",
                epoch,
                train_jepa_loss,
                val_loss,
                best_val,
            )

        logger.info(
            "Finished pretraining. Best validation LeJEPA loss = {:.4f}", best_val
        )

    except Exception as e:
        logger.exception("Exception during pretraining: {}", e)
        raise
    finally:
        wandb.finish()


def train_epoch(
    model: LeJepaEncoderModule,
    loader: Union[DataLoader, PyGDataLoader],
    optimizer: torch.optim.Optimizer,
    device: str,
    training_state: TrainingState,
    n_global_views: int,
    n_local_views: int,
    local_mask,
    global_mask,
    lejepa_lambda: float,
) -> Tuple[float, Optional[float]]:
    model.train()
    for batch in loader:
        model_input = batch.to(device, non_blocking=True)
        loss, pred_loss, sigreg_loss = model.lejepa_loss(
            input=model_input,
            n_global_views=n_global_views,
            n_local_views=n_local_views,
            local_mask=local_mask,
            global_mask=global_mask,
            lambd=lejepa_lambda,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # update counters
        training_state.step(
            num_items=getattr(
                model_input,
                "num_graphs",
                getattr(model_input, "size", lambda: [1])()[0]
                if hasattr(model_input, "size")
                else 1,
            )
        )
        training_state.add_training_step(
            float(loss), float(pred_loss), float(sigreg_loss)
        )

        # light, periodic logging only
        if training_state.should_log:
            wandb.log(
                {
                    "train/lejepa_loss_b": float(loss),
                    "train/lejepa_pred_loss_b": float(pred_loss),
                    "train/lejepa_sigreg_loss_b": float(sigreg_loss),
                },
                step=training_state.get_step(),
            )

    return training_state.finish_epoch()


@torch.no_grad()
def eval_epoch(
    model: LeJepaEncoderModule,
    loader: Union[DataLoader, PyGDataLoader],
    device: str,
    n_global_views: int,
    n_local_views: int,
    local_mask,
    global_mask,
    lejepa_lambda: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss: float = 0.0
    total_pred_loss: float = 0.0
    total_sigreg_loss: float = 0.0
    total_loss: float = 0.0
    n_batches: int = 0

    # Isotropy accumulation
    iso_sums: Dict[str, float] = {}
    iso_count: int = 0

    for batch_idx, batch in enumerate(loader):
        n_batches += 1

        jepa_input = batch.to(device, non_blocking=True)

        loss_val, pred_loss_val, sigreg_loss_val = model.lejepa_loss(
            jepa_input,
            n_global_views=n_global_views,
            n_local_views=n_local_views,
            local_mask=local_mask,
            global_mask=global_mask,
            lambd=lejepa_lambda,
        )
        total_loss += float(loss_val)
        total_pred_loss += float(pred_loss_val)
        total_sigreg_loss += float(sigreg_loss_val)

        embeddings = model.embed([jepa_input])
        iso = isotropy_metrics(embeddings, model.sigreg, prefix="valid/iso/")
        for k, v in iso.items():
            iso_sums[k] = iso_sums.get(k, 0.0) + float(v)
        iso_count += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_pred_loss = total_pred_loss / max(1, n_batches)
    avg_sigreg_loss = total_sigreg_loss / max(1, n_batches)

    metrics: Dict[str, float] = {
        "valid/lejepa_loss": avg_loss,
        "valid/lejepa_pred_loss": avg_pred_loss,
        "valid/lejepa_sigreg_loss": avg_sigreg_loss,
    }
    if iso_count > 0:
        for k, v in iso_sums.items():
            metrics[k] = v / iso_count
    return avg_loss, metrics


if __name__ == "__main__":
    train()
