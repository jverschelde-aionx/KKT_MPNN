import os
import random
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

import configargparse
import numpy as np
import torch
import wandb
from loguru import logger
from torch.utils.data import DataLoader

from data.common import ProblemClass
from data.datasets import GraphDataset, LPDataset, pad_collate, pad_collate_graphs
from models.losses import kkt_loss
from models.models import KKTNetMLP


class TrainingState:
    def __init__(self, save_dir: Path):
        self.step = 0
        self.epoch = 0
        self.best_epoch = 0
        self.best_val_loss = np.inf
        self.save_dir = save_dir

    def increment_epoch(self):
        self.epoch += 1

    def increment_step(self):
        self.step += 1

    def get_step(self) -> int:
        return self.step

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
        t.add_argument("--use_bipartite_graphs", action="store_true", default=True)
        t.add_argument("--epochs", type=int, default=30)
        t.add_argument("--lr", type=float, default=1e-3)
        t.add_argument("--seed", type=int, default=0)
        t.add_argument("--kkt_w_primal", type=float, default=0.1)
        t.add_argument("--kkt_w_dual", type=float, default=0.1)
        t.add_argument("--kkt_w_station", type=float, default=0.6)
        t.add_argument("--kkt_w_comp", type=float, default=0.2)
        t.add_argument("--max_lr", type=float, default=0.001)
        t.add_argument(
            "--log_every",
            type=int,
            default=50,
            help="Log lightweight scalars every N steps.",
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
        train_files, valid_files = [], []

        for problem in args.problems:
            problem_dir = (
                Path(args.data_root) / problem / "BG"
                if args.use_bipartite_graphs
                else "instance"
            )
            for size_dir in (problem_dir / "train").iterdir():
                train_files.extend([str(size_dir / f) for f in os.listdir(size_dir)])
            for size_dir in (problem_dir / "val").iterdir():
                valid_files.extend([str(size_dir / f) for f in os.listdir(size_dir)])

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

        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=pad_collate_graphs if args.use_bipartite_graphs else pad_collate,
        )
        valid_loader = DataLoader(
            valid_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=pad_collate_graphs if args.use_bipartite_graphs else pad_collate,
        )

        (model_input, A, b, c, m_sizes, n_sizes) = next(iter(valid_loader))
        m, n = m_sizes[0], n_sizes[0]

        model = KKTNetMLP(m, n).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        training_state = TrainingState()

        wandb.watch(model, log="gradients", log_graph=False)

        save_dir = Path("exps") / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"train size: {len(train_data)} | valid size: {len(valid_data)}")
        wandb.log(
            {"data/train_size": len(train_data), "data/valid_size": len(valid_data)}
        )

        # Train loop
        for epoch in range(1, args.epochs + 1):
            model.epoch_callback(epoch)

            train_loss = train_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip=args.grad_clip,
                tensor_device=device,
                training_step=training_state,
                log_every=args.log_every,
                scheduler=scheduler,
            )

            val_loss, validation_metrics = eval_epoch(
                model, valid_loader, criterion, device, args.amp
            )

            # logging
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "valid/loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    **validation_metrics,
                },
                step=training_state.get_step(),
            )

            # Model checkpointing
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "last.pt")

            # TODO ik zit hier
            training_state.update(
                val_loss,
            )
            # track best
            improved = val_loss < best_val
            if improved:
                best_val = float(val_loss)
                best_epoch = epoch
                torch.save(ckpt, save_dir / "best.pt")
                wandb.run.summary["best_val_loss"] = best_val
                wandb.run.summary["best_epoch"] = best_epoch
            else:
                if cooldown_left > 0:
                    cooldown_left -= 1
                else:
                    epochs_since_improve += 1

            # ---- early stopping explosion check ----
            if (
                args.early_stop
                and epoch >= args.es_min_epochs
                and args.es_explosion_factor
                and best_val < float("inf")
            ):
                if val_loss > args.es_explosion_factor * best_val:
                    logger.info("Stopping early due to loss explosion")
                    logger.info(val_loss, best_val)

                    logger.info(
                        "Early stop due to loss explosion (val {:.4f} > {:.2f}× best {:.4f})",
                        val_loss,
                        args.es_explosion_factor,
                        best_val,
                    )
                    wandb.run.summary["early_stopped"] = True
                    wandb.run.summary["early_stop_epoch"] = epoch
                    break

            logger.info(
                "Epoch {:03d} | train {:.4f} | valid {:.4f} (best {:.4f})",
                epoch,
                train_loss,
                val_loss,
                best_val,
            )
            # ---- early stopping check ----
            if (
                args.early_stop
                and epoch >= args.es_min_epochs
                and epochs_since_improve >= args.es_patience
            ):
                logger.info(
                    "Early stopping at epoch {:03d} (no improvement for {} epochs). Best {:.4f} @ {:03d}",
                    epoch,
                    epochs_since_improve,
                    best_val,
                    best_epoch,
                )
                wandb.run.summary["early_stopped"] = True
                wandb.run.summary["early_stop_epoch"] = epoch

                if args.es_restore_best and (save_dir / "best.pt").exists():
                    state = torch.load(save_dir / "best.pt", map_location=device)
                    model.load_state_dict(state["model"])
                    logger.info("Restored model weights from best.pt")

                break  # exit training loop
        logger.info("Finished training. Best validation loss = {:.4f}", best_val)

    except Exception as e:
        logger.exception("Exception during training: {}", e)
        raise
    finally:
        if node_loggers is not None:
            for node_logger in node_loggers:
                node_logger.close()

        wandb.finish()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total, steps = 0.0, 0
    for batch in loader:
        flat_input = batch["flat_input"].to(device)
        A = batch["A"].to(device)
        b = batch["b"].to(device)
        c = batch["c"].to(device)
        mask_m = batch["mask_m"].to(device)
        mask_n = batch["mask_n"].to(device)
        y = batch["y"].to(device) if batch["y"] is not None else None

        B, M, N = A.shape[0], A.shape[1], A.shape[2]
        # Ensure model matches current padded sizes (instantiate once with max M,N you plan to use)
        y_pred = model(flat_input)

        loss, parts = kkt_loss(
            y_true=y,
            y_pred=y_pred,
            A=A,
            b=b,
            c=c,
            m=M,
            n=N,
            mask_m=mask_m,
            mask_n=mask_n,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


if __name__ == "__main__":
    train()
