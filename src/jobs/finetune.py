from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import configargparse
import torch
from configargparse import Namespace
from loguru import logger
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.loader import DataLoader as PyGDataLoader

import wandb
from data.common import ProblemClass
from jobs.utils import (
    apply_overrides,
    build_dataloaders,
    device_from_args,
    init_logging_and_dirs,
    pack_by_sizes,
    set_all_seeds,
)
from metrics.optimization import get_optimal_solution, get_optimality_gap, kkt
from models.base import LeJepaEncoderModule
from models.gnn import GNNPolicy
from models.mlp import KKTNetMLP
from models.optimizer import make_optimizer, make_scheduler

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # syncs GPU ops to the line that fails
torch.autograd.set_detect_anomaly(
    True
)  # points to the op that produced NaN/Inf/bad grads


class TrainingState:
    def __init__(self, log_every: int) -> None:
        self.log_every = max(1, int(log_every))
        self.steps: int = 0
        self.epoch: int = 0
        self.items: int = 0  # number of graphs processed in current epoch

        # Running sums (epoch)
        self.kkt_sum: float = 0.0
        self.primal_sum: float = 0.0
        self.dual_sum: float = 0.0
        self.stat_sum: float = 0.0
        self.comp_sum: float = 0.0

    def step(self, n_graphs: int) -> None:
        self.steps += 1
        self.items += int(n_graphs)
        wandb.log({"training/step": self.steps}, step=self.steps)

    @property
    def should_log(self) -> bool:
        return self.steps % self.log_every == 0

    def add(
        self, kkt: float, primal: float, dual: float, stat: float, comp: float
    ) -> None:
        self.kkt_sum += float(kkt)
        self.primal_sum += float(primal)
        self.dual_sum += float(dual)
        self.stat_sum += float(stat)
        self.comp_sum += float(comp)

    def finish_epoch(self) -> Tuple[float, float, float, float, float]:
        denom = max(1, self.items)
        out = (
            self.kkt_sum / denom,
            self.primal_sum / denom,
            self.dual_sum / denom,
            self.stat_sum / denom,
            self.comp_sum / denom,
        )
        self.items = 0
        self.kkt_sum = self.primal_sum = self.dual_sum = self.stat_sum = (
            self.comp_sum
        ) = 0.0
        self.epoch += 1
        return out


def build_arg_parser() -> configargparse.ArgumentParser:
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        default_config_files=["configs/finetune_gnn_kkt_CA_10.yml"],
    )

    # Training
    t = parser.add_argument_group("training")
    t.add_argument("--devices", type=str, default="0")
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--num_workers", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log_every", type=int, default=50)
    t.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adam")
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--wandb_project", type=str, default="kkt_gnn_finetuning")
    t.add_argument("--experiments_dir", type=str, default="./experiments")
    t.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine", "cosine_warmup", "onecycle"],
        default="none",
    )
    t.add_argument("--warmup_pct", type=float, default=0.0)
    t.add_argument("--min_lr_ratio", type=float, default=0.0)
    t.add_argument("--max_grad_norm", type=float, default=0.0)

    # Early stopping
    t.add_argument("--early_stop_patience", type=int, default=20)
    t.add_argument("--early_stop_min_delta", type=float, default=0.0)

    # Finetune mode
    t.add_argument(
        "--finetune_mode",
        type=str,
        choices=["full", "heads"],
        default="full",
        help="'full' updates encoder+heads; 'heads' freezes encoder and trains heads only.",
    )
    t.add_argument(
        "--encoder_path",
        type=str,
        help="Path to encoder-only checkpoint from pretraining (e.g., .../best_encoder.pt).",
    )

    t.add_argument("--primal_weight", type=float, default=0.1)
    t.add_argument("--dual_weight", type=float, default=0.1)
    t.add_argument("--stationarity_weight", type=float, default=0.6)
    t.add_argument("--complementary_slackness_weight", type=float, default=0.2)

    # Data
    d = parser.add_argument_group("data")
    d.add_argument(
        "--use_bipartite_graphs", action="store_true", help="Must be set for GNNPolicy."
    )
    d.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=[ProblemClass.INDEPENDANT_SET, ProblemClass.COMBINATORIAL_AUCTION],
        help="Problem type(s).",
    )
    d.add_argument(
        "--is_sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000]
    )
    d.add_argument(
        "--ca_sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000]
    )
    d.add_argument(
        "--sc_sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000]
    )
    d.add_argument(
        "--cfl_sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000]
    )
    d.add_argument(
        "--rnd_sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000]
    )
    d.add_argument("--data_root", type=str, default="../data")

    # Add shared LeJepa arguments once
    LeJepaEncoderModule.add_args(parser)

    # Add both model-specific arguments (sweep can switch between models)
    # GNN adds its own args
    gnn = parser.add_argument_group("gnn")
    gnn.add_argument("--embedding_size", default=64, type=int)
    gnn.add_argument("--cons_nfeats", default=4, type=int)
    gnn.add_argument("--edge_nfeats", default=1, type=int)
    gnn.add_argument("--var_nfeats", default=18, type=int)
    gnn.add_argument(
        "--num_emb_type", choices=["periodic", "pwl", "linear"], default="periodic"
    )
    gnn.add_argument("--num_emb_bins", type=int, default=32)
    gnn.add_argument("--num_emb_freqs", type=int, default=16)
    gnn.add_argument(
        "--lejepa_local_mask",
        type=float,
        default=0.40,
        help="GNN local view masking ratio",
    )
    gnn.add_argument(
        "--lejepa_global_mask",
        type=float,
        default=0.1,
        help="GNN global view masking ratio",
    )
    gnn.add_argument("--dropout", default=0.1, type=float)
    gnn.add_argument("--lejepa_embed_dim", default=128, type=int)
    gnn.add_argument(
        "--bipartite_conv",
        type=str,
        choices=["gcn", "transformer", "gatv2"],
        default="gcn",
    )
    gnn.add_argument("--attn_heads", type=int, default=4)
    gnn.add_argument(
        "--graph_pool", type=str, choices=["mean", "meanmax"], default="mean"
    )

    # MLP adds its own args
    mlp = parser.add_argument_group("mlp")
    mlp.add_argument("--hidden", type=int, default=256)
    mlp.add_argument("--embed_dim", type=int, default=128)

    return parser


def load_model_and_encoder(model: LeJepaEncoderModule, args: Namespace) -> GNNPolicy:
    if args.encoder_path:
        # Load encoder-only weights
        model.load_encoder(args.encoder_path, strict=True)

    if args.finetune_mode == "heads":
        model.freeze_encoder()
        logger.info("Encoder frozen. Training heads only.")
    else:
        model.unfreeze_encoder()
        logger.info("Encoder unfrozen. Training encoder + heads.")

    # Show param counts
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    heads = total - enc
    logger.info(
        f"Trainable parameters: total={total:,} | encoder={enc:,} | heads={heads:,}"
    )
    return model


def train_epoch(
    model: LeJepaEncoderModule,
    loader: Union[PyGDataLoader, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    state: TrainingState,
    primal_weight: float,
    dual_weight: float,
    stationarity_weight: float,
    complementary_slackness_weight: float,
    max_grad_norm: float,
) -> Tuple[float, float, float, float, float]:
    model.train()
    for batch in loader:
        if isinstance(batch[0], PyGBatch):
            (
                batch_graph,
                A,
                b,
                c,
                mask_m,
                mask_n,
                m_sizes,
                n_sizes,
                *rest,
            ) = batch

            # move to device
            batch_graph = batch_graph.to(device, non_blocking=True)
            A, b, c = A.to(device), b.to(device), c.to(device)
            mask_m, mask_n = mask_m.to(device), mask_n.to(device)

            # forward: flat predictions across all nodes in the merged batch
            x_all, lam_all = model(
                batch_graph.constraint_features,
                batch_graph.edge_index,
                batch_graph.edge_attr,
                batch_graph.variable_features,
            )

            # pack back to [B, max_*]
            x_pred = pack_by_sizes(x_all, n_sizes, c.shape[1])  # [B, n_max]
            lam_pred = pack_by_sizes(lam_all, m_sizes, b.shape[1])  # [B, m_max]
            y_pred = torch.cat([x_pred, lam_pred], dim=1)  # [B, n_max+m_max]

            n_graphs = int(A.size(0))
        else:
            model_input, A, b, c, mask_m, mask_n, *_ = batch

            model_input = model_input.to(device, non_blocking=True)
            A, b, c = A.to(device), b.to(device), c.to(device)
            mask_m, mask_n = mask_m.to(device), mask_n.to(device)

            y_pred = model(model_input)  # [B, n+m]
            n_graphs = int(A.size(0))

        loss, metrics = kkt(
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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0.0:
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        state.step(n_graphs)
        # kkt(...) returns the *weighted* components; there's no "kkt_loss" key
        state.add(
            float(metrics["kkt_loss"]),
            float(metrics["primal_feasibility"]),
            float(metrics["dual_feasibility"]),
            float(metrics["stationarity"]),
            float(metrics["complementary_slackness"]),
        )

        if state.should_log:
            wandb.log(
                {f"train/{key}": value for key, value in metrics.items()},
                step=state.steps,
            )

    return state.finish_epoch()


@torch.no_grad()
def eval_epoch(
    model: LeJepaEncoderModule,
    loader: Union[PyGDataLoader, DataLoader],
    device: torch.device,
    primal_weight: float,
    dual_weight: float,
    stationarity_weight: float,
    complementary_slackness_weight: float,
) -> Dict[str, float]:
    model.eval()

    metrics_sum: Dict[str, float] = {
        "kkt_loss": 0.0,
        "primal_feasibility": 0.0,
        "dual_feasibility": 0.0,
        "stationarity": 0.0,
        "complementary_slackness": 0.0,
        "optimality_gap": 0.0,
        "objective_gap": 0.0,
    }

    n_batches = 0

    for batch in loader:
        n_batches += 1
        if isinstance(batch[0], PyGBatch):
            batch_graph, A, b, c, mask_m, mask_n, m_sizes, n_sizes, sample_paths = batch

            batch_graph = batch_graph.to(device, non_blocking=True)
            A = A.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            mask_m = mask_m.to(device, non_blocking=True)
            mask_n = mask_n.to(device, non_blocking=True)

            x_all, lam_all = model(
                batch_graph.constraint_features,
                batch_graph.edge_index,
                batch_graph.edge_attr,
                batch_graph.variable_features,
            )

            # Pack back to [B, *]
            n_max = c.shape[1]
            m_max = b.shape[1]
            x_pred = pack_by_sizes(x_all, n_sizes, n_max)  # [B, n_max]
            lam_pred = pack_by_sizes(lam_all, m_sizes, m_max)  # [B, m_max]
            y_pred = torch.cat([x_pred, lam_pred], dim=1)  # [B, n_max+m_max]
            B = int(A.size(0))

        else:
            model_input, A, b, c, mask_m, mask_n, sample_paths = batch
            model_input = model_input.to(device, non_blocking=True)
            A = A.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            mask_m = mask_m.to(device, non_blocking=True)
            mask_n = mask_n.to(device, non_blocking=True)

            y_pred = model(model_input)
            B = int(A.size(0))

            n_max = c.shape[1]
            m_max = b.shape[1]

        loss, kkt_metrics = kkt(
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
        for key, value in kkt_metrics.items():
            metrics_sum[key] += float(value)

        optimalitty_batch_sum: float = 0.0
        objective_batch_sum: float = 0.0

        for i in range(B):
            # Actual sizes via masks
            n_vars = int(mask_n[i].sum().item())
            n_cons = int(mask_m[i].sum().item())

            # Unpadded slices
            c_i = c[i, :n_vars]
            b_i = b[i, :n_cons]

            x_i = y_pred[i, :n_vars]
            lambda_i = y_pred[i, n_max : n_max + n_cons]

            opt_gap = get_optimality_gap(x_i, lambda_i, c_i, b_i)
            optimalitty_batch_sum += float(opt_gap)

            input_path = sample_paths[i]

            _chosen_sol, obj_gap = get_optimal_solution(
                input_path=input_path, x_i=x_i, c_i=c_i
            )
            objective_batch_sum += float(obj_gap)

        metrics_sum["optimality_gap"] += optimalitty_batch_sum / float(B)
        metrics_sum["objective_gap"] += objective_batch_sum / float(B)

    denom = max(1, n_batches)
    val_metrics: Dict[str, float] = {
        f"valid/{key}": value / denom for key, value in metrics_sum.items()
    }

    return val_metrics


def finetune(overrides: Optional[Mapping] = None) -> None:
    try:
        parser = build_arg_parser()
        args, _ = parser.parse_known_args()
        if overrides is not None:
            apply_overrides(args, overrides)

        print(f"Finetuning with args: {args}")
        set_all_seeds(args.seed)
        device = device_from_args(args)

        # Data
        train_loader, valid_loader, N_max, M_max = build_dataloaders(
            args, for_pretraining=False
        )
        logger.info(
            f"train size: {len(train_loader.dataset)} | valid size: {len(valid_loader.dataset)}"
        )
        wandb.log(
            {
                "data/train_size": len(train_loader.dataset),
                "data/valid_size": len(valid_loader.dataset),
            }
        )

        model = (
            GNNPolicy(args).to(device)
            if args.use_bipartite_graphs
            else KKTNetMLP(args, M_max, N_max).to(device)
        )
        save_dir = init_logging_and_dirs(args, model)
        model = load_model_and_encoder(model, args)
        optimizer = make_optimizer(model, args)
        scheduler = make_scheduler(optimizer, args, steps_per_epoch=len(train_loader))

        state = TrainingState(log_every=args.log_every)

        best_val: float = float("inf")
        best_epoch: int = 0
        epochs_no_improve: int = 0

        # Train loop
        for epoch in range(1, args.epochs + 1):
            train_kkt, train_pr, train_du, train_st, train_cp = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                state=state,
                primal_weight=args.primal_weight,
                dual_weight=args.dual_weight,
                stationarity_weight=args.stationarity_weight,
                complementary_slackness_weight=args.complementary_slackness_weight,
                max_grad_norm=float(args.max_grad_norm),
            )

            val_metrics = eval_epoch(
                model=model,
                loader=valid_loader,
                device=device,
                primal_weight=args.primal_weight,
                dual_weight=args.dual_weight,
                stationarity_weight=args.stationarity_weight,
                complementary_slackness_weight=args.complementary_slackness_weight,
            )
            val_loss = val_metrics["valid/kkt_loss"]

            log_dict = {
                "epoch": epoch,
                "train/kkt_loss": train_kkt,
                "train/primal_feas": train_pr,
                "train/dual_feas": train_du,
                "train/stationarity": train_st,
                "train/comp_slack": train_cp,
                **val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
            }
            wandb.log(log_dict, step=state.steps)

            # Save checkpoints
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "last.pt")
            # Encoder-only artifact (finetuned encoder)
            model.save_encoder(str(save_dir / "last_encoder.pt"))

            improved = val_loss < (best_val - args.early_stop_min_delta)
            if improved:
                best_val = float(val_loss)
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(ckpt, save_dir / "best.pt")
                model.save_encoder(str(save_dir / "best_encoder.pt"))
                wandb.run.summary["best_val_kkt"] = best_val
                wandb.run.summary["best_epoch"] = best_epoch
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= int(args.early_stop_patience):
                    logger.info(
                        "Early stopping at epoch {:03d}: no val improvement for {} epochs "
                        "(best {:.6f} at epoch {:03d})",
                        epoch,
                        epochs_no_improve,
                        best_val,
                        best_epoch,
                    )
                    wandb.run.summary["early_stop"] = True
                    wandb.run.summary["early_stop_epoch"] = epoch
                    break

            logger.info(
                "Epoch {:03d} | train KKT {:.4f} | valid KKT {:.4f} (best {:.4f} @ {:03d})",
                epoch,
                train_kkt,
                val_loss,
                best_val,
                best_epoch,
            )

        logger.info("Finished finetuning. Best validation KKT loss = {:.6f}", best_val)

    except Exception as e:
        logger.exception("Exception during finetuning: {}", e)
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    finetune()
