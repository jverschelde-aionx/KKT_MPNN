from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import configargparse
import numpy as np
import torch
import wandb
from loguru import logger
from torch.utils.data import DataLoader

from data.common import ProblemClass
from data.split import SplitInstanceDataset, split_instance_collate
from jobs.utils import device_from_args, set_all_seeds
from metrics.isotropy import isotropy_metrics
from models.optimizer import make_scheduler
from models.split import SplitBlockBiJepaPolicy


class TrainingState:
    def __init__(self, log_every: int) -> None:
        self.log_every = max(1, int(log_every))
        self.steps = 0
        self.items = 0
        self.loss_sum = 0.0
        self.pred_sum = 0.0
        self.pred_masked_sum = 0.0
        self.sigreg_sum = 0.0

    @property
    def should_log(self) -> bool:
        return self.steps % self.log_every == 0

    def add(
        self, loss: float, pred: float, pred_masked: float, sigreg: float, n_items: int
    ) -> None:
        self.steps += 1
        self.items += int(n_items)
        self.loss_sum += float(loss)
        self.pred_sum += float(pred)
        self.pred_masked_sum += float(pred_masked)
        self.sigreg_sum += float(sigreg)

    def finish_epoch(self) -> Tuple[float, float, float, float]:
        denom = max(1, self.items)
        out = (
            self.loss_sum / denom,
            self.pred_sum / denom,
            self.pred_masked_sum / denom,
            self.sigreg_sum / denom,
        )
        self.items = 0
        self.loss_sum = self.pred_sum = self.pred_masked_sum = self.sigreg_sum = 0.0
        return out


def build_optimizer(model: SplitBlockBiJepaPolicy, args) -> torch.optim.Optimizer:
    # different LR for encoder/composer is useful during joint training
    enc_params = list(model.encoder.parameters())
    other_params = [
        p for n, p in model.named_parameters() if not n.startswith("_encoder.")
    ]

    if args.optimizer == "adamw":
        opt_cls = torch.optim.AdamW
    else:
        opt_cls = torch.optim.Adam

    return opt_cls(
        [
            {"params": other_params, "lr": args.lr},
            {"params": enc_params, "lr": args.encoder_lr},
        ],
        weight_decay=args.weight_decay,
    )


def maybe_load_warm_encoder(model: SplitBlockBiJepaPolicy, path: str | None) -> None:
    if not path:
        return
    pkg = torch.load(path, map_location="cpu")
    state = pkg["encoder"] if isinstance(pkg, dict) and "encoder" in pkg else pkg
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    logger.info("Warm-loaded encoder from {}", path)
    if missing:
        logger.info("Missing keys: {}", missing)
    if unexpected:
        logger.info("Unexpected keys: {}", unexpected)


def _unpack_lejepa_loss(out):
    if isinstance(out, (tuple, list)):
        if len(out) == 4:
            return out
        if len(out) == 3:
            loss, pred, sigreg = out
            pred_masked = pred
            return loss, pred, pred_masked, sigreg
    raise RuntimeError("Unexpected lejepa_loss return signature")


def _cycling_loader(loader: DataLoader):
    """Yield batches forever, re-shuffling each pass through the dataset."""
    while True:
        yield from loader


def _compute_lambda_steps(
    step: int, base: float, start: float | None, warm_steps: int
) -> float:
    """Linear ramp of LeJEPA lambda over the first `warm_steps` optimizer steps."""
    if start is None or warm_steps <= 0:
        return float(base)
    if step <= warm_steps:
        alpha = step / float(max(1, warm_steps))
        return float(start) * (1.0 - alpha) + float(base) * alpha
    return float(base)


def train_step(
    model: SplitBlockBiJepaPolicy,
    batch,
    optimizer: torch.optim.Optimizer,
    scheduler,
    state: TrainingState,
    lejepa_lambda: float,
    std_loss_weight: float,
    max_grad_norm: float,
) -> None:
    model.train()
    global_views, all_views = model.make_lejepa_views(batch)

    out = model.lejepa_loss(
        input=batch,
        precomputed_views=(global_views, all_views),
        lambd=lejepa_lambda,
        std_loss_weight=std_loss_weight,
    )
    loss, pred_loss, pred_loss_masked, sigreg_loss = _unpack_lejepa_loss(out)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if max_grad_norm and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    state.add(
        float(loss),
        float(pred_loss),
        float(pred_loss_masked),
        float(sigreg_loss),
        n_items=batch.num_graphs,
    )

    if state.should_log:
        wandb.log(
            {
                "train/lejepa_loss_b": float(loss),
                "train/lejepa_pred_loss_b": float(pred_loss),
                "train/lejepa_pred_loss_masked_b": float(pred_loss_masked),
                "train/lejepa_sigreg_loss_b": float(sigreg_loss),
            },
            step=state.steps,
        )
        logger.info(
            "step={} loss={:.5f} pred={:.5f} pred_masked={:.5f} sigreg={:.5f}",
            state.steps,
            float(loss),
            float(pred_loss),
            float(pred_loss_masked),
            float(sigreg_loss),
        )


@torch.no_grad()
def eval_epoch(
    model: SplitBlockBiJepaPolicy,
    loader: DataLoader,
    device: torch.device,
    lejepa_lambda: float,
    std_loss_weight: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()

    total_loss = 0.0
    total_pred = 0.0
    total_pred_masked = 0.0
    total_sigreg = 0.0
    total_graphs = 0

    iso_sums: Dict[str, float] = {}
    iso_count = 0

    for batch in loader:
        B = batch.num_graphs
        total_graphs += B

        out = model.lejepa_loss(
            input=batch,
            precomputed_views=model.make_lejepa_views(batch),
            lambd=lejepa_lambda,
            std_loss_weight=std_loss_weight,
        )
        loss, pred, pred_masked, sigreg = _unpack_lejepa_loss(out)

        total_loss += float(loss) * B
        total_pred += float(pred) * B
        total_pred_masked += float(pred_masked) * B
        total_sigreg += float(sigreg) * B

        emb = model.embed([batch])[0]
        n_c = sum(inst.n_cons for inst in batch.instances)
        c_emb = emb[:n_c]
        v_emb = emb[n_c:]

        iso_all = isotropy_metrics((emb,), model.sigreg, prefix="valid/iso_all/")
        iso_c = {}
        iso_v = {}
        if c_emb.size(0) >= 2:
            iso_c = isotropy_metrics((c_emb,), model.sigreg, prefix="valid/iso_cons/")
        if v_emb.size(0) >= 2:
            iso_v = isotropy_metrics((v_emb,), model.sigreg, prefix="valid/iso_var/")

        for k, v in {**iso_all, **iso_c, **iso_v}.items():
            iso_sums[k] = iso_sums.get(k, 0.0) + float(v)
        iso_count += 1

    avg_loss = total_loss / max(1, total_graphs)
    metrics = {
        "valid/lejepa_loss": avg_loss,
        "valid/lejepa_pred_loss": total_pred / max(1, total_graphs),
        "valid/lejepa_pred_loss_masked": total_pred_masked / max(1, total_graphs),
        "valid/lejepa_sigreg_loss": total_sigreg / max(1, total_graphs),
    }
    if iso_count > 0:
        for k, v in iso_sums.items():
            metrics[k] = v / iso_count

    return avg_loss, metrics


def _parse_args():
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        default_config_files=["configs/pretrain_split/pretrain_split_bijepa_warm.yml"],
    )

    # data
    d = parser.add_argument_group("data")
    d.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=[ProblemClass.INDEPENDANT_SET, ProblemClass.COMBINATORIAL_AUCTION],
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
    d.add_argument("--data_root", type=str, default="./data/instances/milp")
    d.add_argument("--n_instances", type=int, default=None)
    d.add_argument("--val_split", type=float, default=0.15)
    d.add_argument("--max_owned_nodes", type=int, default=50)

    # train
    t = parser.add_argument_group("training")
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--max_steps", type=int, default=10000)
    t.add_argument("--eval_every_steps", type=int, default=500)
    t.add_argument("--save_every_steps", type=int, default=500)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--encoder_lr", type=float, default=1e-4)
    t.add_argument("--num_workers", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log_every", type=int, default=20)
    t.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adam")
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine", "cosine_warmup", "onecycle"],
        default="none",
    )
    t.add_argument("--warmup_pct", type=float, default=0.0)
    t.add_argument("--min_lr_ratio", type=float, default=0.0)
    t.add_argument("--max_grad_norm", type=float, default=1.0)
    t.add_argument("--freeze_encoder_steps", type=int, default=500)
    t.add_argument("--warm_encoder_path", type=str, default="")
    t.add_argument("--early_stop_patience", type=int, default=5)
    t.add_argument("--early_stop_min_delta", type=float, default=0.0)
    t.add_argument("--save_dir", type=str, default="experiments/split_block_bijepa")
    t.add_argument("--devices", type=str, default="0")
    t.add_argument("--lejepa_lambda_start", type=float, default=None)
    t.add_argument("--lejepa_lambda_warm_steps", type=int, default=1000)
    t.add_argument("--wandb_project", type=str, default="split_bijepa_pretraining")
    t.add_argument("--experiments_dir", type=str, default="./experiments")

    SplitBlockBiJepaPolicy.add_args(parser)
    args, _ = parser.parse_known_args()
    return args


def _collect_split_dirs(args, split: str) -> List[Path]:
    """Collect split .pt directories for a given split (train/val/test)."""
    size_cfg = {
        "IS": args.is_sizes,
        "CA": args.ca_sizes,
        "SC": args.sc_sizes,
        "CFL": args.cfl_sizes,
        "RND": args.rnd_sizes,
    }
    data_root = Path(args.data_root)
    if args.num_blocks is not None:
        split_variant = f"halo-{args.halo_hops}-blocks-{args.num_blocks}"
    else:
        split_variant = f"halo-{args.halo_hops}-nodes-{args.max_owned_nodes}"
    dirs: List[Path] = []

    for problem in args.problems:
        sizes = size_cfg.get(problem, [])
        for size in sizes:
            d = data_root / problem / "splits" / split_variant / split / str(size)
            if d.exists():
                dirs.append(d)
            else:
                logger.warning("Missing split dir: {}", d)
    return dirs


def train():
    args = _parse_args()
    set_all_seeds(args.seed)
    device = device_from_args(args)

    # make_scheduler expects args.epochs; we set epochs=1 so total_steps = max_steps
    args.epochs = 1

    from datetime import datetime

    run_name = f"run_{SplitBlockBiJepaPolicy.name(args)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    save_dir = Path(args.experiments_dir) / args.wandb_project / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dirs = _collect_split_dirs(args, "train")
    val_dirs = _collect_split_dirs(args, "val")

    if args.n_instances is not None:
        max_val = round(args.n_instances * args.val_split)
        max_train = args.n_instances - max_val
    else:
        max_train = None
        max_val = None

    train_ds = SplitInstanceDataset(roots=train_dirs, max_instances=max_train)
    val_ds = SplitInstanceDataset(roots=val_dirs, max_instances=max_val)
    logger.info("Train: {} files from {} dirs", len(train_ds), len(train_dirs))
    logger.info("Val:   {} files from {} dirs", len(val_ds), len(val_dirs))
    wandb.log({"data/train_size": len(train_ds), "data/valid_size": len(val_ds)})

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=split_instance_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=split_instance_collate,
    )

    model = SplitBlockBiJepaPolicy(args).to(device)
    maybe_load_warm_encoder(model, args.warm_encoder_path)

    optimizer = build_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args, steps_per_epoch=args.max_steps)

    best_val = np.inf
    best_step = 0
    patience = 0

    # only freeze encoder when warm-starting from a pretrained checkpoint
    encoder_frozen = bool(args.warm_encoder_path) and args.freeze_encoder_steps > 0
    if encoder_frozen:
        model.freeze_encoder()
        logger.info("Encoder frozen for first {} steps (warm-start)", args.freeze_encoder_steps)

    state = TrainingState(args.log_every)
    train_iter = _cycling_loader(train_loader)

    for step in range(1, args.max_steps + 1):
        # staged freeze -> unfreeze
        if encoder_frozen and step > args.freeze_encoder_steps:
            model.unfreeze_encoder()
            encoder_frozen = False
            logger.info("Step {}: encoder unfrozen", step)

        current_lambda = _compute_lambda_steps(
            step=step,
            base=args.lejepa_lambda,
            start=getattr(args, "lejepa_lambda_start", None),
            warm_steps=getattr(args, "lejepa_lambda_warm_steps", 1000),
        )

        batch = next(train_iter)
        train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            lejepa_lambda=current_lambda,
            std_loss_weight=args.lejepa_std_loss_weight,
            max_grad_norm=args.max_grad_norm,
        )

        # save checkpoint
        if step % args.save_every_steps == 0 or step == args.max_steps:
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "last.pt")

        # eval
        if step % args.eval_every_steps == 0 or step == args.max_steps:
            val_loss, val_metrics = eval_epoch(
                model=model,
                loader=val_loader,
                device=device,
                lejepa_lambda=current_lambda,
                std_loss_weight=args.lejepa_std_loss_weight,
            )

            train_loss, train_pred, train_pred_masked, train_sigreg = (
                state.finish_epoch()
            )

            logger.info(
                "Step {:05d} | train {:.5f} | val {:.5f} | pred {:.5f} | pred_masked {:.5f}",
                step,
                train_loss,
                val_loss,
                val_metrics["valid/lejepa_pred_loss"],
                val_metrics["valid/lejepa_pred_loss_masked"],
            )

            wandb.log(
                {
                    "train/lejepa_loss": train_loss,
                    "train/lejepa_pred_loss": train_pred,
                    "train/lejepa_pred_loss_masked": train_pred_masked,
                    "train/lejepa_sigreg_loss": train_sigreg,
                    **val_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                    "lejepa_lambda_used": current_lambda,
                },
                step=state.steps,
            )

            if val_loss < (best_val - args.early_stop_min_delta):
                best_val = float(val_loss)
                best_step = step
                patience = 0
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "args": vars(args),
                }
                torch.save(ckpt, save_dir / "best.pt")
                wandb.run.summary["best_val_loss"] = best_val
                wandb.run.summary["best_step"] = best_step
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    logger.info("Early stopping at step {}", step)
                    wandb.run.summary["early_stop"] = True
                    wandb.run.summary["early_stop_step"] = step
                    break

    wandb.finish()
    logger.info("Finished split BIJEPA pretraining. Best val loss = {:.6f}", best_val)
    return str(save_dir)


if __name__ == "__main__":
    train()
