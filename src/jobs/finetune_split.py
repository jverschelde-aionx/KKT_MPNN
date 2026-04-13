from __future__ import annotations

import math
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import configargparse
import torch
from loguru import logger
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import wandb
from data.common import ProblemClass
from data.split import SplitInstanceBatch, SplitInstanceDataset, split_instance_collate
from jobs.utils import RunningStats, device_from_args, set_all_seeds
from metrics.optimization import (
    binary_feasibility_metrics,
    get_complementary_slackness,
    get_dual_feasibility,
    get_optimal_solution,
    get_primal_feasibility,
    get_stationarity,
    get_surrogate_beta,
    kkt,
    surrogate_loss,
)
from models.optimizer import make_scheduler
from models.split import SplitBlockBiJepaPolicy



class TrainingState:
    def __init__(self, log_every: int) -> None:
        self.log_every = max(1, int(log_every))
        self.steps: int = 0
        self.items: int = 0

        self.kkt_sum: float = 0.0
        self.primal_sum: float = 0.0
        self.dual_sum: float = 0.0
        self.stat_sum: float = 0.0
        self.comp_sum: float = 0.0

    def step(self, n_graphs: int) -> None:
        self.steps += 1
        self.items += int(n_graphs)

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

    def finish_round(self) -> Tuple[float, float, float, float, float]:
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
        return out


def build_arg_parser() -> configargparse.ArgumentParser:
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        default_config_files=["configs/finetune_splits/finetune_split_SC_200_FF.yml"],
    )
    parser.add_argument(
        "--config", is_config_file=True, help="Path to config YAML file"
    )

    # Training
    t = parser.add_argument_group("training")
    t.add_argument("--devices", type=str, default="0")
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--max_steps", type=int, default=10000)
    t.add_argument("--eval_every_steps", type=int, default=500)
    t.add_argument("--save_every_steps", type=int, default=500)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--encoder_lr", type=float, default=1e-4)
    t.add_argument("--num_workers", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log_every", type=int, default=50)
    t.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adam")
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--wandb_project", type=str, default="split_bijepa_finetuning")
    t.add_argument("--experiments_dir", type=str, default="./experiments")
    t.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine", "cosine_warmup", "onecycle"],
        default="none",
    )
    t.add_argument("--warmup_pct", type=float, default=0.0)
    t.add_argument("--min_lr_ratio", type=float, default=0.0)
    t.add_argument("--max_grad_norm", type=float, default=1.0)

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
        default="",
        help="Path to encoder-only checkpoint from pretraining (e.g., .../best_encoder.pt).",
    )
    t.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default="",
        help="Path to full pretrain checkpoint (best.pt) to load full model state.",
    )
    t.add_argument(
        "--unfreeze_step",
        type=int,
        default=0,
        help="When finetune_mode='heads', unfreeze encoder after this many steps. 0 = stay frozen.",
    )
    t.add_argument(
        "--save_step_list", type=int, nargs="+", default=[500, 1000, 2000, 5000]
    )

    # KKT loss weights
    t.add_argument("--primal_weight", type=float, default=0.1)
    t.add_argument("--dual_weight", type=float, default=0.1)
    t.add_argument("--stationarity_weight", type=float, default=0.6)
    t.add_argument("--complementary_slackness_weight", type=float, default=0.2)

    # Surrogate loss
    t.add_argument(
        "--use_surrogate",
        action="store_true",
        help="Add differentiable surrogate loss to KKT loss",
    )
    t.add_argument("--surrogate_alpha", type=float, default=10.0)
    t.add_argument("--surrogate_delta", type=float, default=1.0)
    t.add_argument("--surrogate_beta_final", type=float, default=0.1)
    t.add_argument("--surrogate_warmup_frac", type=float, default=0.3)

    # Data
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
    d.add_argument("--n_instances", type=int, default=None)
    d.add_argument("--data_root", type=str, default="./data/instances/milp")
    d.add_argument("--val_split", type=float, default=0.15)
    d.add_argument("--max_owned_nodes", type=int, default=50)

    # Split model args
    SplitBlockBiJepaPolicy.add_args(parser)

    return parser


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


def _cycling_loader(loader):
    """Yield batches forever, re-shuffling each pass through the dataset."""
    while True:
        yield from loader


def build_optimizer(model: SplitBlockBiJepaPolicy, args) -> torch.optim.Optimizer:
    """Two param-group optimizer: separate LR for encoder vs. heads/composer."""
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


@lru_cache(maxsize=4096)
def _load_kkt_data(lp_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load A, b, c from the .bg (bipartite graph) file corresponding to an .lp path.

    The .bg file stores (A_sparse, v_map, v_nodes, c_nodes, b_vars, b_vec, c_vec)
    where A is [m, n] matching the graph constraint nodes, which is the format
    the KKT loss expects.
    """
    import pickle

    import numpy as np

    bg_path = lp_path.replace("/instance/", "/BG/") + ".bg"
    with open(bg_path, "rb") as f:
        A_sp, _, _, _, _, b_vec, c_vec = pickle.load(f)

    A_dense = A_sp.to_dense() if hasattr(A_sp, "to_dense") else torch.tensor(A_sp)
    b_vec = torch.as_tensor(b_vec, dtype=torch.float32)
    c_vec = torch.as_tensor(np.asarray(c_vec), dtype=torch.float32)
    return A_dense.float(), b_vec, c_vec


def _predict_batch(
    model: SplitBlockBiJepaPolicy,
    batch: SplitInstanceBatch,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[str],
]:
    """
    Run model forward on a SplitInstanceBatch, pad to dense tensors, and return
    (y_pred, A, b, c, mask_m, mask_n, sample_names) ready for KKT loss.
    """
    instances = batch.instances
    B = len(instances)

    # Predict per instance
    x_list, lam_list = [], []
    n_vars_list, n_cons_list = [], []
    for inst in instances:
        x_i, lam_i = model.predict_instance(inst)
        x_list.append(x_i)
        lam_list.append(lam_i)
        n_vars_list.append(inst.n_vars)
        n_cons_list.append(inst.n_cons)

    n_max = max(n_vars_list)
    m_max = max(n_cons_list)

    # Pad and stack to dense [B, ...] tensors
    x_padded = torch.zeros(B, n_max, device=device)
    lam_padded = torch.zeros(B, m_max, device=device)
    A = torch.zeros(B, m_max, n_max, device=device)
    b = torch.zeros(B, m_max, device=device)
    c = torch.zeros(B, n_max, device=device)
    mask_m = torch.zeros(B, m_max, dtype=torch.bool, device=device)
    mask_n = torch.zeros(B, n_max, dtype=torch.bool, device=device)
    sample_names: List[str] = []

    for i, inst in enumerate(instances):
        ni = inst.n_vars
        mi = inst.n_cons

        x_padded[i, :ni] = x_list[i]
        lam_padded[i, :mi] = lam_list[i]

        if inst.A_dense is not None:
            A[i, :mi, :ni] = inst.A_dense.to(device)
            b[i, :mi] = inst.b_vec.to(device)
            c[i, :ni] = inst.c_vec.to(device)
        else:
            A_lp, b_lp, c_lp = _load_kkt_data(inst.name)
            mi_lp, ni_lp = A_lp.shape
            A[i, :mi_lp, :ni_lp] = A_lp.to(device)
            b[i, :mi_lp] = b_lp.to(device)
            c[i, :ni_lp] = c_lp.to(device)

        mask_m[i, :mi] = True
        mask_n[i, :ni] = True
        sample_names.append(inst.name)

    y_pred = torch.cat([x_padded, lam_padded], dim=1)  # [B, n_max + m_max]

    return y_pred, A, b, c, mask_m, mask_n, sample_names


def train_step(
    model: SplitBlockBiJepaPolicy,
    batch: SplitInstanceBatch,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    state: TrainingState,
    primal_weight: float,
    dual_weight: float,
    stationarity_weight: float,
    complementary_slackness_weight: float,
    max_grad_norm: float,
    surrogate_alpha: float = 0.0,
    surrogate_delta: float = 0.0,
    surrogate_beta: float = 0.0,
) -> None:
    model.train()
    n_graphs = batch.num_graphs

    y_pred, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)

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

    if surrogate_alpha > 0.0:
        n = A.shape[2]
        loss_surr, surr_metrics = surrogate_loss(
            x_pred=y_pred[:, :n],
            A=A,
            b=b,
            c=c,
            mask_m=mask_m,
            mask_n=mask_n,
            violation_weight=surrogate_alpha,
            objective_weight=surrogate_delta,
            integrality_weight=surrogate_beta,
        )
        loss = loss + loss_surr
        metrics["kkt_only"] = metrics["kkt_loss"]
        metrics["surrogate_total"] = loss_surr.item()
        metrics["surrogate_viol"] = float(surr_metrics["surrogate_viol"].mean())
        metrics["surrogate_obj"] = float(surr_metrics["surrogate_obj"].mean())
        metrics["surrogate_int"] = float(surr_metrics["surrogate_int"].mean())
        metrics["binary_pressure"] = metrics["surrogate_int"]

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm and max_grad_norm > 0.0:
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    state.step(n_graphs)
    state.add(
        float(metrics["kkt_loss"]) * n_graphs,
        float(metrics["primal_feasibility"]) * n_graphs,
        float(metrics["dual_feasibility"]) * n_graphs,
        float(metrics["stationarity"]) * n_graphs,
        float(metrics["complementary_slackness"]) * n_graphs,
    )

    if state.should_log:
        wandb.log(
            {f"train/{key}": value for key, value in metrics.items()},
            step=state.steps,
        )
        logger.info(
            "step={} kkt={:.5f} primal={:.5f} dual={:.5f} stat={:.5f} comp={:.5f}",
            state.steps,
            float(metrics["kkt_loss"]),
            float(metrics["primal_feasibility"]),
            float(metrics["dual_feasibility"]),
            float(metrics["stationarity"]),
            float(metrics["complementary_slackness"]),
        )


@torch.no_grad()
def eval_epoch(
    model: SplitBlockBiJepaPolicy,
    loader: DataLoader,
    device: torch.device,
    primal_weight: float,
    dual_weight: float,
    stationarity_weight: float,
    complementary_slackness_weight: float,
    surrogate_alpha: float = 0.0,
    surrogate_delta: float = 0.0,
    surrogate_beta: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    use_surrogate = surrogate_alpha > 0.0

    metric_keys: List[str] = [
        "kkt_loss",
        "primal_feasibility",
        "dual_feasibility",
        "stationarity",
        "complementary_slackness",
        "duality_gap",
        "objective_gap",
    ]
    if use_surrogate:
        metric_keys.extend(
            [
                "surrogate_viol",
                "surrogate_obj",
                "surrogate_int",
                "feasibility_rate",
                "viol_sum",
                "viol_max",
                "penalised_obj",
            ]
        )
    stats: Dict[str, RunningStats] = {k: RunningStats() for k in metric_keys}

    for batch in loader:
        y_pred, A, b, c, mask_m, mask_n, sample_names = _predict_batch(
            model, batch, device
        )
        B = int(A.size(0))
        n_max = c.shape[1]
        m_max = b.shape[1]

        x_pred = y_pred[:, :n_max]
        lambda_pred = y_pred[:, n_max : n_max + m_max]

        primal = get_primal_feasibility(x_pred, A, b, mask_m)
        dual = get_dual_feasibility(lambda_pred, mask_m)
        stat = get_stationarity(lambda_pred, A, c, mask_n)
        comp = get_complementary_slackness(x_pred, lambda_pred, A, b, mask_m)

        weighted_primal = primal_weight * primal
        weighted_dual = dual_weight * dual
        weighted_stat = stationarity_weight * stat
        weighted_comp = complementary_slackness_weight * comp

        kkt_loss_per_instance = (
            weighted_primal + weighted_dual + weighted_stat + weighted_comp
        )

        stats["kkt_loss"].update_batch(kkt_loss_per_instance)
        stats["primal_feasibility"].update_batch(weighted_primal)
        stats["dual_feasibility"].update_batch(weighted_dual)
        stats["stationarity"].update_batch(weighted_stat)
        stats["complementary_slackness"].update_batch(weighted_comp)

        if use_surrogate:
            _, surr_metrics = surrogate_loss(
                x_pred=x_pred,
                A=A,
                b=b,
                c=c,
                mask_m=mask_m,
                mask_n=mask_n,
                violation_weight=surrogate_alpha,
                objective_weight=surrogate_delta,
                integrality_weight=surrogate_beta,
            )
            for k in ["surrogate_viol", "surrogate_obj", "surrogate_int"]:
                stats[k].update_batch(surr_metrics[k])

            metrics = binary_feasibility_metrics(
                x_pred=x_pred,
                A=A,
                b=b,
                c=c,
                mask_m=mask_m,
                mask_n=mask_n,
                logits=False,
            )
            for k in metrics.keys():
                stats[k].update_batch(metrics[k])

        # Duality gap
        mask_n_f = mask_n.float()
        mask_m_f = mask_m.float()
        primal_obj = (x_pred * c * mask_n_f).sum(dim=1)
        dual_obj = -(lambda_pred * b * mask_m_f).sum(dim=1)
        opt_gap = (2.0 * (primal_obj - dual_obj).abs()) / (
            primal_obj.abs() + dual_obj.abs() + 1e-9
        )
        stats["duality_gap"].update_batch(opt_gap)

        # Objective gap (needs disk lookup)
        obj_gaps: List[float] = []
        for i in range(B):
            n_vars = int(mask_n[i].sum().item())
            x_i = x_pred[i, :n_vars]
            c_i = c[i, :n_vars]
            input_path = sample_names[i]

            _, obj_gap = get_optimal_solution(
                input_path=input_path,
                x_i=x_i,
                c_i=c_i,
            )
            if obj_gap is not None and not math.isnan(obj_gap):
                obj_gaps.append(float(obj_gap))

        if obj_gaps:
            stats["objective_gap"].update_batch(torch.tensor(obj_gaps))

    out: Dict[str, float] = {}
    for k in metric_keys:
        out[f"valid/{k}"] = stats[k].mean
        out[f"valid/{k}_std"] = stats[k].std(unbiased=False)

    return out


def _load_pretrain_checkpoint(model: SplitBlockBiJepaPolicy, path: str) -> None:
    """Load full model state from a pretrain checkpoint (best.pt / last.pt)."""
    pkg = torch.load(path, map_location="cpu")
    state = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded pretrain checkpoint from {}", path)
    if missing:
        logger.info("Missing keys (expected for new heads): {}", missing)
    if unexpected:
        logger.info("Unexpected keys: {}", unexpected)


def finetune_split(
    overrides: Optional[Mapping] = None,
    config_path: Optional[str] = None,
) -> None:
    try:
        parser = build_arg_parser()
        cli_args = ["--config", config_path] if config_path else []
        args, _ = parser.parse_known_args(args=cli_args or None)

        if overrides is not None:
            from jobs.utils import apply_overrides

            apply_overrides(args, overrides)

        args.epochs = 1

        effective_config = config_path or getattr(args, "config", None)
        run_name = (
            Path(effective_config).stem
            if effective_config
            else f"finetune_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

        if args.use_surrogate:
            if args.surrogate_alpha <= 0.0:
                raise ValueError(
                    f"--use_surrogate requires surrogate_alpha > 0, got {args.surrogate_alpha}"
                )
            if args.surrogate_delta <= 0.0:
                raise ValueError(
                    f"--use_surrogate requires surrogate_delta > 0, got {args.surrogate_delta}"
                )
            if args.surrogate_beta_final <= 0.0:
                raise ValueError(
                    f"--use_surrogate requires surrogate_beta_final > 0, got {args.surrogate_beta_final}"
                )

        print(f"Finetuning split model with args: {args}")
        set_all_seeds(args.seed)
        device = device_from_args(args)

        # Data — reuse the same split dirs as pretrain_split
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

        # Model
        model = SplitBlockBiJepaPolicy(args).to(device)

        # Load weights: full pretrain checkpoint or encoder-only
        if args.pretrain_checkpoint:
            _load_pretrain_checkpoint(model, args.pretrain_checkpoint)
        if args.encoder_path:
            model.load_encoder(args.encoder_path, strict=True)
            logger.info("Loaded encoder from {}", args.encoder_path)

        # Freeze/unfreeze encoder based on finetune_mode
        if args.finetune_mode == "heads":
            model.freeze_encoder()
            logger.info("Encoder frozen. Training heads only.")
        else:
            model.unfreeze_encoder()
            logger.info("Encoder unfrozen. Training encoder + heads.")

        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        logger.info(
            "Trainable parameters: total={:,} | encoder={:,} | heads={:,}",
            total,
            enc,
            total - enc,
        )

        save_dir = Path(args.experiments_dir) / args.wandb_project / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        optimizer = build_optimizer(model, args)
        scheduler = make_scheduler(optimizer, args, steps_per_epoch=args.max_steps)

        state = TrainingState(log_every=args.log_every)

        best_val: float = float("inf")
        best_step: int = 0
        encoder_unfrozen: bool = False
        print(f"device: {device}")

        train_iter = _cycling_loader(train_loader)

        for step in range(1, args.max_steps + 1):
            # Gradual encoder unfreeze
            if (
                not encoder_unfrozen
                and args.finetune_mode == "heads"
                and args.unfreeze_step > 0
                and step == args.unfreeze_step
            ):
                model.unfreeze_encoder()
                optimizer = build_optimizer(model, args)
                scheduler = make_scheduler(
                    optimizer, args, steps_per_epoch=args.max_steps
                )
                encoder_unfrozen = True
                total = sum(p.numel() for p in model.parameters() if p.requires_grad)
                enc = sum(
                    p.numel() for p in model.encoder.parameters() if p.requires_grad
                )
                logger.info(
                    "Unfreezing encoder at step {}. Trainable params: total={:,} | encoder={:,} | heads={:,}",
                    step,
                    total,
                    enc,
                    total - enc,
                )
                wandb.log({"encoder_unfrozen": True, "step": step}, step=state.steps)

            # Surrogate beta schedule
            surr_beta = (
                get_surrogate_beta(
                    epoch=step,
                    total_epochs=args.max_steps,
                    integrality_weight_final=args.surrogate_beta_final,
                    warmup_frac=args.surrogate_warmup_frac,
                )
                if args.use_surrogate
                else 0.0
            )
            surr_alpha = args.surrogate_alpha if args.use_surrogate else 0.0
            surr_delta = args.surrogate_delta if args.use_surrogate else 0.0

            batch = next(train_iter)
            train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                state=state,
                primal_weight=args.primal_weight,
                dual_weight=args.dual_weight,
                stationarity_weight=args.stationarity_weight,
                complementary_slackness_weight=args.complementary_slackness_weight,
                max_grad_norm=float(args.max_grad_norm),
                surrogate_alpha=surr_alpha,
                surrogate_delta=surr_delta,
                surrogate_beta=surr_beta,
            )

            # Periodic checkpoints
            if step % args.save_every_steps == 0 or step == args.max_steps:
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }
                torch.save(ckpt, save_dir / "last.pt")
                model.save_encoder(str(save_dir / "last_encoder.pt"))

            if step in args.save_step_list:
                logger.info("Saving checkpoint for step {}", step)
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }
                torch.save(ckpt, save_dir / f"step_{step:06d}.pt")

            # Eval
            if step % args.eval_every_steps == 0 or step == args.max_steps:
                val_metrics = eval_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    primal_weight=args.primal_weight,
                    dual_weight=args.dual_weight,
                    stationarity_weight=args.stationarity_weight,
                    complementary_slackness_weight=args.complementary_slackness_weight,
                    surrogate_alpha=surr_alpha,
                    surrogate_delta=surr_delta,
                    surrogate_beta=args.surrogate_beta_final
                    if args.use_surrogate
                    else 0.0,
                )
                val_loss = val_metrics["valid/kkt_loss"]

                train_kkt, train_pr, train_du, train_st, train_cp = state.finish_round()

                log_dict = {
                    "step": step,
                    "train/kkt_loss": train_kkt,
                    "train/primal_feas": train_pr,
                    "train/dual_feas": train_du,
                    "train/stationarity": train_st,
                    "train/comp_slack": train_cp,
                    **val_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                wandb.log(log_dict, step=state.steps)

                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }

                if val_loss < best_val:
                    best_val = float(val_loss)
                    best_step = step
                    torch.save(ckpt, save_dir / "best.pt")
                    model.save_encoder(str(save_dir / "best_encoder.pt"))
                    wandb.run.summary["best_val_kkt"] = best_val
                    wandb.run.summary["best_step"] = best_step

                logger.info(
                    "Step {:05d} | train KKT {:.4f} | valid KKT {:.4f} (best {:.4f} @ step {})",
                    step,
                    train_kkt,
                    val_loss,
                    best_val,
                    best_step,
                )

        logger.info(
            "Finished split finetuning. Best validation KKT loss = {:.6f}", best_val
        )

    except Exception as e:
        logger.exception("Exception during split finetuning: {}", e)
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    finetune_split()
