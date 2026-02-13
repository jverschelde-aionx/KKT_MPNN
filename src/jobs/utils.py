import math
import os
import random
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from configargparse import Namespace
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

import wandb
from data.common import CONS_PAD, VARS_PAD
from data.datasets import (
    GraphDataset,
    LPDataset,
    PadFeaturesTransform,
    lejepa_views_collate,
    make_pad_collate,
    pad_collate_graphs,
)
from models.base import LeJepaEncoderModule


class RunningStats:
    """
    Online mean/std over individual samples using mergeable batch stats (Welford/Chan).

    Stores:
      n   = number of samples seen
      mean
      m2  = sum of squared deviations from mean (population form)
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update_batch(self, x: torch.Tensor) -> None:
        """
        Update stats with a 1D tensor of per-instance values (length B).

        Non-finite values (NaN/inf) are ignored.
        """
        if x is None:
            return

        x = x.detach()
        if x.numel() == 0:
            return

        x = x[torch.isfinite(x)]
        nb = int(x.numel())
        if nb == 0:
            return

        mean_b = float(x.mean().item())
        m2_b = float(((x - mean_b) ** 2).sum().item())  # sum of squared deviations

        if self.n == 0:
            self.n = nb
            self.mean = mean_b
            self.m2 = m2_b
            return

        # Merge two groups: (n, mean, m2) with (nb, mean_b, m2_b)
        delta = mean_b - self.mean
        n_total = self.n + nb

        self.mean = self.mean + delta * (nb / n_total)
        self.m2 = self.m2 + m2_b + (delta * delta) * (self.n * nb / n_total)
        self.n = n_total

    def std(self, unbiased: bool = False) -> float:
        if self.n < 2:
            return 0.0
        denom = (self.n - 1) if unbiased else self.n
        return math.sqrt(self.m2 / denom)


def compute_lambda(
    epoch: int, base: float, start: Optional[float], warm_epochs: int
) -> float:
    """
    Linear ramp of LeJEPA λ over the first `warm_epochs` epochs.
    epoch: 1-based current epoch.
    Returns `base` if start is None or warm_epochs<=0.
    """
    if start is None or warm_epochs <= 0:
        return float(base)
    if epoch <= warm_epochs:
        alpha = epoch / float(max(1, warm_epochs))  # 0..1
        return float(start) * (1.0 - alpha) + float(base) * alpha
    return float(base)


def pack_by_sizes(flat: torch.Tensor, sizes: List[int], max_size: int) -> torch.Tensor:
    B = len(sizes)
    out = flat.new_zeros((B, max_size))
    cursor = 0
    for i, sz in enumerate(sizes):
        out[i, :sz] = flat[cursor : cursor + sz]
        cursor += sz
    return out


def apply_overrides(args: Namespace, overrides: Optional[Mapping]) -> None:
    if not overrides:
        return

    def apply_block(block: Mapping) -> None:
        for k, v in block.items():
            if isinstance(v, Mapping):
                apply_block(v)
            elif hasattr(args, k):
                setattr(args, k, v)

    apply_block(overrides)


def set_all_seeds(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_args(args: Namespace) -> torch.device:
    if torch.cuda.is_available() and args.devices:
        return torch.device("cuda")
    return torch.device("cpu")


def init_logging_and_dirs(args: Namespace, model: LeJepaEncoderModule) -> Path:
    project_name = args.wandb_project
    experiments_dir = args.experiments_dir

    if project_name is None:
        raise ValueError("Please provide a wandb_project argument")

    if experiments_dir is None:
        raise ValueError("Please provide an experiments_dir argument")

    run_name = f"run_{model.name(args)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if wandb.run is None:
        wandb.init(project=project_name, name=run_name, config=vars(args))

    save_dir = Path(experiments_dir) / project_name / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb.define_metric("training/step")
    wandb.define_metric("*", step_metric="training/step")
    wandb.define_metric("epoch")
    wandb.define_metric("valid/*", step_metric="epoch")
    return save_dir


def collect_files(args: Namespace) -> Tuple[list[str], list[str]]:
    # Setup data
    print("args", args)
    size_cfg = {
        "IS": args.is_sizes,
        "CA": args.ca_sizes,
        "SC": args.sc_sizes,
        "CFL": args.cfl_sizes,
        "RND": args.rnd_sizes,
    }
    max_instances_per_problem = (
        args.n_instances // len(args.problems) if args.n_instances else None
    )

    max_instances_per_split = {
        "train": round(max_instances_per_problem * (1 - args.val_split))
        if max_instances_per_problem
        else None,
        "val": round(max_instances_per_problem * args.val_split)
        if max_instances_per_problem
        else None,
    }

    print(max_instances_per_split)

    train_files, valid_files, test_files = [], [], []
    for problem in args.problems:
        problem_dir = (
            Path(args.data_root)
            / problem
            / ("BG" if args.use_bipartite_graphs else "instance")
        )
        sizes = size_cfg.get(problem, [])
        for split, files in (
            ("train", train_files),
            ("val", valid_files),
            ("test", test_files),
        ):
            split_dir = problem_dir / split
            if sizes:
                for size in sizes:
                    size_dir = split_dir / str(size)
                    if not size_dir.exists():
                        raise ValueError(f"Missing size dir: {size_dir}")
                    if split == "test":
                        max_instances = len(os.listdir(size_dir))
                    else:
                        max_instances = (
                            min(
                                max_instances_per_split[split],
                                len(os.listdir(size_dir)),
                            )
                            if max_instances_per_split
                            else len(os.listdir(size_dir))
                        )
                    files.extend(
                        [
                            str(size_dir / file)
                            for file in os.listdir(size_dir)[:max_instances]
                        ]
                    )
            else:
                raise ValueError(
                    f"Should contain at least one problem size for class {problem}"
                )

    train_files = sorted(train_files)
    valid_files = sorted(valid_files)
    test_files = sorted(test_files)

    print(
        f"Collected {len(train_files)} training files, {len(valid_files)} validation files, and {len(test_files)} test files."
    )

    return train_files, valid_files, test_files


def build_dataloaders(
    args: Namespace,
    M_max: Optional[int],
    N_max: Optional[int],
    for_pretraining: bool = True,
) -> Tuple[
    Union[PyGDataLoader, DataLoader],
    Union[PyGDataLoader, DataLoader],
    Optional[int],
    Optional[int],
]:
    train_files, valid_files, test_files = collect_files(args)

    train_data = (
        GraphDataset(train_files, transform=PadFeaturesTransform(CONS_PAD, VARS_PAD))
        if args.use_bipartite_graphs
        else LPDataset(train_files)
    )
    valid_data = (
        GraphDataset(valid_files, transform=PadFeaturesTransform(CONS_PAD, VARS_PAD))
        if args.use_bipartite_graphs
        else LPDataset(valid_files)
    )

    test_data = (
        GraphDataset(test_files, transform=PadFeaturesTransform(CONS_PAD, VARS_PAD))
        if args.use_bipartite_graphs
        else LPDataset(test_files)
    )
    if M_max is None:
        M_max = (
            max(
                [m for m, _ in train_data.shapes]
                + [m for m, _ in valid_data.shapes]
                + [m for m, _ in test_data.shapes]
            )
            if not args.use_bipartite_graphs
            else None
        )
    if N_max is None:
        N_max = (
            max(
                [n for _, n in train_data.shapes]
                + [n for _, n in valid_data.shapes]
                + [n for _, n in test_data.shapes]
            )
            if not args.use_bipartite_graphs
            else None
        )

    if args.use_bipartite_graphs:
        if for_pretraining:
            train_loader = DataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.num_workers),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=lejepa_views_collate,
            )
            valid_loader = DataLoader(
                valid_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(1, args.num_workers),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=lejepa_views_collate,
            )
            test_loader = DataLoader(
                test_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(1, args.num_workers),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=lejepa_views_collate,
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
                collate_fn=pad_collate_graphs,
            )
            valid_loader = DataLoader(
                valid_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(1, args.num_workers),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=pad_collate_graphs,
            )
            test_loader = DataLoader(
                test_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(1, args.num_workers),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=pad_collate_graphs,
            )

    else:
        # MLP case: always need pad_collate to handle variable-sized tensors
        pad_collate = make_pad_collate(M_fixed=M_max, N_fixed=N_max)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.num_workers),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=pad_collate,
        )
        valid_loader = DataLoader(
            valid_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=pad_collate,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=pad_collate,
        )
    return train_loader, valid_loader, test_loader, N_max, M_max
