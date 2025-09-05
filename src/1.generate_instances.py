#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generator for MILP bipartite graphs.
"""

from __future__ import annotations

from pathlib import Path

import configargparse
import torch

from instances.common import ProblemClass, Settings
from instances.generators import generate_instances

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate() -> None:
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        description="[KKT] Generate instances",
    )
    parser.add_argument(
        "--configs", is_config_file=True, required=False, default="config.yml"
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
        "--n_instances", type=int, default=1000, help="Number of instances per size"
    )
    d.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Fraction of instances for the test set",
    )
    d.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of instances for the validation set",
    )
    d.add_argument("--data_root", type=str, default="../../data")
    d.add_argument(
        "--solve", action="store_true", help="Run Gurobi to collect solution pools"
    )
    d.add_argument(
        "--add_pos_feat",
        action="store_true",
        help="Add positional features to variable features",
    )
    d.add_argument(
        "--norm_pos_feat", action="store_true", help="Normalize positional features"
    )
    d.add_argument(
        "--gurobi_threads", type=int, default=4, help="Number of threads for Gurobi"
    )
    d.add_argument(
        "--n_jobs",
        type=int,
        default=32,
        help="Number of parallel jobs for data generation",
    )

    args, _ = parser.parse_known_args()

    settings = Settings(
        problems=tuple(args.problems),
        is_sizes=tuple(args.is_sizes),
        ca_sizes=tuple(args.ca_sizes),
        test_split=args.test_split,
        val_split=args.val_split,
        add_positional_features=args.add_pos_feat,
        normalize_positional_features=args.norm_pos_feat,
        n_instances=args.n_instances,
        data_root=Path(args.data_root),
        solve=args.solve,
        gurobi_threads=args.gurobi_threads,
        n_jobs=args.n_jobs,
    )

    generate_instances(settings)


if __name__ == "__main__":
    generate()
