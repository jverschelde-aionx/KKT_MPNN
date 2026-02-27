#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

import wandb
from jobs.pretrain import train


def load_sweep_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"sweep file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("sweep.yaml must parse to a dict")
    return cfg


def agent_entry() -> None:
    """
    One sweep trial. W&B agent calls this per-run.
    We:
      1) init a run
      2) update config with any training-time overrides (e.g., epochs)
      3) call your train(overrides=...) so your parser picks them up
    """
    run = wandb.init()
    assert run is not None, "wandb.init() failed"

    # Flattened config dict for your pretrain.train(overrides=...)
    overrides: Dict[str, Any] = dict(wandb.config)

    # Make sure W&B shows the *effective* config
    wandb.config.update(overrides, allow_val_change=True)

    # Call your training function with these overrides
    train(overrides=overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch W&B sweep for LeJEPA params.")
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("sweeps/configs/lejepa_pretraining_gnn_max_sphericity.yaml"),
        help="Path to sweep.yaml",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="kkt_gnn_pretraining",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity", type=str, default=None, help="W&B entity/org (optional)"
    )
    parser.add_argument(
        "--count", type=int, default=1000, help="Number of runs for this agent"
    )

    args = parser.parse_args()

    sweep_cfg = load_sweep_config(args.sweep)

    # Create the sweep on W&B
    sweep_id = wandb.sweep(sweep=sweep_cfg, project=args.project, entity=args.entity)
    print(f"[wandb] Created sweep: {sweep_id}")

    # Launch an agent locally to execute N runs sequentially
    wandb.agent(
        sweep_id,
        function=lambda: agent_entry(),
        count=int(args.count),
        project=args.project,
        entity=args.entity,
    )


if __name__ == "__main__":
    main()
