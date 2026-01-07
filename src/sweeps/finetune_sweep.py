from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

import wandb
from jobs.finetune import finetune


def load_sweep_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"sweep file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("sweep.yaml must parse to a dict")
    return cfg


def _scenario_overrides(
    scenario: str,
    encoder_path_cli: Optional[Path],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns a dict of overrides for the selected scenario.

    Priority for encoder_path:
      1) sweep config: wandb.config.get("encoder_path")
      2) CLI --encoder_path (if provided)
    """
    o: Dict[str, Any] = {}

    # Prefer encoder_path from sweep config if present; else CLI arg
    encoder_path_cfg = cfg.get("encoder_path", None)
    encoder_path = Path(encoder_path_cfg) if encoder_path_cfg else encoder_path_cli

    if scenario == "baseline":
        # Train from scratch
        o["finetune_mode"] = "full"
        o["encoder_path"] = None

    elif scenario == "mlp_baseline":
        # Train from scratch
        o["finetune_mode"] = "full"
        o["encoder_path"] = None
        o["use_bipartite_graphs"] = False

    elif scenario == "pretrained_frozen":
        if not encoder_path or not encoder_path.exists():
            raise FileNotFoundError(
                "pretrained_frozen scenario requires a valid encoder_path "
                "(provide in sweep.yaml or via --encoder_path)."
            )
        o["finetune_mode"] = "heads"
        o["encoder_path"] = str(encoder_path)

    elif scenario == "pretrained_full":
        if not encoder_path or not encoder_path.exists():
            raise FileNotFoundError(
                "pretrained_full scenario requires a valid encoder_path "
                "(provide in sweep.yaml or via --encoder_path)."
            )
        o["finetune_mode"] = "full"
        o["encoder_path"] = str(encoder_path)

    elif scenario == "random_frozen":
        # DIAGNOSTIC 4: Train with random frozen encoder (control for pretrained_frozen)
        # This tests whether pretrained encoder provides any benefit over random encoder
        o["finetune_mode"] = "heads"
        o["encoder_path"] = None

    else:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            "Expected one of: baseline, pretrained_frozen, pretrained_full, random_frozen."
        )

    return o


def agent_entry(encoder_path_cli: Optional[Path]) -> None:
    """
    One sweep trial executed by the W&B agent.
    - Start run
    - Read selected 'scenario' from wandb.config
    - Build overrides: (all sweep params) + (scenario-specific params)
    - Call finetune_train(overrides=...)
    """
    run = wandb.init()
    if run is None:
        raise RuntimeError("wandb.init() returned None")

    # All sweep parameters are visible here
    cfg: Dict[str, Any] = dict(wandb.config)

    # Which scenario?
    scenario = str(cfg["scenario"])

    # Start with the sweep params so everything you listed in sweep.yaml is forwarded
    overrides: Dict[str, Any] = dict(cfg)

    # Apply scenario-specific overrides (encoder_path + finetune_mode)
    overrides.update(_scenario_overrides(scenario, encoder_path_cli, cfg))

    # For baseline scenarios, explicitly remove encoder_path to prevent accidental loading
    if scenario in ["baseline", "mlp_baseline", "random_frozen"]:
        overrides.pop("encoder_path", None)

    # Reflect effective config in W&B UI
    wandb.config.update(overrides, allow_val_change=True)

    # Launch finetuning
    finetune(overrides=overrides)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="W&B sweep for finetuning: baseline vs pretrained (frozen/full)."
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("sweeps/configs/diagnostic_frozen_comparison.yaml"),
        help="Path to W&B sweep.yaml describing parameters (must include 'scenario').",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="kkt_gnn_finetuning",
        help="W&B project name",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of runs this agent executes (3 scenarios).",
    )
    parser.add_argument(
        "--encoder_path",
        type=Path,
        default=None,
        help="Fallback: path to pretrained encoder (used if not provided in sweep.yaml).",
    )
    args, _ = parser.parse_known_args()

    sweep_cfg = load_sweep_config(args.sweep)

    # Create the sweep on W&B
    sweep_id = wandb.sweep(sweep=sweep_cfg, project=args.project)
    print(f"[wandb] Created sweep: {sweep_id}")

    # Launch an agent locally to execute the runs sequentially
    wandb.agent(
        sweep_id,
        function=lambda: agent_entry(args.encoder_path),
        count=int(args.count),
        project=args.project,
    )


if __name__ == "__main__":
    main()
