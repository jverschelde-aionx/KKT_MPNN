"""
RQ3 Experiment Evaluation
=========================
Evaluates four already-trained model variants across halo depths {0, 1, 2}:

  1. Full unsplit baseline   — GNNPolicy on the full graph (halo-independent)
  2. Raw split baseline      — split encoder, composer skipped (skip_composer=1)
  3. Block-GNN composer      — split encoder + learned block-GNN composer
  4. Pretrained composer     — same as (3) but initialized from split-pretrained checkpoint

All models are already trained. This script:
  1. Loads each checkpoint
  2. Builds the corresponding model + validation data
  3. Runs eval_epoch to collect KKT metrics
  4. Produces a CSV comparison table + grouped bar chart

Usage:
    # From src/ directory, with graph-aug conda env active:
    python -m jobs.run_rq3_experiments
    python -m jobs.run_rq3_experiments --output_dir ./results/rq3
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent.parent  # .../KKT_MPNN/src

SPLIT_EXPERIMENTS_DIR = SRC_DIR / "experiments" / "split_bijepa_finetuning"
FULL_EXPERIMENTS_DIR = SRC_DIR / "experiments" / "gnn_finetuning"

# Full unsplit baseline config (needed to rebuild model + data)
FULL_BASELINE_CONFIG = (
    SRC_DIR / "configs" / "finetune" / "finetune_CA_200"
    / "finetune_CA_200_gnn_baseline.yml"
)

# Split finetune configs (used to rebuild model + data for split variants)
SPLIT_CONFIG_DIR = SRC_DIR / "configs" / "finetune_splits"

HALO_DEPTHS = [0, 1, 2]
SPLIT_VARIANTS = ["RAW", "COMPOSER", "PRETRAINED"]

# ---------------------------------------------------------------------------
# Experiment registry: (label, config_path, checkpoint_path, pipeline)
# pipeline: "full" = GNNPolicy via jobs.finetune, "split" = SplitBlockBiJepaPolicy
# ---------------------------------------------------------------------------

def _find_full_baseline_checkpoint() -> Optional[Path]:
    """Find step_005000.pt for the CA_200 GNN baseline (matches 5000-step budget of split models)."""
    base = FULL_EXPERIMENTS_DIR / "finetune_CA_200" / "finetune_CA_200_gnn_baseline"
    candidates = sorted(base.rglob("step_005000.pt"))
    return candidates[0] if candidates else None


def _find_split_checkpoint(variant: str, halo: int) -> Optional[Path]:
    """Find best.pt for a split variant."""
    name = f"finetune_split_CA_200_{variant}_h{halo}"
    ckpt = SPLIT_EXPERIMENTS_DIR / name / "best.pt"
    return ckpt if ckpt.exists() else None


def _find_split_config(variant: str, halo: int) -> Optional[Path]:
    """Find the config used to train a split variant."""
    name = f"finetune_split_CA_200_{variant}_h{halo}"
    cfg = SPLIT_CONFIG_DIR / f"{name}.yml"
    return cfg if cfg.exists() else None


def build_experiment_registry() -> List[Dict]:
    """Build the list of all experiments to evaluate."""
    experiments = []

    # Full-graph GNN baseline
    ckpt = _find_full_baseline_checkpoint()
    experiments.append({
        "name": "Full-graph GNN",
        "label": "Full-graph GNN",
        "halo": "-",
        "pipeline": "full",
        "config": FULL_BASELINE_CONFIG,
        "checkpoint": ckpt,
    })

    # Split variants
    label_map = {
        "RAW": "Split (no composer)",
        "COMPOSER": "Split + Composer",
        "PRETRAINED": "Split + Composer (pretrained)",
    }
    for halo in HALO_DEPTHS:
        for variant in SPLIT_VARIANTS:
            ckpt = _find_split_checkpoint(variant, halo)
            cfg = _find_split_config(variant, halo)
            experiments.append({
                "name": f"{label_map[variant]} h={halo}",
                "label": f"{label_map[variant]} h={halo}",
                "halo": halo,
                "pipeline": "split",
                "config": cfg,
                "checkpoint": ckpt,
            })

    return experiments


# ---------------------------------------------------------------------------
# Evaluation: full (GNNPolicy) model
# ---------------------------------------------------------------------------

def eval_full_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> Dict[str, float]:
    """Load a full GNNPolicy checkpoint and evaluate on test set."""
    from jobs.finetune import build_arg_parser, eval_epoch
    from jobs.utils import build_dataloaders, set_all_seeds
    from models.gnn import GNNPolicy

    parser = build_arg_parser()
    args, _ = parser.parse_known_args(args=["--config", str(config_path)])
    GNNPolicy.add_args(parser)
    args, _ = parser.parse_known_args(args=["--config", str(config_path)])
    args.epochs = 1

    set_all_seeds(args.seed)

    _, _, test_loader, N_max, M_max = build_dataloaders(
        args, None, None, for_pretraining=False
    )

    model = GNNPolicy(args).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = eval_epoch(
        model=model,
        loader=test_loader,
        device=device,
        primal_weight=args.primal_weight,
        dual_weight=args.dual_weight,
        stationarity_weight=args.stationarity_weight,
        complementary_slackness_weight=args.complementary_slackness_weight,
    )
    return metrics


# ---------------------------------------------------------------------------
# Evaluation: split (SplitBlockBiJepaPolicy) model
# ---------------------------------------------------------------------------

def eval_split_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> Dict[str, float]:
    """Load a SplitBlockBiJepaPolicy checkpoint and evaluate on test set."""
    from torch.utils.data import DataLoader

    from data.split import SplitInstanceDataset, split_instance_collate
    from jobs.finetune_split import _collect_split_dirs, build_arg_parser, eval_epoch
    from jobs.utils import set_all_seeds
    from models.split import SplitBlockBiJepaPolicy

    parser = build_arg_parser()
    args, _ = parser.parse_known_args(args=["--config", str(config_path)])
    args.epochs = 1

    set_all_seeds(args.seed)

    # Build test loader
    test_dirs = _collect_split_dirs(args, "test")

    if args.n_instances is not None:
        max_test = round(args.n_instances * args.val_split)
    else:
        max_test = None

    test_ds = SplitInstanceDataset(roots=test_dirs, max_instances=max_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=split_instance_collate,
    )

    # Build and load model
    model = SplitBlockBiJepaPolicy(args).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = eval_epoch(
        model=model,
        loader=test_loader,
        device=device,
        primal_weight=args.primal_weight,
        dual_weight=args.dual_weight,
        stationarity_weight=args.stationarity_weight,
        complementary_slackness_weight=args.complementary_slackness_weight,
    )
    return metrics


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(output_dir: Path, device: torch.device) -> None:
    """Evaluate all RQ3 experiments and produce results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = build_experiment_registry()

    rows: List[Dict] = []

    for exp in experiments:
        name = exp["name"]
        logger.info("Evaluating: {}", name)

        if exp["checkpoint"] is None:
            logger.warning("  {} — checkpoint not found, skipping", name)
            rows.append(_empty_row(exp, status="missing_checkpoint"))
            continue

        if exp["config"] is None:
            logger.warning("  {} — config not found, skipping", name)
            rows.append(_empty_row(exp, status="missing_config"))
            continue

        try:
            if exp["pipeline"] == "full":
                metrics = eval_full_model(exp["config"], exp["checkpoint"], device)
            else:
                metrics = eval_split_model(exp["config"], exp["checkpoint"], device)

            row = {
                "name": exp["name"],
                "label": exp["label"],
                "halo": exp["halo"],
                "kkt_loss": metrics.get("valid/kkt_loss", ""),
                "primal": metrics.get("valid/primal_feasibility", ""),
                "dual": metrics.get("valid/dual_feasibility", ""),
                "stationarity": metrics.get("valid/stationarity", ""),
                "comp_slack": metrics.get("valid/complementary_slackness", ""),
                "objective_gap": metrics.get("valid/objective_gap", ""),
                "optimality_gap": metrics.get("valid/duality_gap", ""),  # eval_epoch calls it duality_gap
                "status": "ok",
            }
            rows.append(row)
            logger.info("  {} — KKT={:.6f}  obj_gap={:.6f}",
                        name, row["kkt_loss"], row["objective_gap"] if isinstance(row["objective_gap"], float) else float("nan"))

        except Exception as e:
            logger.error("  {} — evaluation failed: {}", name, e)
            rows.append(_empty_row(exp, status=f"error: {e}"))

    # Write CSV
    csv_path = output_dir / "rq3_results.csv"
    fieldnames = ["name", "label", "halo", "kkt_loss", "primal", "dual",
                  "stationarity", "comp_slack", "objective_gap", "optimality_gap", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved results to {}", csv_path)

    _print_summary_table(rows)
    _generate_plot(rows, output_dir)


def _empty_row(exp: Dict, status: str) -> Dict:
    return {
        "name": exp["name"], "label": exp["label"], "halo": exp["halo"],
        "kkt_loss": "", "primal": "", "dual": "", "stationarity": "",
        "comp_slack": "", "objective_gap": "", "optimality_gap": "", "status": status,
    }


# ---------------------------------------------------------------------------
# Output: summary table
# ---------------------------------------------------------------------------

def _print_summary_table(rows: List[Dict]) -> None:
    header = f"{'Label':<25s} {'Halo':<6s} {'KKT Loss':<12s} {'Obj Gap':<12s} {'Opt Gap':<12s} {'Status':<10s}"
    print("\n" + "=" * len(header))
    print("RQ3 Results Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for row in rows:
        kkt = f"{row['kkt_loss']:.6f}" if isinstance(row['kkt_loss'], (int, float)) else str(row['kkt_loss'])
        obj = f"{row['objective_gap']:.6f}" if isinstance(row['objective_gap'], (int, float)) else str(row['objective_gap'])
        dg = f"{row['optimality_gap']:.6f}" if isinstance(row['optimality_gap'], (int, float)) else str(row['optimality_gap'])
        print(f"{row['label']:<25s} {str(row['halo']):<6s} {kkt:<12s} {obj:<12s} {dg:<12s} {row['status']:<10s}")

    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Output: comparison plot
# ---------------------------------------------------------------------------

def _generate_plot(rows: List[Dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available — skipping plot generation")
        return

    variant_labels = [
        ("Full-graph GNN", "Full-graph GNN"),
        ("Split (no composer)", "Split (no composer)"),
        ("Split + Composer", "Split + Composer"),
        ("Split + Composer (pretrained)", "Split + Composer (pretrained)"),
    ]
    halo_labels = ["h=0", "h=1", "h=2"]

    data = {row["label"]: row for row in rows}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_to_plot = [
        ("kkt_loss", "KKT Loss"),
        ("objective_gap", "Objective Gap"),
        ("optimality_gap", "Optimality Gap"),
    ]

    for ax, (metric, title) in zip(axes, metrics_to_plot):
        x = np.arange(len(halo_labels))
        width = 0.2

        for i, (variant_key, variant_display) in enumerate(variant_labels):
            values = []
            for halo in HALO_DEPTHS:
                if variant_key == "Full-graph GNN":
                    label = "Full-graph GNN"
                else:
                    label = f"{variant_key} h={halo}"
                val = data.get(label, {}).get(metric, None)
                values.append(val if isinstance(val, (int, float)) else 0)

            if variant_key == "Full-graph GNN":
                full_val = data.get("Full-graph GNN", {}).get(metric, None)
                if isinstance(full_val, (int, float)):
                    ax.axhline(y=full_val, color="black", linestyle="--",
                               linewidth=1.5, label="Full-graph GNN")
            else:
                offset = (i - 1.5) * width
                ax.bar(x + offset, values, width, label=variant_display)

        ax.set_xlabel("Halo Depth")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(halo_labels)

    # Single shared legend below the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("RQ3: Model Variant Comparison Across Halo Depths", fontsize=13)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    plot_path = output_dir / "rq3_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to {}", plot_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def replot(output_dir: Path) -> None:
    """Regenerate table + plot from existing CSV without re-running evaluation."""
    csv_path = output_dir / "rq3_results.csv"
    if not csv_path.exists():
        logger.error("No CSV found at {} — run evaluation first.", csv_path)
        return

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Convert numeric fields back from strings
            for key in ["kkt_loss", "primal", "dual", "stationarity",
                        "comp_slack", "objective_gap", "optimality_gap"]:
                val = row.get(key, "")
                if val != "":
                    try:
                        row[key] = float(val)
                    except ValueError:
                        pass
            rows.append(row)

    _print_summary_table(rows)
    _generate_plot(rows, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 experiment evaluation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: experiments/rq3/results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for evaluation (default: cuda:0)",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Regenerate table + plot from existing CSV without re-running evaluation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SRC_DIR / "experiments" / "rq3" / "results"

    if args.plot_only:
        replot(output_dir)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        run_evaluation(output_dir, device)


if __name__ == "__main__":
    main()
