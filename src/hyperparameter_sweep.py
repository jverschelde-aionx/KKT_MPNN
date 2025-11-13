"""
Hyperparameter Sweep for Model Scaling Experiments

Tests model variants across problem sizes to measure JEPA impact and scaling behavior.

Model Variants:
- Baselines: MLP, GNN (normalized and unnormalized)
- JEPA: MLP/GNN × EMA/SimSiam (normalized and unnormalized)

Problem Scaling:
- Types: RND (Random LP), CA (Combinatorial Auction)
- Sizes: 2, 5, 10, 20, 50, 100

Usage:
    python hyperparameter_sweep.py --problem-types RND CA --problem-sizes 2 5 10 --epochs 50

    Results automatically exported to Excel after sweep completes.
"""

import argparse
from typing import Mapping

import pandas as pd
import wandb

from train import train


def _wandb_config_to_dict(cfg) -> dict:
    """Convert wandb.config or a plain mapping to a regular Python dict."""
    if hasattr(cfg, "as_dict"):
        return cfg.as_dict()
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {}


def main():
    """Main training function called by WandB sweep agent."""
    run = wandb.init(project="kkt_model_scaling")

    # Build overrides from sweep configuration
    overrides = _wandb_config_to_dict(wandb.config)

    # Flatten nested configuration for train() function
    flat_overrides = {}

    # Data configuration
    if "data" in overrides:
        data_cfg = overrides["data"]
        flat_overrides["problems"] = [data_cfg["problem_type"]]

        # Set sizes based on problem type
        problem_type = data_cfg["problem_type"]
        size = data_cfg["problem_size"]
        if problem_type == "RND":
            flat_overrides["rnd_sizes"] = [size]
        elif problem_type == "CA":
            flat_overrides["ca_sizes"] = [size]
        elif problem_type == "IS":
            flat_overrides["is_sizes"] = [size]
        elif problem_type == "SC":
            flat_overrides["sc_sizes"] = [size]
        elif problem_type == "CFL":
            flat_overrides["cfl_sizes"] = [size]

    # Model configuration
    if "model" in overrides:
        model_cfg = overrides["model"]
        flat_overrides["use_bipartite_graphs"] = model_cfg.get("use_bipartite_graphs", False)

        # Normalization
        if "normalize_features" in model_cfg:
            flat_overrides["normalize_features"] = model_cfg["normalize_features"]

        # JEPA configuration
        if model_cfg.get("use_jepa", False):
            flat_overrides["use_jepa"] = True
            flat_overrides["jepa_mode"] = model_cfg.get("jepa_mode", "ema")
            flat_overrides["jepa_weight"] = model_cfg.get("jepa_weight", 0.2)
            flat_overrides["jepa_pretrain_epochs"] = model_cfg.get("jepa_pretrain_epochs", 3)

            if model_cfg.get("use_bipartite_graphs"):
                # GNN masking
                flat_overrides["jepa_mask_ratio_nodes"] = model_cfg.get("jepa_mask_ratio_nodes", 0.3)
            else:
                # MLP masking
                flat_overrides["jepa_mask_entry_online"] = model_cfg.get("jepa_mask_entry_online", 0.40)
                flat_overrides["jepa_mask_row_online"] = model_cfg.get("jepa_mask_row_online", 0.20)
                flat_overrides["jepa_mask_col_online"] = model_cfg.get("jepa_mask_col_online", 0.20)
                flat_overrides["jepa_mask_entry_target"] = model_cfg.get("jepa_mask_entry_target", 0.10)
                flat_overrides["jepa_mask_row_target"] = model_cfg.get("jepa_mask_row_target", 0.05)
                flat_overrides["jepa_mask_col_target"] = model_cfg.get("jepa_mask_col_target", 0.05)

    # Training configuration
    if "training" in overrides:
        train_cfg = overrides["training"]
        for key in ["batch_size", "epochs", "lr", "devices", "seed"]:
            if key in train_cfg:
                flat_overrides[key] = train_cfg[key]

    # Call training with flattened overrides
    train(overrides=flat_overrides)
    run.finish()


# =============================================================================
# Model Variant Definitions
# =============================================================================

MODEL_VARIANTS = {
    # =========================================================================
    # Baseline Models (No JEPA)
    # =========================================================================
    "mlp_baseline_norm": {
        "use_bipartite_graphs": False,
        "use_jepa": False,
        "normalize_features": True,
    },
    "mlp_baseline_unnorm": {
        "use_bipartite_graphs": False,
        "use_jepa": False,
        "normalize_features": False,
    },
    "gnn_baseline_norm": {
        "use_bipartite_graphs": True,
        "use_jepa": False,
        "normalize_features": True,
    },
    "gnn_baseline_unnorm": {
        "use_bipartite_graphs": True,
        "use_jepa": False,
        "normalize_features": False,
    },

    # =========================================================================
    # JEPA Models - MLP with EMA
    # =========================================================================
    "mlp_jepa_ema_norm": {
        "use_bipartite_graphs": False,
        "use_jepa": True,
        "jepa_mode": "ema",
        "jepa_weight": 0.2,
        "jepa_pretrain_epochs": 3,
        "normalize_features": True,
        "jepa_mask_entry_online": 0.40,
        "jepa_mask_row_online": 0.20,
        "jepa_mask_col_online": 0.20,
        "jepa_mask_entry_target": 0.10,
        "jepa_mask_row_target": 0.05,
        "jepa_mask_col_target": 0.05,
    },
    "mlp_jepa_ema_unnorm": {
        "use_bipartite_graphs": False,
        "use_jepa": True,
        "jepa_mode": "ema",
        "jepa_weight": 0.2,
        "jepa_pretrain_epochs": 3,
        "normalize_features": False,
        "jepa_mask_entry_online": 0.40,
        "jepa_mask_row_online": 0.20,
        "jepa_mask_col_online": 0.20,
        "jepa_mask_entry_target": 0.10,
        "jepa_mask_row_target": 0.05,
        "jepa_mask_col_target": 0.05,
    },

    # =========================================================================
    # JEPA Models - MLP with SimSiam
    # =========================================================================
    "mlp_jepa_simsiam_norm": {
        "use_bipartite_graphs": False,
        "use_jepa": True,
        "jepa_mode": "simsiam",
        "jepa_weight": 0.2,
        "jepa_pretrain_epochs": 0,
        "normalize_features": True,
        "jepa_mask_entry_online": 0.40,
        "jepa_mask_row_online": 0.20,
        "jepa_mask_col_online": 0.20,
        "jepa_mask_entry_target": 0.10,
        "jepa_mask_row_target": 0.05,
        "jepa_mask_col_target": 0.05,
    },
    "mlp_jepa_simsiam_unnorm": {
        "use_bipartite_graphs": False,
        "use_jepa": True,
        "jepa_mode": "simsiam",
        "jepa_weight": 0.2,
        "jepa_pretrain_epochs": 0,
        "normalize_features": False,
        "jepa_mask_entry_online": 0.40,
        "jepa_mask_row_online": 0.20,
        "jepa_mask_col_online": 0.20,
        "jepa_mask_entry_target": 0.10,
        "jepa_mask_row_target": 0.05,
        "jepa_mask_col_target": 0.05,
    },

    # =========================================================================
    # JEPA Models - GNN with EMA
    # =========================================================================
    "gnn_jepa_ema_norm": {
        "use_bipartite_graphs": True,
        "use_jepa": True,
        "jepa_mode": "ema",
        "jepa_weight": 0.2,
        "jepa_pretrain_epochs": 3,
        "normalize_features": True,
        "jepa_mask_ratio_nodes": 0.3,
    },
    "gnn_jepa_ema_unnorm": {
        "use_bipartite_graphs": True,
        "use_jepa": True,
        "jepa_mode": "ema",
        "jepa_weight": 0.2,
        "jepa_pretrain_epochs": 3,
        "normalize_features": False,
        "jepa_mask_ratio_nodes": 0.3,
    },

    # =========================================================================
    # JEPA Models - GNN with SimSiam
    # =========================================================================
    "gnn_jepa_simsiam_norm": {
        "use_bipartite_graphs": True,
        "use_jepa": True,
        "jepa_mode": "simsiam",
        "jepa_weight": 0.2,
        "jepa_pretrain_epochs": 0,
        "normalize_features": True,
        "jepa_mask_ratio_nodes": 0.3,
    },
    "gnn_jepa_simsiam_unnorm": {
        "use_bipartite_graphs": True,
        "use_jepa": True,
        "jepa_mode": "simsiam",
        "jepa_weight": 0.2,
        "jepa_pretrain_epochs": 0,
        "normalize_features": False,
        "jepa_mask_ratio_nodes": 0.3,
    },
}


# =============================================================================
# Sweep Configuration for WandB
# =============================================================================

def create_sweep_config(
    problem_types=["RND", "CA"],
    problem_sizes=[2, 5, 10, 20, 50, 100],
    model_variants=None,
    epochs=50,
    batch_size=256,
):
    """
    Create WandB sweep configuration for grid search over models and problem sizes.

    Args:
        problem_types: List of problem types to test (e.g., ["RND", "CA"])
        problem_sizes: List of problem sizes to test (e.g., [2, 5, 10, 20, 50, 100])
        model_variants: List of model variant keys to test (None = all)
        epochs: Number of training epochs
        batch_size: Batch size for training (256 for MLP, reduce to 128 for GNN if needed)

    Returns:
        Dictionary with WandB sweep configuration
    """
    if model_variants is None:
        model_variants = list(MODEL_VARIANTS.keys())

    # Build model variant configurations for sweep
    model_configs = []
    for variant_key in model_variants:
        if variant_key in MODEL_VARIANTS:
            config = MODEL_VARIANTS[variant_key].copy()
            config["variant_name"] = variant_key
            model_configs.append(config)

    sweep_config = {
        "name": "kkt_model_scaling_experiments",
        "method": "grid",  # Grid search for complete coverage
        "metric": {"name": "valid/loss", "goal": "minimize"},
        "parameters": {
            "data": {
                "parameters": {
                    "problem_type": {"values": problem_types},
                    "problem_size": {"values": problem_sizes},
                }
            },
            "model": {
                "values": model_configs
            },
            "training": {
                "parameters": {
                    "epochs": {"value": epochs},
                    "batch_size": {"value": batch_size},
                    "lr": {"value": 0.001},
                    "devices": {"value": "0"},
                    "seed": {"value": 42},
                }
            },
        },
    }

    return sweep_config


# =============================================================================
# Results Export to Excel
# =============================================================================

def export_results_to_excel(
    project_name="kkt_model_scaling",
    output_file="sweep_results.xlsx",
    entity=None,
):
    """
    Export WandB sweep results to Excel for analysis.

    Args:
        project_name: WandB project name
        output_file: Output Excel filename
        entity: WandB entity/username (None = default)
    """
    api = wandb.Api()

    # Get all runs from project
    if entity:
        runs = api.runs(f"{entity}/{project_name}")
    else:
        runs = api.runs(project_name)

    print(f"\nFound {len(runs)} runs in project '{project_name}'")

    # Collect data
    results = []
    for run in runs:
        if run.state != "finished":
            continue

        # Extract configuration
        config = run.config
        summary = run.summary

        result = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,

            # Problem configuration
            "problem_type": config.get("data", {}).get("problem_type", ""),
            "problem_size": config.get("data", {}).get("problem_size", ""),

            # Model configuration
            "model_variant": config.get("model", {}).get("variant_name", ""),
            "architecture": "GNN" if config.get("model", {}).get("use_bipartite_graphs") else "MLP",
            "normalized": config.get("model", {}).get("normalize_features", True),
            "use_jepa": config.get("model", {}).get("use_jepa", False),
            "jepa_mode": config.get("model", {}).get("jepa_mode", ""),

            # Training configuration
            "epochs": config.get("training", {}).get("epochs", ""),
            "batch_size": config.get("training", {}).get("batch_size", ""),
            "lr": config.get("training", {}).get("lr", ""),

            # Results
            "final_train_loss": summary.get("train/loss", ""),
            "final_valid_loss": summary.get("valid/loss", ""),
            "best_valid_loss": summary.get("best_valid_loss", ""),
            "final_train_loss_kkt": summary.get("train/loss_kkt", ""),
            "final_train_loss_jepa": summary.get("train/loss_jepa", ""),

            # KKT components
            "final_primal": summary.get("valid/primal", ""),
            "final_dual": summary.get("valid/dual", ""),
            "final_stationarity": summary.get("valid/stationarity", ""),
            "final_comp_slack": summary.get("valid/complementary_slackness", ""),

            # Training metadata
            "runtime_seconds": summary.get("_runtime", ""),
            "created_at": run.created_at,
        }

        results.append(result)

    # Write to Excel
    if results:
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✅ Exported {len(results)} results to '{output_file}'")
    else:
        print("⚠️  No finished runs found to export")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model scaling experiments sweep - launches WandB sweep and exports results to Excel"
    )
    parser.add_argument(
        "--project",
        default="kkt_model_scaling",
        help="WandB project name (default: kkt_model_scaling)",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="WandB entity/username (default: your default entity)",
    )
    parser.add_argument(
        "--problem-types",
        nargs="+",
        default=["RND", "CA"],
        help="Problem types to test (default: RND CA)",
    )
    parser.add_argument(
        "--problem-sizes",
        type=int,
        nargs="+",
        default=[2, 5, 10, 20, 50, 100],
        help="Problem sizes to test (default: 2 5 10 20 50 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256 for MLP, reduce to 128 for GNN if needed)",
    )
    parser.add_argument(
        "--model-variants",
        nargs="+",
        default=None,
        help="Model variants to test (default: all 12 variants)",
    )
    parser.add_argument(
        "--export-file",
        default="sweep_results.xlsx",
        help="Output Excel filename (default: sweep_results.xlsx)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip automatic export after sweep (default: auto-export enabled)",
    )

    args = parser.parse_args()

    # Create and launch sweep
    sweep_config = create_sweep_config(
        problem_types=args.problem_types,
        problem_sizes=args.problem_sizes,
        model_variants=args.model_variants,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    num_variants = len(sweep_config['parameters']['model']['values'])
    num_types = len(args.problem_types)
    num_sizes = len(args.problem_sizes)
    total_configs = num_variants * num_types * num_sizes

    print("\n" + "="*70)
    print("  KKT Model Scaling Experiments - WandB Sweep")
    print("="*70)
    print(f"Project: {args.project}")
    print(f"Model variants: {num_variants}")
    print(f"Problem types: {args.problem_types}")
    print(f"Problem sizes: {args.problem_sizes}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Total experiments: {total_configs}")
    print("="*70 + "\n")

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)

    print(f"✅ Sweep created with ID: {sweep_id}\n")
    print("To launch additional parallel agents, run:")
    print(f"  wandb agent {sweep_id}\n")
    print("Starting sweep agent...\n")

    # Launch agent
    try:
        wandb.agent(sweep_id, function=main, project=args.project)
    except KeyboardInterrupt:
        print("\n\n⚠️  Sweep interrupted by user")
    finally:
        # Auto-export results unless disabled
        if not args.no_export:
            print("\n" + "="*70)
            print("  Exporting Results to Excel")
            print("="*70 + "\n")
            export_results_to_excel(
                project_name=args.project,
                output_file=args.export_file,
                entity=args.entity,
            )
            print("\n✅ Sweep complete! Results saved to:", args.export_file)
        else:
            print("\n✅ Sweep complete! (Auto-export disabled)")
