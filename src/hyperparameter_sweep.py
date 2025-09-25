from typing import Mapping

import wandb
from trainer import train


def _wandb_config_to_dict(cfg) -> dict:
    """Convert wandb.config or a plain mapping to a regular Python dict."""
    if hasattr(cfg, "as_dict"):
        return cfg.as_dict()
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {}


def main():
    run = wandb.init(project="kkt_transformer_sweep")
    # Build overrides for your trainer from the run config.
    overrides = _wandb_config_to_dict(wandb.config)
    # Call the training entrypoint. It reuses the active W&B run.
    train(overrides=overrides)
    run.finish()


sweep_configuration = {
    "name": "kkt_transformer",
    "method": "random",
    "metric": {"name": "valid/loss", "goal": "minimize"},
    "early_terminate": {"type": "hyperband", "min_iter": 6},  # stop weak runs early
    "parameters": {
        # Model (mostly fixed, sweep key capacity knobs)
        "gnn_policy": {
            "parameters": {
                "embedding_size": {"values": [32, 64, 128, 256]},
                "cons_nfeats": {"value": 4},
                "edge_nfeats": {"value": 1},
                "var_nfeats": {"value": 18},
                "num_emb_type": {"values": ["periodic", "pwl", "linear"]},
                "num_emb_bins": {"values": [16, 32, 64]},
                "num_emb_freqs": {"values": [8, 16, 24]},
            }
        },
        "transformer": {
            "parameters": {
                "d_model": {"value": 128},
                "nhead": {"values": [4, 8]},
                "dim_feedforward": {"values": [256, 384, 512, 1024]},
                "transformer_dropout": {
                    "distribution": "uniform",
                    "min": 0.1,
                    "max": 0.5,
                },
                "transformer_activation": {"value": "relu"},
                "num_encoder_layers": {"values": [3, 4, 6]},
                "transformer_norm_input": {"value": True},
                "num_encoder_layers_masked": {"value": 0},
                "transformer_prenorm": {"values": [True]},
                "pos_encoder": {"values": [True, False]},
            }
        },
        "training": {
            "parameters": {
                "devices": {"value": "0"},
                "batch_size": {"values": [16, 32, 64]},
                "epochs": {"value": 20},
                "num_workers": {"value": 0},
                "lr": {"values": [1e-4, 2e-4, 3e-4, 5e-4, 8e-4]},
                "max_lr": {"values": [0.001, 0.0015, 0.002]},
                "pct_start": {"values": [0.1, 0.15, 0.3]},
                # Regularization / stability
                "weight_decay": {
                    "distribution": "log_uniform",
                    "min": 1e-6,
                    "max": 3e-2,
                },
                "grad_clip": {"values": [0.5, 1.0, 1.5]},
                "amp": {"value": False},
                "kkt_w_primal": {"values": [0.05, 0.1, 0.15, 0.3, 0.4]},
                "kkt_w_dual": {"values": [0.05, 0.1, 0.15, 0.3, 0.4]},
                "kkt_w_station": {"values": [0.3, 0.4, 0.5, 0.6, 0.7]},
                "kkt_w_comp": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
                # Scheduler and misc
                "scheduler": {"values": ["onecycle", "plateau", "cosine"]},
                "seed": {"value": 42},
            }
        },
    },
}


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="kkt_transformer")
    wandb.agent(sweep_id, function=main)
