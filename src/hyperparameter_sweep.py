from typing import Mapping

import wandb

from trainer_old import train


def _wandb_config_to_dict(cfg) -> dict:
    """Convert wandb.config or a plain mapping to a regular Python dict."""
    if hasattr(cfg, "as_dict"):
        return cfg.as_dict()
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {}


def main():
    run = wandb.init(project="kkt_transformer_sweep_2")
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
        "gnn_policy": {
            "parameters": {
                "embedding_size": {"values": [128]},
                "cons_nfeats": {"value": 4},
                "edge_nfeats": {"value": 1},
                "var_nfeats": {"value": 18},
                "num_emb_type": {"value": "periodic"},
                "num_emb_bins": {"value": 32},
                "num_emb_freqs": {"values": [16]},
            }
        },
        "transformer": {
            "parameters": {
                "d_model": {"value": 128},
                "nhead": {"value": 8},
                "dim_feedforward": {"values": [512, 1024, 256]},
                "transformer_dropout": {
                    "distribution": "uniform",
                    "min": 0.08,
                    "max": 0.60,
                },
                "transformer_activation": {"value": "relu"},
                "num_encoder_layers": {"value": 4},
                "transformer_norm_input": {"value": True},
                "num_encoder_layers_masked": {"value": 0},
                "transformer_prenorm": {"value": True},
                "pos_encoder": {"value": False},
            }
        },
        "training": {
            "parameters": {
                "devices": {"value": "0"},
                "batch_size": {"values": [16, 32, 64]},
                "epochs": {"value": 20},
                "num_workers": {"value": 0},
                "lr": {
                    "distribution": "log_uniform",
                    "min": 2e-4,
                    "max": 2.2e-3,
                },
                "max_lr": {
                    "distribution": "uniform",
                    "min": 1.5e-3,
                    "max": 2.2e-3,
                },
                "pct_start": {"values": [0.1, 0.2, 0.3]},
                "weight_decay": {
                    "distribution": "log_uniform",
                    "min": 1e-6,
                    "max": 3e-2,
                },
                "grad_clip": {"value": 1.5},
                "amp": {"value": False},
                "kkt_w_primal": {"value": 0.1},
                "kkt_w_dual": {"value": 0.1},
                "kkt_w_station": {"value": 0.6},
                "kkt_w_comp": {"value": 0.2},
                "scheduler": {"values": ["onecycle", "cosine", "plateau"]},
                "seed": {"value": 42},
                "early_stop": {"value": True},
            }
        },
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="kkt_transformer")
    wandb.agent(sweep_id, function=main)
