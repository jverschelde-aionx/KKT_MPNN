"""
Plot predicted vs optimal (solver pool best) variable values for a single test case.

Usage example:
  python plot_case_predictions.py \
      --checkpoint exps/IS+GNNT=gcn+emb=300+lr=0.0005+0923-191400/best.pt \
      --bg_path ../data/IS/BG/test/some_instance.lp.bg \
      --outdir plots/case_001 --topk 40 --device cuda:0

This script expects your training checkpoint to contain an "args" dict (as saved
by your trainer) so it can reconstruct the model exactly.
"""

import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import configargparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# --- Your project imports (match trainer.py) ---
from models.gnn_transformer import GNNTransformer
from models.policy_encoder import GraphDataset, PolicyEncoder, collate


# ----------------------------- Helpers (copied from trainer) ----------------------------- #
def _sol_path_from_bg(bg_path: str) -> Path:
    """Map .../BG/<size>/<name>.lp.bg  ->  .../solution/<size>/<name>.lp.sol"""
    p = Path(bg_path)
    parts = list(p.parts)
    try:
        idx = parts.index("BG")
    except ValueError:
        return Path()  # not a standard BG path
    parts[idx] = "solution"
    stem = p.name[:-3] if p.name.endswith(".bg") else p.name
    sol_name = stem + ".sol"
    return Path(*parts[:-1]) / sol_name


def _load_bg_meta(bg_path: str) -> Optional[Tuple[dict, np.ndarray]]:
    """
    Returns (v_map: name->idx, b_vars: np.int64[nb]) or None if failed.
    v_map comes from the generator's BG pickle (name order -> model index).
    """
    try:
        with open(bg_path, "rb") as f:
            A, v_map, v_nodes, c_nodes, b_vars, b_vec, c_vec = pickle.load(f)
        b_vars = np.asarray(b_vars, dtype=np.int64)
        return v_map, b_vars
    except Exception:
        return None


def _reorder_pool_to_model(
    v_map: dict, var_names: list, sols: np.ndarray
) -> Optional[np.ndarray]:
    """Reorder solution pool columns to match model (BG) order using v_map."""
    try:
        perm = np.asarray([v_map[name] for name in var_names], dtype=np.int64)
    except KeyError:
        return None
    out = np.empty_like(sols)
    out[:, perm] = sols
    return out


# ----------------------------- Checkpoint / Model loading ----------------------------- #
def _namespace_from_args_dict(d: dict) -> SimpleNamespace:
    """Convert the 'args' dict saved in the checkpoint to a Namespace with safe defaults."""
    ns = SimpleNamespace(**d)
    # Gentle defaults if missing
    if not hasattr(ns, "devices"):
        ns.devices = "0"
    if not hasattr(ns, "transformer_norm_input"):
        ns.transformer_norm_input = True
    if not hasattr(ns, "transformer_prenorm"):
        ns.transformer_prenorm = True
    if not hasattr(ns, "pos_encoder"):
        ns.pos_encoder = False
    return ns


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "args" not in ckpt:
        raise RuntimeError(
            "Checkpoint is missing 'args'. Re-train with trainer.py that saves args."
        )
    args = _namespace_from_args_dict(ckpt["args"])

    # Build model in the exact same way as trainer.py
    node_encoder = PolicyEncoder(args)
    model = GNNTransformer(args=args, gnn_node=node_encoder).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print("[warn] load_state_dict: missing:", missing, "unexpected:", unexpected)
    model.eval()
    return model, args


# ----------------------------- Inference on a single BG case ----------------------------- #
@torch.no_grad()
def predict_single_case(model, bg_path: str, device: torch.device):
    """Return (x_pred, lam_pred, c_vec, (m, n)) for the single instance at bg_path."""
    ds = GraphDataset([bg_path])
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    batch = next(iter(dl))

    (batch_graph, A_list, b_pad, c_pad, b_mask, c_mask, m_sizes, n_sizes, sources) = (
        batch
    )
    assert len(sources) == 1, "Expected a single instance"
    batch_graph = batch_graph.to(device)
    b_pad = b_pad.to(device)
    c_pad = c_pad.to(device)

    x_hat, lam_hat = model(batch_graph)  # concatenated across batch

    m, n = int(m_sizes[0]), int(n_sizes[0])
    x_pred = x_hat[:n].detach().cpu().numpy()
    lam_pred = lam_hat[:m].detach().cpu().numpy()
    c_vec = c_pad[0, :n].detach().cpu().numpy()
    return x_pred, lam_pred, c_vec, (m, n), sources[0]


def load_optimal_solution(bg_path: str, c_vec: np.ndarray) -> Optional[np.ndarray]:
    """
    Load the solution pool, reorder columns to model order, and return the BEST solution
    under min objective (c^T x). Returns None if not available.
    """
    sol_path = _sol_path_from_bg(bg_path)
    if (not sol_path) or (not sol_path.exists()):
        print(f"[warn] No .sol file found for {bg_path} -> {sol_path}")
        return None

    try:
        with open(sol_path, "rb") as f:
            sol_data = pickle.load(f)
        var_names = sol_data["var_names"]
        sols = np.asarray(sol_data["sols"])  # (S, n) in Gurobi order
    except Exception as e:
        print(f"[warn] Failed to read solution pool at {sol_path}: {e}")
        return None

    meta = _load_bg_meta(bg_path)
    if meta is None:
        print(f"[warn] BG meta not found for {bg_path}")
        return None
    v_map, _b_vars = meta

    sols_model = _reorder_pool_to_model(v_map, var_names, sols)
    if sols_model is None:
        print("[warn] Could not reorder solution pool to model order.")
        return None

    # Pick the pool member with best (minimum) objective
    obj_vals = sols_model @ c_vec  # shape (S,)
    best_idx = int(np.argmin(obj_vals))
    return sols_model[best_idx].astype(np.float32)


# ----------------------------- Plotting ----------------------------- #
def _ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_pred_vs_opt(
    x_pred: np.ndarray, x_opt: np.ndarray, outdir: str, case_name: str, topk: int = 40
):
    outdir_p = _ensure_outdir(outdir)
    n = x_pred.shape[0]
    idx = np.arange(n)

    # 1) Full plot (may be dense if n is large)
    fig, ax = plt.subplots(figsize=(min(20, 2 + 0.02 * n), 5))
    width = 0.45
    print("x_opt:", x_opt)
    print("x_pred:", x_pred)
    ax.bar(idx - width / 2, x_opt, width, label="Optimal")
    ax.bar(idx + width / 2, x_pred, width, label="Predicted")
    ax.set_title(f"Predicted vs Optimal (n={n})")
    ax.set_xlabel("Variable index")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir_p / f"{case_name}_pred_vs_opt_full.png", dpi=200)
    plt.close(fig)

    # 2) Top-K largest absolute deviations
    diff = np.abs(x_pred - x_opt)
    order = np.argsort(-diff)
    k = min(topk, n)
    top_idx = order[:k]
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * k), 5))
    ax.bar(np.arange(k) - 0.45 / 2, x_opt[top_idx], 0.45, label="Optimal")
    ax.bar(np.arange(k) + 0.45 / 2, x_pred[top_idx], 0.45, label="Predicted")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([str(i) for i in top_idx], rotation=90)
    ax.set_title(f"Top-{k} |Δx| variables")
    ax.set_xlabel("Variable index (by |Δx|)")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir_p / f"{case_name}_pred_vs_opt_top{k}.png", dpi=200)
    plt.close(fig)


# ----------------------------- CLI ----------------------------- #
def main():
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        description="evaluate a trained GNN-Transformer and plot embeddings",
        default_config_files=["config.yml"],
    )

    t = parser.add_argument_group("testing")

    t.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")
    t.add_argument("--bg_path", type=str, required=True, help="Path to single .bg file")
    t.add_argument("--outdir", type=str, default="outputs", help="Where to save plots")
    t.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cuda:0 | cuda | cpu (if CUDA not available, falls back to cpu)",
    )
    t.add_argument(
        "--topk", type=int, default=40, help="Top-K variables to show in zoomed plot"
    )
    args, _ = parser.parse_known_args()

    # Device
    if args.device is not None:
        device = torch.device(
            f"cuda:{args.device}"
            if args.device != "cpu" and torch.cuda.is_available()
            else "cpu"
        )

    print(f"[info] Using device: {device}")

    # Load model
    model, margs = load_model_from_checkpoint(args.ckpt, device)

    # Predict on single case
    x_pred, lam_pred, c_vec, (m, n), src_path = predict_single_case(
        model, args.bg_path, device
    )
    print(f"[info] Case: {src_path} | m={m}, n={n}")

    # Load optimal from pool
    x_opt = load_optimal_solution(src_path, c_vec)
    if x_opt is None:
        raise SystemExit("[error] Could not retrieve optimal solution for plotting.")

    # Objectives + gaps
    eps = 1e-8
    obj_pred = float(x_pred @ c_vec)
    obj_opt = float(x_opt @ c_vec)
    gap_abs = obj_pred - obj_opt
    gap_rel = gap_abs / (abs(obj_opt) + eps)

    print(f"[metrics] Objective(optimal)   = {obj_opt:,.6f}")
    print(f"[metrics] Objective(predicted) = {obj_pred:,.6f}")
    print(f"[metrics] Objective gap (abs)  = {gap_abs:,.6f}")
    print(f"[metrics] Objective gap (rel)  = {gap_rel:.6%}")

    # Plots
    case_name = Path(src_path).stem.replace(".lp", "")
    plot_pred_vs_opt(x_pred, x_opt, args.outdir, case_name, topk=args.topk)

    # Save arrays for further analysis (optional)
    outdir_p = _ensure_outdir(args.outdir)
    np.save(outdir_p / f"{case_name}_x_pred.npy", x_pred)
    np.save(outdir_p / f"{case_name}_x_opt.npy", x_opt)
    np.save(outdir_p / f"{case_name}_c.npy", c_vec)
    print(f"[done] Saved plots and arrays to: {outdir_p.resolve()}")


if __name__ == "__main__":
    main()
