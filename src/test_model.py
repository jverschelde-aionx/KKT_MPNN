from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import configargparse
import numpy as np
import pyscipopt as scp
import torch
import torch.nn as nn
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from graphtrans.modules.utils import pad_batch

# --- Your project imports (update paths if your src/ layout differs) ---
from models.gnn_transformer import GNNTransformer
from models.losses import KKTLoss, kkt_metrics
from models.policy_encoder import GraphDataset, PolicyEncoder, collate

# ------------------------------- Utils ------------------------------------- #


def _load_ckpt_args(ckpt_path: Path) -> SimpleNamespace:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "args" not in ckpt:
        raise RuntimeError(f"'args' not found in checkpoint: {ckpt_path}")
    args = SimpleNamespace(**ckpt["args"])
    return args, ckpt


def _find_ckpt_file(maybe_dir_or_file: str) -> Path:
    p = Path(maybe_dir_or_file)
    if p.is_file():
        return p
    elif p.is_dir():
        best = p / "best.pt"
        if best.exists():
            return best
        raise FileNotFoundError(f"'best.pt' not found in directory: {p}")
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")


def _list_split_files(data_root: str, problems: List[str], split: str) -> List[str]:
    files = []
    for problem in problems:
        dir_bg = Path(data_root) / problem / "BG" / split
        if dir_bg.exists():
            for sub in sorted([d for d in dir_bg.iterdir() if d.is_dir()]):
                files.extend([str(sub / f) for f in os.listdir(sub)])
        else:
            logger.warning(f"[{problem}] BG/{split} not found under {dir_bg.parent}.")
    return sorted(files)


def _sol_path_from_bg(bg_path: str) -> Path:
    """
    Map .../BG/<size>/<name>.lp.bg  ->  .../solution/<size>/<name>.lp.sol
    (benign if not standard)
    """
    p = Path(bg_path)
    parts = list(p.parts)
    try:
        idx = parts.index("BG")
    except ValueError:
        return Path()  # not a standard BG path
    parts[idx] = "solution"
    # .bg -> .sol
    stem = p.name[:-3] if p.name.endswith(".bg") else p.name
    sol_name = stem + ".sol"
    return Path(*parts[:-1]) / sol_name


def _lp_path_from_bg(bg_path: str) -> Path:
    """
    Map .../BG/<size>/<name>.lp.bg  ->  .../instance/<size>/<name>.lp
    (benign if not standard)
    """
    p = Path(bg_path)
    parts = list(p.parts)
    try:
        idx = parts.index("BG")
    except ValueError:
        return Path()  # not a standard BG path
    parts[idx] = "instance"
    # .bg -> .sol
    stem = p.name[:-3] if p.name.endswith(".bg") else p.name
    sol_name = stem
    return Path(*parts[:-1]) / sol_name


def _load_bg_meta(bg_path: str) -> Optional[Tuple[dict, np.ndarray]]:
    """
    Returns (v_map: name->idx, b_vars: np.int64[nb]) or None if failed.
    """
    try:
        import pickle

        with open(bg_path, "rb") as f:
            A, v_map, v_nodes, c_nodes, b_vars, b_vec, c_vec = pickle.load(f)
        b_vars = np.asarray(b_vars, dtype=np.int64)
        return v_map, b_vars
    except Exception:
        return None


def _reorder_pool_to_model(
    v_map: dict, var_names: list[str], sols: np.ndarray
) -> Optional[np.ndarray]:
    try:
        perm = np.asarray([v_map[name] for name in var_names], dtype=np.int64)
    except KeyError:
        return None
    out = np.empty_like(sols)
    out[:, perm] = sols
    return out


def _safe_norm_denom(x: float) -> float:
    return float(max(1.0, abs(x)))


# ------------------------- Embedding extraction ---------------------------- #


@torch.no_grad()
def encode_nodes_like_forward(
    model: GNNTransformer,
    batched_data,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reproduce the forward path up to final hidden node embeddings (before heads),
    returning:
      valid_h         : [N_tot, d] embeddings aligned with unpadded node order
      var_mask_1d     : [N_tot] bool mask selecting variables
      con_mask_1d     : [N_tot] bool mask selecting constraints
      graph_id_1d     : [N_tot] int graph index for each node embedding
    """
    device = next(model.parameters()).device

    # GNN node embeddings -> linear to d_model
    h_node = model.gnn_node(batched_data, perturb=None)
    h_node = model.gnn2transformer(
        h_node
    )  # [S_all_nodes, d_model] grouped by graph via Data.batch

    # Left pad to [S, B, d] + collect masks/counts
    padded_h_node, src_padding_mask, _num_nodes, _mask, max_num_nodes = pad_batch(
        h_node, batched_data.batch, get_mask=True
    )  # [S, B, d], [B, S] True at pad positions (left)
    S, B, d = padded_h_node.size(0), padded_h_node.size(1), padded_h_node.size(2)

    # Build node-type layout consistent with your forward()
    batch_vec = batched_data.batch.to(device)
    is_constr = batched_data.is_constr_node.to(device)
    is_var = batched_data.is_var_node.to(device)
    valid_counts = (~src_padding_mask).sum(dim=1).tolist()  # [B]
    m_sizes, n_sizes = [], []
    for i in range(B):
        sel = batch_vec == i
        m_i = int(is_constr[sel].sum().item())
        n_i = int(is_var[sel].sum().item())
        m_sizes.append(m_i)
        n_sizes.append(n_i)
        assert m_i + n_i == valid_counts[i], (m_i, n_i, valid_counts[i])

    # Build [B, S] grids aligned with left padding (same as in forward)
    node_type_grid = torch.zeros(
        (B, S), dtype=torch.long, device=device
    )  # 0=constraint by default
    var_mask_grid = torch.zeros((B, S), dtype=torch.bool, device=device)
    con_mask_grid = torch.zeros((B, S), dtype=torch.bool, device=device)
    for i in range(B):
        n = valid_counts[i]
        if n == 0:
            continue
        m_i, n_i = m_sizes[i], n_sizes[i]
        start = S - n
        split = start + m_i
        con_mask_grid[i, start:split] = True
        var_mask_grid[i, split:S] = True
        node_type_grid[i, split:S] = 1  # 1 = variable

    # Node-type embedding like in your model
    type_emb = model.node_type_embed(node_type_grid)  # [B, S, d]
    transformer_out = padded_h_node + type_emb.transpose(0, 1)  # [S, B, d]

    # Optional positional encoding
    if model.pos_encoder is not None:
        transformer_out = model.pos_encoder(transformer_out)

    # Optional masked encoder layer(s)
    if (
        getattr(model, "num_encoder_layers_masked", 0)
        and model.num_encoder_layers_masked > 0
    ):
        adj_list = getattr(batched_data, "adj_list", None)
        if adj_list is not None:
            padded_adj_list = torch.zeros(
                (len(adj_list), max_num_nodes, max_num_nodes), device=device
            )
            for idx, adj_list_item in enumerate(adj_list):
                N, _ = adj_list_item.shape
                padded_adj_list[idx, 0:N, 0:N] = torch.from_numpy(adj_list_item)
            transformer_out = model.masked_transformer_encoder(
                transformer_out.transpose(0, 1),
                attn_mask=padded_adj_list,
                valid_input_mask=src_padding_mask,
            ).transpose(0, 1)

    # Vanilla encoder
    if getattr(model, "num_encoder_layers", 0) and model.num_encoder_layers > 0:
        transformer_out, _ = model.transformer_encoder(
            transformer_out, src_padding_mask
        )  # [S, B, d], [B, S]

    # Unpad
    out_B_S_d = transformer_out.transpose(0, 1)  # [B, S, d]
    valid_mask = ~src_padding_mask  # [B, S] True where real tokens
    valid_h = out_B_S_d[valid_mask]  # [N_tot, d]
    var_mask_1d = var_mask_grid[valid_mask]  # [N_tot]
    con_mask_1d = con_mask_grid[valid_mask]  # [N_tot]

    # Build graph-id index for valid nodes
    graph_id_1d = []
    for i in range(B):
        graph_id_1d.extend([i] * int(valid_counts[i]))
    graph_id_1d = torch.tensor(graph_id_1d, device=device, dtype=torch.long)

    return valid_h, var_mask_1d, con_mask_1d, graph_id_1d


def _graph_pool_means(H: torch.Tensor, graph_ids: torch.Tensor, B: int) -> torch.Tensor:
    """
    Mean pool over valid nodes per graph.
      H: [N_tot, d], graph_ids: [N_tot] in [0, B-1]
    Returns:
      G: [B, d]
    """
    d = H.size(1)
    G = torch.zeros(B, d, device=H.device)
    counts = torch.zeros(B, device=H.device).float()
    G.index_add_(0, graph_ids, H)
    counts.index_add_(0, graph_ids, torch.ones_like(graph_ids, dtype=torch.float32))
    counts = counts.clamp_min(1.0).unsqueeze(-1)
    G = G / counts
    return G


# ------------------------------- Evaluation -------------------------------- #


@torch.no_grad()
def evaluate_and_collect(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: KKTLoss,
    device: torch.device,
    use_amp: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate over the loader and aggregate:
      - KKT metrics (primal/dual/stationarity/compl_slack)
      - avg objective gap to best pool solution (obj_gap_best)
      - predicted optimality gap: (c^T x - b^T λ) normalized
    Returns: (kkt_avg, extras)
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    term_sums = {"primal": 0.0, "dual": 0.0, "stationarity": 0.0, "compl_slack": 0.0}

    # "Closeness" accumulators
    obj_gap_sum, obj_gap_count = 0.0, 0
    opt_gap_sum, opt_gap_count = 0.0, 0
    eps = 1e-8

    amp_enabled = bool(use_amp)
    autocast = torch.cuda.amp.autocast if amp_enabled else torch.cpu.amp.autocast

    for (
        batch_graph,
        A_list,
        b_pad,
        c_pad,
        b_mask,
        c_mask,
        m_sizes,
        n_sizes,
        sources,
    ) in loader:
        batch_graph = batch_graph.to(device)
        b_pad, c_pad = b_pad.to(device), c_pad.to(device)
        A_list = [A.to(device) if A.device != device else A for A in A_list]

        with autocast():
            x_hat, lam_hat = model(batch_graph)
            A_list = [
                A if A.device == x_hat.device else A.to(x_hat.device, non_blocking=True)
                for A in A_list
            ]
            loss = criterion(x_hat, lam_hat, A_list, b_pad, c_pad, m_sizes, n_sizes)
            kkt = kkt_metrics(x_hat, lam_hat, A_list, b_pad, c_pad, m_sizes, n_sizes)

        total_loss += float(loss.item())
        n_batches += 1
        for k in term_sums:
            term_sums[k] += float(kkt[k])

        # ---- Objective gap to best solution in pool ----
        off_x = 0
        for i in range(len(m_sizes)):
            m_i, n_i = int(m_sizes[i]), int(n_sizes[i])
            x_i = x_hat[off_x : off_x + n_i].detach().float().cpu()
            c_i = c_pad[i, :n_i].detach().float().cpu().numpy()
            off_x += n_i

            sol_path = _sol_path_from_bg(sources[i])
            if not sol_path or not sol_path.exists():
                continue
            try:
                with open(sol_path, "rb") as f:
                    sol_data = pickle.load(f)
                var_names = sol_data["var_names"]
                sols = sol_data["sols"]  # (S, n) Gurobi order
            except Exception:
                continue
            meta = _load_bg_meta(sources[i])
            if meta is None:
                continue
            v_map, _b_vars = meta
            sols_model = _reorder_pool_to_model(v_map, var_names, sols)
            if sols_model is None:
                continue
            pool_obj_min = float(
                (sols_model @ c_i).min()
            )  # best (min) objective in the pool
            x_obj = float((x_i.cpu().numpy() @ c_i))
            obj_gap = (x_obj - pool_obj_min) / (_safe_norm_denom(pool_obj_min) + eps)
            obj_gap_sum += obj_gap
            obj_gap_count += 1

        # ---- Predicted optimality gap ----
        # For each graph, use b^T lambda_hat as dual and c^T x_hat as primal.
        # Slice λ the same way:
        off_l = 0
        off_x = 0
        B = len(m_sizes)
        for i in range(B):
            m_i, n_i = int(m_sizes[i]), int(n_sizes[i])
            lam_i = lam_hat[off_l : off_l + m_i].detach().float()
            x_i = x_hat[off_x : off_x + n_i].detach().float()
            b_i = b_pad[i, :m_i].detach().float()
            c_i = c_pad[i, :n_i].detach().float()
            off_l += m_i
            off_x += n_i

            primal_obj = float(torch.dot(x_i, c_i).item())
            dual_obj = float(
                torch.dot(lam_i.clamp(min=0.0), b_i).item()
            )  # clip negatives if any
            opt_gap_norm = (primal_obj - dual_obj) / (
                _safe_norm_denom(primal_obj) + eps
            )
            opt_gap_sum += opt_gap_norm
            opt_gap_count += 1

    kkt_avg = {k: term_sums[k] / max(n_batches, 1) for k in term_sums}
    extras = {
        "valid/loss": total_loss / max(n_batches, 1),
        "close/obj_gap_best": (obj_gap_sum / obj_gap_count)
        if obj_gap_count
        else float("nan"),
        "close/opt_gap_pred": (opt_gap_sum / opt_gap_count)
        if opt_gap_count
        else float("nan"),
        "close/num_evaluated": obj_gap_count,
    }
    return kkt_avg, extras


# ------------------------------- Plotting ---------------------------------- #


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _plot_2d(
    X2: np.ndarray,
    labels: List[str],
    title: str,
    outfile: Path,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    # Map labels to ints for coloring
    uniq = sorted(list(set(labels)))
    lut = {lab: i for i, lab in enumerate(uniq)}
    colors = [lut[l] for l in labels]
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=colors, s=6, alpha=0.6)
    # Build legend (sample up to 12 entries to keep it readable)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", label=lab) for lab in uniq[:12]
    ]
    plt.legend(handles=handles, title="Label (truncated)", loc="best", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=160)
    plt.close()


def _dimreduce_2d(X: np.ndarray, method: str = "pca", seed: int = 42) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    elif method == "tsne":
        return TSNE(
            n_components=2, learning_rate="auto", init="pca", random_state=seed
        ).fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_embeddings(
    X_all: torch.Tensor,
    all_lab_prob: List[str],
    all_lab_size: List[str],
    G_all: torch.Tensor,
    out_dir: Path,
    run_args: SimpleNamespace,
    args: argparse.Namespace,
    X_con: torch.Tensor,
    X_var: torch.Tensor,
):
    # --- PCA plots ---
    for name, X, labs in [
        ("node_pca_all", X_all, all_lab_prob[: len(X_all)]),
        ("node_pca_constraints", X_con, [all_lab_prob[i] for i in range(len(X_con))]),
        ("node_pca_variables", X_var, [all_lab_prob[i] for i in range(len(X_var))]),
    ]:
        if len(X) >= 2:
            X2 = _dimreduce_2d(X, method="pca", seed=getattr(run_args, "seed", 42))
            _plot_2d(
                X2, labs, f"{name} (PCA, colored by problem)", out_dir / f"{name}.png"
            )
            # Also by size if sizes parsed
            if len(all_lab_size) >= len(X2):
                _plot_2d(
                    X2,
                    all_lab_size[: len(X2)],
                    f"{name} (PCA, colored by size)",
                    out_dir / f"{name}_by_size.png",
                )

    if G_all.shape[0] >= 2:
        G2 = _dimreduce_2d(G_all, method="pca", seed=getattr(run_args, "seed", 42))
        # Build graph labels: approximate by the first occurrence per graph in the concatenation order
        graph_labels = [f"graph_{i}" for i in range(G_all.shape[0])]
        _plot_2d(G2, graph_labels, "graph_pca_all (PCA)", out_dir / "graph_pca_all.png")

    # --- Optional t-SNE (slower) ---
    if args.tsne:
        for name, X, labs in [
            ("node_tsne_all", X_all, all_lab_prob[: len(X_all)]),
            (
                "node_tsne_constraints",
                X_con,
                [all_lab_prob[i] for i in range(len(X_con))],
            ),
            (
                "node_tsne_variables",
                X_var,
                [all_lab_prob[i] for i in range(len(X_var))],
            ),
        ]:
            if len(X) >= 2:
                X2 = _dimreduce_2d(X, method="tsne", seed=getattr(run_args, "seed", 42))
                _plot_2d(
                    X2,
                    labs,
                    f"{name} (t-SNE, colored by problem)",
                    out_dir / f"{name}.png",
                )

        if G_all.shape[0] >= 2:
            G2 = _dimreduce_2d(G_all, method="tsne", seed=getattr(run_args, "seed", 42))
            graph_labels = [f"graph_{i}" for i in range(G_all.shape[0])]
            _plot_2d(
                G2,
                graph_labels,
                "graph_tsne_all (t-SNE)",
                out_dir / "graph_tsne_all.png",
            )

    logger.info("Saved metrics and plots to {}", out_dir)


def load_optimal_solutions(bg_path: str):
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

    max_obj_value = sol_data["objs"].max()
    best_sol_idx = np.argwhere(sol_data["objs"] == max_obj_value)
    best_sols = sols_model[best_sol_idx]
    return best_sols


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
    print("xhat:", x_hat, "lamhat:", lam_hat)
    m, n = int(m_sizes[0]), int(n_sizes[0])
    x_pred = x_hat[:n].detach().cpu().numpy()
    lam_pred = lam_hat[:m].detach().cpu().numpy()
    print("xhat:", x_pred, "lamhat:", lam_pred)


def opt_vs_pred_comparison(
    model: nn.Module, bg_path: str, device: torch.device, outdir: Path
):
    # read in lp file
    lp_path = _lp_path_from_bg(bg_path)
    problem = scp.Model()
    problem.hideOutput(True)
    problem.readProblem(str(lp_path))

    x_opts = load_optimal_solutions(bg_path)
    for x_opt in x_opts:
        print("sol:", x_opt)

    x_pred = predict_single_case(model, bg_path, device)


# --------------------------------- Main ------------------------------------ #


def main():
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        description="evaluate a trained GNN-Transformer and plot embeddings",
        default_config_files=["config.yml"],
    )

    t = parser.add_argument_group("testing")
    t.add_argument(
        "--ckpt", required=True, help="Path to best.pt or dir containing it."
    )
    t.add_argument("--split", choices=["test", "val"], default="test")
    t.add_argument("--out_dir", type=str, default="outputs/eval-embeddings")
    t.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device like '0' or 'cpu' (default: from ckpt args/devices).",
    )
    t.add_argument(
        "--max_points",
        type=int,
        default=50000,
        help="Max node embeddings to plot (downsample if larger).",
    )
    t.add_argument(
        "--tsne",
        action="store_true",
        help="Also produce t-SNE plots (in addition to PCA).",
    )
    t.add_argument("--bg_path", type=str, required=True, help="Path to single .bg file")

    args, _ = parser.parse_known_args()

    print(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoint & its original args ---
    ckpt_file = _find_ckpt_file(args.ckpt)
    run_args, ckpt = _load_ckpt_args(ckpt_file)

    # Device
    if args.device is not None:
        device = torch.device(
            f"cuda:{args.device}"
            if args.device != "cpu" and torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and getattr(run_args, "devices", "")
            else "cpu"
        )
    logger.info(f"Using device: {device}")

    # --- Build dataset (test preferred, else val) ---
    problems = list(getattr(run_args, "problems", ["IS", "CA"]))
    data_root = getattr(run_args, "data_root", "../data")
    files = _list_split_files(data_root, problems, args.split)
    if len(files) == 0 and args.split == "test":
        logger.warning("No 'test' files found; falling back to 'val'.")
        files = _list_split_files(data_root, problems, "val")
    if len(files) == 0:
        raise RuntimeError("No files found for evaluation.")

    dataset = GraphDataset(files)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=getattr(
            run_args, "eval_batch_size", getattr(run_args, "batch_size", 32)
        ),
        shuffle=False,
        num_workers=getattr(run_args, "num_workers", 0),
        collate_fn=collate,
    )

    # --- Infer (m, n) to construct KKTLoss ---
    (batch_graph, _A_list, _b, _c, _bm, _cm, m_sizes, n_sizes, _sources) = next(
        iter(loader)
    )
    m0, n0 = int(m_sizes[0]), int(n_sizes[0])

    # --- Rebuild model exactly as in training ---
    # fill derived feature sizes
    run_args.cons_nfeats = batch_graph.constraint_features.size(1)
    run_args.var_nfeats = batch_graph.variable_features.size(1)
    run_args.edge_nfeats = batch_graph.edge_attr.size(1)

    node_encoder = PolicyEncoder(run_args)
    model = GNNTransformer(args=run_args, gnn_node=node_encoder)
    model = model.to(device)

    state = ckpt["model"]
    # If you ever used DataParallel/Distributed, this strips 'module.' prefixes:
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[debug] load_state_dict: missing:", missing[:10], " ...total:", len(missing))
    print(
        "[debug] load_state_dict: unexpected:",
        unexpected[:10],
        " ...total:",
        len(unexpected),
    )

    # 2) Sanity-check a few key weights/biases
    named = dict(model.named_parameters())
    for key in [
        "gnn2transformer.weight",
        "head.to_xhat.weight",
        "head.to_xhat.bias",
        "head.to_lam.0.weight",
        "head.to_lam.0.bias",
    ]:
        if key in named:
            w = named[key].detach().cpu()
            print(
                f"[debug] {key}: L2={w.norm().item():.6f}  mean={w.mean().item():.6e}  absmean={w.abs().mean().item():.6e}"
            )
        else:
            print(f"[debug] {key}: NOT FOUND in model")

    # 3) Probe representation variation on a single batch
    batch = next(iter(loader))
    batch_graph = batch[0].to(device)
    with torch.no_grad():
        h_node = model.gnn_node(batch_graph)  # [N_tot, emb]
        h_tr = model.gnn2transformer(h_node)  # [N_tot, d_model]
        print(
            "[debug] h_node  std:",
            float(h_node.std().cpu()),
            "  mean:",
            float(h_node.mean().cpu()),
        )
        print(
            "[debug] h_tr    std:",
            float(h_tr.std().cpu()),
            "  mean:",
            float(h_tr.mean().cpu()),
        )

    raise SystemExit()

    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    opt_vs_pred_comparison(model, args.bg_path, device, out_dir)

    # --- Criterion with same KKT weights ---
    kkt_loss = KKTLoss(
        m0,
        n0,
        w_primal=getattr(run_args, "kkt_w_primal", 0.1),
        w_dual=getattr(run_args, "kkt_w_dual", 0.1),
        w_stat=getattr(run_args, "kkt_w_station", 0.6),
        w_comp=getattr(run_args, "kkt_w_comp", 0.2),
    ).to(device)

    # --- Evaluate on the whole split ---
    kkt_avg, extras = evaluate_and_collect(
        model=model,
        loader=loader,
        criterion=kkt_loss,
        device=device,
        use_amp=getattr(run_args, "amp", False),
    )
    metrics = {**{f"kkt/{k}": v for k, v in kkt_avg.items()}, **extras}
    logger.info("Evaluation metrics:\n{}", json.dumps(metrics, indent=2))

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Collect embeddings over the split (may be large; we concatenate lazily) ---
    all_H, all_lab_prob, all_lab_size, all_varmask, all_conmask, all_graphids = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with torch.no_grad():
        for (
            batch_graph,
            _A_list,
            _b,
            _c,
            _bm,
            _cm,
            m_sizes,
            n_sizes,
            sources,
        ) in loader:
            batch_graph = batch_graph.to(device)
            H, vmask, cmask, gids = encode_nodes_like_forward(
                model, batch_graph
            )  # [N_tot, d], [N_tot], [N_tot], [N_tot]
            all_H.append(H)
            all_varmask.append(vmask)
            all_conmask.append(cmask)
            all_graphids.append(gids)

            # Labels from file paths (problem + size)
            for src in sources:
                # src: .../<Problem>/BG/<size>/<file>.bg
                parts = Path(src).parts
                # naive parse: problem is parts[-5] (…/<Problem>/BG/<size>/file)
                try:
                    prob = parts[-5]
                except Exception:
                    prob = "UNK"
                # size is directory under BG
                try:
                    size = parts[-2]
                except Exception:
                    size = "UNK"
                all_lab_prob.extend(
                    [prob] * (int(len(H)) // len(sources))
                )  # balanced approx
                all_lab_size.extend([size] * (int(len(H)) // len(sources)))

    H = torch.cat(all_H, dim=0)  # [N_all, d]
    varmask = torch.cat(all_varmask, dim=0)  # [N_all]
    conmask = torch.cat(all_conmask, dim=0)  # [N_all]
    graphids = torch.cat(all_graphids, dim=0)  # [N_all]

    # Downsample nodes for plotting if needed
    N_all = H.size(0)
    if N_all > args.max_points:
        idx = torch.randperm(N_all)[: args.max_points]
        H = H[idx]
        varmask = varmask[idx]
        conmask = conmask[idx]
        graphids = graphids[idx]
        # labels were approximate; for visuals, just align length
        L = int(H.size(0))
        all_lab_prob = all_lab_prob[:L]
        all_lab_size = all_lab_size[:L]

    X_all = _to_np(H)
    X_con = _to_np(H[conmask])
    X_var = _to_np(H[varmask])

    # Graph-level mean pooling
    B_total = int(graphids.max().item()) + 1 if graphids.numel() > 0 else 0
    G_all = (
        _to_np(_graph_pool_means(H, graphids, B_total))
        if B_total > 0
        else np.zeros((0, H.size(1)))
    )

    plot_embeddings(
        X_all,
        all_lab_prob,
        all_lab_size,
        G_all,
        out_dir,
        run_args,
        args,
        X_con,
        X_var,
    )

    opt_vs_pred_comparison(model, args.bg_path, device, out_dir)


if __name__ == "__main__":
    main()
