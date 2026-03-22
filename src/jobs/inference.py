"""
Standalone inference script.

Loads a trained model (GNN or MLP), runs it on a data split, and saves
predicted primal solutions x and dual multipliers lambda to a JSON file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import configargparse
import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Batch as PyGBatch

from data.common import ProblemClass
from jobs.utils import build_dataloaders, device_from_args, pack_by_sizes, set_all_seeds
from metrics.optimization import get_optimal_solution
from models.gnn import GNNPolicy
from models.genetic import EvoConfig, evolve_binary_solution_batch
from models.mlp import KKTNetMLP


def build_arg_parser() -> configargparse.ArgumentParser:
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        default_config_files=[
            "configs/finetune/finetune_CA_200/finetune_CA_200_gnn_baseline.yml"
        ],
    )
    parser.add_argument("--config", is_config_file=True, help="Path to config YAML")

    # Inference
    t = parser.add_argument_group("inference")
    t.add_argument("--devices", type=str, default="0")
    t.add_argument("--batch_size", type=int, default=64)
    t.add_argument("--num_workers", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
    )
    t.add_argument(
        "--checkpoint",
        type=str,
        default="/home/joachim-verschelde/Repos/KKT_MPNN/src/experiments/kkt_gnn_node_finetuning_experiments/200/50-genial-wind-ca-gnn/best.pt",
        help="Path to full model checkpoint (.pt with 'model' key). "
        "Takes precedence over --encoder_path.",
    )
    t.add_argument(
        "--encoder_path",
        type=str,
        default=None,
        help="Path to encoder-only checkpoint (used when --checkpoint is not given).",
    )
    t.add_argument(
        "--output_dir",
        type=str,
        default="results/inference",
    )
    t.add_argument("--method_tag", type=str, default=None)

    # Data (mirrors finetune parser)
    d = parser.add_argument_group("data")
    d.add_argument("--use_bipartite_graphs", action="store_true")
    d.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=[ProblemClass.INDEPENDANT_SET, ProblemClass.COMBINATORIAL_AUCTION],
    )
    d.add_argument("--is_sizes", type=int, nargs="+", default=[200])
    d.add_argument("--ca_sizes", type=int, nargs="+", default=[200])
    d.add_argument("--sc_sizes", type=int, nargs="+", default=[200])
    d.add_argument("--cfl_sizes", type=int, nargs="+", default=[200])
    d.add_argument("--rnd_sizes", type=int, nargs="+", default=[200])
    d.add_argument("--n_instances", type=int, default=50000)
    d.add_argument("--data_root", type=str, default="data/instances/milp")
    d.add_argument("--val_split", type=float, default=0.15)

    # GA decoding
    g = parser.add_argument_group("ga_decode")
    g.add_argument("--ga_pop_size", type=int, default=128)
    g.add_argument("--ga_generations", type=int, default=60)
    g.add_argument("--ga_logits", action="store_true", default=True)
    g.add_argument("--ga_temperature", type=float, default=1.0)
    g.add_argument("--ga_rho_multiplier", type=float, default=10.0)
    g.add_argument("--ga_gamma_lambda", type=float, default=1.0)
    g.add_argument("--ga_maximize", action="store_true", default=False)
    g.add_argument("--ga_prefer_feasible", action="store_true", default=False)

    return parser


def _infer_method_tag(args) -> str:
    if args.method_tag:
        return args.method_tag
    if args.checkpoint:
        return Path(args.checkpoint).stem
    if args.encoder_path:
        return Path(args.encoder_path).stem
    return "model"


@torch.no_grad()
def run_inference(args) -> List[dict]:
    set_all_seeds(args.seed)
    device = device_from_args(args)

    # --- data ---
    train_loader, valid_loader, test_loader, N_max, M_max = build_dataloaders(
        args, None, None, for_pretraining=False
    )
    loader = {"train": train_loader, "val": valid_loader, "test": test_loader}[
        args.split
    ]
    logger.info("Split '{}': {} samples", args.split, len(loader.dataset))

    # --- model ---
    if args.use_bipartite_graphs:
        model = GNNPolicy(args).to(device)
    else:
        model = KKTNetMLP(args, M_max, N_max).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("Loaded full checkpoint from {}", args.checkpoint)
    elif args.encoder_path:
        model.load_encoder(args.encoder_path, strict=True)
        logger.info("Loaded encoder from {}", args.encoder_path)
    else:
        logger.warning("No checkpoint provided — running with random weights")

    model.eval()

    # GA config
    evo_cfg = EvoConfig(
        pop_size=args.ga_pop_size,
        generations=args.ga_generations,
        logits=args.ga_logits,
        temperature=args.ga_temperature,
        rho_multiplier=args.ga_rho_multiplier,
        gamma_lambda=args.ga_gamma_lambda,
        maximize=args.ga_maximize,
        prefer_feasible=args.ga_prefer_feasible,
        seed=args.seed,
    )

    # --- forward pass ---
    results: List[dict] = []

    for batch_idx, batch in enumerate(loader):
        if isinstance(batch[0], PyGBatch):
            batch_graph, A, b, c, mask_m, mask_n, m_sizes, n_sizes, sample_paths = batch
            batch_graph = batch_graph.to(device, non_blocking=True)
            A = A.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            mask_n = mask_n.to(device, non_blocking=True)
            mask_m = mask_m.to(device, non_blocking=True)

            x_all, lam_all = model(
                batch_graph.constraint_features,
                batch_graph.edge_index,
                batch_graph.edge_attr,
                batch_graph.variable_features,
            )

            n_max = c.shape[1]
            m_max = b.shape[1]
            x_pred = pack_by_sizes(x_all, n_sizes, n_max)  # [B, n_max]
            lam_pred = pack_by_sizes(lam_all, m_sizes, m_max)  # [B, m_max]
        else:
            model_input, A, b, c, mask_m, mask_n, sample_paths = batch
            model_input = model_input.to(device, non_blocking=True)
            A = A.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            mask_n = mask_n.to(device, non_blocking=True)
            mask_m = mask_m.to(device, non_blocking=True)

            y_pred = model(model_input)  # [B, n+m]
            n_max = c.shape[1]
            m_max = b.shape[1]
            x_pred = y_pred[:, :n_max]
            lam_pred = y_pred[:, n_max : n_max + m_max]

        # --- GA decode ---
        x_int_batch, ga_infos = evolve_binary_solution_batch(
            x_pred=x_pred,
            lambda_pred=lam_pred,
            A=A,
            b=b,
            c=c,
            mask_m=mask_m,
            mask_n=mask_n,
            cfg=evo_cfg,
        )

        B = int(A.size(0))
        for i in range(B):
            n_vars = int(mask_n[i].sum().item())
            m_cons = int(mask_m[i].sum().item())

            x_i = x_pred[i, :n_vars]
            c_i = c[i, :n_vars]
            lam_i = lam_pred[i, :m_cons]
            x_int_i = x_int_batch[i, :n_vars]

            entry = {
                "path": sample_paths[i] if sample_paths[i] else "",
                "x": x_i.cpu().tolist(),
                "lambda": lam_i.cpu().tolist(),
                "x_int": x_int_i.cpu().tolist(),
                "ga_obj": ga_infos[i].get("obj_min", ""),
                "ga_viol_sum": ga_infos[i].get("viol_sum", ""),
                "ga_viol_max": ga_infos[i].get("viol_max", ""),
            }

            # Fetch optimal solution from disk
            input_path = sample_paths[i]
            if input_path:
                try:
                    opt_sol, obj_gap = get_optimal_solution(
                        input_path=input_path, x_i=x_i, c_i=c_i
                    )
                    if opt_sol is not None:
                        entry["x_opt"] = opt_sol.tolist()
                        entry["objective_gap"] = float(obj_gap)
                        # Count matching integer variables (GA)
                        x_int_cpu = x_int_i.cpu()
                        int_correct = int((x_int_cpu == opt_sol).sum().item())
                        entry["int_correct"] = int_correct
                        entry["int_accuracy"] = int_correct / n_vars
                        # Count matching after naive rounding
                        x_rounded = (x_i.cpu() >= 0.5).float()
                        round_correct = int((x_rounded == opt_sol).sum().item())
                        entry["round_correct"] = round_correct
                        entry["round_accuracy"] = round_correct / n_vars
                except Exception as e:
                    logger.debug("Could not load optimal for {}: {}", input_path, e)

            results.append(entry)

        if (batch_idx + 1) % 10 == 0:
            logger.info("Processed {}/{} batches", batch_idx + 1, len(loader))

    logger.info("Inference complete: {} instances", len(results))
    return results


def main() -> None:
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()

    # Add model-specific args then re-parse
    if args.use_bipartite_graphs:
        GNNPolicy.add_args(parser)
    else:
        KKTNetMLP.add_args(parser)
    args, _ = parser.parse_known_args()

    results = run_inference(args)

    # --- print comparison for all instances with optimal solutions ---
    compared = 0
    obj_gaps = []
    l2_dists = []
    int_corrects = []
    int_accs = []
    round_corrects = []
    round_accs = []
    for idx, r in enumerate(results):
        if "x_opt" not in r:
            continue
        x_pred_arr = np.array(r["x"])
        x_opt_arr = np.array(r["x_opt"])
        n = len(x_pred_arr)
        l2 = float(np.sqrt(((x_pred_arr - x_opt_arr) ** 2).sum()))
        obj_gap = r["objective_gap"]
        ic = r.get("int_correct", "")
        ia = r.get("int_accuracy", "")
        rc = r.get("round_correct", "")
        ra = r.get("round_accuracy", "")
        obj_gaps.append(obj_gap)
        l2_dists.append(l2)
        if ic != "":
            int_corrects.append(ic)
            int_accs.append(ia)
        if rc != "":
            round_corrects.append(rc)
            round_accs.append(ra)

        print(
            f"[{idx:5d}] {r['path']:<60s}  "
            f"obj_gap={obj_gap:+.6f}  L2={l2:.4f}  "
            f"ga={ic}/{n}  round={rc}/{n}"
        )
        compared += 1

    if compared > 0:
        obj_gaps_arr = np.array(obj_gaps)
        l2_dists_arr = np.array(l2_dists)
        print("\n" + "=" * 70)
        print(f"Summary: {compared}/{len(results)} instances with optimal solutions")
        print(
            f"  Objective gap    — mean: {obj_gaps_arr.mean():.6f}  "
            f"std: {obj_gaps_arr.std():.6f}  "
            f"median: {np.median(obj_gaps_arr):.6f}  "
            f"max: {obj_gaps_arr.max():.6f}"
        )
        print(
            f"  L2 distance      — mean: {l2_dists_arr.mean():.4f}  "
            f"std: {l2_dists_arr.std():.4f}  "
            f"median: {np.median(l2_dists_arr):.4f}  "
            f"max: {l2_dists_arr.max():.4f}"
        )
        if int_accs:
            int_accs_arr = np.array(int_accs)
            int_corrects_arr = np.array(int_corrects)
            print(
                f"  GA int accuracy  — mean: {int_accs_arr.mean():.4f}  "
                f"std: {int_accs_arr.std():.4f}  "
                f"median: {np.median(int_accs_arr):.4f}  "
                f"min: {int_accs_arr.min():.4f}"
            )
            print(
                f"  GA int correct   — mean: {int_corrects_arr.mean():.1f}  "
                f"std: {int_corrects_arr.std():.1f}  "
                f"median: {np.median(int_corrects_arr):.0f}  "
                f"min: {int_corrects_arr.min()}"
            )
        if round_accs:
            round_accs_arr = np.array(round_accs)
            round_corrects_arr = np.array(round_corrects)
            print(
                f"  Round accuracy   — mean: {round_accs_arr.mean():.4f}  "
                f"std: {round_accs_arr.std():.4f}  "
                f"median: {np.median(round_accs_arr):.4f}  "
                f"min: {round_accs_arr.min():.4f}"
            )
            print(
                f"  Round correct    — mean: {round_corrects_arr.mean():.1f}  "
                f"std: {round_corrects_arr.std():.1f}  "
                f"median: {np.median(round_corrects_arr):.0f}  "
                f"min: {round_corrects_arr.min()}"
            )
        print("=" * 70 + "\n")
    else:
        print("\n[info] No optimal solutions found on disk for comparison.\n")

    # --- save ---
    tag = _infer_method_tag(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_{args.split}.csv"

    fieldnames = [
        "path",
        "objective_gap",
        "l2_distance",
        "int_correct",
        "int_accuracy",
        "round_correct",
        "round_accuracy",
        "n_vars",
        "x",
        "lambda",
        "x_int",
        "ga_obj",
        "ga_viol_sum",
        "ga_viol_max",
        "x_opt",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            x_pred_arr = np.array(r["x"])
            x_opt_arr = np.array(r.get("x_opt", []))
            l2 = (
                float(np.sqrt(((x_pred_arr - x_opt_arr) ** 2).sum()))
                if x_opt_arr.size > 0
                else ""
            )
            writer.writerow(
                {
                    "path": r.get("path", ""),
                    "objective_gap": r.get("objective_gap", ""),
                    "l2_distance": l2,
                    "int_correct": r.get("int_correct", ""),
                    "int_accuracy": r.get("int_accuracy", ""),
                    "round_correct": r.get("round_correct", ""),
                    "round_accuracy": r.get("round_accuracy", ""),
                    "n_vars": len(r["x"]),
                    "x": r["x"],
                    "lambda": r["lambda"],
                    "x_int": r.get("x_int", ""),
                    "ga_obj": r.get("ga_obj", ""),
                    "ga_viol_sum": r.get("ga_viol_sum", ""),
                    "ga_viol_max": r.get("ga_viol_max", ""),
                    "x_opt": r.get("x_opt", ""),
                }
            )
    logger.info("Saved {} results to {}", len(results), out_path)


if __name__ == "__main__":
    main()
