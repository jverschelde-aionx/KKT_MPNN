"""
Halo Embedding Recovery Evaluation
===================================
Measures how much of the full-graph encoder embedding is lost when the
bipartite graph is split into METIS partitions, and how much halo expansion
recovers.

For each LP instance:
1. Run the frozen encoder on the **full** graph → teacher embeddings.
2. Split with METIS (once) into K partitions.
3. For each halo depth in [0, 1, 2, 4]:
   - Build halo subgraphs for each partition.
   - Run the frozen encoder **independently** on each subgraph.
   - Compare owned-node embeddings to teacher via cosine similarity and MSE.
4. Report metrics per partition, per halo depth, with boundary vs interior
   breakdown.

No trainable model is added — this is the baseline measurement before
building a block-GNN composition model.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import configargparse
import torch
import torch.nn.functional as F

from data.common import CONS_PAD, VARS_PAD
from data.datasets import BipartiteNodeData, right_pad
from data.generators import get_bipartite_graph
from models.decomposition import (
    CouplingDiagnostics,
    PartitionSpec,
    build_halo_subgraph,
    compute_coupling_diagnostics_from_specs,
    compute_halo_expansion_ratio,
    identify_boundary_nodes,
    n_splits_for,
    split_bipartite_graph_metis,
    validate_partition,
)
from models.gnn import GNNEncoder, GNNPolicy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NodeRecoveryMetrics:
    """Embedding recovery metrics for a set of nodes."""

    n_nodes: int
    mean_cosine_sim: float
    mean_mse: float


@dataclass
class PartitionResult:
    """Recovery metrics for one partition at one halo depth."""

    instance: str
    part_id: int
    halo_depth: int
    n_owned_cons: int
    n_owned_vars: int
    n_halo_cons: int
    n_halo_vars: int
    cons_all: NodeRecoveryMetrics
    cons_boundary: NodeRecoveryMetrics
    cons_interior: NodeRecoveryMetrics
    vars_all: NodeRecoveryMetrics
    vars_boundary: NodeRecoveryMetrics
    vars_interior: NodeRecoveryMetrics


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------


def _load_encoder(
    args: configargparse.Namespace,
    device: torch.device,
) -> GNNEncoder:
    """Load a frozen GNNEncoder using the standard load_model_and_encoder flow.

    Expects ``args.encoder_path`` (path to best_encoder.pt) and
    ``args.finetune_mode`` (should be ``"heads"`` to freeze the encoder).
    """
    model = GNNPolicy(args).to(device)
    model.load_model_and_encoder(args, logger)
    return model.encoder


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _compute_full_graph_embeddings(
    encoder: GNNEncoder,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen encoder on the full graph.

    Returns
    -------
    c_full : [m, embedding_size]
    v_full : [n, embedding_size]
    """
    c_padded = right_pad(c_nodes, CONS_PAD).to(device)
    v_padded = right_pad(v_nodes, VARS_PAD).to(device)
    ei = edge_index.to(device)
    ea = edge_attr.to(device)

    c_full, v_full = encoder.encode_nodes(c_padded, ei, ea, v_padded)
    return c_full.cpu(), v_full.cpu()


@torch.inference_mode()
def _compute_subgraph_embeddings(
    encoder: GNNEncoder,
    sg: BipartiteNodeData,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen encoder on a single halo subgraph.

    Subgraph features are already padded by extract_subgraph.

    Returns
    -------
    c_sub : [n_cons_local, embedding_size]
    v_sub : [n_vars_local, embedding_size]
    """
    c_feat = sg.constraint_features.to(device)
    v_feat = sg.variable_features.to(device)
    ei = sg.edge_index.to(device)
    ea = sg.edge_attr.to(device)

    c_sub, v_sub = encoder.encode_nodes(c_feat, ei, ea, v_feat)
    return c_sub.cpu(), v_sub.cpu()


# ---------------------------------------------------------------------------
# Boundary node identification
# ---------------------------------------------------------------------------


_identify_boundary_nodes = identify_boundary_nodes  # backward compat alias


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_recovery_metrics(
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor,
) -> NodeRecoveryMetrics:
    """Compute cosine similarity and MSE between teacher and student embeddings.

    Both tensors must have the same shape [N, d].
    """
    n = teacher_emb.size(0)
    if n == 0:
        return NodeRecoveryMetrics(
            n_nodes=0, mean_cosine_sim=float("nan"), mean_mse=float("nan")
        )

    cos = F.cosine_similarity(teacher_emb, student_emb, dim=-1).mean().item()
    mse = (teacher_emb - student_emb).pow(2).mean().item()
    return NodeRecoveryMetrics(n_nodes=n, mean_cosine_sim=cos, mean_mse=mse)


# ---------------------------------------------------------------------------
# Per-partition evaluation
# ---------------------------------------------------------------------------


def _evaluate_partition(
    encoder: GNNEncoder,
    part: PartitionSpec,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    c_full: torch.Tensor,
    v_full: torch.Tensor,
    cons_is_boundary: torch.Tensor,
    vars_is_boundary: torch.Tensor,
    halo_depth: int,
    instance_name: str,
    device: torch.device,
) -> PartitionResult:
    """Evaluate embedding recovery for one partition at one halo depth."""
    # Build halo subgraph
    sg = build_halo_subgraph(
        part, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=halo_depth
    )

    # Encode subgraph
    c_sub, v_sub = _compute_subgraph_embeddings(encoder, sg, device)

    # --- Constraints ---
    owned_cons_mask = sg.owned_cons_mask.bool()
    owned_cons_global = sg.orig_cons_ids[owned_cons_mask]
    c_teacher = c_full[owned_cons_global]
    c_student = c_sub[owned_cons_mask]

    cons_bnd_mask = cons_is_boundary[owned_cons_global]
    cons_int_mask = ~cons_bnd_mask

    cons_all = _compute_recovery_metrics(c_teacher, c_student)
    cons_boundary = _compute_recovery_metrics(
        c_teacher[cons_bnd_mask], c_student[cons_bnd_mask]
    )
    cons_interior = _compute_recovery_metrics(
        c_teacher[cons_int_mask], c_student[cons_int_mask]
    )

    # --- Variables ---
    owned_var_mask = sg.owned_var_mask.bool()
    owned_var_global = sg.orig_var_ids[owned_var_mask]
    v_teacher = v_full[owned_var_global]
    v_student = v_sub[owned_var_mask]

    vars_bnd_mask = vars_is_boundary[owned_var_global]
    vars_int_mask = ~vars_bnd_mask

    vars_all = _compute_recovery_metrics(v_teacher, v_student)
    vars_boundary = _compute_recovery_metrics(
        v_teacher[vars_bnd_mask], v_student[vars_bnd_mask]
    )
    vars_interior = _compute_recovery_metrics(
        v_teacher[vars_int_mask], v_student[vars_int_mask]
    )

    # Halo counts
    n_total_cons = sg.constraint_features.size(0)
    n_total_vars = sg.variable_features.size(0)
    n_owned_cons = int(owned_cons_mask.sum().item())
    n_owned_vars = int(owned_var_mask.sum().item())

    return PartitionResult(
        instance=instance_name,
        part_id=part.part_id,
        halo_depth=halo_depth,
        n_owned_cons=n_owned_cons,
        n_owned_vars=n_owned_vars,
        n_halo_cons=n_total_cons - n_owned_cons,
        n_halo_vars=n_total_vars - n_owned_vars,
        cons_all=cons_all,
        cons_boundary=cons_boundary,
        cons_interior=cons_interior,
        vars_all=vars_all,
        vars_boundary=vars_boundary,
        vars_interior=vars_interior,
    )


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------


def evaluate_instance(
    encoder: GNNEncoder,
    lp_path: Path,
    specs: List[PartitionSpec],
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    c_full: torch.Tensor,
    v_full: torch.Tensor,
    cons_is_boundary: torch.Tensor,
    vars_is_boundary: torch.Tensor,
    halo_depths: List[int],
    device: torch.device,
) -> List[PartitionResult]:
    """Evaluate all (partition, halo_depth) combinations for one LP instance.

    The METIS split is done once; only halo expansion varies.
    """
    instance_name = str(lp_path)
    results: List[PartitionResult] = []

    for depth in halo_depths:
        logger.info("  halo_depth=%d", depth)
        for part in specs:
            result = _evaluate_partition(
                encoder,
                part,
                c_nodes,
                v_nodes,
                edge_index,
                edge_attr,
                c_full,
                v_full,
                cons_is_boundary,
                vars_is_boundary,
                depth,
                instance_name,
                device,
            )
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format_summary(
    results: List[PartitionResult],
    coupling_by_instance: Optional[Dict[str, CouplingDiagnostics]] = None,
) -> str:
    """Format a console summary grouped by (instance, halo_depth)."""
    if not results:
        return "No results."

    lines: List[str] = []

    # --- Coupling diagnostics block ---
    if coupling_by_instance:
        c_header = (
            f"{'instance':<25s} │ {'cut_edges':>9s} {'cut%':>6s} │ "
            f"{'bnd_c%':>6s} {'bnd_v%':>6s} │ {'cpl_c%':>6s} {'avg_blk':>7s}"
        )
        c_sep = "─" * len(c_header)
        lines.append(c_sep)
        lines.append("Coupling diagnostics")
        lines.append(c_sep)
        lines.append(c_header)
        lines.append(c_sep)
        for inst, diag in sorted(coupling_by_instance.items()):
            lines.append(
                f"{inst:<25s} │ {diag.edge_cut_count:>9d} "
                f"{100.0 * diag.edge_cut_fraction:>5.1f}% │ "
                f"{100.0 * diag.boundary_cons_fraction:>5.1f}% "
                f"{100.0 * diag.boundary_vars_fraction:>5.1f}% │ "
                f"{100.0 * diag.coupling_fraction:>5.1f}% "
                f"{diag.avg_blocks_per_constraint:>7.2f}"
            )
        lines.append(c_sep)
        lines.append("")

    # --- Recovery table ---
    # Group by (instance, halo_depth)
    groups: Dict[Tuple[str, int], List[PartitionResult]] = {}
    for r in results:
        key = (r.instance, r.halo_depth)
        groups.setdefault(key, []).append(r)

    header = (
        f"{'instance':<25s} {'halo':>4s} │ {'cons_cos':>8s} {'vars_cos':>8s} "
        f"{'cons_mse':>8s} {'vars_mse':>8s} │ {'bnd_cos':>7s} {'int_cos':>7s} "
        f"│ {'exp_rat':>7s}"
    )
    sep = "─" * len(header)
    lines.append(sep)
    lines.append("Embedding recovery")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for (instance, depth), group in sorted(groups.items()):

        def _wavg(items):
            valid = [(v, w) for v, w in items if w > 0 and math.isfinite(v)]
            total_w = sum(w for _, w in valid)
            if total_w == 0:
                return float("nan")
            return sum(v * w for v, w in valid) / total_w

        cons_cos = _wavg([(r.cons_all.mean_cosine_sim, r.n_owned_cons) for r in group])
        vars_cos = _wavg([(r.vars_all.mean_cosine_sim, r.n_owned_vars) for r in group])
        cons_mse = _wavg([(r.cons_all.mean_mse, r.n_owned_cons) for r in group])
        vars_mse = _wavg([(r.vars_all.mean_mse, r.n_owned_vars) for r in group])

        # Boundary/interior (combined across partitions)
        bnd_cons_items = [
            (r.cons_boundary.mean_cosine_sim, r.cons_boundary.n_nodes) for r in group
        ]
        bnd_vars_items = [
            (r.vars_boundary.mean_cosine_sim, r.vars_boundary.n_nodes) for r in group
        ]
        int_cons_items = [
            (r.cons_interior.mean_cosine_sim, r.cons_interior.n_nodes) for r in group
        ]
        int_vars_items = [
            (r.vars_interior.mean_cosine_sim, r.vars_interior.n_nodes) for r in group
        ]

        bnd_cos = _wavg(
            [(v, w) for v, w in bnd_cons_items + bnd_vars_items if not math.isnan(v)]
        )
        int_cos = _wavg(
            [(v, w) for v, w in int_cons_items + int_vars_items if not math.isnan(v)]
        )

        # Average expansion ratio across partitions
        exp_ratios = [
            compute_halo_expansion_ratio(
                r.n_owned_cons,
                r.n_owned_vars,
                r.n_halo_cons,
                r.n_halo_vars,
            )
            for r in group
        ]
        avg_exp = sum(exp_ratios) / len(exp_ratios) if exp_ratios else float("nan")

        lines.append(
            f"{instance:<25s} {depth:>4d} │ {cons_cos:>8.4f} {vars_cos:>8.4f} "
            f"{cons_mse:>8.5f} {vars_mse:>8.5f} │ {bnd_cos:>7.4f} {int_cos:>7.4f} "
            f"│ {avg_exp:>7.3f}"
        )

    lines.append(sep)
    return "\n".join(lines)


def _write_csv(results: List[PartitionResult], path: Path) -> None:
    """Write results to CSV. Two rows per partition (constraints + variables)."""
    fieldnames = [
        "instance",
        "part_id",
        "halo_depth",
        "node_type",
        "n_owned",
        "n_halo",
        "expansion_ratio",
        "n_boundary",
        "n_interior",
        "mean_cosine_all",
        "mean_cosine_boundary",
        "mean_cosine_interior",
        "mean_mse_all",
        "mean_mse_boundary",
        "mean_mse_interior",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            exp_ratio = compute_halo_expansion_ratio(
                r.n_owned_cons,
                r.n_owned_vars,
                r.n_halo_cons,
                r.n_halo_vars,
            )
            # Constraint row
            writer.writerow(
                {
                    "instance": r.instance,
                    "part_id": r.part_id,
                    "halo_depth": r.halo_depth,
                    "node_type": "constraint",
                    "n_owned": r.n_owned_cons,
                    "n_halo": r.n_halo_cons,
                    "expansion_ratio": f"{exp_ratio:.4f}",
                    "n_boundary": r.cons_boundary.n_nodes,
                    "n_interior": r.cons_interior.n_nodes,
                    "mean_cosine_all": r.cons_all.mean_cosine_sim,
                    "mean_cosine_boundary": r.cons_boundary.mean_cosine_sim,
                    "mean_cosine_interior": r.cons_interior.mean_cosine_sim,
                    "mean_mse_all": r.cons_all.mean_mse,
                    "mean_mse_boundary": r.cons_boundary.mean_mse,
                    "mean_mse_interior": r.cons_interior.mean_mse,
                }
            )
            # Variable row
            writer.writerow(
                {
                    "instance": r.instance,
                    "part_id": r.part_id,
                    "halo_depth": r.halo_depth,
                    "node_type": "variable",
                    "n_owned": r.n_owned_vars,
                    "n_halo": r.n_halo_vars,
                    "expansion_ratio": f"{exp_ratio:.4f}",
                    "n_boundary": r.vars_boundary.n_nodes,
                    "n_interior": r.vars_interior.n_nodes,
                    "mean_cosine_all": r.vars_all.mean_cosine_sim,
                    "mean_cosine_boundary": r.vars_boundary.mean_cosine_sim,
                    "mean_cosine_interior": r.vars_interior.mean_cosine_sim,
                    "mean_mse_all": r.vars_all.mean_mse,
                    "mean_mse_boundary": r.vars_boundary.mean_mse,
                    "mean_mse_interior": r.vars_interior.mean_mse,
                }
            )


def _write_coupling_csv(
    coupling_by_instance: Dict[str, CouplingDiagnostics],
    path: Path,
) -> None:
    """Write coupling diagnostics to CSV (one row per instance)."""
    fieldnames = [
        "instance",
        "edge_cut_count",
        "n_total_edges",
        "edge_cut_fraction",
        "n_boundary_cons",
        "n_total_constraints",
        "boundary_cons_fraction",
        "n_boundary_vars",
        "n_total_vars",
        "boundary_vars_fraction",
        "n_coupling_constraints",
        "coupling_fraction",
        "avg_blocks_per_constraint",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for inst, diag in sorted(coupling_by_instance.items()):
            writer.writerow(
                {
                    "instance": inst,
                    "edge_cut_count": diag.edge_cut_count,
                    "n_total_edges": diag.n_total_edges,
                    "edge_cut_fraction": f"{diag.edge_cut_fraction:.6f}",
                    "n_boundary_cons": diag.n_boundary_cons,
                    "n_total_constraints": diag.n_total_constraints,
                    "boundary_cons_fraction": f"{diag.boundary_cons_fraction:.6f}",
                    "n_boundary_vars": diag.n_boundary_vars,
                    "n_total_vars": diag.n_total_vars,
                    "boundary_vars_fraction": f"{diag.boundary_vars_fraction:.6f}",
                    "n_coupling_constraints": diag.n_coupling_constraints,
                    "coupling_fraction": f"{diag.coupling_fraction:.6f}",
                    "avg_blocks_per_constraint": f"{diag.avg_blocks_per_constraint:.4f}",
                }
            )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        default_config_files=["configs/decomposition/eval_halo_embedding.yml"],
    )

    g = parser.add_argument_group("eval_halo_embedding")
    g.add_argument(
        "--lp_path",
        type=str,
        default="",
        help="Single .lp file to evaluate",
    )
    g.add_argument(
        "--lp_dir",
        type=str,
        default="",
        help="Directory of .lp files (evaluates all)",
    )
    g.add_argument(
        "--encoder_path",
        type=str,
        required=True,
        help="Path to encoder-only checkpoint (e.g. best_encoder.pt)",
    )
    g.add_argument(
        "--finetune_mode",
        type=str,
        choices=["full", "heads"],
        default="heads",
        help="'heads' freezes encoder (default for eval).",
    )
    g.add_argument(
        "--max_subgraph_ratio",
        type=float,
        default=0.2,
        help="METIS partition sizing (fraction of total nodes)",
    )
    g.add_argument(
        "--halo_depths",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4],
        help="Halo depths to sweep",
    )
    g.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Path to write CSV results",
    )
    g.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda:N",
    )
    g.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Add GNN model args so checkpoint can be loaded with matching architecture
    GNNPolicy.add_args(parser)

    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@torch.inference_mode()
def main() -> List[PartitionResult]:
    args = _parse_args()

    if not args.lp_path and not args.lp_dir:
        raise ValueError("Must provide --lp_path or --lp_dir")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # Load frozen encoder
    encoder = _load_encoder(args, device)

    # Collect LP files
    lp_paths: List[Path] = []
    if args.lp_path:
        p = Path(args.lp_path)
        if not p.exists():
            raise FileNotFoundError(f"LP file not found: {p}")
        lp_paths.append(p)
    if args.lp_dir:
        d = Path(args.lp_dir)
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")
        lp_paths.extend(sorted(d.glob("*.lp")))

    if not lp_paths:
        raise ValueError("No .lp files found")

    ratio = args.max_subgraph_ratio
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"max_subgraph_ratio must be in (0, 1], got {ratio}")

    halo_depths = sorted(args.halo_depths)

    logger.info("Encoder          : %s", args.encoder_path)
    logger.info("LP files         : %d", len(lp_paths))
    logger.info("max_subgraph_ratio: %.2f", ratio)
    logger.info("Halo depths      : %s", halo_depths)
    logger.info("Device           : %s", device)

    all_results: List[PartitionResult] = []
    coupling_by_instance: Dict[str, CouplingDiagnostics] = {}

    for lp_path in lp_paths:
        logger.info("=== Evaluating: %s ===", lp_path.name)

        # Load bipartite graph
        A, v_map, v_nodes, c_nodes, b_vars, b_vec, c_vec = get_bipartite_graph(lp_path)
        edge_index = A.edge_index
        edge_attr = A.edge_attr
        n_cons = c_nodes.size(0)
        n_vars = v_nodes.size(0)

        logger.info(
            "  %d constraints, %d variables, %d edges",
            n_cons,
            n_vars,
            edge_index.size(1),
        )

        # Teacher embeddings (full graph)
        c_full, v_full = _compute_full_graph_embeddings(
            encoder, c_nodes, v_nodes, edge_index, edge_attr, device
        )

        # METIS partition (done ONCE, reused across halo depths)
        max_subgraph_size = max(1, int((n_cons + n_vars) * ratio))
        num_parts = n_splits_for(n_cons + n_vars, max_subgraph_size)

        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=num_parts
        )
        validate_partition(specs, n_cons, n_vars)

        logger.info("  Split into %d partitions (ratio=%.2f)", len(specs), ratio)

        # Identify boundary nodes (once per instance)
        cons_is_boundary, vars_is_boundary = _identify_boundary_nodes(
            specs, edge_index, n_cons, n_vars
        )
        n_bnd_c = int(cons_is_boundary.sum().item())
        n_bnd_v = int(vars_is_boundary.sum().item())
        logger.info(
            "  Boundary nodes: %d/%d cons (%.1f%%), %d/%d vars (%.1f%%)",
            n_bnd_c,
            n_cons,
            100.0 * n_bnd_c / max(n_cons, 1),
            n_bnd_v,
            n_vars,
            100.0 * n_bnd_v / max(n_vars, 1),
        )

        # Coupling diagnostics (once per instance, independent of halo depth)
        instance_name = str(lp_path)
        diag = compute_coupling_diagnostics_from_specs(
            specs,
            n_cons,
            n_vars,
            edge_index,
            cons_is_boundary=cons_is_boundary,
            vars_is_boundary=vars_is_boundary,
        )
        coupling_by_instance[instance_name] = diag
        logger.info(
            "  Edge cut: %d/%d (%.1f%%)",
            diag.edge_cut_count,
            diag.n_total_edges,
            100.0 * diag.edge_cut_fraction,
        )
        logger.info(
            "  Coupling constraints: %d/%d (%.1f%%)",
            diag.n_coupling_constraints,
            diag.n_total_constraints,
            100.0 * diag.coupling_fraction,
        )
        logger.info(
            "  Avg blocks/constraint: %.2f",
            diag.avg_blocks_per_constraint,
        )

        # Evaluate for each halo depth
        results = evaluate_instance(
            encoder,
            lp_path,
            specs,
            c_nodes,
            v_nodes,
            edge_index,
            edge_attr,
            c_full,
            v_full,
            cons_is_boundary,
            vars_is_boundary,
            halo_depths,
            device,
        )
        all_results.extend(results)

    # Output summary
    summary = _format_summary(all_results, coupling_by_instance)
    logger.info("\n%s", summary)

    # Write CSV if requested
    if args.output_csv:
        csv_path = Path(args.output_csv)
        _write_csv(all_results, csv_path)
        logger.info("Results written to %s", csv_path)

        coupling_csv_path = csv_path.with_name(
            csv_path.stem + "_coupling" + csv_path.suffix
        )
        _write_coupling_csv(coupling_by_instance, coupling_csv_path)
        logger.info("Coupling diagnostics written to %s", coupling_csv_path)

    return all_results


if __name__ == "__main__":
    main()
