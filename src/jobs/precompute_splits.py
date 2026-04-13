from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import configargparse
import torch
from tqdm import tqdm

from data.common import ProblemClass
from data.generators import get_bipartite_graph
from data.split import SplitInstanceData, SplitPartitionData
from models.split import (
    build_block_graph,
    build_halo_subgraph,
    compute_boundary_features,
    identify_boundary_nodes,
    split_bipartite_graph_metis,
    validate_partition,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _normalize_boundary_features(x: torch.Tensor) -> torch.Tensor:
    """
    Expected columns:
      [total_degree, n_cross_edges, cut_fraction, sum_abs_cut_coeff, is_boundary]
    """
    x = x.clone().float()
    if x.size(1) < 5:
        raise ValueError(f"Expected boundary feature dim >= 5, got {x.size(1)}")
    x[:, 0] = torch.log1p(x[:, 0])
    x[:, 1] = torch.log1p(x[:, 1])
    x[:, 3] = torch.log1p(x[:, 3])
    return x


def _owned_bounds_lower_bound(
    n_cons: int,
    n_vars: int,
    max_owned_nodes: Optional[int],
    max_owned_vars: Optional[int],
) -> int:
    """
    Lower bound on number of parts implied by owned-node budgets.
    """
    lbs = [1]
    if max_owned_nodes is not None and max_owned_nodes > 0:
        lbs.append(math.ceil((n_cons + n_vars) / max_owned_nodes))
    if max_owned_vars is not None and max_owned_vars > 0:
        lbs.append(math.ceil(n_vars / max_owned_vars))
    return max(lbs)


def _build_halos_and_stats(
    specs,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    halo_hops: int,
) -> Tuple[List, Dict[str, float]]:
    """
    Build halo subgraphs once and compute summary statistics.
    """
    halo_sgs = []

    max_owned_nodes = 0
    max_owned_vars = 0
    max_halo_nodes = 0
    max_halo_edges = 0
    expansions: List[float] = []

    for p in specs:
        owned_cons = int(p.owned_cons_ids.numel())
        owned_vars = int(p.owned_var_ids.numel())
        owned_total = owned_cons + owned_vars

        sg = build_halo_subgraph(
            p, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=halo_hops
        )
        halo_sgs.append(sg)

        total_nodes = int(sg.constraint_features.size(0) + sg.variable_features.size(0))
        total_edges = int(sg.edge_index.size(1))

        max_owned_nodes = max(max_owned_nodes, owned_total)
        max_owned_vars = max(max_owned_vars, owned_vars)
        max_halo_nodes = max(max_halo_nodes, total_nodes)
        max_halo_edges = max(max_halo_edges, total_edges)
        expansions.append(total_nodes / max(owned_total, 1))

    stats = {
        "max_owned_nodes": float(max_owned_nodes),
        "max_owned_vars": float(max_owned_vars),
        "max_halo_nodes": float(max_halo_nodes),
        "max_halo_edges": float(max_halo_edges),
        "avg_expansion_ratio": float(sum(expansions) / max(len(expansions), 1)),
    }
    return halo_sgs, stats


def _stats_within_budget(
    stats: Dict[str, float],
    *,
    max_owned_nodes: Optional[int],
    max_owned_vars: Optional[int],
    max_halo_nodes: Optional[int],
    max_halo_edges: Optional[int],
) -> bool:
    if max_owned_nodes is not None and stats["max_owned_nodes"] > max_owned_nodes:
        return False
    if max_owned_vars is not None and stats["max_owned_vars"] > max_owned_vars:
        return False
    if max_halo_nodes is not None and stats["max_halo_nodes"] > max_halo_nodes:
        return False
    if max_halo_edges is not None and stats["max_halo_edges"] > max_halo_edges:
        return False
    return True


def choose_num_parts_for_budget(
    *,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    halo_hops: int,
    max_owned_nodes: Optional[int],
    max_owned_vars: Optional[int],
    max_halo_nodes: Optional[int],
    max_halo_edges: Optional[int],
) -> Tuple[int, List, List, Dict[str, float]]:
    """
    Choose the smallest feasible number of METIS parts satisfying the requested budgets.
    Returns:
      num_parts, specs, halo_sgs, stats
    """
    n_cons = int(c_nodes.size(0))
    n_vars = int(v_nodes.size(0))
    n_total = n_cons + n_vars

    lower = _owned_bounds_lower_bound(
        n_cons, n_vars, max_owned_nodes=max_owned_nodes, max_owned_vars=max_owned_vars
    )

    cache: Dict[int, Tuple[bool, List, List, Dict[str, float]]] = {}

    def eval_k(k: int) -> Tuple[bool, List, List, Dict[str, float]]:
        k = int(max(1, min(k, n_total)))
        if k in cache:
            return cache[k]

        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=k
        )
        validate_partition(specs, n_cons, n_vars)
        halo_sgs, stats = _build_halos_and_stats(
            specs,
            c_nodes,
            v_nodes,
            edge_index,
            edge_attr,
            halo_hops=halo_hops,
        )
        ok = _stats_within_budget(
            stats,
            max_owned_nodes=max_owned_nodes,
            max_owned_vars=max_owned_vars,
            max_halo_nodes=max_halo_nodes,
            max_halo_edges=max_halo_edges,
        )
        cache[k] = (ok, specs, halo_sgs, stats)
        return cache[k]

    # 1) Find a feasible upper bound by doubling.
    high = max(1, lower)
    ok, specs, halo_sgs, stats = eval_k(high)
    while not ok:
        if high >= n_total:
            raise RuntimeError(
                "Could not find a feasible partition count within budgets. "
                f"Even num_parts={n_total} violates budgets. Last stats={stats}"
            )
        high = min(n_total, high * 2)
        ok, specs, halo_sgs, stats = eval_k(high)

    # 2) Binary search for the smallest feasible K in [1, high].
    low_bad = 0
    high_good = high
    best = (high_good, specs, halo_sgs, stats)

    while high_good - low_bad > 1:
        mid = (low_bad + high_good) // 2
        ok, specs_mid, halo_sgs_mid, stats_mid = eval_k(mid)
        if ok:
            high_good = mid
            best = (mid, specs_mid, halo_sgs_mid, stats_mid)
        else:
            low_bad = mid

    return best


def _parse_args():
    parser = configargparse.ArgParser(
        allow_abbrev=False,
        default_config_files=["configs/precompute_splits/block-5-halo-2-CA-100.yml"],
    )

    # Data
    d = parser.add_argument_group("data")
    d.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=[ProblemClass.INDEPENDANT_SET, ProblemClass.COMBINATORIAL_AUCTION],
    )
    d.add_argument(
        "--is_sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000]
    )
    d.add_argument(
        "--ca_sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000]
    )
    d.add_argument(
        "--sc_sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000]
    )
    d.add_argument(
        "--cfl_sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000]
    )
    d.add_argument(
        "--rnd_sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000]
    )
    d.add_argument("--data_root", type=str, default="./data/instances/milp/finetune")

    # Split params
    g = parser.add_argument_group("split_precompute")
    g.add_argument("--num_blocks", type=int)
    g.add_argument("--halo_hops", type=int, default=0)
    g.add_argument("--max_owned_nodes", type=int, default=None)
    g.add_argument("--max_owned_vars", type=int, default=None)
    g.add_argument("--max_halo_nodes", type=int, default=None)
    g.add_argument("--max_halo_edges", type=int, default=None)
    g.add_argument("--store_kkt", action="store_true")
    return parser.parse_args()


def _collect_lp_files(args) -> List[Tuple[Path, Path]]:
    """
    Collect (lp_path, save_path) pairs by iterating over problems, sizes, and splits.
    Output mirrors input:
      data_root/<problem>/instance/<split>/<size>/foo.lp
      -> data_root/<problem>/splits/halo-{halo_hops}-nodes-{max_owned_nodes}/<split>/<size>/foo.pt
    """
    size_cfg = {
        "IS": args.is_sizes,
        "CA": args.ca_sizes,
        "SC": args.sc_sizes,
        "CFL": args.cfl_sizes,
        "RND": args.rnd_sizes,
    }
    data_root = Path(args.data_root)
    if args.num_blocks is not None:
        split_variant = f"halo-{args.halo_hops}-blocks-{args.num_blocks}"
    else:
        split_variant = f"halo-{args.halo_hops}-nodes-{args.max_owned_nodes}"
    pairs: List[Tuple[Path, Path]] = []

    for problem in args.problems:
        sizes = size_cfg.get(problem, [])
        if not sizes:
            raise ValueError(f"No sizes configured for problem {problem}")
        for split in ("train", "val", "test"):
            for size in sizes:
                lp_dir = data_root / problem / "instance" / split / str(size)
                out_dir = (
                    data_root / problem / "splits" / split_variant / split / str(size)
                )
                if not lp_dir.exists():
                    logger.warning("Skipping missing dir: %s", lp_dir)
                    continue
                for lp_path in sorted(lp_dir.glob("*.lp")):
                    save_path = out_dir / f"{lp_path.stem}.pt"
                    pairs.append((lp_path, save_path))

    return pairs


def main():
    args = _parse_args()

    if args.num_blocks is None:
        if (
            args.max_owned_nodes is None
            and args.max_owned_vars is None
            and args.max_halo_nodes is None
            and args.max_halo_edges is None
        ):
            raise ValueError(
                "Either provide --num_blocks or at least one of "
                "--max_owned_nodes / --max_owned_vars / --max_halo_nodes / --max_halo_edges."
            )
    elif args.num_blocks <= 0:
        raise ValueError(f"--num_blocks must be positive, got {args.num_blocks}")

    pairs = _collect_lp_files(args)
    if not pairs:
        raise ValueError("No LP files found for the given problems/sizes.")

    skipped = 0
    for lp_path, save_path in tqdm(pairs, desc="precompute_splits"):
        if save_path.exists():
            skipped += 1
            continue

        logger.debug("%s", lp_path)

        A_sparse, _v_map, v_nodes, c_nodes, _b_vars, b_vec, c_vec = get_bipartite_graph(
            lp_path
        )
        edge_index = A_sparse.edge_index
        edge_attr = A_sparse.edge_attr

        n_cons = c_nodes.size(0)
        n_vars = v_nodes.size(0)

        # hard sanity checks
        _b_len = b_vec.numel() if isinstance(b_vec, torch.Tensor) else b_vec.size
        _c_len = c_vec.numel() if isinstance(c_vec, torch.Tensor) else c_vec.size
        assert n_cons == _b_len, (
            f"{lp_path}: constraint node count {n_cons} != b_vec rows {_b_len}"
        )
        assert n_vars == _c_len, (
            f"{lp_path}: variable node count {n_vars} != c_vec length {_c_len}"
        )

        # ------------------------------------------------------------
        # Choose / validate partition count and build halo subgraphs
        # ------------------------------------------------------------
        if args.num_blocks is not None:
            num_parts = int(args.num_blocks)
            specs = split_bipartite_graph_metis(
                c_nodes, v_nodes, edge_index, edge_attr, num_parts=num_parts
            )
            validate_partition(specs, n_cons, n_vars)
            halo_sgs, split_stats = _build_halos_and_stats(
                specs,
                c_nodes,
                v_nodes,
                edge_index,
                edge_attr,
                halo_hops=args.halo_hops,
            )
            if not _stats_within_budget(
                split_stats,
                max_owned_nodes=args.max_owned_nodes,
                max_owned_vars=args.max_owned_vars,
                max_halo_nodes=args.max_halo_nodes,
                max_halo_edges=args.max_halo_edges,
            ):
                raise RuntimeError(
                    "Explicit num_blocks does not satisfy the requested budgets.\n"
                    f"num_blocks={num_parts}, stats={split_stats}, "
                    f"budgets={{max_owned_nodes={args.max_owned_nodes}, "
                    f"max_owned_vars={args.max_owned_vars}, "
                    f"max_halo_nodes={args.max_halo_nodes}, "
                    f"max_halo_edges={args.max_halo_edges}}}"
                )
        else:
            num_parts, specs, halo_sgs, split_stats = choose_num_parts_for_budget(
                c_nodes=c_nodes,
                v_nodes=v_nodes,
                edge_index=edge_index,
                edge_attr=edge_attr,
                halo_hops=args.halo_hops,
                max_owned_nodes=args.max_owned_nodes,
                max_owned_vars=args.max_owned_vars,
                max_halo_nodes=args.max_halo_nodes,
                max_halo_edges=args.max_halo_edges,
            )

        logger.debug(
            "  chosen num_parts=%d | max_owned_nodes=%d max_owned_vars=%d "
            "max_halo_nodes=%d max_halo_edges=%d avg_expansion=%.3f",
            num_parts,
            int(split_stats["max_owned_nodes"]),
            int(split_stats["max_owned_vars"]),
            int(split_stats["max_halo_nodes"]),
            int(split_stats["max_halo_edges"]),
            split_stats["avg_expansion_ratio"],
        )

        # block ids
        cons_block_id = torch.full((n_cons,), -1, dtype=torch.long)
        vars_block_id = torch.full((n_vars,), -1, dtype=torch.long)
        for k, p in enumerate(specs):
            cons_block_id[p.owned_cons_ids] = k
            vars_block_id[p.owned_var_ids] = k
        assert torch.all(cons_block_id >= 0)
        assert torch.all(vars_block_id >= 0)

        # boundary features / masks
        cons_boundary_feat, vars_boundary_feat = compute_boundary_features(
            specs, edge_index, edge_attr, n_cons, n_vars
        )
        cons_boundary_feat = _normalize_boundary_features(cons_boundary_feat)
        vars_boundary_feat = _normalize_boundary_features(vars_boundary_feat)

        cons_is_boundary, vars_is_boundary = identify_boundary_nodes(
            specs, edge_index, n_cons, n_vars
        )

        # block graph
        bg = build_block_graph(
            specs, edge_index, edge_attr, n_cons=n_cons, n_vars=n_vars
        )

        # build fixed halo subgraphs (already computed above)
        partitions = []
        for p, sg in zip(specs, halo_sgs):
            parts = SplitPartitionData(
                part_id=p.part_id,
                graph=sg,
                orig_cons_ids=sg.orig_cons_ids.clone(),
                orig_var_ids=sg.orig_var_ids.clone(),
                owned_cons_local=sg.owned_cons_mask.nonzero(as_tuple=False)
                .view(-1)
                .clone(),
                owned_var_local=sg.owned_var_mask.nonzero(as_tuple=False)
                .view(-1)
                .clone(),
            )
            partitions.append(parts)

        # optional KKT tensors
        A_dense = None
        if args.store_kkt:
            if not hasattr(A_sparse, "to_dense"):
                raise RuntimeError(
                    "A_sparse has no to_dense(); adjust this script to your matrix representation."
                )
            A_dense = A_sparse.to_dense().cpu()

        inst = SplitInstanceData(
            name=str(lp_path),
            partitions=partitions,
            block_edge_index=bg.block_edge_index.cpu(),
            block_edge_attr=bg.block_edge_attr.cpu(),
            cons_block_id=cons_block_id.cpu(),
            vars_block_id=vars_block_id.cpu(),
            cons_boundary_feat=cons_boundary_feat.cpu(),
            vars_boundary_feat=vars_boundary_feat.cpu(),
            cons_is_boundary=cons_is_boundary.cpu(),
            vars_is_boundary=vars_is_boundary.cpu(),
            n_cons=n_cons,
            n_vars=n_vars,
            A_dense=A_dense,
            b_vec=b_vec.cpu() if args.store_kkt else None,
            c_vec=c_vec.cpu() if args.store_kkt else None,
        )

        # Optional metadata attached to the saved object for later inspection
        inst.num_parts = num_parts
        inst.max_owned_nodes = int(split_stats["max_owned_nodes"])
        inst.max_owned_vars = int(split_stats["max_owned_vars"])
        inst.max_halo_nodes = int(split_stats["max_halo_nodes"])
        inst.max_halo_edges = int(split_stats["max_halo_edges"])
        inst.avg_expansion_ratio = float(split_stats["avg_expansion_ratio"])

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(inst, save_path)
        logger.info("  saved -> %s", save_path)

    logger.info(
        "Done. %d processed, %d skipped (already existed).",
        len(pairs) - skipped,
        skipped,
    )


if __name__ == "__main__":
    main()
