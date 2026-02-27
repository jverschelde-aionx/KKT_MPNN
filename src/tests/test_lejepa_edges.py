"""
Diagnostic Test 2: Does LeJEPA actually use graph edges?

Tests whether the LeJEPA pretraining loss depends on message passing layers
by running ablations: shuffled edges, zero edge features, and skipped convolutions.

If LeJEPA loss is similar across all conditions, it suggests pretraining is not
training the message passing layers effectively.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import yaml
from loguru import logger
from torch_geometric.data import Batch

from data.datasets import GraphDataset, lejepa_views_collate
from models.base import LeJepaEncoderModule
from models.gnn import GNNPolicy


def load_pretrained_model(
    checkpoint_path: Path, cfg: Dict, device: str
) -> LeJepaEncoderModule:
    """Load pretrained model from checkpoint using config."""
    from argparse import Namespace

    # Create a Namespace from config for GNNPolicy
    args = Namespace(
        embedding_size=cfg["gnn"].get("embedding_size", 128),
        cons_nfeats=cfg["data"].get("cons_nfeats", 4),
        var_nfeats=cfg["data"].get("var_nfeats", 18),
        edge_nfeats=cfg["data"].get("edge_nfeats", 1),
        num_emb_freqs=cfg["gnn"].get("num_emb_freqs", 16),
        num_emb_bins=cfg["gnn"].get("num_emb_bins", 32),
        num_emb_type=cfg["gnn"].get("num_emb_type", "periodic"),
        bipartite_conv=cfg["gnn"].get("bipartite_conv", "gatv2"),
        attn_heads=cfg["gnn"].get("attn_heads", 4),
        dropout=cfg["gnn"].get("dropout", 0.0),
        lejepa_embed_dim=cfg["model"].get("lejepa_embed_dim", 128),
        graph_pool=cfg["gnn"].get("graph_pool", "mean"),
        sigreg_slices=cfg["model"].get("sigreg_slices", 512),
        sigreg_points=cfg["model"].get("sigreg_points", 13),
    )

    # Create model
    model = GNNPolicy(args).to(device)

    # Load encoder weights
    model.load_encoder(str(checkpoint_path), strict=True)
    model.eval()

    return model


def compute_lejepa_loss(
    model: LeJepaEncoderModule,
    batch: Dict,
    device: str,
    lejepa_lambda: float = 0.126,
    std_loss_weight: float = 1.0,
) -> float:
    """Compute LeJEPA loss for a batch."""
    base = batch["base"].to(device)
    global_views = [v.to(device) for v in batch["global_views"]]
    all_views = [v.to(device) for v in batch["all_views"]]

    with torch.no_grad():
        loss, _, _ = model.lejepa_loss(
            input=base,
            precomputed_views=(global_views, all_views),
            lambd=lejepa_lambda,
            std_loss_weight=std_loss_weight,
        )

    return loss.item()


def shuffle_edges_in_batch(batch: Batch) -> Batch:
    """
    Randomly REWIRE edges by shuffling target nodes.

    This actually changes graph connectivity, not just edge ordering.
    For a bipartite graph with edges from constraints to variables,
    we shuffle which variable each constraint connects to.
    """
    if not hasattr(batch, "edge_index"):
        return batch

    # edge_index is [2, num_edges]: [source_nodes, target_nodes]
    source_nodes = batch.edge_index[0].clone()
    target_nodes = batch.edge_index[1].clone()

    # Randomly permute target nodes to rewire the graph
    perm = torch.randperm(target_nodes.size(0))
    shuffled_targets = target_nodes[perm]

    # Create new edge index with rewired connections
    batch.edge_index = torch.stack([source_nodes, shuffled_targets], dim=0)

    # Also shuffle edge features (keep them aligned with new edges)
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        batch.edge_attr = batch.edge_attr[perm]

    return batch


def zero_edge_features(batch: Batch) -> Batch:
    """Set all edge features to zero."""
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        batch.edge_attr = torch.zeros_like(batch.edge_attr)
    return batch


def test_edge_attr_statistics(dataloader, num_batches: int = 5) -> Dict[str, float]:
    """
    DIAGNOSTIC TEST A: Check if edge attributes are nearly constant.

    If edge_attr has very low variance, zeroing it won't change much.
    """
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC A: Edge Attribute Statistics")
    logger.info("=" * 60)

    all_edge_attrs = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        if hasattr(batch["base"], "edge_attr") and batch["base"].edge_attr is not None:
            all_edge_attrs.append(batch["base"].edge_attr)

    if not all_edge_attrs:
        logger.warning("No edge attributes found in batches!")
        return {}

    # Concatenate all edge attributes
    ea = torch.cat(all_edge_attrs, dim=0)

    # Compute statistics
    stats = {
        "shape": ea.shape,
        "min": ea.min().item(),
        "max": ea.max().item(),
        "mean": ea.mean().item(),
        "std": ea.std(unbiased=False).item(),
        "num_unique": torch.unique(ea).numel() if ea.numel() < 2_000_000 else -1,
    }

    logger.info(f"Edge attribute shape: {stats['shape']}")
    logger.info(f"Min/Max: {stats['min']:.6f} / {stats['max']:.6f}")
    logger.info(f"Mean/Std: {stats['mean']:.6f} / {stats['std']:.6f}")
    if stats["num_unique"] > 0:
        logger.info(f"Unique values: {stats['num_unique']}")

    # Verdict
    if stats["std"] < 0.01:
        logger.warning(
            "\n⚠️  ISSUE DETECTED: Edge attributes are nearly constant!\n"
            f"   Std = {stats['std']:.6f} < 0.01\n"
            "   → Zeroing edge_attr will have minimal effect.\n"
            "   → Model may ignore edge features entirely."
        )
    elif stats["std"] < 0.1:
        logger.info(
            f"\n📊 Edge attributes have low variance (std={stats['std']:.4f}).\n"
            "   May limit edge feature utility."
        )
    else:
        logger.info(
            f"\n✓ Edge attributes have reasonable variance (std={stats['std']:.4f})."
        )

    return stats


def test_layernorm_destruction(model: LeJepaEncoderModule) -> bool:
    """
    DIAGNOSTIC TEST B: Check if LayerNorm(1) destroys scalar edge features.

    If edge_proj uses LayerNorm(1), it normalizes scalar features to constant.
    """
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC B: LayerNorm(1) Architecture Check")
    logger.info("=" * 60)

    # Inspect edge_proj architecture
    edge_proj = model.encoder.edge_proj
    logger.info(f"edge_proj architecture: {edge_proj}")

    # Check first layer
    has_layernorm_1 = False
    if len(edge_proj) > 0:
        first_layer = edge_proj[0]
        logger.info(f"First layer: {first_layer}")

        # Check if it's LayerNorm with normalized_shape=(1,)
        if isinstance(first_layer, torch.nn.LayerNorm):
            if first_layer.normalized_shape == (
                1,
            ) or first_layer.normalized_shape == torch.Size([1]):
                has_layernorm_1 = True
                logger.error(
                    "\n🔴 CRITICAL ISSUE: LayerNorm(1) detected!\n"
                    f"   {first_layer}\n"
                    "   → This destroys scalar edge features!"
                )

                # Demonstrate the problem
                logger.info("\nDemonstrating LayerNorm(1) destruction:")
                test_inputs = torch.tensor([[0.5], [1.0], [2.0], [10.0]])
                with torch.no_grad():
                    outputs = first_layer(test_inputs)
                logger.info(f"  Inputs:  {test_inputs.squeeze().tolist()}")
                logger.info(f"  Outputs: {outputs.squeeze().tolist()}")
                logger.warning(
                    "  → All different inputs produce (nearly) identical outputs!\n"
                    "  → Edge features are replaced by learned bias."
                )

    if not has_layernorm_1:
        logger.info("\n✓ No LayerNorm(1) issue detected in edge_proj.")

    return has_layernorm_1


def rel_diff(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute relative difference between two tensors."""
    return ((a - b).norm() / (a.norm() + eps)).item()


def test_random_encoder_sensitivity(
    cfg: Dict,
    dataloader,
    device: str,
    num_batches: int = 3,
) -> Dict[str, float]:
    """
    CONTROL TEST: Test edge sensitivity on a random (untrained) encoder.

    If random encoder shows 0% sensitivity: test is broken
    If random encoder shows >5% sensitivity: test is valid
    """
    from argparse import Namespace

    logger.info("\n" + "=" * 60)
    logger.info("CONTROL TEST: Random Encoder Sensitivity")
    logger.info("=" * 60)

    # Create fresh random model (same architecture)
    args = Namespace(
        embedding_size=cfg["gnn"].get("embedding_size", 128),
        cons_nfeats=cfg["data"].get("cons_nfeats", 4),
        var_nfeats=cfg["data"].get("var_nfeats", 18),
        edge_nfeats=cfg["data"].get("edge_nfeats", 1),
        num_emb_freqs=cfg["gnn"].get("num_emb_freqs", 16),
        num_emb_bins=cfg["gnn"].get("num_emb_bins", 32),
        num_emb_type=cfg["gnn"].get("num_emb_type", "periodic"),
        bipartite_conv=cfg["gnn"].get("bipartite_conv", "gatv2"),
        attn_heads=cfg["gnn"].get("attn_heads", 4),
        dropout=0.0,  # No dropout for deterministic test
        lejepa_embed_dim=cfg["model"].get("lejepa_embed_dim", 256),
        graph_pool=cfg["gnn"].get("graph_pool", "mean"),
        sigreg_slices=cfg["model"].get("sigreg_slices", 512),
        sigreg_points=cfg["model"].get("sigreg_points", 13),
    )

    random_model = GNNPolicy(args).to(device)
    random_model.eval()

    logger.info("Testing random (untrained) encoder...")

    node_emb_diffs = {"shuffled": [], "zero": []}

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        base_graph = batch["base"].to(device)
        base_shuffled = shuffle_edges_in_batch(batch["base"].clone()).to(device)
        base_zero = zero_edge_features(batch["base"].clone()).to(device)

        with torch.no_grad():
            # Baseline node embeddings
            c0, v0 = random_model.encoder.encode_nodes(
                base_graph.constraint_features,
                base_graph.edge_index,
                base_graph.edge_attr,
                base_graph.variable_features,
            )

            # Shuffled edges node embeddings
            c1, v1 = random_model.encoder.encode_nodes(
                base_shuffled.constraint_features,
                base_shuffled.edge_index,
                base_shuffled.edge_attr,
                base_shuffled.variable_features,
            )

            # Zero edges node embeddings
            c2, v2 = random_model.encoder.encode_nodes(
                base_zero.constraint_features,
                base_zero.edge_index,
                base_zero.edge_attr,
                base_zero.variable_features,
            )

            # Compute relative differences
            node_diff_shuffled = (rel_diff(c0, c1) + rel_diff(v0, v1)) / 2
            node_diff_zero = (rel_diff(c0, c2) + rel_diff(v0, v2)) / 2
            node_emb_diffs["shuffled"].append(node_diff_shuffled)
            node_emb_diffs["zero"].append(node_diff_zero)

            logger.info(
                f"  Batch {i + 1}/{num_batches}: "
                f"node_emb_diff: shuffled={node_diff_shuffled:.4f}, zero={node_diff_zero:.4f}"
            )

    # Average
    avg_diffs = {
        "shuffled": sum(node_emb_diffs["shuffled"]) / len(node_emb_diffs["shuffled"]),
        "zero": sum(node_emb_diffs["zero"]) / len(node_emb_diffs["zero"]),
    }

    logger.info("\nRandom encoder average node embedding differences:")
    for condition, diff in avg_diffs.items():
        logger.info(f"  {condition:20s}: {diff:.4f} ({diff * 100:.2f}%)")

    # Validation
    if avg_diffs["shuffled"] < 0.05 and avg_diffs["zero"] < 0.05:
        logger.error(
            "\n❌ TEST VALIDATION FAILED!\n"
            "  Random encoder ALSO shows ~0% sensitivity.\n"
            "  → Either shuffle is a no-op (only reorders, doesn't change connectivity)\n"
            "  → Or measurement is comparing identical tensors by mistake.\n"
            "  → Fix the test before trusting trained model results!"
        )
    else:
        logger.info(
            "\n✅ TEST VALIDATION PASSED!\n"
            f"  Random encoder shows {avg_diffs['shuffled'] * 100:.2f}% sensitivity to shuffled edges.\n"
            f"  Random encoder shows {avg_diffs['zero'] * 100:.2f}% sensitivity to zero edges.\n"
            "  → Test is working correctly.\n"
            "  → Trained model's 0% sensitivity is REAL (learned to ignore edges)."
        )

    return avg_diffs


def test_edge_ablations(
    model: LeJepaEncoderModule,
    dataloader,
    device: str,
    num_batches: int = 10,
) -> Dict[str, float]:
    """
    Run edge ablation tests.

    Returns:
        Dictionary with average losses and representation changes.
    """
    results = {
        "baseline": 0.0,
        "shuffled_edges": 0.0,
        "zero_edges": 0.0,
    }

    # Track representation changes (A & B)
    node_emb_diffs = {"shuffled": [], "zero": []}
    pooled_emb_diffs = {"shuffled": [], "zero": []}

    logger.info(f"Running edge ablation tests on {num_batches} batches...")

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        # 1. Baseline (normal)
        loss_baseline = compute_lejepa_loss(model, batch, device)
        results["baseline"] += loss_baseline

        # 2. Shuffled edges
        batch_shuffled = {
            "base": shuffle_edges_in_batch(batch["base"].clone()),
            "global_views": [
                shuffle_edges_in_batch(v.clone()) for v in batch["global_views"]
            ],
            "all_views": [
                shuffle_edges_in_batch(v.clone()) for v in batch["all_views"]
            ],
        }
        loss_shuffled = compute_lejepa_loss(model, batch_shuffled, device)
        results["shuffled_edges"] += loss_shuffled

        # 3. Zero edge features
        batch_zero = {
            "base": zero_edge_features(batch["base"].clone()),
            "global_views": [
                zero_edge_features(v.clone()) for v in batch["global_views"]
            ],
            "all_views": [zero_edge_features(v.clone()) for v in batch["all_views"]],
        }
        loss_zero = compute_lejepa_loss(model, batch_zero, device)
        results["zero_edges"] += loss_zero

        # TEST A: Does message passing change node embeddings?
        base_graph = batch["base"].to(device)
        base_shuffled = shuffle_edges_in_batch(batch["base"].clone()).to(device)
        base_zero = zero_edge_features(batch["base"].clone()).to(device)

        with torch.no_grad():
            # Baseline node embeddings
            c0, v0 = model.encoder.encode_nodes(
                base_graph.constraint_features,
                base_graph.edge_index,
                base_graph.edge_attr,
                base_graph.variable_features,
            )

            # Shuffled edges node embeddings
            c1, v1 = model.encoder.encode_nodes(
                base_shuffled.constraint_features,
                base_shuffled.edge_index,
                base_shuffled.edge_attr,
                base_shuffled.variable_features,
            )

            # Zero edges node embeddings
            c2, v2 = model.encoder.encode_nodes(
                base_zero.constraint_features,
                base_zero.edge_index,
                base_zero.edge_attr,
                base_zero.variable_features,
            )

            # Compute relative differences for node embeddings
            node_diff_shuffled = (rel_diff(c0, c1) + rel_diff(v0, v1)) / 2
            node_diff_zero = (rel_diff(c0, c2) + rel_diff(v0, v2)) / 2
            node_emb_diffs["shuffled"].append(node_diff_shuffled)
            node_emb_diffs["zero"].append(node_diff_zero)

            # TEST B: Does the pooled LeJEPA embedding change?
            from torch_geometric.nn import global_mean_pool

            # Get batch indices
            cb = base_graph.constraint_features_batch
            vb = base_graph.variable_features_batch
            cb1 = base_shuffled.constraint_features_batch
            vb1 = base_shuffled.variable_features_batch
            cb2 = base_zero.constraint_features_batch
            vb2 = base_zero.variable_features_batch

            # Baseline pooled embedding
            ce0 = model.encoder.cons_lejepa_proj(c0)
            ve0 = model.encoder.var_lejepa_proj(v0)
            c_pool0 = global_mean_pool(ce0, cb)
            v_pool0 = global_mean_pool(ve0, vb)
            g0 = torch.cat([c_pool0, v_pool0], dim=-1)
            z0 = model.encoder.graph_proj(g0)

            # Shuffled pooled embedding
            ce1 = model.encoder.cons_lejepa_proj(c1)
            ve1 = model.encoder.var_lejepa_proj(v1)
            c_pool1 = global_mean_pool(ce1, cb1)
            v_pool1 = global_mean_pool(ve1, vb1)
            g1 = torch.cat([c_pool1, v_pool1], dim=-1)
            z1 = model.encoder.graph_proj(g1)

            # Zero pooled embedding
            ce2 = model.encoder.cons_lejepa_proj(c2)
            ve2 = model.encoder.var_lejepa_proj(v2)
            c_pool2 = global_mean_pool(ce2, cb2)
            v_pool2 = global_mean_pool(ve2, vb2)
            g2 = torch.cat([c_pool2, v_pool2], dim=-1)
            z2 = model.encoder.graph_proj(g2)

            # Compute relative differences for pooled embeddings
            pooled_diff_shuffled = rel_diff(z0, z1)
            pooled_diff_zero = rel_diff(z0, z2)
            pooled_emb_diffs["shuffled"].append(pooled_diff_shuffled)
            pooled_emb_diffs["zero"].append(pooled_diff_zero)

        logger.info(
            f"Batch {i + 1}/{num_batches}: "
            f"loss: baseline={loss_baseline:.4f}, shuffled={loss_shuffled:.4f}, zero={loss_zero:.4f} | "
            f"node_emb_diff: shuffled={node_diff_shuffled:.4f}, zero={node_diff_zero:.4f} | "
            f"pooled_diff: shuffled={pooled_diff_shuffled:.4f}, zero={pooled_diff_zero:.4f}"
        )

    # Average over batches
    for key in results:
        results[key] /= num_batches

    # Compute average representation differences
    avg_node_emb_diffs = {
        "shuffled": sum(node_emb_diffs["shuffled"]) / len(node_emb_diffs["shuffled"]),
        "zero": sum(node_emb_diffs["zero"]) / len(node_emb_diffs["zero"]),
    }
    avg_pooled_emb_diffs = {
        "shuffled": sum(pooled_emb_diffs["shuffled"])
        / len(pooled_emb_diffs["shuffled"]),
        "zero": sum(pooled_emb_diffs["zero"]) / len(pooled_emb_diffs["zero"]),
    }

    return results, avg_node_emb_diffs, avg_pooled_emb_diffs


def main():
    parser = argparse.ArgumentParser(
        description="Test whether LeJEPA actually uses graph edges"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to pretrained encoder checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/instances"),
        help="Path to graph data",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain_gnn_gatv2_conv_CA_10.yml"),
        help="Path to config file (for data loading params)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches to test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load model
    logger.info(f"Loading pretrained model from {args.checkpoint}")
    model = load_pretrained_model(args.checkpoint, cfg, args.device)

    # Load dataset
    logger.info(f"Loading validation data from {args.data_path}")

    # Collect validation files
    from data.common import CONS_PAD, VARS_PAD
    from data.datasets import PadFeaturesTransform

    problems = cfg["data"].get("problems", ["CA"])
    print(f"Testing problems: {problems}")
    size_cfg = {
        "IS": cfg["data"].get("is_sizes", []),
        "CA": cfg["data"].get("ca_sizes", []),
        "SC": cfg["data"].get("sc_sizes", []),
        "CFL": cfg["data"].get("cfl_sizes", []),
        "RND": cfg["data"].get("rnd_sizes", []),
    }

    val_files = []
    for problem in problems:
        problem_dir = Path(args.data_path) / problem / "BG"
        split_dir = problem_dir / "val"
        sizes = size_cfg.get(problem, [])
        for size in sizes:
            size_dir = split_dir / str(size)
            if size_dir.exists():
                val_files.extend([str(f) for f in size_dir.iterdir() if f.is_file()])
            else:
                logger.warning(f"Directory not found: {size_dir}")

    if not val_files:
        raise RuntimeError(f"No validation files found in {args.data_path}")

    logger.info(f"Found {len(val_files)} validation files")

    # Limit to small number for quick testing
    val_files = sorted(val_files)[:50]

    dataset = GraphDataset(
        val_files, transform=PadFeaturesTransform(CONS_PAD, VARS_PAD)
    )

    # Create dataloader with LeJEPA views
    from functools import partial

    from torch.utils.data import DataLoader

    collate_fn = partial(
        lejepa_views_collate,
        n_global_views=cfg["model"].get("lejepa_n_global_views", 2),
        n_local_views=cfg["model"].get("lejepa_n_local_views", 8),
        global_keep=cfg["gnn"].get("lejepa_global_mask", 0.95),
        local_keep=cfg["gnn"].get("lejepa_local_mask", 0.4),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # PRE-FLIGHT DIAGNOSTICS: Check why edge_attr might have zero effect
    logger.info("\n" + "=" * 80)
    logger.info("STEP 0: PRE-FLIGHT DIAGNOSTICS - Why Edge Features Might Be Ignored")
    logger.info("=" * 80)

    # Test B: Check for LayerNorm(1) architecture issue
    has_layernorm_issue = test_layernorm_destruction(model)

    # Test A: Check if edge attributes are nearly constant
    edge_stats = test_edge_attr_statistics(dataloader, num_batches=5)

    # Summary of pre-flight diagnostics
    if has_layernorm_issue and edge_stats.get("std", 1.0) < 0.01:
        logger.error(
            "\n🔴 BOTH ISSUES DETECTED:\n"
            "   1. LayerNorm(1) destroys scalar edge features\n"
            "   2. Edge attributes are nearly constant in dataset\n"
            "   → Zero edge test WILL show ~0% change (this is expected!)\n"
            "   → Model cannot learn from edge features even if it wanted to."
        )
    elif has_layernorm_issue:
        logger.warning(
            "\n⚠️  LayerNorm(1) issue detected.\n"
            "   → Edge features are destroyed by architecture.\n"
            "   → Zero edge test will show ~0% change."
        )
    elif edge_stats.get("std", 1.0) < 0.01:
        logger.warning(
            "\n⚠️  Edge attributes are nearly constant.\n"
            "   → Zero edge test may show minimal change.\n"
            "   → Model has little signal to learn from."
        )
    else:
        logger.info(
            "\n✓ No obvious issues detected.\n"
            "   If zero edge test shows ~0% change, it indicates learned insensitivity."
        )

    # CONTROL TEST: Run on random encoder first to validate test
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: CONTROL TEST - Validate measurement on random encoder")
    logger.info("=" * 80)
    random_encoder_diffs = test_random_encoder_sensitivity(
        cfg, dataloader, args.device, num_batches=3
    )

    # Run ablation tests on pretrained model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: PRETRAINED MODEL TEST")
    logger.info("=" * 80)
    results, node_emb_diffs, pooled_emb_diffs = test_edge_ablations(
        model, dataloader, args.device, args.num_batches
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("EDGE ABLATION TEST RESULTS")
    logger.info("=" * 60)
    logger.info("\n📊 LOSS VALUES:")
    for condition, loss in results.items():
        logger.info(f"  {condition:20s}: {loss:.6f}")

    # Compute relative differences for losses
    baseline = results["baseline"]
    logger.info("\n📉 Loss relative differences from baseline:")
    for condition, loss in results.items():
        if condition != "baseline":
            rel_diff_pct = abs(loss - baseline) / baseline * 100
            logger.info(f"  {condition:20s}: {rel_diff_pct:.2f}%")

    # TEST A RESULTS: Node embedding differences
    logger.info("\n" + "=" * 60)
    logger.info("TEST A: NODE EMBEDDING SENSITIVITY")
    logger.info("=" * 60)
    logger.info("(Does message passing change node embeddings?)")
    for condition, diff in node_emb_diffs.items():
        logger.info(f"  {condition:20s}: {diff:.4f} ({diff * 100:.2f}%)")

    # TEST B RESULTS: Pooled embedding differences
    logger.info("\n" + "=" * 60)
    logger.info("TEST B: POOLED EMBEDDING SENSITIVITY")
    logger.info("=" * 60)
    logger.info("(Does the final LeJEPA embedding change?)")
    for condition, diff in pooled_emb_diffs.items():
        logger.info(f"  {condition:20s}: {diff:.4f} ({diff * 100:.2f}%)")

    # Interpretation
    logger.info("\n" + "=" * 60)
    logger.info("COMPREHENSIVE INTERPRETATION")
    logger.info("=" * 60)

    # Analyze based on all three tests
    loss_insensitive = all(
        abs(results[k] - baseline) / baseline < 0.05 for k in results if k != "baseline"
    )
    node_insensitive_shuffled = node_emb_diffs["shuffled"] < 0.05
    node_insensitive_zero = node_emb_diffs["zero"] < 0.05
    pooled_insensitive_shuffled = pooled_emb_diffs["shuffled"] < 0.05
    pooled_insensitive_zero = pooled_emb_diffs["zero"] < 0.05

    if node_insensitive_shuffled and node_insensitive_zero:
        logger.warning(
            "\n🔴 CRITICAL FINDING (Test A):\n"
            "  Node embeddings are INVARIANT to edge structure (<5% change)!\n"
            "  → Message passing layers are NOT learning from graph topology.\n"
            "  → Convolutions are functionally edge-invariant/ignored during pretraining."
        )
    elif loss_insensitive and not (node_insensitive_shuffled and node_insensitive_zero):
        logger.warning(
            "\n🟡 PARTIAL ISSUE (Tests A & Loss):\n"
            "  Node embeddings DO change with edge structure,\n"
            "  BUT LeJEPA loss is insensitive to these changes.\n"
            "  → Pooling/projections are hiding node-level differences.\n"
            "  → Graph-level loss doesn't incentivize structure learning."
        )

        if pooled_insensitive_shuffled and pooled_insensitive_zero:
            logger.warning(
                "  Test B confirms: Pooled embeddings are also invariant!\n"
                "  → The pooling + projection pipeline destroys structural information."
            )
    elif not loss_insensitive:
        logger.info(
            "\n✅ GOOD NEWS:\n"
            "  LeJEPA loss IS sensitive to edge structure (>5% change).\n"
            "  → Message passing is contributing to the learned representations."
        )

        if not (node_insensitive_shuffled and node_insensitive_zero):
            logger.info(
                "  Test A confirms: Node embeddings change with edge structure.\n"
                "  → Convolutions are learning graph topology."
            )
        if not (pooled_insensitive_shuffled and pooled_insensitive_zero):
            logger.info(
                "  Test B confirms: Pooled embeddings preserve structural information.\n"
                "  → Graph-level representations depend on message passing."
            )
    else:
        logger.info(
            "\n📊 MIXED RESULTS:\n"
            "  Some sensitivity detected, but not consistent across all tests."
        )

    # Summary recommendation
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATION")
    logger.info("=" * 60)
    if node_insensitive_shuffled and node_insensitive_zero:
        logger.warning(
            "❌ Pretraining is NOT training message passing layers effectively.\n"
            "   Consider:\n"
            "   1. Node-level contrastive loss (match nodes across views)\n"
            "   2. Multi-task pretraining (graph + node objectives)\n"
            "   3. Check if issue existed in pre-LeJEPA commits\n"
        )
    elif loss_insensitive:
        logger.warning(
            "⚠️  Graph-level loss destroys node-level structural information.\n"
            "   Consider adding node-level auxiliary task during pretraining."
        )
    else:
        logger.info(
            "✓ Pretraining appears to be using message passing effectively.\n"
            "  If downstream performance is still poor, investigate:\n"
            "  - Task mismatch (pretraining vs finetuning objectives)\n"
            "  - Hyperparameters (learning rate, weight decay)\n"
            "  - Catastrophic forgetting during finetuning"
        )

    # Final comparison: Random vs Trained
    logger.info("\n" + "=" * 80)
    logger.info("FINAL COMPARISON: Random vs Trained Encoder")
    logger.info("=" * 80)
    logger.info(f"\nNode embedding sensitivity (shuffled edges):")
    logger.info(
        f"  Random encoder    : {random_encoder_diffs['shuffled']:.4f} ({random_encoder_diffs['shuffled'] * 100:.2f}%)"
    )
    logger.info(
        f"  Trained encoder   : {node_emb_diffs['shuffled']:.4f} ({node_emb_diffs['shuffled'] * 100:.2f}%)"
    )
    logger.info(f"\nNode embedding sensitivity (zero edges):")
    logger.info(
        f"  Random encoder    : {random_encoder_diffs['zero']:.4f} ({random_encoder_diffs['zero'] * 100:.2f}%)"
    )
    logger.info(
        f"  Trained encoder   : {node_emb_diffs['zero']:.4f} ({node_emb_diffs['zero'] * 100:.2f}%)"
    )

    # Verdict
    if random_encoder_diffs["shuffled"] > 0.05 and node_emb_diffs["shuffled"] < 0.05:
        logger.warning(
            "\n🔴 CONFIRMED: Trained model LEARNED TO IGNORE graph edges!\n"
            f"   Random model: {random_encoder_diffs['shuffled'] * 100:.2f}% sensitivity\n"
            f"   Trained model: {node_emb_diffs['shuffled'] * 100:.2f}% sensitivity\n"
            "   → Pretraining actively suppressed edge-based learning."
        )
    elif random_encoder_diffs["shuffled"] < 0.05 and node_emb_diffs["shuffled"] < 0.05:
        logger.error(
            "\n❌ TEST BROKEN: Both random and trained show 0% sensitivity!\n"
            "   → Fix the shuffle/measurement before drawing conclusions."
        )
    elif random_encoder_diffs["shuffled"] > 0.05 and node_emb_diffs["shuffled"] > 0.05:
        logger.info(
            "\n✅ Both random and trained models respond to edge changes.\n"
            "   Pretraining preserved edge sensitivity."
        )


if __name__ == "__main__":
    main()
