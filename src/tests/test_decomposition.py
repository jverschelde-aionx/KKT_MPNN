"""Tests for models.decomposition."""

from __future__ import annotations

import pytest
import torch

from models.decomposition import (
    BlockGraph,
    CouplingDiagnostics,
    PartitionSpec,
    _bfs_expand_bipartite,
    balanced_chunks,
    build_block_graph,
    build_halo_subgraph,
    build_halo_subgraphs,
    compute_block_features,
    compute_coupling_diagnostics,
    compute_coupling_diagnostics_from_specs,
    compute_halo_expansion_ratio,
    extract_subgraph,
    extract_subgraph_by_constraints,
    extract_subgraph_by_variables,
    log_block_graph_diagnostics,
    n_splits_for,
    split_bipartite_graph_metis,
    split_by_constraints,
    split_by_variables,
    validate_partition,
)


# ---------------------------------------------------------------------------
# Helpers: build a small bipartite graph for testing
# ---------------------------------------------------------------------------


def _make_graph():
    """Create a small bipartite graph:

    5 constraints (c0..c4), 8 variables (v0..v7), 12 edges.
    Features are just row indices so we can verify extraction.

    Edges (constraint, variable):
        c0-v0, c0-v1, c0-v2
        c1-v1, c1-v3
        c2-v2, c2-v4, c2-v5
        c3-v5, c3-v6
        c4-v6, c4-v7
    """
    n_cons, n_vars = 5, 8
    c_nodes = torch.arange(n_cons, dtype=torch.float32).unsqueeze(1).expand(-1, 4)
    v_nodes = torch.arange(n_vars, dtype=torch.float32).unsqueeze(1).expand(-1, 6)

    rows = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    cols = [0, 1, 2, 1, 3, 2, 4, 5, 5, 6, 6, 7]
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.arange(len(rows), dtype=torch.float32).unsqueeze(1)

    return c_nodes, v_nodes, edge_index, edge_attr


# ---------------------------------------------------------------------------
# Tests: pure helpers
# ---------------------------------------------------------------------------


class TestNSplitsFor:
    def test_exact_division(self):
        assert n_splits_for(100, 50) == 2

    def test_no_split_needed(self):
        assert n_splits_for(100, 100) == 1

    def test_rounding_up(self):
        assert n_splits_for(100, 33) == 4  # ceil(100/33) = 4

    def test_single_element(self):
        assert n_splits_for(1, 1) == 1

    def test_zero_total(self):
        assert n_splits_for(0, 10) == 1  # max(1, ceil(0/10)) = 1


class TestBalancedChunks:
    def test_even_split(self):
        assert balanced_chunks(9, 3) == [3, 3, 3]

    def test_uneven_split(self):
        assert balanced_chunks(10, 3) == [4, 3, 3]

    def test_single_chunk(self):
        assert balanced_chunks(5, 1) == [5]


# ---------------------------------------------------------------------------
# Tests: extract_subgraph (vectorised)
# ---------------------------------------------------------------------------


class TestExtractSubgraph:
    def test_basic_extraction(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()

        # Extract subgraph with constraints {0, 1} and variables {0, 1, 2, 3}
        cons_ids = torch.tensor([0, 1])
        var_ids = torch.tensor([0, 1, 2, 3])
        sg = extract_subgraph(cons_ids, var_ids, c_nodes, v_nodes, edge_index, edge_attr)

        assert sg.constraint_features.size(0) == 2
        assert sg.variable_features.size(0) == 4
        assert torch.equal(sg.orig_cons_ids, torch.tensor([0, 1]))
        assert torch.equal(sg.orig_var_ids, torch.tensor([0, 1, 2, 3]))

        # Edge indices must be in valid range
        assert sg.edge_index[0].max() < 2
        assert sg.edge_index[1].max() < 4

        # Expected edges: c0-v0, c0-v1, c0-v2, c1-v1, c1-v3
        # Remapped: (0,0), (0,1), (0,2), (1,1), (1,3)
        assert sg.edge_index.size(1) == 5

    def test_unsorted_ids(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()

        # Pass unsorted IDs — should still work
        cons_ids = torch.tensor([1, 0])
        var_ids = torch.tensor([3, 1, 0, 2])
        sg = extract_subgraph(cons_ids, var_ids, c_nodes, v_nodes, edge_index, edge_attr)

        assert torch.equal(sg.orig_cons_ids, torch.tensor([0, 1]))
        assert torch.equal(sg.orig_var_ids, torch.tensor([0, 1, 2, 3]))
        assert sg.edge_index.size(1) == 5

    def test_duplicate_ids(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()

        # Pass duplicate IDs — should deduplicate
        cons_ids = torch.tensor([0, 0, 1, 1])
        var_ids = torch.tensor([0, 1, 1, 2])
        sg = extract_subgraph(cons_ids, var_ids, c_nodes, v_nodes, edge_index, edge_attr)

        assert sg.constraint_features.size(0) == 2
        assert sg.variable_features.size(0) == 3  # {0, 1, 2}

    def test_features_match_original(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()

        cons_ids = torch.tensor([2, 4])
        var_ids = torch.tensor([2, 4, 5, 6, 7])
        sg = extract_subgraph(cons_ids, var_ids, c_nodes, v_nodes, edge_index, edge_attr)

        # First 4 columns of constraint features should match c_nodes[2] and c_nodes[4]
        assert torch.allclose(sg.constraint_features[0, :4], c_nodes[2])
        assert torch.allclose(sg.constraint_features[1, :4], c_nodes[4])


class TestExtractSubgraphByConstraints:
    def test_induced_variables(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()

        # c0 connects to v0, v1, v2
        sg = extract_subgraph_by_constraints(
            torch.tensor([0]), c_nodes, v_nodes, edge_index, edge_attr
        )
        assert sg.constraint_features.size(0) == 1
        assert torch.equal(sg.orig_var_ids, torch.tensor([0, 1, 2]))


class TestExtractSubgraphByVariables:
    def test_induced_constraints(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()

        # v5 connects to c2 and c3
        sg = extract_subgraph_by_variables(
            torch.tensor([5]), c_nodes, v_nodes, edge_index, edge_attr
        )
        assert sg.variable_features.size(0) == 1
        assert torch.equal(sg.orig_cons_ids, torch.tensor([2, 3]))


# ---------------------------------------------------------------------------
# Tests: splitting
# ---------------------------------------------------------------------------


class TestSplitByConstraints:
    def test_basic_split(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()

        subgraphs = split_by_constraints(c_nodes, v_nodes, edge_index, edge_attr, 3)
        assert len(subgraphs) == 2  # ceil(5/3)=2 → chunks [3, 2]
        # Each subgraph has orig_cons_ids
        all_cons = torch.cat([sg.orig_cons_ids for sg in subgraphs])
        assert all_cons.unique().size(0) == 5  # all constraints covered


class TestSplitByVariables:
    def test_basic_split(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()

        subgraphs = split_by_variables(c_nodes, v_nodes, edge_index, edge_attr, 4)
        assert len(subgraphs) == 2  # ceil(8/4)=2 → chunks [4, 4]
        all_vars = torch.cat([sg.orig_var_ids for sg in subgraphs])
        assert all_vars.unique().size(0) == 8


class TestSplitBipartiteGraphMetis:
    def test_single_partition(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()

        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=1
        )
        assert len(specs) == 1
        assert specs[0].owned_cons_ids.size(0) == 5
        assert specs[0].owned_var_ids.size(0) == 8

    def test_two_partitions_coverage(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()

        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=2
        )
        assert len(specs) == 2

        # All constraints and variables covered
        all_cons = torch.cat([s.owned_cons_ids for s in specs])
        all_vars = torch.cat([s.owned_var_ids for s in specs])
        assert all_cons.unique().size(0) == 5
        assert all_vars.unique().size(0) == 8

    def test_disconnected_components(self):
        """Two disconnected bipartite components should partition cleanly."""
        # Component 1: c0-v0, c0-v1
        # Component 2: c1-v2, c1-v3
        c_nodes = torch.randn(2, 4)
        v_nodes = torch.randn(4, 6)
        edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long)
        edge_attr = torch.ones(4, 1)

        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=2
        )
        assert len(specs) == 2

        # Validate partition
        validate_partition(specs, n_cons=2, n_vars=4)


# ---------------------------------------------------------------------------
# Tests: coupling diagnostics
# ---------------------------------------------------------------------------


class TestCouplingDiagnostics:
    def test_no_coupling(self):
        """Perfectly separated subgraphs → zero coupling."""
        # Build a graph with two disconnected components
        c_nodes = torch.randn(4, 4)
        v_nodes = torch.randn(4, 6)
        # c0-v0, c0-v1 (component 1) and c1-v2, c1-v3 (component 2)
        # c2-v0 (component 1) and c3-v3 (component 2)
        edge_index = torch.tensor([[0, 0, 2, 1, 1, 3], [0, 1, 0, 2, 3, 3]], dtype=torch.long)
        edge_attr = torch.ones(6, 1)

        sg1 = extract_subgraph(
            torch.tensor([0, 2]),
            torch.tensor([0, 1]),
            c_nodes, v_nodes, edge_index, edge_attr,
        )
        sg2 = extract_subgraph(
            torch.tensor([1, 3]),
            torch.tensor([2, 3]),
            c_nodes, v_nodes, edge_index, edge_attr,
        )

        diag = compute_coupling_diagnostics([sg1, sg2], 4, 4, edge_index)
        assert isinstance(diag, CouplingDiagnostics)
        assert diag.edge_cut_count == 0
        assert diag.n_coupling_constraints == 0
        # New boundary fields
        assert diag.n_boundary_cons == 0
        assert diag.n_boundary_vars == 0
        assert diag.boundary_cons_fraction == 0.0
        assert diag.boundary_vars_fraction == 0.0
        assert diag.n_total_vars == 4

    def test_with_coupling(self):
        """Overlapping variable assignments → non-zero coupling."""
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()

        # v2 appears in both subgraphs → coupling via c0 and c2
        sg1 = extract_subgraph(
            torch.tensor([0, 1]),
            torch.tensor([0, 1, 2, 3]),
            c_nodes, v_nodes, edge_index, edge_attr,
        )
        sg2 = extract_subgraph(
            torch.tensor([2, 3, 4]),
            torch.tensor([2, 4, 5, 6, 7]),  # v2 shared!
            c_nodes, v_nodes, edge_index, edge_attr,
        )

        diag = compute_coupling_diagnostics([sg1, sg2], 5, 8, edge_index)
        # sg2 overwrites v2's block, so some edges of c0 cross blocks
        assert diag.edge_cut_count > 0
        # New boundary fields should be populated
        assert diag.n_boundary_cons >= 0
        assert diag.n_boundary_vars >= 0
        assert diag.n_total_vars == 8
        assert diag.boundary_cons_fraction == diag.n_boundary_cons / 5
        assert diag.boundary_vars_fraction == diag.n_boundary_vars / 8


# ---------------------------------------------------------------------------
# Tests: validate_partition
# ---------------------------------------------------------------------------


class TestValidatePartition:
    def test_valid_partition(self):
        specs = [
            PartitionSpec(0, torch.tensor([0, 1]), torch.tensor([0, 1, 2])),
            PartitionSpec(1, torch.tensor([2, 3, 4]), torch.tensor([3, 4, 5, 6, 7])),
        ]
        # Should not raise
        validate_partition(specs, n_cons=5, n_vars=8)

    def test_overlapping_constraints(self):
        specs = [
            PartitionSpec(0, torch.tensor([0, 1, 2]), torch.tensor([0, 1])),
            PartitionSpec(1, torch.tensor([2, 3, 4]), torch.tensor([2, 3])),  # c2 overlap
        ]
        with pytest.raises(ValueError, match="overlap"):
            validate_partition(specs, n_cons=5, n_vars=4)

    def test_missing_variables(self):
        specs = [
            PartitionSpec(0, torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2])),
            # v3..v7 missing
        ]
        with pytest.raises(ValueError, match="missing"):
            validate_partition(specs, n_cons=5, n_vars=8)

    def test_empty_partition_list(self):
        with pytest.raises(ValueError, match="Empty"):
            validate_partition([], n_cons=5, n_vars=8)

    def test_empty_partition_entry(self):
        specs = [
            PartitionSpec(0, torch.tensor([0, 1, 2, 3, 4]), torch.tensor([], dtype=torch.long)),
        ]
        with pytest.raises(ValueError, match="empty"):
            validate_partition(specs, n_cons=5, n_vars=0)


# ---------------------------------------------------------------------------
# Tests: _bfs_expand_bipartite
# ---------------------------------------------------------------------------


class TestBfsExpandBipartite:
    def test_zero_hops(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        cons, vars_ = _bfs_expand_bipartite(
            torch.tensor([0]), torch.tensor([7]),
            edge_index, 5, 8, hops=0,
        )
        assert torch.equal(cons, torch.tensor([0]))
        assert torch.equal(vars_, torch.tensor([7]))

    def test_one_hop_from_constraint(self):
        """c0 --1 hop--> v0, v1, v2."""
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        cons, vars_ = _bfs_expand_bipartite(
            torch.tensor([0]), torch.tensor([], dtype=torch.long),
            edge_index, 5, 8, hops=1,
        )
        assert torch.equal(cons, torch.tensor([0]))
        assert torch.equal(vars_, torch.tensor([0, 1, 2]))

    def test_two_hops_from_constraint(self):
        """c0 --1 hop--> v0,v1,v2 --2 hops--> c1,c2 (via v1->c1, v2->c2)."""
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        cons, vars_ = _bfs_expand_bipartite(
            torch.tensor([0]), torch.tensor([], dtype=torch.long),
            edge_index, 5, 8, hops=2,
        )
        assert torch.equal(cons, torch.tensor([0, 1, 2]))
        assert torch.equal(vars_, torch.tensor([0, 1, 2]))

    def test_full_graph_at_sufficient_hops(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        cons, vars_ = _bfs_expand_bipartite(
            torch.tensor([0]), torch.tensor([], dtype=torch.long),
            edge_index, 5, 8, hops=8,
        )
        assert cons.size(0) == 5
        assert vars_.size(0) == 8


# ---------------------------------------------------------------------------
# Tests: build_halo_subgraph
# ---------------------------------------------------------------------------


class TestBuildHaloSubgraph:
    def test_halo_zero_matches_extract(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        part = PartitionSpec(0, torch.tensor([0, 1]), torch.tensor([0, 1, 2, 3]))

        sg_halo = build_halo_subgraph(
            part, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=0,
        )
        sg_direct = extract_subgraph(
            part.owned_cons_ids, part.owned_var_ids,
            c_nodes, v_nodes, edge_index, edge_attr,
        )

        assert torch.equal(sg_halo.constraint_features, sg_direct.constraint_features)
        assert torch.equal(sg_halo.variable_features, sg_direct.variable_features)
        assert torch.equal(sg_halo.edge_index, sg_direct.edge_index)
        assert torch.equal(sg_halo.edge_attr, sg_direct.edge_attr)
        assert sg_halo.owned_cons_mask.all()
        assert sg_halo.owned_var_mask.all()
        assert sg_halo.halo_depth.item() == 0

    def test_halo_one_adds_neighbors(self):
        """Partition {c3,c4}/{v5,v6,v7} + 1 hop → adds c2 from v5->c2."""
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        part = PartitionSpec(0, torch.tensor([3, 4]), torch.tensor([5, 6, 7]))

        sg = build_halo_subgraph(
            part, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=1,
        )

        assert torch.equal(sg.orig_cons_ids, torch.tensor([2, 3, 4]))
        assert torch.equal(sg.orig_var_ids, torch.tensor([5, 6, 7]))
        # c2 is halo, c3 and c4 are owned
        assert sg.owned_cons_mask.tolist() == [False, True, True]
        assert sg.owned_var_mask.tolist() == [True, True, True]

    def test_halo_two_expands_further(self):
        """Partition {c3,c4}/{v5,v6,v7} + 2 hops → adds v2,v4 from c2."""
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        part = PartitionSpec(0, torch.tensor([3, 4]), torch.tensor([5, 6, 7]))

        sg = build_halo_subgraph(
            part, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=2,
        )

        assert torch.equal(sg.orig_cons_ids, torch.tensor([2, 3, 4]))
        assert torch.equal(sg.orig_var_ids, torch.tensor([2, 4, 5, 6, 7]))
        assert sg.owned_cons_mask.tolist() == [False, True, True]
        assert sg.owned_var_mask.tolist() == [False, False, True, True, True]

    def test_owned_mask_counts(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        part = PartitionSpec(0, torch.tensor([0, 1]), torch.tensor([0, 1, 2, 3]))

        for hops in [0, 1, 2, 4]:
            sg = build_halo_subgraph(
                part, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=hops,
            )
            assert sg.owned_cons_mask.sum().item() == part.owned_cons_ids.numel()
            assert sg.owned_var_mask.sum().item() == part.owned_var_ids.numel()

    def test_monotonic_size(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        part = PartitionSpec(0, torch.tensor([0]), torch.tensor([0]))

        prev_total = 0
        for hops in [0, 1, 2, 4]:
            sg = build_halo_subgraph(
                part, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=hops,
            )
            total = sg.constraint_features.size(0) + sg.variable_features.size(0)
            assert total >= prev_total
            prev_total = total

    def test_edge_validity(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        part = PartitionSpec(0, torch.tensor([3, 4]), torch.tensor([5, 6, 7]))

        for hops in [0, 1, 2, 4]:
            sg = build_halo_subgraph(
                part, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=hops,
            )
            n_c = sg.constraint_features.size(0)
            n_v = sg.variable_features.size(0)
            assert (sg.edge_index[0] < n_c).all()
            assert (sg.edge_index[1] < n_v).all()

    def test_build_halo_subgraphs_batch(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=2,
        )

        sgs = build_halo_subgraphs(
            specs, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=1,
        )
        assert len(sgs) == 2
        for sg in sgs:
            assert hasattr(sg, "owned_cons_mask")
            assert hasattr(sg, "owned_var_mask")
            assert hasattr(sg, "halo_depth")
            assert sg.halo_depth.item() == 1

    def test_negative_halo_hops_raises(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        part = PartitionSpec(0, torch.tensor([0]), torch.tensor([0]))

        with pytest.raises(ValueError, match="halo_hops"):
            build_halo_subgraph(
                part, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=-1,
            )


# ---------------------------------------------------------------------------
# Tests: compute_coupling_diagnostics_from_specs
# ---------------------------------------------------------------------------


class TestCouplingDiagnosticsFromSpecs:
    def test_no_coupling(self):
        """Two disconnected components in separate partitions → zero coupling."""
        # Component 1: c0-v0, c0-v1
        # Component 2: c1-v2, c1-v3
        c_nodes = torch.randn(2, 4)
        v_nodes = torch.randn(4, 6)
        edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long)

        specs = [
            PartitionSpec(0, torch.tensor([0]), torch.tensor([0, 1])),
            PartitionSpec(1, torch.tensor([1]), torch.tensor([2, 3])),
        ]
        diag = compute_coupling_diagnostics_from_specs(specs, 2, 4, edge_index)

        assert diag.edge_cut_count == 0
        assert diag.n_coupling_constraints == 0
        assert diag.n_boundary_cons == 0
        assert diag.n_boundary_vars == 0
        assert diag.boundary_cons_fraction == 0.0
        assert diag.boundary_vars_fraction == 0.0

    def test_with_coupling(self):
        """METIS split of _make_graph produces cross-partition edges."""
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=2,
        )
        diag = compute_coupling_diagnostics_from_specs(
            specs, 5, 8, edge_index,
        )

        assert diag.edge_cut_count >= 0
        assert diag.n_total_edges == 12
        assert diag.n_total_vars == 8
        assert diag.n_total_constraints == 5
        # Boundary fractions consistent with counts
        assert diag.boundary_cons_fraction == diag.n_boundary_cons / 5
        assert diag.boundary_vars_fraction == diag.n_boundary_vars / 8

    def test_boundary_masks_passthrough(self):
        """Pre-computed boundary masks are used directly."""
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = [
            PartitionSpec(0, torch.tensor([0, 1]), torch.tensor([0, 1, 2, 3])),
            PartitionSpec(1, torch.tensor([2, 3, 4]), torch.tensor([4, 5, 6, 7])),
        ]

        # Manually set boundary masks
        cons_bnd = torch.tensor([False, False, True, True, False])
        vars_bnd = torch.tensor([False, False, True, False, True, False, False, False])

        diag = compute_coupling_diagnostics_from_specs(
            specs, 5, 8, edge_index,
            cons_is_boundary=cons_bnd,
            vars_is_boundary=vars_bnd,
        )

        assert diag.n_boundary_cons == 2
        assert diag.n_boundary_vars == 2
        assert diag.boundary_cons_fraction == 2 / 5
        assert diag.boundary_vars_fraction == 2 / 8


# ---------------------------------------------------------------------------
# Tests: compute_halo_expansion_ratio
# ---------------------------------------------------------------------------


class TestHaloExpansionRatio:
    def test_zero_halo(self):
        assert compute_halo_expansion_ratio(10, 20, 0, 0) == 1.0

    def test_with_halo(self):
        # 10+20 owned, 5+10 halo → (30+15)/30 = 1.5
        assert compute_halo_expansion_ratio(10, 20, 5, 10) == 1.5

    def test_empty_partition(self):
        assert compute_halo_expansion_ratio(0, 0, 5, 5) == float("inf")


# ---------------------------------------------------------------------------
# Tests: build_block_graph
# ---------------------------------------------------------------------------


class TestBuildBlockGraph:
    def test_basic_construction(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=2,
        )
        bg = build_block_graph(
            specs, edge_index, edge_attr, n_cons=5, n_vars=8,
        )

        assert isinstance(bg, BlockGraph)
        assert bg.n_blocks == 2
        assert bg.block_edge_index.dim() == 2
        assert bg.block_edge_index.size(0) == 2
        assert bg.block_edge_index.size(1) >= 1  # partitions are coupled
        assert bg.block_edge_attr.size(0) == bg.block_edge_index.size(1)
        assert bg.block_edge_attr.size(1) == 4
        # Block indices in range
        assert (bg.block_edge_index >= 0).all()
        assert (bg.block_edge_index < 2).all()
        assert bg.block_features is None

    def test_edge_features_positive(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=2,
        )
        bg = build_block_graph(
            specs, edge_index, edge_attr, n_cons=5, n_vars=8,
        )

        if bg.block_edge_index.size(1) > 0:
            assert (bg.block_edge_attr[:, 0] >= 1).all()  # cut count >= 1
            assert (bg.block_edge_attr[:, 1] >= 0).all()  # sum abs coeff >= 0
            assert (bg.block_edge_attr[:, 2] >= 1).all()  # bnd cons >= 1
            assert (bg.block_edge_attr[:, 3] >= 1).all()  # bnd vars >= 1

    def test_disconnected_partitions_no_edges(self):
        """Two disconnected components → 0 block edges."""
        c_nodes = torch.randn(2, 4)
        v_nodes = torch.randn(4, 6)
        edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long)
        edge_attr = torch.ones(4, 1)

        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=2,
        )
        bg = build_block_graph(
            specs, edge_index, edge_attr, n_cons=2, n_vars=4,
        )

        assert bg.n_blocks == 2
        assert bg.block_edge_index.size(1) == 0

    def test_single_partition(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=1,
        )
        bg = build_block_graph(
            specs, edge_index, edge_attr, n_cons=5, n_vars=8,
        )

        assert bg.n_blocks == 1
        assert bg.block_edge_index.size(1) == 0

    def test_cut_count_matches_diagnostics(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=2,
        )
        bg = build_block_graph(
            specs, edge_index, edge_attr, n_cons=5, n_vars=8,
        )
        diag = compute_coupling_diagnostics_from_specs(specs, 5, 8, edge_index)

        # Total cut edges from block graph should match coupling diagnostics
        total_cut = int(bg.block_edge_attr[:, 0].sum().item())
        assert total_cut == diag.edge_cut_count


# ---------------------------------------------------------------------------
# Tests: compute_block_features
# ---------------------------------------------------------------------------


class TestComputeBlockFeatures:
    def test_basic_shape(self):
        d = 8
        specs = [
            PartitionSpec(0, torch.tensor([0, 1]), torch.tensor([0, 1, 2, 3])),
            PartitionSpec(1, torch.tensor([2, 3, 4]), torch.tensor([4, 5, 6, 7])),
        ]
        c_emb = torch.randn(5, d)
        v_emb = torch.randn(8, d)

        features = compute_block_features(specs, c_emb, v_emb)
        assert features.shape == (2, 4 * d)

    def test_with_metadata(self):
        d = 8
        specs = [
            PartitionSpec(0, torch.tensor([0, 1]), torch.tensor([0, 1, 2, 3])),
            PartitionSpec(1, torch.tensor([2, 3, 4]), torch.tensor([4, 5, 6, 7])),
        ]
        c_emb = torch.randn(5, d)
        v_emb = torch.randn(8, d)

        features = compute_block_features(
            specs, c_emb, v_emb, include_metadata=True,
        )
        assert features.shape == (2, 4 * d + 4)

    def test_values_correctness(self):
        d = 4
        specs = [
            PartitionSpec(0, torch.tensor([0]), torch.tensor([0, 1])),
        ]
        c_emb = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        v_emb = torch.tensor([[10.0, 20.0, 30.0, 40.0],
                               [2.0, 4.0, 6.0, 8.0]])

        features = compute_block_features(specs, c_emb, v_emb)

        # mean_v = [6, 12, 18, 24], max_v = [10, 20, 30, 40]
        # mean_c = [1, 2, 3, 4],    max_c = [1, 2, 3, 4]
        expected = torch.tensor([[
            6, 12, 18, 24,   # mean_v
            10, 20, 30, 40,  # max_v
            1, 2, 3, 4,      # mean_c
            1, 2, 3, 4,      # max_c
        ]], dtype=torch.float)

        assert torch.allclose(features, expected)

    def test_single_partition(self):
        d = 8
        specs = [
            PartitionSpec(0, torch.arange(5), torch.arange(8)),
        ]
        c_emb = torch.randn(5, d)
        v_emb = torch.randn(8, d)

        features = compute_block_features(specs, c_emb, v_emb)
        assert features.shape == (1, 4 * d)


# ---------------------------------------------------------------------------
# Tests: log_block_graph_diagnostics
# ---------------------------------------------------------------------------


class TestLogBlockGraphDiagnostics:
    def test_runs_without_error(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=2,
        )
        bg = build_block_graph(
            specs, edge_index, edge_attr, n_cons=5, n_vars=8,
        )
        # Should not raise
        log_block_graph_diagnostics(bg)
