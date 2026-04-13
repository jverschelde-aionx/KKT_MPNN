from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from data.datasets import BipartiteNodeData


@dataclass
class SplitViewMasks:
    """
    Lightweight view: just boolean masks per partition, no graph cloning.
    masks[k][i] = (cons_mask, var_mask, edge_mask) for block k, instance i.
    """

    masks: List[List[tuple]]  # [K][B] of (cons_mask, var_mask, edge_mask)


@dataclass
class SplitPartitionData:
    """
    One owned partition + fixed halo subgraph.
    """

    part_id: int
    graph: BipartiteNodeData
    orig_cons_ids: torch.Tensor  # local row -> original constraint id
    orig_var_ids: torch.Tensor  # local row -> original variable id
    owned_cons_local: torch.Tensor  # local indices of owned constraints
    owned_var_local: torch.Tensor  # local indices of owned variables


@dataclass
class SplitInstanceData:
    """
    Self-contained split representation of one original master problem.

    No full graph is needed anymore after this is created.
    """

    name: str

    partitions: List[SplitPartitionData]

    # block graph
    block_edge_index: torch.Tensor  # [2, E_block]
    block_edge_attr: torch.Tensor  # [E_block, d_edge]

    # per-node global structural metadata
    cons_block_id: torch.Tensor  # [n_cons]
    vars_block_id: torch.Tensor  # [n_vars]
    cons_boundary_feat: torch.Tensor  # [n_cons, 5]
    vars_boundary_feat: torch.Tensor  # [n_vars, 5]
    cons_is_boundary: torch.Tensor  # [n_cons] bool
    vars_is_boundary: torch.Tensor  # [n_vars] bool

    n_cons: int
    n_vars: int

    # optional downstream tensors (for later KKT fine-tuning)
    A_dense: Optional[torch.Tensor] = None  # [m_kkt, n]
    b_vec: Optional[torch.Tensor] = None  # [m_kkt]
    c_vec: Optional[torch.Tensor] = None  # [n]

    @property
    def num_blocks(self) -> int:
        return len(self.partitions)

    def make_view(
        self,
        *,
        cons_mask: float,
        var_mask: float,
        edge_drop: float,
        generator: Optional[torch.Generator] = None,
    ) -> "SplitInstanceData":
        """
        Node-preserving augmentation applied directly to precomputed split subgraphs.
        """
        parts: List[SplitPartitionData] = []
        for p in self.partitions:
            g = p.graph.clone()
            dev = g.constraint_features.device

            g.constraint_features = g.constraint_features.clone()
            g.variable_features = g.variable_features.clone()

            cons_drop = (
                torch.rand(
                    g.constraint_features.size(0),
                    generator=generator,
                    device=dev,
                )
                < cons_mask
            )
            var_drop = (
                torch.rand(
                    g.variable_features.size(0),
                    generator=generator,
                    device=dev,
                )
                < var_mask
            )

            g.constraint_features[cons_drop] = 0
            g.variable_features[var_drop] = 0

            if edge_drop > 0.0 and g.edge_index.numel() > 0:
                E = g.edge_index.size(1)
                keep = torch.rand(E, generator=generator, device=dev) >= edge_drop
                g.edge_index = g.edge_index[:, keep]
                if g.edge_attr is not None:
                    g.edge_attr = g.edge_attr[keep]

            parts.append(
                SplitPartitionData(
                    part_id=p.part_id,
                    graph=g,
                    orig_cons_ids=p.orig_cons_ids,
                    orig_var_ids=p.orig_var_ids,
                    owned_cons_local=p.owned_cons_local,
                    owned_var_local=p.owned_var_local,
                )
            )

        return SplitInstanceData(
            name=self.name,
            partitions=parts,
            block_edge_index=self.block_edge_index,
            block_edge_attr=self.block_edge_attr,
            cons_block_id=self.cons_block_id,
            vars_block_id=self.vars_block_id,
            cons_boundary_feat=self.cons_boundary_feat,
            vars_boundary_feat=self.vars_boundary_feat,
            cons_is_boundary=self.cons_is_boundary,
            vars_is_boundary=self.vars_is_boundary,
            n_cons=self.n_cons,
            n_vars=self.n_vars,
            A_dense=self.A_dense,
            b_vec=self.b_vec,
            c_vec=self.c_vec,
        )


@dataclass
class SplitInstanceBatch:
    instances: List[SplitInstanceData]

    @property
    def num_graphs(self) -> int:
        return len(self.instances)

    def make_view(
        self,
        *,
        cons_mask: float,
        var_mask: float,
        edge_drop: float,
        generator: Optional[torch.Generator] = None,
    ) -> "SplitInstanceBatch":
        return SplitInstanceBatch(
            instances=[
                inst.make_view(
                    cons_mask=cons_mask,
                    var_mask=var_mask,
                    edge_drop=edge_drop,
                    generator=generator,
                )
                for inst in self.instances
            ]
        )


class SplitInstanceDataset(Dataset):
    def __init__(
        self,
        root: str | Path | None = None,
        roots: list | None = None,
        max_instances: int | None = None,
    ):
        paths: list[Path] = []
        if roots is not None:
            for r in roots:
                paths.extend(Path(r).rglob("*.pt"))
        elif root is not None:
            paths.extend(Path(root).rglob("*.pt"))
        else:
            raise ValueError("Provide either root or roots")
        self.paths = sorted(paths)
        if not self.paths:
            raise ValueError(f"No .pt split instances found")
        if max_instances is not None and max_instances < len(self.paths):
            self.paths = self.paths[:max_instances]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> SplitInstanceData:
        return torch.load(self.paths[idx], map_location="cpu")


def split_instance_collate(items: List[SplitInstanceData]) -> SplitInstanceBatch:
    return SplitInstanceBatch(instances=items)
