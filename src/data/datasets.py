import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch_geometric
from pyscipopt import Model
from torch.utils.data import Dataset

from data.common import CONS_PAD, SCIP_INF, VARS_PAD


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath: str):
        with open(filepath, "rb") as f:
            BG = pickle.load(f)
        if not (isinstance(BG, tuple) and len(BG) == 7):
            raise RuntimeError(
                f"{filepath} is not in the correct format. Expected 7‑tuple (A, v_map, v_nodes, c_nodes, b_vars, b_vec, c_vec)."
            )
        return BG

    def get(self, index: int):
        A, v_map, v_nodes, c_nodes, b_vars, b_vec, c_vec = self.process_sample(
            self.sample_files[index]
        )

        edge_indices = A._indices()
        edge_features = A._values().unsqueeze(1)  # real coefficients

        variable_features = torch.as_tensor(v_nodes, dtype=torch.float32)
        constraint_features = torch.as_tensor(c_nodes, dtype=torch.float32)
        b_vec = torch.as_tensor(b_vec, dtype=torch.float32)
        c_vec = torch.as_tensor(c_vec, dtype=torch.float32)

        graph = BipartiteNodeData(
            constraint_features=constraint_features,
            edge_indices=torch.LongTensor(edge_indices),
            edge_features=edge_features,
            variable_features=variable_features,
        )
        graph.num_nodes = constraint_features.size(0) + variable_features.size(0)

        graph.b_vec = b_vec
        graph.c_vec = c_vec
        graph.sample_path = self.sample_files[index]
        return graph


class BipartiteNodeData(torch_geometric.data.Data):
    """
    Node‑bipartite graph for MILP problems.
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    Must support zero‑arg construction so that PyG's Batch can create
    a template object internally.
    """

    def __init__(
        self,
        constraint_features: Optional[torch.Tensor] = None,
        edge_indices: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        variable_features: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # allow empty init (template object)
        if constraint_features is None:
            return

        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features

        m = constraint_features.size(0)
        n = variable_features.size(0)

        self.is_var_node = torch.cat(
            [torch.zeros(m, dtype=torch.bool), torch.ones(n, dtype=torch.bool)]
        )

        self.is_constr_node = ~self.is_var_node

    def to_node_tensor(self) -> torch.Tensor:
        """
        Return a unified [N, D] tensor of node features by concatenating
        constraint and variable node features.
        """
        c_feat = self.constraint_features
        v_feat = self.variable_features
        if c_feat is None or v_feat is None:
            raise ValueError(
                "Both constraint_features and variable_features must be set."
            )

        # Handle unequal feature dimensions
        if c_feat.size(1) != v_feat.size(1):
            # Pad the smaller one to match the larger dimension
            max_d = max(c_feat.size(1), v_feat.size(1))
            if c_feat.size(1) < max_d:
                pad = (0, max_d - c_feat.size(1))
                c_feat = torch.nn.functional.pad(c_feat, pad)
            if v_feat.size(1) < max_d:
                pad = (0, max_d - v_feat.size(1))
                v_feat = torch.nn.functional.pad(v_feat, pad)

        # Stack all nodes together
        return torch.cat([c_feat, v_feat], dim=0)

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __str__(self):
        return (
            f"BipartiteNodeData(num_constraint_nodes={self.constraint_features.size(0)}, "
            f"num_variable_nodes={self.variable_features.size(0)}, "
            f"num_edges={self.edge_index.size(1)}, "
            f"constraint_features={self.constraint_features}, "
            f"variable_features={self.variable_features}, "
            f"edge_index={self.edge_index}, "
            f"edge_attr={self.edge_attr}, "
            f"is_var_node={self.is_var_node}, "
            f"is_constr_node={self.is_constr_node})"
        )


class LPDataset(Dataset):
    """
    Reads a folder/glob of .lp files and returns (A,b,c). No labels required.
    If you *do* have labels elsewhere, you can extend this to load y and add it.
    """

    def __init__(self, lp_files: List[Union[str, Path]]):
        self.files = lp_files

        self.shapes = []
        for file in self.files:
            A, b, c = self.read_lp_file(file)
            self.shapes.append((A.shape[0], A.shape[1]))  # (m, n)

    def __len__(self):
        return len(self.files)

    def read_lp_file(
        self, lp_path: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = Model()
        model.readProblem(lp_path)

        variables = model.getVars()
        n_variables = len(variables)

        rows = []
        rhs = []

        # Row constraints
        for constraint in model.getConss():
            row = np.zeros(n_variables, dtype=np.float32)
            constraint_vars, constraint_coefficients = (
                constraint.getVars(),
                constraint.getValsLinear(),
            )  # coefficients
            for constraint_variable, constraint_coefficient in zip(
                constraint_vars, constraint_coefficients
            ):
                j = variables.index(constraint_variable)
                row[j] = constraint_coefficient

            # SCIP stores lhs <= a^T x <= rhs (could be +/- inf)
            lhs = constraint.getLhs()
            rhs_r = constraint.getRhs()

            # handle three cases: <=, >=, == (lhs == rhs)
            # equality: split into two inequalities
            if (
                lhs > -SCIP_INF
                and rhs_r < SCIP_INF
                and math.isclose(lhs, rhs_r, rel_tol=1e-12, abs_tol=1e-12)
            ):
                rows.append(row.copy())
                rhs.append(rhs_r)
                rows.append((-row).copy())
                rhs.append(-lhs)
            else:
                # upper bound (<= rhs) if finite
                if rhs_r < SCIP_INF:
                    rows.append(row.copy())
                    rhs.append(rhs_r)
                # lower bound (lhs <=) => -a^T x <= -lhs
                if lhs > -SCIP_INF:
                    rows.append((-row).copy())
                    rhs.append(-lhs)

        # Variable bounds as constraints
        for j, v in enumerate(variables):
            lb = v.getLb()
            ub = v.getUb()
            # x_j <= ub
            if ub < SCIP_INF:
                row = np.zeros(n_variables, dtype=np.float32)
                row[j] = 1.0
                rows.append(row)
                rhs.append(ub)
            # -x_j <= -lb   (i.e., x_j >= lb)
            if lb > -SCIP_INF:
                row = np.zeros(n_variables, dtype=np.float32)
                row[j] = -1.0
                rows.append(row)
                rhs.append(-lb)

        A = np.stack(rows, axis=0).astype(np.float32)  # [m, n]
        b = np.asarray(rhs, dtype=np.float32)  # [m]

        # Objective c
        c = np.asarray(
            [model.getObjCoef(v) for v in variables], dtype=np.float32
        )  # [n]

        return A, b, c

    def __getitem__(self, idx):
        lp_path = self.files[idx]
        A, b, c = self.read_lp_file(lp_path)
        m, n = A.shape
        return {
            "A": torch.from_numpy(A),  # [m, n]
            "b": torch.from_numpy(b),  # [m]
            "c": torch.from_numpy(c),  # [n]
            "m": m,
            "n": n,
            "path": lp_path,
        }


def pad_collate(batch: List[Dict]):
    """
    Pads A (m,n), b (m), c (n) to (M,N) in the batch. Returns masks for real rows/cols.
    Also returns the flat input [A_flat, b, c] expected by the MLP.
    """
    m_list = [item["m"] for item in batch]
    n_list = [item["n"] for item in batch]

    A, b, c = [], [], []
    mask_m, mask_n = [], []

    for item in batch:
        A, b, c = item["A"], item["b"], item["c"]
        m, n = item["m"], item["n"]

        # pad A to [M, N]
        A_full = torch.zeros((max(m_list), max(n_list)), dtype=A.dtype)
        A_full[:m, :n] = A
        A.append(A_full)

        # pad b to [M], c to [N]
        b_full = torch.zeros((max(m_list),), dtype=b.dtype)
        b_full[:m] = b
        c_full = torch.zeros((max(n_list),), dtype=c.dtype)
        c_full[:n] = c
        b.append(b_full)
        c.append(c_full)

        # masks
        mask_m.append(torch.arange(max(m_list)) < m)  # [M]
        mask_n.append(torch.arange(max(n_list)) < n)  # [N]

    A = torch.stack(A, dim=0)  # [B, M, N]
    b = torch.stack(b, dim=0)  # [B, M]
    c = torch.stack(c, dim=0)  # [B, N]
    mask_m = torch.stack(mask_m, dim=0).float()  # [B, M]
    mask_n = torch.stack(mask_n, dim=0).float()  # [B, N]

    # flat input = [A_flat, b, c]
    B = A.shape[0]
    flat_A = A.view(B, -1)
    flat_input = torch.cat([flat_A, b, c], dim=1)  # [B, M*N + M + N]

    return {
        "flat_input": flat_input,
        "A": A,
        "b": b,
        "c": c,
        "mask_m": mask_m,
        "mask_n": mask_n,
    }


def pad_collate_graphs(
    batch: List["BipartiteNodeData"],
) -> Tuple[
    torch_geometric.data.Batch,
    List[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[int],
    List[int],
    List[str],
]:
    graphs: List["BipartiteNodeData"] = []
    A, b_list, c_list = [], [], []
    m_sizes, n_sizes = [], []

    for data in batch:
        # Encoder padding
        data.constraint_features = right_pad(data.constraint_features, CONS_PAD)
        data.variable_features = right_pad(data.variable_features, VARS_PAD)

        m = int(data.b_vec.numel())
        n = int(data.c_vec.numel())

        rows, cols = data.edge_index
        A_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([rows, cols]),
            values=data.edge_attr.squeeze(-1),
            size=(m, n),
        ).coalesce()
        assert A_sparse.size(0) == m and A_sparse.size(1) == n

        graphs.append(data)
        A.append(A_sparse)
        b_list.append(data.b_vec)
        c_list.append(data.c_vec)
        m_sizes.append(m)
        n_sizes.append(n)

    B = len(batch)
    max_m, max_n = max(m_sizes), max(n_sizes)
    b = torch.zeros(B, max_m, dtype=torch.float32)
    c = torch.zeros(B, max_n, dtype=torch.float32)

    for i, (b, m) in enumerate(zip(b_list, m_sizes)):
        b[i, :m] = b
    for i, (c, n) in enumerate(zip(c_list, n_sizes)):
        c[i, :n] = c

    batch_graph = torch_geometric.data.Batch.from_data_list(graphs)
    return batch_graph, A, b, c, m_sizes, n_sizes


def right_pad(x: torch.Tensor, target: int) -> torch.Tensor:
    # Pads the last dim of `x` with zeros on the right up to `target`.
    if x.size(1) == target:
        return x
    pad_w = target - x.size(1)
    pad = x.new_zeros((x.size(0), pad_w))
    return torch.cat([x, pad], dim=1)
