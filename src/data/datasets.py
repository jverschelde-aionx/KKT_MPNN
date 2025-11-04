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
        model.hideOutput(False)
        model.readProblem(lp_path)

        variables = model.getVars()
        n_variables = len(variables)
        variable_idx = {
            str(variable): index for index, variable in enumerate(variables)
        }

        rows = []
        rhs = []

        # Row constraints
        for constraint in model.getConss():
            row = np.zeros(n_variables, dtype=np.float32)

            constraint_coefficients = model.getValsLinear(constraint)

            for v, coef in constraint_coefficients.items():
                row[variable_idx[str(v)]] += float(coef)

            # SCIP stores lhs <= a^T x <= rhs (could be +/- inf)
            lhs = model.getLhs(constraint)
            rhs_r = model.getRhs(constraint)

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
            lb = v.getLbOriginal()
            ub = v.getUbOriginal()
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
        c = np.asarray([v.getObj() for v in variables], dtype=np.float32)  # [n]

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


def make_pad_collate(M_fixed: int, N_fixed: int):
    def _pad_collate(batch: List[Dict]):
        # infer device/dtype from first sample
        A0, b0, c0 = batch[0]["A"], batch[0]["b"], batch[0]["c"]
        device = A0.device
        dtypeA, dtypeb, dtypec = A0.dtype, b0.dtype, c0.dtype

        B = len(batch)
        A_batch = torch.zeros((B, M_fixed, N_fixed), dtype=dtypeA, device=device)
        b_batch = torch.zeros((B, M_fixed), dtype=dtypeb, device=device)
        c_batch = torch.zeros((B, N_fixed), dtype=dtypec, device=device)
        mask_m = torch.zeros((B, M_fixed), dtype=torch.float32, device=device)
        mask_n = torch.zeros((B, N_fixed), dtype=torch.float32, device=device)

        for i, item in enumerate(batch):
            A_i, b_i, c_i = item["A"], item["b"], item["c"]
            m_i, n_i = item["m"], item["n"]  # actual sizes of THIS sample

            # copy into the top‑left block
            A_batch[i, :m_i, :n_i] = A_i
            b_batch[i, :m_i] = b_i
            c_batch[i, :n_i] = c_i
            mask_m[i, :m_i] = 1.0
            mask_n[i, :n_i] = 1.0

        # flat input = [vec(A), b, c]
        flat_input = torch.cat([A_batch.view(B, -1), b_batch, c_batch], dim=1)
        return flat_input, A_batch, b_batch, c_batch, mask_m, mask_n

    return _pad_collate


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
    A_sp_list, b_list, c_list = [], [], []
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

        graphs.append(data)
        A_sp_list.append(A_sparse)
        b_list.append(data.b_vec)
        c_list.append(data.c_vec)
        m_sizes.append(m)
        n_sizes.append(n)

    B = len(batch)
    max_m, max_n = max(m_sizes), max(n_sizes)
    A = torch.zeros(B, max_m, max_n, dtype=torch.float32)
    b = torch.zeros(B, max_m, dtype=torch.float32)
    c = torch.zeros(B, max_n, dtype=torch.float32)
    mask_m = torch.zeros(B, max_m, dtype=torch.float32)
    mask_n = torch.zeros(B, max_n, dtype=torch.float32)

    for i, (A_sp, b_i, c_i, m, n) in enumerate(
        zip(A_sp_list, b_list, c_list, m_sizes, n_sizes)
    ):
        A[i, :m, :n] = A_sp.to_dense()
        b[i, :m] = b_i
        c[i, :n] = c_i
        mask_m[i, :m] = 1.0
        mask_n[i, :n] = 1.0

    batch_graph = torch_geometric.data.Batch.from_data_list(graphs)
    return batch_graph, A, b, c, mask_m, mask_n, m_sizes, n_sizes


def right_pad(x: torch.Tensor, target: int) -> torch.Tensor:
    # Pads the last dim of `x` with zeros on the right up to `target`.
    if x.size(1) == target:
        return x
    pad_w = target - x.size(1)
    pad = x.new_zeros((x.size(0), pad_w))
    return torch.cat([x, pad], dim=1)
