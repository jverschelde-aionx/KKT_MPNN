import pickle
from typing import List, Optional, Tuple

import torch
import torch_geometric
from rtdl_num_embeddings import PeriodicEmbeddings, PiecewiseLinearEmbeddings

_CONS_PAD: int = 4  # global max constraint feature width
_VAR_PAD: int = 18  # global max variable   feature width


class GNNPolicy(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        BipartiteGraphConvolution.add_args(parser)
        group = parser.add_argument_group("GNNPolicy - MPNN Config")
        group.add_argument(
            "--cons_nfeats",
            default=4,
            type=int,
            help="Number of features for constraints",
        )
        group.add_argument(
            "--edge_nfeats", default=1, type=int, help="Number of features for edges"
        )
        group.add_argument(
            "--var_nfeats", default=6, type=int, help="Number of features for variables"
        )
        group.add_argument(
            "--num_emb_type", choices=["periodic", "pwl", "linear"], default="periodic"
        )
        group.add_argument("--num_emb_bins", type=int, default=32)  # for PWL
        group.add_argument("--num_emb_freqs", type=int, default=16)  # for periodic

    @staticmethod
    def name(args):
        name = "gnn_policy"
        name += f"-embedding_size={args.embedding_size}"
        name += f"-cons_nfeats={args.cons_nfeats}"
        name += f"-edge_nfeats={args.edge_nfeats}"
        name += f"-var_nfeats={args.var_nfeats}"
        name += f"-num_emb_type={args.num_emb_type}"
        if args.num_emb_type == "pwl":
            name += f"-num_emb_bins={args.num_emb_bins}"
        elif args.num_emb_type == "periodic":
            name += f"-num_emb_freqs={args.num_emb_freqs}"
        return name

    def __init__(self, args):
        super().__init__()
        d_out = args.embedding_size
        # Constraint embedding
        self.cons_num_emb, cons_out_dim = self._build_numeric_block(
            in_feats=args.cons_nfeats,
            emb_type=args.num_emb_type,
            n_freq=args.num_emb_freqs,
            n_bins=args.num_emb_bins,
        )
        self.cons_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_out_dim),
            torch.nn.Linear(cons_out_dim, d_out),
            torch.nn.ReLU(),
            torch.nn.Linear(d_out, d_out),
            torch.nn.ReLU(),
        )

        # Variable embedding
        self.var_num_emb, var_out_dim = self._build_numeric_block(
            in_feats=args.var_nfeats,
            emb_type=args.num_emb_type,
            n_freq=args.num_emb_freqs,
            n_bins=args.num_emb_bins,
        )
        self.var_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(var_out_dim),
            torch.nn.Linear(var_out_dim, d_out),
            torch.nn.ReLU(),
            torch.nn.Linear(d_out, d_out),
            torch.nn.ReLU(),
        )

        # Edge embedding (A_ij coefficients)
        self.edge_num_emb, edge_out_dim = self._build_numeric_block(
            in_feats=args.edge_nfeats,
            emb_type=args.num_emb_type,
            n_freq=args.num_emb_freqs,
            n_bins=args.num_emb_bins,
        )
        self.edge_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_out_dim),
            torch.nn.Linear(edge_out_dim, d_out),
            torch.nn.ReLU(),
            torch.nn.Linear(d_out, d_out),
            torch.nn.ReLU(),
        )

        # Message‑passing layers
        self.conv_v_to_c = BipartiteGraphConvolution(args)
        self.conv_c_to_v = BipartiteGraphConvolution(args)
        self.conv_v_to_c2 = BipartiteGraphConvolution(args)
        self.conv_c_to_v2 = BipartiteGraphConvolution(args)

        # Output module
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(d_out, d_out),
            torch.nn.ReLU(),
            torch.nn.Linear(d_out, 1, bias=False),
        )

    def forward(
        self,
        constraint_features,  # (n_c, cons_nfeats)
        edge_indices,  # (2, |E|)
        edge_features,  # (|E|, edge_nfeats)
        variable_features,
    ):  # (n_v, var_nfeats)
        print("Raw constraint features:", constraint_features)
        print("Raw variable features:", variable_features)
        print("Raw edge indices:", edge_indices)
        print("Raw edge features:", edge_features)
        rev_idx = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # numeric embeddings + small MLP
        c = self.cons_proj(self.cons_num_emb(constraint_features))
        v = self.var_proj(self.var_num_emb(variable_features))
        e = self.edge_proj(self.edge_num_emb(edge_features))

        print("Raw constraint emb:", c.shape, "\n", c)
        print("Raw variable emb:", v.shape, "\n", v)
        print("Raw edge emb:", e.shape, "\n", e)

        # two rounds of bipartite convolution
        c = self.conv_v_to_c(v, rev_idx, e, c)
        v = self.conv_c_to_v(c, edge_indices, e, v)
        c = self.conv_v_to_c2(v, rev_idx, e, c)
        v = self.conv_c_to_v2(c, edge_indices, e, v)

        # optional scalar head
        return self.output_module(v).squeeze(-1)  # (n_v,)

    @staticmethod
    def _build_numeric_block(
        in_feats: int,
        emb_type: str,
        n_freq: int,
        n_bins: int,
    ) -> Tuple[torch.nn.Module, int]:
        # Build numeric embedding and return (module, output_dim).

        if emb_type == "periodic":
            emb_layer = PeriodicEmbeddings(
                n_features=in_feats,
                n_frequencies=n_freq,
                lite=True,  # keep your current setting
            )
        elif emb_type == "pwl":
            emb_layer = PiecewiseLinearEmbeddings(
                n_features=in_feats,
                num_bins=n_bins,
            )
        else:  # "linear"
            return torch.nn.Identity(), in_feats

        # Determine the true output width by a forward pass on a dummy tensor.
        with torch.no_grad():
            dummy: torch.Tensor = torch.zeros(1, in_feats)
            out_dim: int = emb_layer(dummy).numel()

        return torch.nn.Sequential(emb_layer, torch.nn.Flatten(start_dim=1)), out_dim


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("BipartiteGraphConvolution - MPNN Config")
        # The embedding size is the size of the node embeddings
        group.add_argument(
            "--embedding_size", default=64, type=int, help="Size of the node embeddings"
        )

    @staticmethod
    def name(args):
        name = "bipartite_graph_convolution"
        name += f"-embedding_size={args.embedding_size}"
        return name

    def __init__(self, args):
        super().__init__("add")

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(args.embedding_size, args.embedding_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(args.embedding_size, args.embedding_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(args.embedding_size, args.embedding_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(args.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.embedding_size, args.embedding_size),
        )

        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(args.embedding_size)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * args.embedding_size, args.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.embedding_size, args.embedding_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """

        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output


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


# custom class that wraps the GNNPolicy to be used as an encoder that can be used in a GraphTrans model.
class PolicyEncoder(torch.nn.Module):
    """
    Wrapper that turns GNNPolicy into a pure encoder.
    `out_dim` is used later by the transformer.
    """

    def __init__(self, args):
        super().__init__()
        self.base = GNNPolicy(args)
        self.out_dim = args.embedding_size  # 64 by default

    def forward(self, data, perturb=None):
        c = data.constraint_features  # [n_c, f_c]
        v = data.variable_features  # [n_v, f_v]
        e_idx = data.edge_index  # shape [2, |E|]
        e_attr = data.edge_attr  # [|E|, f_e]

        # identical to first part of original GNNPolicy.forward
        rev_idx = torch.stack([e_idx[1], e_idx[0]], dim=0)
        c = self.base.cons_proj(self.base.cons_num_emb(c))
        e = self.base.edge_proj(self.base.edge_num_emb(data.edge_attr))
        v = self.base.var_proj(self.base.var_num_emb(v))

        c = self.base.conv_v_to_c(v, rev_idx, e, c)
        v = self.base.conv_c_to_v(c, e_idx, e, v)
        c = self.base.conv_v_to_c2(v, rev_idx, e, c)
        v = self.base.conv_c_to_v2(c, e_idx, e, v)

        h = torch.cat([c, v], dim=0)  # [n_c+n_v, 64]
        if perturb is not None:
            h = h + perturb
        return h


def _right_pad(x: torch.Tensor, target: int) -> torch.Tensor:
    # Pads the last dim of `x` with zeros on the right up to `target`.
    if x.size(1) == target:
        return x
    pad_w = target - x.size(1)
    pad = x.new_zeros((x.size(0), pad_w))
    return torch.cat([x, pad], dim=1)


def collate(
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
    A_list, b_list, c_list = [], [], []
    m_sizes, n_sizes, sources = [], [], []

    for data in batch:
        # Encoder padding
        data.constraint_features = _right_pad(data.constraint_features, _CONS_PAD)
        data.variable_features = _right_pad(data.variable_features, _VAR_PAD)

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
        A_list.append(A_sparse)
        b_list.append(data.b_vec)
        c_list.append(data.c_vec)
        m_sizes.append(m)
        n_sizes.append(n)
        sources.append(getattr(data, "sample_path", "<unknown>"))

    B = len(batch)
    max_m, max_n = max(m_sizes), max(n_sizes)
    b_pad = torch.zeros(B, max_m, dtype=torch.float32)
    b_mask = torch.zeros(B, max_m, dtype=torch.bool)
    c_pad = torch.zeros(B, max_n, dtype=torch.float32)
    c_mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, (b, m) in enumerate(zip(b_list, m_sizes)):
        b_pad[i, :m] = b
        b_mask[i, :m] = True
    for i, (c, n) in enumerate(zip(c_list, n_sizes)):
        c_pad[i, :n] = c
        c_mask[i, :n] = True

    batch_graph = torch_geometric.data.Batch.from_data_list(graphs)
    return batch_graph, A_list, b_pad, c_pad, b_mask, c_mask, m_sizes, n_sizes, sources


def collate_graph_only(batch):
    # same as default collate, but does NOT build A_sparse/A_list
    # return an empty list for A_list to keep unpacking compatible
    graphs, b_list, c_list, m_sizes, n_sizes, sources = [], [], [], [], [], []
    for data in batch:
        data.constraint_features = _right_pad(data.constraint_features, _CONS_PAD)
        data.variable_features = _right_pad(data.variable_features, _VAR_PAD)
        graphs.append(data)
        b_list.append(data.b_vec)
        c_list.append(data.c_vec)
        m_sizes.append(int(data.b_vec.numel()))
        n_sizes.append(int(data.c_vec.numel()))
        sources.append(getattr(data, "sample_path", "<unknown>"))

    B = len(batch)
    max_m, max_n = max(m_sizes), max(n_sizes)
    b_pad = torch.zeros(B, max_m)
    b_mask = torch.zeros(B, max_m, dtype=torch.bool)
    c_pad = torch.zeros(B, max_n)
    c_mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (b, m) in enumerate(zip(b_list, m_sizes)):
        b_pad[i, :m] = b
        b_mask[i, :m] = True
    for i, (c, n) in enumerate(zip(c_list, n_sizes)):
        c_pad[i, :n] = c
        c_mask[i, :n] = True

    batch_graph = torch_geometric.data.Batch.from_data_list(graphs)
    A_list = []  # placeholder
    return batch_graph, A_list, b_pad, c_pad, b_mask, c_mask, m_sizes, n_sizes, sources
