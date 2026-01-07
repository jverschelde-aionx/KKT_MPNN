#!/usr/bin/env python3
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import configargparse
import torch
from pyscipopt import Model

from data.utils import lp_path_from_bg

# ----------------------------
# Utilities
# ----------------------------


def _var_name(v) -> str:
    # PySCIPOpt Var sometimes has .name; otherwise str(v) is usually the name
    return getattr(v, "name", None) or str(v)


def sparse_from_edges(
    edge_index: torch.Tensor, edge_attr: torch.Tensor, size: Tuple[int, int]
) -> torch.Tensor:
    """
    edge_index: [2, nnz] long
    edge_attr:  [nnz] or [nnz, 1] float
    """
    if edge_attr.dim() == 2 and edge_attr.size(1) == 1:
        edge_attr = edge_attr.squeeze(1)
    A = torch.sparse_coo_tensor(edge_index, edge_attr, size=size).coalesce()
    return A


def assert_sparse_equal(
    A: torch.Tensor, B: torch.Tensor, atol=1e-6, rtol=1e-6, name="A"
):
    A = A.coalesce()
    B = B.coalesce()
    if A.shape != B.shape:
        raise AssertionError(f"{name}: shape mismatch {A.shape} vs {B.shape}")

    print(f"[check] {name}: shape {A.shape} ✓")

    ia = A.indices()
    ib = B.indices()
    va = A.values()
    vb = B.values()

    if ia.numel() != ib.numel() or not torch.equal(ia, ib):
        # show a small diff
        raise AssertionError(
            f"{name}: indices mismatch.\n"
            f"A nnz={va.numel()}  B nnz={vb.numel()}\n"
            f"A head idx:\n{ia[:, : min(10, ia.size(1))]}\n"
            f"B head idx:\n{ib[:, : min(10, ib.size(1))]}"
        )

    print(f"[check] {name}: indices match (nnz={va.numel()}) ✓")

    if not torch.allclose(va, vb, atol=atol, rtol=rtol):
        diff = (va - vb).abs()
        mx = diff.max().item() if diff.numel() else 0.0
        raise AssertionError(f"{name}: values mismatch (max abs diff={mx})")

    if va.numel() > 0:
        diff = (va - vb).abs()
        mx = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"[check] {name}: values match (max_diff={mx:.2e}, mean_diff={mean_diff:.2e}) ✓")


# ----------------------------
# LP parsing in two “views”
# ----------------------------


def parse_lp_original_bipartite(
    lp_path: Path,
) -> Tuple[torch.Tensor, Dict[str, int], int, int]:
    """
    Returns:
      A_orig_sparse: (m_orig, n) with coefficients for original linear constraints only
      v_map: name->index following sorted variable names
      m_orig, n
    Matches your get_bipartite_graph() ordering:
      vars sorted by name
      constraints sorted by (nnz, str(constraint))
    """
    m = Model()
    m.hideOutput(True)
    m.readProblem(str(lp_path))

    vars_ = m.getVars()
    vars_.sort(key=lambda v: _var_name(v))
    v_map = {_var_name(v): i for i, v in enumerate(vars_)}
    n = len(vars_)

    conss = [c for c in m.getConss() if len(m.getValsLinear(c)) > 0]
    conss.sort(key=lambda c: (len(m.getValsLinear(c)), str(c)))
    m_orig = len(conss)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for i, c in enumerate(conss):
        coeffs = m.getValsLinear(c)  # dict[var]->coef
        for var_obj, coef in coeffs.items():
            coef = float(coef)
            if coef == 0.0:
                continue
            name = _var_name(var_obj)
            j = v_map[name]
            rows.append(i)
            cols.append(j)
            vals.append(coef)

    if len(vals) == 0:
        A = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=(m_orig, n),
        ).coalesce()
    else:
        A = torch.sparse_coo_tensor(
            torch.tensor([rows, cols], dtype=torch.long),
            torch.tensor(vals, dtype=torch.float32),
            size=(m_orig, n),
        ).coalesce()
    return A, v_map, m_orig, n


def parse_lp_kkt_leq(
    lp_path: Path, inf=1e40
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds the same “<= rows + bounds rows” system as your get_bipartite_graph() for:
      - A_kkt (sparse)
      - b_leq (dense)
      - c_vec (dense objective coeffs)
    Note: assumes the .lp has already been normalized to MIN when generated.
    """
    prob = Model()
    prob.hideOutput(True)
    prob.readProblem(str(lp_path))

    vars_ = prob.getVars()
    vars_.sort(key=lambda v: _var_name(v))
    v_map = {_var_name(v): i for i, v in enumerate(vars_)}
    n = len(vars_)

    # objective
    c_vec = torch.zeros(n, dtype=torch.float32)
    for v in vars_:
        c_vec[v_map[_var_name(v)]] = float(v.getObj())

    conss = [c for c in prob.getConss() if len(prob.getValsLinear(c)) > 0]
    conss.sort(key=lambda c: (len(prob.getValsLinear(c)), str(c)))

    ind_rows: List[int] = []
    ind_cols: List[int] = []
    ind_vals: List[float] = []
    b_leq: List[float] = []

    def add_leq_row(coeffs_dict, rhs_val: float):
        row = len(b_leq)
        for var_obj, coef in coeffs_dict.items():
            cval = float(coef)
            if cval == 0.0:
                continue
            j = v_map[_var_name(var_obj)]
            ind_rows.append(row)
            ind_cols.append(j)
            ind_vals.append(cval)
        b_leq.append(float(rhs_val))

    # fold original rows into <= form
    for c in conss:
        coeffs = prob.getValsLinear(c)
        rhs = float(prob.getRhs(c))
        lhs = float(prob.getLhs(c))

        if abs(rhs - lhs) < 1e-12:
            add_leq_row(coeffs, rhs)
            add_leq_row({v: -coef for v, coef in coeffs.items()}, -lhs)
        elif rhs < 1e20:
            add_leq_row(coeffs, rhs)
        else:
            add_leq_row({v: -coef for v, coef in coeffs.items()}, -lhs)

    # add finite variable bounds
    for v in vars_:
        j = v_map[_var_name(v)]
        try:
            lb = float(v.getLb())
            ub = float(v.getUb())
        except Exception:
            lb, ub = -inf, inf
        if ub < inf:
            add_leq_row({v: +1.0}, ub)
        if lb > -inf:
            add_leq_row({v: -1.0}, -lb)

    m_kkt = len(b_leq)
    if len(ind_vals) == 0:
        A = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=(m_kkt, n),
        ).coalesce()
    else:
        A = torch.sparse_coo_tensor(
            torch.tensor([ind_rows, ind_cols], dtype=torch.long),
            torch.tensor(ind_vals, dtype=torch.float32),
            size=(m_kkt, n),
        ).coalesce()

    b = torch.tensor(b_leq, dtype=torch.float32)
    return A, b, c_vec


# ----------------------------
# Main verification
# ----------------------------


def main():
    ap = configargparse.ArgumentParser(
        allow_abbrev=False,
        default_config_files=["configs/check_bg_vs_lp.yml"],
    )
    ap.add_argument("--bg", type=str, required=True, help="Path to a .bg file")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-6)
    args = ap.parse_args()

    bg_path = Path(args.bg)
    lp_path = lp_path_from_bg(bg_path)

    print(f"[info] BG: {bg_path}")
    print(f"[info] LP: {lp_path}")

    # Load BG tuple
    with open(bg_path, "rb") as f:
        A_bg, v_map_bg, v_nodes, c_nodes, b_vars, b_vec_bg, c_vec_bg = pickle.load(f)

    # Convert numpy arrays to PyTorch tensors if needed
    if not isinstance(b_vec_bg, torch.Tensor):
        b_vec_bg = torch.from_numpy(b_vec_bg).float()
    if not isinstance(c_vec_bg, torch.Tensor):
        c_vec_bg = torch.from_numpy(c_vec_bg).float()

    # --- Parse LP (ground truth)
    A_orig_lp, v_map_lp, m_orig, n = parse_lp_original_bipartite(lp_path)
    A_kkt_lp, b_kkt_lp, c_lp = parse_lp_kkt_leq(lp_path)

    # --- Check v_map consistency (names + ordering)
    # We only check that indices match for all names in the BG mapping.
    for name, idx in v_map_bg.items():
        if name not in v_map_lp:
            raise AssertionError(f"v_map: name '{name}' not found in LP variables")
        if v_map_lp[name] != idx:
            raise AssertionError(
                f"v_map: index mismatch for '{name}': BG={idx}, LP={v_map_lp[name]}"
            )

    # --- Check KKT A/b/c match .lp
    A_bg_kkt = A_bg.coalesce()
    assert_sparse_equal(
        A_bg_kkt, A_kkt_lp, atol=args.atol, rtol=args.rtol, name="A_kkt"
    )

    if b_vec_bg.numel() != b_kkt_lp.numel() or not torch.allclose(
        b_vec_bg, b_kkt_lp, atol=args.atol, rtol=args.rtol
    ):
        mx = (b_vec_bg - b_kkt_lp).abs().max().item()
        raise AssertionError(f"b_vec: mismatch vs LP (max abs diff={mx})")
    else:
        diff = (b_vec_bg - b_kkt_lp).abs()
        mx = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"[check] b_vec: match (size={b_vec_bg.numel()}, max_diff={mx:.2e}, mean_diff={mean_diff:.2e}) ✓")

    if c_vec_bg.numel() != c_lp.numel() or not torch.allclose(
        c_vec_bg, c_lp, atol=args.atol, rtol=args.rtol
    ):
        mx = (c_vec_bg - c_lp).abs().max().item()
        raise AssertionError(f"c_vec: mismatch vs LP (max abs diff={mx})")
    else:
        diff = (c_vec_bg - c_lp).abs()
        mx = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"[check] c_vec: match (size={c_vec_bg.numel()}, max_diff={mx:.2e}, mean_diff={mean_diff:.2e}) ✓")

    # --- Check original constraint bipartite edges if present as artifacts
    if hasattr(A_bg, "edge_index") and hasattr(A_bg, "edge_attr"):
        A_bg_orig = sparse_from_edges(A_bg.edge_index, A_bg.edge_attr, size=(m_orig, n))
        assert_sparse_equal(
            A_bg_orig,
            A_orig_lp,
            atol=args.atol,
            rtol=args.rtol,
            name="A_orig(artifacts)",
        )
    else:
        print(
            "[warn] BG has no A.edge_index/A.edge_attr artifacts; skipping original-constraint edge check."
        )

    # --- Now check what your GraphDataset / DataLoader would produce
    # Import lazily so this script can run from repo root
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from data.datasets import GraphDataset

    ds = GraphDataset([str(bg_path)], transform=None)
    g = ds.get(0)

    # This is the key “bug-catcher”: edge_index rows must be within constraint_features
    row_max = int(g.edge_index[0].max().item()) if g.edge_index.numel() else -1
    m_feat = int(g.constraint_features.size(0))
    if row_max >= m_feat:
        raise AssertionError(
            f"Dataset graph has edge_index rows up to {row_max}, "
            f"but constraint_features has only {m_feat} rows. "
            "This indicates you're using the KKT A edges with original constraint features."
        )

    # Check dataset edge matrix equals original constraint matrix from LP
    A_ds = sparse_from_edges(g.edge_index, g.edge_attr, size=(m_feat, n))
    # m_feat should be m_orig if everything is consistent
    if m_feat != m_orig:
        raise AssertionError(
            f"constraint_features rows={m_feat} but LP original constraints={m_orig}"
        )
    assert_sparse_equal(
        A_ds, A_orig_lp, atol=args.atol, rtol=args.rtol, name="A_orig(dataset)"
    )

    # Dataloader check (PyG Batch)
    loader = PyGDataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        follow_batch=["constraint_features", "variable_features"],
    )
    batch = next(iter(loader))
    # batch is a Batch object; for batch_size=1, the first graph is easy:
    g2 = batch.to_data_list()[0]
    A_dl = sparse_from_edges(
        g2.edge_index, g2.edge_attr, size=(int(g2.constraint_features.size(0)), n)
    )
    assert_sparse_equal(
        A_dl, A_orig_lp, atol=args.atol, rtol=args.rtol, name="A_orig(dataloader)"
    )

    print("[ok] All checks passed.")


if __name__ == "__main__":
    main()
