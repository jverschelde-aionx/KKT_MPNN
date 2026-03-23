#!/usr/bin/env python3
"""
Plot halo embedding recovery results.

Inputs
------
1) recovery CSV written by _write_csv(...)
2) coupling CSV written by _write_coupling_csv(...)

Outputs
-------
Saves a set of PNG figures and a few aggregated CSVs to --outdir.

Requirements
------------
pip install pandas matplotlib numpy
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------
# Utilities
# ----------------------------

NUMERIC_RECOVERY_COLS = [
    "part_id",
    "halo_depth",
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

NUMERIC_COUPLING_COLS = [
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


def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = (
        values.notna()
        & weights.notna()
        & np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if not mask.any():
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def sem(series: pd.Series) -> float:
    s = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=1) / np.sqrt(len(s)))


def mean_no_nan(series: pd.Series) -> float:
    s = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(s) == 0:
        return np.nan
    return float(s.mean())


def savefig(fig: plt.Figure, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Aggregation
# ----------------------------


def build_instance_level_recovery(recovery: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate partition-level recovery rows to instance-level rows
    using weighted averages.
    """
    rows = []
    group_cols = ["instance", "halo_depth", "node_type"]

    for keys, g in recovery.groupby(group_cols, sort=True):
        instance, halo_depth, node_type = keys

        row = {
            "instance": instance,
            "halo_depth": int(halo_depth),
            "node_type": node_type,
            "n_owned_total": g["n_owned"].sum(),
            "n_boundary_total": g["n_boundary"].sum(),
            "n_interior_total": g["n_interior"].sum(),
            "cos_all": weighted_mean(g["mean_cosine_all"], g["n_owned"]),
            "mse_all": weighted_mean(g["mean_mse_all"], g["n_owned"]),
            "cos_boundary": weighted_mean(g["mean_cosine_boundary"], g["n_boundary"]),
            "mse_boundary": weighted_mean(g["mean_mse_boundary"], g["n_boundary"]),
            "cos_interior": weighted_mean(g["mean_cosine_interior"], g["n_interior"]),
            "mse_interior": weighted_mean(g["mean_mse_interior"], g["n_interior"]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_partition_base(recovery: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate the recovery CSV to one row per (instance, part_id, halo_depth)
    and compute total owned/halo nodes for expansion-weighting.
    """
    pivot = recovery.pivot_table(
        index=["instance", "part_id", "halo_depth", "expansion_ratio"],
        columns="node_type",
        values=["n_owned", "n_halo"],
        aggfunc="first",
    ).reset_index()

    # Flatten MultiIndex columns
    pivot.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in pivot.columns.to_flat_index()
    ]

    for col in [
        "n_owned_constraint",
        "n_owned_variable",
        "n_halo_constraint",
        "n_halo_variable",
    ]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["owned_total"] = pivot["n_owned_constraint"].fillna(0) + pivot[
        "n_owned_variable"
    ].fillna(0)
    pivot["halo_total"] = pivot["n_halo_constraint"].fillna(0) + pivot[
        "n_halo_variable"
    ].fillna(0)
    pivot["halo_fraction_total"] = pivot["halo_total"] / pivot["owned_total"].replace(
        0, np.nan
    )

    return pivot


def build_instance_level_expansion(recovery: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate partition-level expansion ratios to instance-level by halo depth.
    """
    part = build_partition_base(recovery)
    rows = []

    for (instance, halo_depth), g in part.groupby(
        ["instance", "halo_depth"], sort=True
    ):
        rows.append(
            {
                "instance": instance,
                "halo_depth": int(halo_depth),
                "expansion_ratio_mean": weighted_mean(
                    g["expansion_ratio"], g["owned_total"]
                ),
                "halo_fraction_mean": weighted_mean(
                    g["halo_fraction_total"], g["owned_total"]
                ),
                "owned_total": g["owned_total"].sum(),
            }
        )

    return pd.DataFrame(rows)


def build_instance_level_gains(
    inst_rec: pd.DataFrame,
    inst_exp: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each instance/node_type/halo_depth>0, compute gain relative to halo=0.
    Positive cos_gain is better. Positive mse_gain is better.
    """
    base = (
        inst_rec[inst_rec["halo_depth"] == 0]
        .rename(
            columns={
                "cos_all": "cos_all_h0",
                "mse_all": "mse_all_h0",
                "cos_boundary": "cos_boundary_h0",
                "mse_boundary": "mse_boundary_h0",
                "cos_interior": "cos_interior_h0",
                "mse_interior": "mse_interior_h0",
            }
        )
        .drop(columns=["halo_depth"])
    )

    gains = inst_rec[inst_rec["halo_depth"] > 0].merge(
        base,
        on=["instance", "node_type"],
        how="left",
        suffixes=("", "_drop"),
    )

    gains["cos_gain"] = gains["cos_all"] - gains["cos_all_h0"]
    gains["mse_gain"] = (
        gains["mse_all_h0"] - gains["mse_all"]
    )  # positive is improvement

    gains["cos_boundary_gain"] = gains["cos_boundary"] - gains["cos_boundary_h0"]
    gains["mse_boundary_gain"] = gains["mse_boundary_h0"] - gains["mse_boundary"]

    gains["cos_interior_gain"] = gains["cos_interior"] - gains["cos_interior_h0"]
    gains["mse_interior_gain"] = gains["mse_interior_h0"] - gains["mse_interior"]

    gains = gains.merge(inst_exp, on=["instance", "halo_depth"], how="left")
    return gains


# ----------------------------
# Plot helpers
# ----------------------------


def grouped_errorbar(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    *,
    marker: str = "o",
    capsize: int = 3,
) -> None:
    for group_value, g in df.groupby(group_col, sort=True):
        xs, ys, es = [], [], []
        for x, gx in sorted(g.groupby(x_col), key=lambda t: t[0]):
            y = mean_no_nan(gx[y_col])
            e = sem(gx[y_col])
            if np.isfinite(y):
                xs.append(x)
                ys.append(y)
                es.append(e)
        if xs:
            ax.errorbar(
                xs, ys, yerr=es, marker=marker, capsize=capsize, label=str(group_value)
            )


def scatter_by_depth(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    depth_col: str = "halo_depth",
    annotate: bool = False,
) -> None:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, (depth, g) in enumerate(sorted(df.groupby(depth_col), key=lambda t: t[0])):
        x = pd.to_numeric(g[x_col], errors="coerce")
        y = pd.to_numeric(g[y_col], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.scatter(
                x[mask],
                y[mask],
                marker=markers[i % len(markers)],
                label=f"halo={depth}",
            )
            if annotate:
                for _, row in g.loc[mask].iterrows():
                    ax.annotate(
                        str(row["instance"]),
                        (row[x_col], row[y_col]),
                        fontsize=7,
                        alpha=0.7,
                    )


# ----------------------------
# Plot functions
# ----------------------------


def plot_halo_vs_overall_recovery(inst_rec: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    grouped_errorbar(axes[0], inst_rec, "halo_depth", "cos_all", "node_type")
    axes[0].set_title("Halo depth vs overall cosine recovery")
    axes[0].set_xlabel("Halo depth")
    axes[0].set_ylabel("Weighted mean cosine similarity")
    axes[0].legend()

    grouped_errorbar(axes[1], inst_rec, "halo_depth", "mse_all", "node_type")
    axes[1].set_title("Halo depth vs overall MSE recovery")
    axes[1].set_xlabel("Halo depth")
    axes[1].set_ylabel("Weighted mean MSE")
    axes[1].legend()

    savefig(fig, outdir / "01_halo_vs_overall_recovery.png")


def plot_boundary_vs_interior(inst_rec: pd.DataFrame, outdir: Path) -> None:
    node_types = sorted(inst_rec["node_type"].dropna().unique().tolist())
    if not node_types:
        return

    fig, axes = plt.subplots(
        len(node_types), 2, figsize=(12, 4.5 * len(node_types)), sharex=True
    )
    if len(node_types) == 1:
        axes = np.array([axes])

    for r, node_type in enumerate(node_types):
        sub = inst_rec[inst_rec["node_type"] == node_type]

        # Cosine
        ax = axes[r, 0]
        for label, col in [("boundary", "cos_boundary"), ("interior", "cos_interior")]:
            xs, ys, es = [], [], []
            for depth, g in sorted(sub.groupby("halo_depth"), key=lambda t: t[0]):
                y = mean_no_nan(g[col])
                e = sem(g[col])
                if np.isfinite(y):
                    xs.append(depth)
                    ys.append(y)
                    es.append(e)
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label)
        ax.set_title(f"{node_type}: boundary vs interior cosine")
        ax.set_xlabel("Halo depth")
        ax.set_ylabel("Weighted mean cosine similarity")
        ax.legend()

        # MSE
        ax = axes[r, 1]
        for label, col in [("boundary", "mse_boundary"), ("interior", "mse_interior")]:
            xs, ys, es = [], [], []
            for depth, g in sorted(sub.groupby("halo_depth"), key=lambda t: t[0]):
                y = mean_no_nan(g[col])
                e = sem(g[col])
                if np.isfinite(y):
                    xs.append(depth)
                    ys.append(y)
                    es.append(e)
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label)
        ax.set_title(f"{node_type}: boundary vs interior MSE")
        ax.set_xlabel("Halo depth")
        ax.set_ylabel("Weighted mean MSE")
        ax.legend()

    savefig(fig, outdir / "02_boundary_vs_interior.png")


def plot_expansion_vs_gain(gains: pd.DataFrame, outdir: Path) -> None:
    node_types = sorted(gains["node_type"].dropna().unique().tolist())
    if not node_types:
        return

    fig, axes = plt.subplots(len(node_types), 2, figsize=(12, 4.5 * len(node_types)))
    if len(node_types) == 1:
        axes = np.array([axes])

    for r, node_type in enumerate(node_types):
        sub = gains[gains["node_type"] == node_type]

        ax = axes[r, 0]
        scatter_by_depth(ax, sub, "expansion_ratio_mean", "cos_gain")
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"{node_type}: expansion ratio vs cosine gain")
        ax.set_xlabel("Mean expansion ratio")
        ax.set_ylabel("Cosine gain vs halo=0")
        ax.legend()

        ax = axes[r, 1]
        scatter_by_depth(ax, sub, "expansion_ratio_mean", "mse_gain")
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"{node_type}: expansion ratio vs MSE gain")
        ax.set_xlabel("Mean expansion ratio")
        ax.set_ylabel("MSE reduction vs halo=0")
        ax.legend()

    savefig(fig, outdir / "03_expansion_vs_gain.png")


def plot_coupling_vs_halo0_recovery(
    inst_rec: pd.DataFrame, coupling: pd.DataFrame, outdir: Path
) -> None:
    halo0 = inst_rec[inst_rec["halo_depth"] == 0].merge(
        coupling, on="instance", how="left"
    )
    node_types = sorted(halo0["node_type"].dropna().unique().tolist())
    if not node_types:
        return

    x_vars = [
        ("edge_cut_fraction", "Edge cut fraction"),
        ("coupling_fraction", "Coupling fraction"),
        ("avg_blocks_per_constraint", "Avg blocks per constraint"),
    ]

    fig, axes = plt.subplots(
        len(node_types), len(x_vars), figsize=(5 * len(x_vars), 4.2 * len(node_types))
    )
    if len(node_types) == 1:
        axes = np.array([axes])

    for r, node_type in enumerate(node_types):
        sub = halo0[halo0["node_type"] == node_type]
        for c, (x_col, x_label) in enumerate(x_vars):
            ax = axes[r, c]
            x = pd.to_numeric(sub[x_col], errors="coerce")
            y = pd.to_numeric(sub["cos_all"], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], marker="o")
            ax.set_title(f"{node_type}: halo=0 cosine vs {x_label}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Cosine similarity at halo=0")

    savefig(fig, outdir / "04_coupling_vs_halo0_recovery.png")


def plot_coupling_vs_gain(
    gains: pd.DataFrame, coupling: pd.DataFrame, outdir: Path
) -> None:
    merged = gains.merge(coupling, on="instance", how="left")
    node_types = sorted(merged["node_type"].dropna().unique().tolist())
    if not node_types:
        return

    # Often most interesting to inspect largest halo depth as the "best recovered" setting.
    max_halo = int(merged["halo_depth"].max())
    submax = merged[merged["halo_depth"] == max_halo]

    x_vars = [
        ("edge_cut_fraction", "Edge cut fraction"),
        ("coupling_fraction", "Coupling fraction"),
        ("avg_blocks_per_constraint", "Avg blocks per constraint"),
    ]

    fig, axes = plt.subplots(
        len(node_types), len(x_vars), figsize=(5 * len(x_vars), 4.2 * len(node_types))
    )
    if len(node_types) == 1:
        axes = np.array([axes])

    for r, node_type in enumerate(node_types):
        sub = submax[submax["node_type"] == node_type]
        for c, (x_col, x_label) in enumerate(x_vars):
            ax = axes[r, c]
            x = pd.to_numeric(sub[x_col], errors="coerce")
            y = pd.to_numeric(sub["cos_gain"], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], marker="o")
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.set_title(f"{node_type}: cosine gain at halo={max_halo} vs {x_label}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Cosine gain vs halo=0")

    savefig(fig, outdir / "05_coupling_vs_gain.png")


def plot_partition_distributions(recovery: pd.DataFrame, outdir: Path) -> None:
    node_types = sorted(recovery["node_type"].dropna().unique().tolist())
    halo_depths = sorted(
        pd.to_numeric(recovery["halo_depth"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not node_types or not halo_depths:
        return

    # Cosine distribution
    fig, axes = plt.subplots(len(node_types), 2, figsize=(12, 4.5 * len(node_types)))
    if len(node_types) == 1:
        axes = np.array([axes])

    for r, node_type in enumerate(node_types):
        sub = recovery[recovery["node_type"] == node_type]

        # cosine
        data = []
        labels = []
        for depth in halo_depths:
            vals = (
                pd.to_numeric(
                    sub.loc[sub["halo_depth"] == depth, "mean_cosine_all"],
                    errors="coerce",
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(vals) > 0:
                data.append(vals.values)
                labels.append(str(depth))
        if data:
            axes[r, 0].boxplot(data, labels=labels)
        axes[r, 0].set_title(f"{node_type}: partition cosine distribution")
        axes[r, 0].set_xlabel("Halo depth")
        axes[r, 0].set_ylabel("Partition mean cosine")

        # MSE
        data = []
        labels = []
        for depth in halo_depths:
            vals = (
                pd.to_numeric(
                    sub.loc[sub["halo_depth"] == depth, "mean_mse_all"], errors="coerce"
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(vals) > 0:
                data.append(vals.values)
                labels.append(str(depth))
        if data:
            axes[r, 1].boxplot(data, labels=labels)
        axes[r, 1].set_title(f"{node_type}: partition MSE distribution")
        axes[r, 1].set_xlabel("Halo depth")
        axes[r, 1].set_ylabel("Partition mean MSE")

    savefig(fig, outdir / "06_partition_distributions.png")


def plot_boundary_fraction_vs_penalty(recovery: pd.DataFrame, outdir: Path) -> None:
    df = recovery.copy()
    df["boundary_fraction"] = df["n_boundary"] / df["n_owned"].replace(0, np.nan)
    df["boundary_penalty_cos"] = df["mean_cosine_interior"] - df["mean_cosine_boundary"]
    df["boundary_penalty_mse"] = (
        df["mean_mse_boundary"] - df["mean_mse_interior"]
    )  # >0 means boundary worse

    node_types = sorted(df["node_type"].dropna().unique().tolist())
    if not node_types:
        return

    fig, axes = plt.subplots(len(node_types), 2, figsize=(12, 4.5 * len(node_types)))
    if len(node_types) == 1:
        axes = np.array([axes])

    for r, node_type in enumerate(node_types):
        sub = df[df["node_type"] == node_type]

        ax = axes[r, 0]
        scatter_by_depth(ax, sub, "boundary_fraction", "boundary_penalty_cos")
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"{node_type}: boundary fraction vs cosine penalty")
        ax.set_xlabel("Boundary fraction")
        ax.set_ylabel("Interior cosine - boundary cosine")

        ax = axes[r, 1]
        scatter_by_depth(ax, sub, "boundary_fraction", "boundary_penalty_mse")
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"{node_type}: boundary fraction vs MSE penalty")
        ax.set_xlabel("Boundary fraction")
        ax.set_ylabel("Boundary MSE - interior MSE")

    savefig(fig, outdir / "07_boundary_fraction_vs_penalty.png")


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot halo embedding recovery results."
    )
    parser.add_argument(
        "--recovery_csv",
        type=Path,
        default="eval_halo_embedding.csv",
        help="Path to recovery CSV.",
    )
    parser.add_argument(
        "--coupling_csv",
        type=Path,
        default="eval_halo_embedding_coupling.csv",
        help="Path to coupling CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default="plots",
        help="Directory to write plots.",
    )
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    recovery = pd.read_csv(args.recovery_csv)
    coupling = pd.read_csv(args.coupling_csv)

    recovery = _to_numeric(recovery, NUMERIC_RECOVERY_COLS)
    coupling = _to_numeric(coupling, NUMERIC_COUPLING_COLS)

    # Build aggregated tables
    inst_rec = build_instance_level_recovery(recovery)
    inst_exp = build_instance_level_expansion(recovery)
    gains = build_instance_level_gains(inst_rec, inst_exp)

    # Save aggregated tables for inspection
    inst_rec.to_csv(outdir / "instance_level_recovery.csv", index=False)
    inst_exp.to_csv(outdir / "instance_level_expansion.csv", index=False)
    gains.to_csv(outdir / "instance_level_gains.csv", index=False)

    # Generate plots
    plot_halo_vs_overall_recovery(inst_rec, outdir)
    plot_boundary_vs_interior(inst_rec, outdir)
    plot_expansion_vs_gain(gains, outdir)
    plot_coupling_vs_halo0_recovery(inst_rec, coupling, outdir)
    plot_coupling_vs_gain(gains, coupling, outdir)
    plot_partition_distributions(recovery, outdir)
    plot_boundary_fraction_vs_penalty(recovery, outdir)

    print(f"Wrote plots and aggregated CSVs to: {outdir}")


if __name__ == "__main__":
    main()
