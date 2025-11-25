from __future__ import annotations

from math import sqrt
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from models.base import SigRegWrapper

EPS = 1e-9


def _center(z: Tensor) -> Tensor:
    return z - z.mean(dim=0, keepdim=True)


def _cov(z: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Sample covariance of N x D matrix (centered inside).
    """
    zc = _center(z)
    n = max(zc.shape[0] - 1, 1)
    c = (zc.T @ zc) / float(n)
    d = c.shape[0]
    return c + eps * torch.eye(d, device=zc.device, dtype=zc.dtype)


def _corr_from_cov(cov: Tensor) -> Tensor:
    d = cov.shape[0]
    std = torch.sqrt(torch.clamp(torch.diag(cov), min=EPS))
    denom = (std[:, None] * std[None, :]).clamp_min(EPS)
    corr = cov / denom
    corr = torch.clamp(corr, -1.0, 1.0)
    return corr


def _sphericity(cov: Tensor) -> float:
    """
    Psi = (trace(C)^2) / (D * trace(C^2))
    In [1/D, 1], identity (up to scale) → 1.
    """
    d = float(cov.shape[0])
    tr = torch.trace(cov)
    tr2 = torch.trace(cov @ cov)
    psi = (tr * tr) / (d * tr2 + EPS)
    return float(psi)


def _kappa(cov: Tensor) -> float:
    ev = torch.linalg.eigvalsh(cov)
    return float((ev.max() / ev.min().clamp_min(EPS)).item())


def _mean_abs_corr(corr: Tensor) -> float:
    d = corr.shape[0]
    mask = ~torch.eye(d, dtype=torch.bool, device=corr.device)
    vals = torch.abs(corr[mask])
    return float(vals.mean().item()) if vals.numel() > 0 else 0.0


def _chi_radius_expectations(d: int) -> Tuple[float, float]:
    """
    For r = ||N(0, I_d)||_2 (Chi with k=d):
    E[r] = sqrt(2) * Γ((d+1)/2) / Γ(d/2)
    Var[r] = d - E[r]^2
    """
    # Use torch.lgamma for stability: Γ(a) = exp(lgamma(a))
    a = torch.tensor([(d + 1) * 0.5, d * 0.5], dtype=torch.float64)
    num = torch.exp(torch.lgamma(a[0]))
    den = torch.exp(torch.lgamma(a[1]))
    mean_r = float(sqrt(2.0) * (num / den))
    var_r = max(float(d) - mean_r * mean_r, 0.0)
    std_r = var_r**0.5
    return mean_r, std_r


@torch.no_grad()
def isotropy_metrics(
    embeddings: Tuple[Tensor],
    sigreg: SigRegWrapper,
    prefix: str = "iso/",
) -> Dict[str, float]:
    """
    Compute isotropy metrics on embeddings Z (N x D).
    Returns a flat dict with prefixed keys.
    """
    Z = torch.cat(embeddings, dim=0)
    if Z.ndim != 2:
        Z = Z.reshape(-1, Z.shape[-1])
    n, d = int(Z.shape[0]), int(Z.shape[1])
    out: Dict[str, float] = {
        f"{prefix}N": float(n),
        f"{prefix}D": float(d),
    }
    if n < 2 or d < 1:
        return out

    cov = _cov(Z)
    corr = _corr_from_cov(cov)
    sphi = _sphericity(cov)
    kappa = _kappa(cov)
    mac = _mean_abs_corr(corr)

    zc = _center(Z)
    r = torch.linalg.norm(zc, dim=1)
    mean_r = float(r.mean().item())
    std_r = float(r.std(unbiased=True).item())
    exp_mean_r, exp_std_r = _chi_radius_expectations(d)
    mean_ratio = mean_r / (exp_mean_r + EPS)
    std_ratio = std_r / (exp_std_r + EPS if exp_std_r > 0 else 1.0)

    out.update(
        {
            f"{prefix}sphericity": sphi,
            f"{prefix}kappa": kappa,
            f"{prefix}mean_abs_corr": mac,
            f"{prefix}radius_mean": mean_r,
            f"{prefix}radius_std": std_r,
            f"{prefix}radius_mean_ratio": float(mean_ratio),
            f"{prefix}radius_std_ratio": float(std_ratio),
            f"{prefix}sigreg": float(sigreg(Z).item()),
        }
    )

    return out


def isotropy_pass(
    metrics: Mapping[str, float],
    prefix: str,
    thresholds: Mapping[str, float],
) -> bool:
    """
    Decide pass/fail given metric dict and thresholds:
    - sphericity ≥ sphericity_min
    - kappa ≤ kappa_max
    - mean_abs_corr ≤ mean_abs_corr_max
    - sigreg ≤ sigreg_max         (if present)
    """
    s_ok = metrics.get(f"{prefix}sphericity", 0.0) >= thresholds.get(
        "sphericity_min", 0.95
    )
    k_ok = metrics.get(f"{prefix}kappa", 1e9) <= thresholds.get("kappa_max", 1.5)
    c_ok = metrics.get(f"{prefix}mean_abs_corr", 1.0) <= thresholds.get(
        "mean_abs_corr_max", 0.05
    )
    if f"{prefix}sigreg" in metrics:
        g_ok = metrics[f"{prefix}sigreg"] <= thresholds.get("sigreg_max", 0.15)
    else:
        g_ok = True
    return bool(s_ok and k_ok and c_ok and g_ok)
