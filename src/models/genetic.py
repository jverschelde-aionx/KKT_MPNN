# decoders/gp_decode.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class EvoConfig:
    # Evolution settings
    pop_size: int = 128
    generations: int = 60
    elite_frac: float = 0.05
    tournament_k: int = 3
    crossover_prob: float = 0.9
    mutation_prob: Optional[float] = None  # if None -> use 1/n (clipped)

    # How we interpret x_pred
    logits: bool = True  # True if x_pred are logits (your current GNN head output)
    temperature: float = 1.0  # affects sampling from sigmoid(logits / temperature)
    threshold: float = 0.5  # seed rounding threshold

    # Fitness/penalties
    rho_multiplier: float = 10.0  # base penalty multiplier for L1 violation
    gamma_lambda: float = 1.0  # weight for lambda-weighted violation term
    maximize: bool = False  # True if your problem is maximize; will flip objective sign

    # Output choice
    prefer_feasible: bool = (
        False  # if True: return best feasible if found; else best penalized
    )

    # Repro
    seed: Optional[int] = None


def _evaluate_population(
    X: torch.Tensor,  # [P, n] float {0,1}
    A: torch.Tensor,  # [m, n]
    b: torch.Tensor,  # [m]
    c: torch.Tensor,  # [n]
    lam: Optional[torch.Tensor],  # [m] or None
    *,
    rho: float,
    gamma_lambda: float,
    maximize: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      fitness:   [P]  (lower is better)
      obj_min:   [P]  objective in minimization convention
      viol_sum:  [P]  L1 sum of violations
      viol_max:  [P]  max violation
    """
    # Objective
    obj = X @ c  # [P]
    obj_min = -obj if maximize else obj  # convert to minimization convention

    # Violations
    Ax = X @ A.t()  # [P, m]
    viol = torch.relu(Ax - b)  # [P, m]
    viol_sum = viol.sum(dim=1)  # [P]
    viol_max = viol.max(dim=1).values  # [P]

    # Penalties
    penalty = rho * viol_sum
    if lam is not None and gamma_lambda != 0.0:
        lam_pos = torch.relu(lam)  # be safe; your model already makes it >=0
        penalty = penalty + gamma_lambda * (viol * lam_pos).sum(dim=1)

    fitness = obj_min + penalty
    return fitness, obj_min, viol_sum, viol_max


def _tournament_select(
    fitness: torch.Tensor, num: int, k: int, g: torch.Generator
) -> torch.Tensor:
    """Tournament selection indices (minimize fitness)."""
    P = fitness.numel()
    idx = torch.randint(0, P, (num, k), generator=g, device=fitness.device)
    f = fitness[idx]  # [num, k]
    winners = idx[torch.arange(num, device=fitness.device), torch.argmin(f, dim=1)]
    return winners


def _uniform_crossover(
    p1: torch.Tensor, p2: torch.Tensor, p_cross: float, g: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform crossover for bit-vectors represented as float {0,1}.
    p1, p2: [num_pairs, n]
    """
    num_pairs, n = p1.shape
    device = p1.device

    do_cross = torch.rand((num_pairs, 1), generator=g, device=device) < p_cross
    mask = torch.rand((num_pairs, n), generator=g, device=device) < 0.5

    c1 = p1.clone()
    c2 = p2.clone()

    if do_cross.any():
        m = mask & do_cross  # broadcast do_cross to [num_pairs, n]
        c1[m] = p2[m]
        c2[m] = p1[m]
    return c1, c2


def _mutate_bits(X: torch.Tensor, p_mut: float, g: torch.Generator) -> torch.Tensor:
    """Bit-flip mutation on float {0,1} matrix."""
    if p_mut <= 0.0:
        return X
    flip = torch.rand(X.shape, generator=g, device=X.device) < p_mut
    X = X.clone()
    X[flip] = 1.0 - X[flip]
    return X


@torch.no_grad()
def evolve_binary_solution_instance(
    x_pred: torch.Tensor,  # [n] logits or probs
    lambda_pred: Optional[torch.Tensor],  # [m]
    A: torch.Tensor,  # [m, n]
    b: torch.Tensor,  # [m]
    c: torch.Tensor,  # [n]
    *,
    cfg: EvoConfig,
    tol: float = 1e-9,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns:
      x_best: [n] float {0,1}
      info: summary dict (fitness, violations, objective)
    """
    device = x_pred.device
    n = x_pred.numel()
    m = b.numel()
    lam = lambda_pred if lambda_pred is not None else None

    # RNG
    g = torch.Generator(device=device)
    if cfg.seed is not None:
        g.manual_seed(int(cfg.seed))
    else:
        # deterministic-ish but different across calls
        g.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), device=device).item()))

    # Convert x_pred -> sampling probabilities
    if cfg.logits:
        p = torch.sigmoid(x_pred / max(cfg.temperature, 1e-6))
    else:
        p = x_pred.clamp(0.0, 1.0)

    P = int(cfg.pop_size)

    # --- Initialize population (seeded by model output) ---
    pop = torch.bernoulli(p.expand(P, n), generator=g)  # [P, n]

    # Seed 0: simple rounding
    pop[0] = (p >= cfg.threshold).float()

    # Seed 1: Lagrangian-style rounding using lambda_pred if available
    # For minimization: choose x_j=1 if (c_j + (A^T lambda)_j) < 0
    if lam is not None and P > 1:
        lag_coef = c + (A.t() @ torch.relu(lam))  # [n]
        pop[1] = (lag_coef < 0).float()

    # Penalty scale rho (per instance)
    rho = float(cfg.rho_multiplier * max(float(c.abs().max().item()), 1.0))

    # Evaluate initial population
    fitness, obj_min, viol_sum, viol_max = _evaluate_population(
        pop, A, b, c, lam, rho=rho, gamma_lambda=cfg.gamma_lambda, maximize=cfg.maximize
    )

    # Track best overall and best feasible
    best_idx = int(torch.argmin(fitness).item())
    best = pop[best_idx].clone()
    best_fit = float(fitness[best_idx].item())
    best_obj = float(obj_min[best_idx].item())
    best_vs = float(viol_sum[best_idx].item())
    best_vm = float(viol_max[best_idx].item())

    feasible_mask = viol_max <= tol
    best_feas = None
    best_feas_fit = float("inf")
    best_feas_obj = float("inf")
    if feasible_mask.any():
        feas_idx = int(
            torch.argmin(fitness.masked_fill(~feasible_mask, float("inf"))).item()
        )
        best_feas = pop[feas_idx].clone()
        best_feas_fit = float(fitness[feas_idx].item())
        best_feas_obj = float(obj_min[feas_idx].item())

    elite = max(1, int(round(cfg.elite_frac * P)))
    p_mut = (
        cfg.mutation_prob
        if cfg.mutation_prob is not None
        else min(0.25, 1.0 / max(n, 1))
    )

    # --- Evolution loop ---
    for _gen in range(int(cfg.generations)):
        order = torch.argsort(fitness)
        elites = pop[order[:elite]].clone()

        num_off = P - elite
        if num_off <= 0:
            pop = elites
            break

        # Make offspring count even
        if num_off % 2 == 1:
            num_off -= 1
        if num_off <= 0:
            pop = elites
            break

        # Select parents via tournament
        num_pairs = num_off // 2
        idx1 = _tournament_select(fitness, num_pairs, cfg.tournament_k, g)
        idx2 = _tournament_select(fitness, num_pairs, cfg.tournament_k, g)

        p1 = pop[idx1]
        p2 = pop[idx2]

        # Crossover + mutation
        c1, c2 = _uniform_crossover(p1, p2, cfg.crossover_prob, g)
        children = torch.cat([c1, c2], dim=0)
        children = _mutate_bits(children, p_mut, g)

        pop = torch.cat([elites, children], dim=0)

        # If we trimmed, pad to size
        if pop.shape[0] < P:
            extra = torch.bernoulli(p.view(1, n), generator=g)
            pop = torch.cat([pop, extra], dim=0)

        # Evaluate new pop
        fitness, obj_min, viol_sum, viol_max = _evaluate_population(
            pop,
            A,
            b,
            c,
            lam,
            rho=rho,
            gamma_lambda=cfg.gamma_lambda,
            maximize=cfg.maximize,
        )

        # Update bests
        idx = int(torch.argmin(fitness).item())
        f = float(fitness[idx].item())
        if f < best_fit:
            best_fit = f
            best = pop[idx].clone()
            best_obj = float(obj_min[idx].item())
            best_vs = float(viol_sum[idx].item())
            best_vm = float(viol_max[idx].item())

        feas_mask = viol_max <= tol
        if feas_mask.any():
            feas_idx = int(
                torch.argmin(fitness.masked_fill(~feas_mask, float("inf"))).item()
            )
            feas_fit = float(fitness[feas_idx].item())
            if feas_fit < best_feas_fit:
                best_feas_fit = feas_fit
                best_feas = pop[feas_idx].clone()
                best_feas_obj = float(obj_min[feas_idx].item())

    # Decide what to return
    if cfg.prefer_feasible and best_feas is not None:
        x_out = best_feas
        info = {
            "returned": "best_feasible",
            "fitness": best_feas_fit,
            "obj_min": best_feas_obj,
            "viol_sum": 0.0,
            "viol_max": 0.0,
            "best_any_fitness": best_fit,
            "best_any_obj_min": best_obj,
            "best_any_viol_sum": best_vs,
            "best_any_viol_max": best_vm,
        }
    else:
        x_out = best
        info = {
            "returned": "best_any",
            "fitness": best_fit,
            "obj_min": best_obj,
            "viol_sum": best_vs,
            "viol_max": best_vm,
        }
        if best_feas is not None:
            info.update(
                {
                    "best_feasible_fitness": best_feas_fit,
                    "best_feasible_obj_min": best_feas_obj,
                }
            )

    return x_out, info


@torch.no_grad()
def evolve_binary_solution_batch(
    x_pred: torch.Tensor,  # [B, n_max] logits or probs
    lambda_pred: torch.Tensor,  # [B, m_max]
    A: torch.Tensor,  # [B, m_max, n_max]
    b: torch.Tensor,  # [B, m_max]
    c: torch.Tensor,  # [B, n_max]
    mask_m: torch.Tensor,  # [B, m_max] bool
    mask_n: torch.Tensor,  # [B, n_max] bool
    *,
    cfg: EvoConfig,
    tol: float = 1e-9,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """
    Runs evolutionary decoding per instance in the batch (simple + robust).
    Returns:
      x_bin: [B, n_max] float {0,1}
      infos: list of info dicts length B
    """
    B, n_max = x_pred.shape
    x_out = torch.zeros((B, n_max), device=x_pred.device, dtype=torch.float32)
    infos: List[Dict[str, float]] = []

    base_seed = cfg.seed
    for i in range(B):
        n_i = int(mask_n[i].sum().item())
        m_i = int(mask_m[i].sum().item())

        inst_cfg = cfg
        if base_seed is not None:
            inst_cfg = EvoConfig(**{**cfg.__dict__, "seed": int(base_seed + i)})

        xi, info = evolve_binary_solution_instance(
            x_pred=x_pred[i, :n_i],
            lambda_pred=lambda_pred[i, :m_i] if lambda_pred is not None else None,
            A=A[i, :m_i, :n_i],
            b=b[i, :m_i],
            c=c[i, :n_i],
            cfg=inst_cfg,
            tol=tol,
        )
        x_out[i, :n_i] = xi
        infos.append(info)

    return x_out, infos


@dataclass
class BlockMasterGAConfig:
    # Decomposition
    n_blocks: int = 5
    split_strategy: str = "contiguous"  # "contiguous" or "random"

    # Candidate pool per block
    candidates_per_block: int = 12
    thresholds: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)  # threshold candidates
    n_samples: int = 4  # random Bernoulli samples per block
    sample_temperature: float = 1.0
    include_all_zero: bool = True
    include_lagrangian: bool = True  # uses lambda_pred

    # GA
    pop_size: int = 128
    generations: int = 60
    elite_frac: float = 0.1
    tournament_k: int = 3
    crossover_prob: float = 0.9
    mutation_prob: Optional[float] = None  # if None -> 1/n_blocks

    # Fitness penalties (minimization convention)
    rho_multiplier: float = 10.0  # L1 violation penalty scale
    gamma_lambda: float = 1.0  # lambda-weighted violation term
    maximize: bool = False  # if true, flips objective sign (so lower fitness better)
    prefer_feasible: bool = False  # if true, return best feasible if any found

    # Interpret x_pred
    logits: bool = True  # your x_pred from model is logits by default

    # Repro
    seed: Optional[int] = None


def _split_indices(n: int, n_blocks: int, strategy: str, device) -> List[torch.Tensor]:
    assert n_blocks > 0
    if strategy == "random":
        perm = torch.randperm(n, device=device)
    elif strategy == "contiguous":
        perm = torch.arange(n, device=device)
    else:
        raise ValueError(f"Unknown split_strategy: {strategy}")

    # Split into near-equal chunks
    blocks = []
    base = n // n_blocks
    rem = n % n_blocks
    start = 0
    for k in range(n_blocks):
        size = base + (1 if k < rem else 0)
        blocks.append(perm[start : start + size])
        start += size
    return blocks


def _probs_from_xpred(
    x_pred_1d: torch.Tensor, logits: bool, temperature: float
) -> torch.Tensor:
    if logits:
        return torch.sigmoid(x_pred_1d / max(temperature, 1e-6))
    return x_pred_1d.clamp(0.0, 1.0)


def _make_block_candidates(
    x_block: torch.Tensor,  # [n_k] logits or probs
    A_block_T: torch.Tensor,  # [n_k, m] (transpose of A[:, block])
    c_block: torch.Tensor,  # [n_k]
    lambda_pred: Optional[torch.Tensor],  # [m]
    cfg: BlockMasterGAConfig,
    g: torch.Generator,
) -> torch.Tensor:
    """
    Returns candidates: [M, n_k] float {0,1}, M == cfg.candidates_per_block
    """
    device = x_block.device
    n_k = x_block.numel()
    cand: List[torch.Tensor] = []

    p = _probs_from_xpred(x_block, cfg.logits, cfg.sample_temperature)

    # Always useful fallback
    if cfg.include_all_zero:
        cand.append(torch.zeros(n_k, device=device))

    # Threshold candidates
    for t in cfg.thresholds:
        cand.append((p >= t).float())

    # Lagrangian-style candidate using lambda_pred (global coupling info)
    if cfg.include_lagrangian and (lambda_pred is not None):
        lam = torch.relu(lambda_pred)  # safety
        # reduced/Lagrangian coefficient for minimization: c + A^T lam
        # Here A_block_T is [n_k, m], so (A^T lam)[block] = A_block_T @ lam
        lag = c_block + (A_block_T @ lam)  # [n_k]
        cand.append((lag < 0.0).float())

    # Random samples around model probabilities
    for _ in range(int(cfg.n_samples)):
        cand.append(torch.bernoulli(p, generator=g))

    # De-duplicate a bit (optional, cheap)
    # Keep only unique rows up to some cap
    uniq = []
    seen = set()
    for x in cand:
        key = x.to(torch.uint8).cpu().numpy().tobytes()
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    cand = uniq

    # Trim/pad to fixed size M
    M = int(cfg.candidates_per_block)
    if len(cand) >= M:
        return torch.stack(cand[:M], dim=0)

    # Pad with additional random samples if needed
    while len(cand) < M:
        cand.append(torch.bernoulli(p, generator=g))
    return torch.stack(cand, dim=0)


def _evaluate_population(
    pop_idx: torch.Tensor,  # [P, K] integer candidate indices
    lhs_contrib: torch.Tensor,  # [K, M, m]
    obj_contrib: torch.Tensor,  # [K, M]
    b: torch.Tensor,  # [m]
    lambda_pred: Optional[torch.Tensor],
    rho: float,
    gamma_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized evaluation.
    Returns:
      fitness [P], obj_min [P], viol_sum [P], viol_max [P]
      (lower fitness is better)
    """
    device = pop_idx.device
    P, K = pop_idx.shape
    K0, M, m = lhs_contrib.shape
    assert K0 == K

    k_idx = torch.arange(K, device=device).view(1, K).expand(P, K)  # [P,K]

    # Gather contributions for each genome
    lhs = lhs_contrib[k_idx, pop_idx]  # [P, K, m]
    lhs = lhs.sum(dim=1)  # [P, m]

    obj = obj_contrib[k_idx, pop_idx].sum(dim=1)  # [P]  (already min convention)
    viol = torch.relu(lhs - b)  # [P, m]
    viol_sum = viol.sum(dim=1)  # [P]
    viol_max = viol.max(dim=1).values  # [P]

    penalty = rho * viol_sum
    if lambda_pred is not None and gamma_lambda != 0.0:
        lam = torch.relu(lambda_pred)
        penalty = penalty + gamma_lambda * (viol * lam).sum(dim=1)

    fitness = obj + penalty
    return fitness, obj, viol_sum, viol_max


def _tournament_select(
    fitness: torch.Tensor, num: int, k: int, g: torch.Generator
) -> torch.Tensor:
    P = fitness.numel()
    idx = torch.randint(0, P, (num, k), generator=g, device=fitness.device)
    f = fitness[idx]  # [num,k]
    winners = idx[torch.arange(num, device=fitness.device), torch.argmin(f, dim=1)]
    return winners


def _crossover_indices(
    p1: torch.Tensor, p2: torch.Tensor, p_cross: float, g: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform crossover on integer genes.
    p1,p2: [num_pairs,K]
    """
    device = p1.device
    num_pairs, K = p1.shape
    do = torch.rand((num_pairs, 1), generator=g, device=device) < p_cross
    mask = torch.rand((num_pairs, K), generator=g, device=device) < 0.5
    mask = mask & do
    c1 = p1.clone()
    c2 = p2.clone()
    c1[mask] = p2[mask]
    c2[mask] = p1[mask]
    return c1, c2


def _mutate_indices(
    pop: torch.Tensor, M: int, p_mut: float, g: torch.Generator
) -> torch.Tensor:
    if p_mut <= 0.0:
        return pop
    device = pop.device
    flip = torch.rand(pop.shape, generator=g, device=device) < p_mut
    new_vals = torch.randint(0, M, pop.shape, generator=g, device=device)
    out = pop.clone()
    out[flip] = new_vals[flip]
    return out


@torch.no_grad()
def decode_master_via_block_ga_instance(
    x_pred: torch.Tensor,  # [n] logits/probs
    lambda_pred: torch.Tensor,  # [m]
    A: torch.Tensor,  # [m,n]
    b: torch.Tensor,  # [m]
    c: torch.Tensor,  # [n]
    *,
    cfg: BlockMasterGAConfig,
    tol: float = 1e-9,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns binary x [n] assembled from block candidates via master GA.
    """
    device = x_pred.device
    n = x_pred.numel()
    m = b.numel()

    # RNG
    g = torch.Generator(device=device)
    if cfg.seed is not None:
        g.manual_seed(int(cfg.seed))

    # Split variables into blocks
    blocks = _split_indices(n, int(cfg.n_blocks), cfg.split_strategy, device=device)
    K = len(blocks)

    # Prepare objective convention (minimize)
    # If original is maximize, flip sign so "lower is better"
    c_min = -c if cfg.maximize else c

    # Build candidate pools and precompute contributions
    M = int(cfg.candidates_per_block)
    lhs_contrib = torch.zeros((K, M, m), device=device, dtype=torch.float32)
    obj_contrib = torch.zeros((K, M), device=device, dtype=torch.float32)
    candidates_per_block: List[torch.Tensor] = []

    for k, idx in enumerate(blocks):
        # Block slices
        x_blk = x_pred[idx]  # [n_k]
        c_blk = c_min[idx]  # [n_k]
        A_blk = A[:, idx]  # [m, n_k]
        A_blk_T = A_blk.t().contiguous()  # [n_k, m]

        cand = _make_block_candidates(
            x_block=x_blk,
            A_block_T=A_blk_T,
            c_block=c_blk,
            lambda_pred=lambda_pred,
            cfg=cfg,
            g=g,
        )  # [M, n_k]
        candidates_per_block.append(cand)

        # Precompute contributions
        # lhs: [M, m] = cand @ A_blk.T
        lhs_k = cand @ A_blk.t()
        lhs_contrib[k] = lhs_k
        obj_contrib[k] = cand @ c_blk

    # penalty scale
    rho = float(cfg.rho_multiplier * max(float(c.abs().max().item()), 1.0))

    # Initialize GA population of candidate indices [P,K]
    P = int(cfg.pop_size)
    pop = torch.randint(0, M, (P, K), generator=g, device=device)

    # Seed: naive rounding candidate (we included thresholds; try to pick the threshold 0.5 if present)
    # If thresholds include 0.5 and include_all_zero maybe shifted, easiest: just set all genes to the first nonzero-threshold candidate.
    # We'll seed pop[0] to "candidate index 0" for all blocks, which is all-zero if include_all_zero=True.
    pop[0] = 0

    # Evaluate
    fitness, obj_min, viol_sum, viol_max = _evaluate_population(
        pop,
        lhs_contrib,
        obj_contrib,
        b,
        lambda_pred,
        rho=rho,
        gamma_lambda=cfg.gamma_lambda,
    )

    # Track best and best feasible
    best_idx = int(torch.argmin(fitness).item())
    best_gene = pop[best_idx].clone()
    best_fit = float(fitness[best_idx].item())
    best_obj = float(obj_min[best_idx].item())
    best_vs = float(viol_sum[best_idx].item())
    best_vm = float(viol_max[best_idx].item())

    feas_mask = viol_max <= tol
    best_feas_gene = None
    best_feas_fit = float("inf")
    best_feas_obj = float("inf")
    if feas_mask.any():
        feas_idx = int(
            torch.argmin(fitness.masked_fill(~feas_mask, float("inf"))).item()
        )
        best_feas_gene = pop[feas_idx].clone()
        best_feas_fit = float(fitness[feas_idx].item())
        best_feas_obj = float(obj_min[feas_idx].item())

    elite = max(1, int(round(cfg.elite_frac * P)))
    p_mut = (
        cfg.mutation_prob
        if cfg.mutation_prob is not None
        else min(0.5, 1.0 / max(K, 1))
    )

    # GA loop
    for _ in range(int(cfg.generations)):
        order = torch.argsort(fitness)
        elites = pop[order[:elite]].clone()

        # offspring count
        num_off = P - elite
        if num_off <= 0:
            pop = elites
            break
        if num_off % 2 == 1:
            num_off -= 1
        if num_off <= 0:
            pop = elites
            break
        num_pairs = num_off // 2

        # parent selection
        p1_idx = _tournament_select(fitness, num_pairs, cfg.tournament_k, g)
        p2_idx = _tournament_select(fitness, num_pairs, cfg.tournament_k, g)
        p1 = pop[p1_idx]
        p2 = pop[p2_idx]

        # crossover + mutation
        c1, c2 = _crossover_indices(p1, p2, cfg.crossover_prob, g)
        children = torch.cat([c1, c2], dim=0)
        children = _mutate_indices(children, M, p_mut, g)

        pop = torch.cat([elites, children], dim=0)
        if pop.shape[0] < P:
            extra = torch.randint(
                0, M, (P - pop.shape[0], K), generator=g, device=device
            )
            pop = torch.cat([pop, extra], dim=0)

        fitness, obj_min, viol_sum, viol_max = _evaluate_population(
            pop,
            lhs_contrib,
            obj_contrib,
            b,
            lambda_pred,
            rho=rho,
            gamma_lambda=cfg.gamma_lambda,
        )

        idx = int(torch.argmin(fitness).item())
        f = float(fitness[idx].item())
        if f < best_fit:
            best_fit = f
            best_gene = pop[idx].clone()
            best_obj = float(obj_min[idx].item())
            best_vs = float(viol_sum[idx].item())
            best_vm = float(viol_max[idx].item())

        feas_mask = viol_max <= tol
        if feas_mask.any():
            feas_idx = int(
                torch.argmin(fitness.masked_fill(~feas_mask, float("inf"))).item()
            )
            feas_fit = float(fitness[feas_idx].item())
            if feas_fit < best_feas_fit:
                best_feas_fit = feas_fit
                best_feas_gene = pop[feas_idx].clone()
                best_feas_obj = float(obj_min[feas_idx].item())

    # Choose which gene to output
    if cfg.prefer_feasible and best_feas_gene is not None:
        gene = best_feas_gene
        returned = "best_feasible"
        out_fit, out_obj = best_feas_fit, best_feas_obj
        out_vs, out_vm = 0.0, 0.0
    else:
        gene = best_gene
        returned = "best_any"
        out_fit, out_obj = best_fit, best_obj
        out_vs, out_vm = best_vs, best_vm

    # Assemble x from block candidates
    x = torch.zeros(n, device=device, dtype=torch.float32)
    for k, idx in enumerate(blocks):
        x[idx] = candidates_per_block[k][int(gene[k].item())]

    info = {
        "returned": returned,
        "fitness": float(out_fit),
        "obj_min": float(out_obj),
        "viol_sum": float(out_vs),
        "viol_max": float(out_vm),
        "best_any_fit": float(best_fit),
        "best_any_obj_min": float(best_obj),
        "best_any_viol_sum": float(best_vs),
        "best_any_viol_max": float(best_vm),
    }
    if best_feas_gene is not None:
        info.update(
            {
                "best_feasible_fit": float(best_feas_fit),
                "best_feasible_obj_min": float(best_feas_obj),
            }
        )
    return x, info


@torch.no_grad()
def decode_master_via_block_ga_batch(
    x_pred: torch.Tensor,  # [B, n_max]
    lambda_pred: torch.Tensor,  # [B, m_max]
    A: torch.Tensor,  # [B, m_max, n_max]
    b: torch.Tensor,  # [B, m_max]
    c: torch.Tensor,  # [B, n_max]
    mask_m: torch.Tensor,  # [B, m_max] bool
    mask_n: torch.Tensor,  # [B, n_max] bool
    *,
    cfg: BlockMasterGAConfig,
    tol: float = 1e-9,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """
    Runs the block-master GA per instance in the batch.
    Returns:
      x_bin: [B, n_max] binary float
      infos: list of dicts
    """
    device = x_pred.device
    B, n_max = x_pred.shape
    x_out = torch.zeros((B, n_max), device=device, dtype=torch.float32)
    infos: List[Dict[str, float]] = []

    base_seed = cfg.seed
    for i in range(B):
        n_i = int(mask_n[i].sum().item())
        m_i = int(mask_m[i].sum().item())

        inst_cfg = cfg
        if base_seed is not None:
            inst_cfg = BlockMasterGAConfig(
                **{**cfg.__dict__, "seed": int(base_seed + i)}
            )

        xi, info = decode_master_via_block_ga_instance(
            x_pred=x_pred[i, :n_i],
            lambda_pred=lambda_pred[i, :m_i],
            A=A[i, :m_i, :n_i],
            b=b[i, :m_i],
            c=c[i, :n_i],
            cfg=inst_cfg,
            tol=tol,
        )
        x_out[i, :n_i] = xi
        infos.append(info)

    return x_out, infos
