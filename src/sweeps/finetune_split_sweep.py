"""Sweep over all finetune_split config files in a directory.

Usage:
    python -m sweeps.finetune_split_sweep --config_dir configs/finetune_splits
    python -m sweeps.finetune_split_sweep --config_dir configs/finetune_splits --dry_run
    python -m sweeps.finetune_split_sweep --config_dir configs/finetune_splits --parallel 9
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from loguru import logger


def _get_run_dir(config_path: Path) -> Path | None:
    """Derive the specific run directory for a config.

    finetune_split saves to: {experiments_dir}/{wandb_project}/{config_stem}
    """
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        experiments_dir = data.get("experiments_dir", "./experiments")
        wandb_project = data.get("wandb_project", "split_bijepa_finetuning")
        run_name = config_path.stem
        return Path(experiments_dir) / wandb_project / run_name
    except Exception:
        return None


def _is_already_run(config_path: Path) -> bool:
    """Check if a config has already been run by looking for best.pt in its run directory."""
    run_dir = _get_run_dir(config_path)
    if run_dir is None or not run_dir.exists():
        return False
    return (run_dir / "best.pt").exists()


def collect_configs(config_dir: Path) -> list[Path]:
    """Recursively find all .yml config files, sorted for determinism."""
    configs = sorted(config_dir.rglob("*.yml"))
    return configs


def _run_config_subprocess(cfg: Path) -> tuple[Path, bool, str]:
    """Run a single config as a subprocess. Returns (config_path, success, message)."""
    cmd = [sys.executable, "-m", "jobs.finetune_split", "--config", str(cfg)]
    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return (cfg, True, "")
        else:
            return (cfg, False, f"exit code {result.returncode}")
    except Exception as e:
        return (cfg, False, str(e))


def _run_sequential(configs: list[Path]) -> list[tuple[int, Path, str]]:
    """Run configs sequentially in-process."""
    from jobs.finetune_split import finetune_split

    failed: list[tuple[int, Path, str]] = []
    for i, cfg in enumerate(configs, 1):
        logger.info(f"[{i}/{len(configs)}] Running finetune_split with config: {cfg}")
        try:
            finetune_split(config_path=str(cfg))
        except Exception as e:
            logger.error(f"[{i}/{len(configs)}] FAILED: {cfg} — {e}")
            failed.append((i, cfg, str(e)))
            continue
        logger.info(f"[{i}/{len(configs)}] Completed: {cfg}")
    return failed


def _run_parallel(configs: list[Path], max_workers: int) -> list[tuple[int, Path, str]]:
    """Run configs in parallel as subprocesses."""
    failed: list[tuple[int, Path, str]] = []
    logger.info(f"Launching {len(configs)} configs with max {max_workers} parallel workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_cfg = {
            executor.submit(_run_config_subprocess, cfg): (i, cfg)
            for i, cfg in enumerate(configs, 1)
        }
        for future in as_completed(future_to_cfg):
            i, cfg = future_to_cfg[future]
            config_path, success, err_msg = future.result()
            if success:
                logger.info(f"[{i}/{len(configs)}] Completed: {cfg}")
            else:
                logger.error(f"[{i}/{len(configs)}] FAILED: {cfg} — {err_msg}")
                failed.append((i, cfg, err_msg))

    return failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run finetune_split for each config in a directory"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs/finetune_splits",
        help="Root directory to search for .yml config files (recursive)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List config files that would be run without executing them",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Max number of configs to run in parallel (0 = sequential)",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        logger.error(f"Config directory does not exist: {config_dir}")
        sys.exit(1)

    all_configs = collect_configs(config_dir)
    if not all_configs:
        logger.error(f"No .yml files found in {config_dir}")
        sys.exit(1)

    skipped = [c for c in all_configs if _is_already_run(c)]
    configs = [c for c in all_configs if not _is_already_run(c)]

    logger.info(
        f"Found {len(all_configs)} config files in {config_dir}: "
        f"{len(configs)} to run, {len(skipped)} already completed"
    )
    for cfg in skipped:
        logger.info(f"  [skip] {cfg}")
    for i, cfg in enumerate(configs, 1):
        logger.info(f"  [{i}/{len(configs)}] {cfg}")

    if args.dry_run:
        logger.info("Dry run — exiting without running any experiments.")
        return

    if not configs:
        logger.info("All configs already completed — nothing to run.")
        return

    if args.parallel > 0:
        failed = _run_parallel(configs, max_workers=args.parallel)
    else:
        failed = _run_sequential(configs)

    logger.info(
        f"Sweep finished: {len(configs) - len(failed)}/{len(configs)} succeeded"
    )
    if failed:
        logger.warning("Failed configs:")
        for idx, cfg, err in failed:
            logger.warning(f"  [{idx}] {cfg}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
