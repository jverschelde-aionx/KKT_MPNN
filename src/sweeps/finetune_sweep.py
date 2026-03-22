"""Sweep over all finetune config files in a directory.

Usage:
    python -m sweeps.finetune_sweep --config_dir configs/finetune/milp
    python -m sweeps.finetune_sweep --config_dir configs/finetune/milp/IS
    python -m sweeps.finetune_sweep --config_dir configs/finetune/milp --dry_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

from jobs.finetune import finetune


def _get_experiments_dir(config_path: Path) -> Path | None:
    """Extract experiments_dir from a config YAML."""
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        # experiments_dir can be top-level or nested under 'training'
        if isinstance(data, dict):
            if "experiments_dir" in data:
                return Path(data["experiments_dir"])
            training = data.get("training", {})
            if isinstance(training, dict) and "experiments_dir" in training:
                return Path(training["experiments_dir"])
    except Exception:
        pass
    return None


def _is_already_run(config_path: Path) -> bool:
    """Check if a config has already been run by looking for best.pt in its experiments_dir."""
    exp_dir = _get_experiments_dir(config_path)
    if exp_dir is None:
        return False
    return any(exp_dir.rglob("best.pt"))


def collect_configs(config_dir: Path) -> list[Path]:
    """Recursively find all .yml config files, sorted for determinism."""
    configs = sorted(config_dir.rglob("*.yml"))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run finetune for each config in a directory"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs/finetune/milp",
        help="Root directory to search for .yml config files (recursive)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List config files that would be run without executing them",
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

    failed: list[tuple[int, Path, str]] = []
    for i, cfg in enumerate(configs, 1):
        logger.info(f"[{i}/{len(configs)}] Running finetune with config: {cfg}")
        try:
            finetune(config_path=str(cfg))
        except Exception as e:
            logger.error(f"[{i}/{len(configs)}] FAILED: {cfg} — {e}")
            failed.append((i, cfg, str(e)))
            continue
        logger.info(f"[{i}/{len(configs)}] Completed: {cfg}")

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
