"""Run one A1 configuration on a Colab GPU with persistent offline metrics."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path


def _rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # A killed VM can leave its final append incomplete.
            raise
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object in {path}:{index + 1}")
        result.append(row)
    return result


def read_history(path: str | Path, checkpoint_ceiling: int | None = None) -> list[dict]:
    """Merge train/eval rows; ceiling is the number of completed updates."""
    merged = {}
    for row in _rows(path):
        if "optimizer_step" not in row:
            continue
        step = int(row["optimizer_step"])
        if checkpoint_ceiling is not None and step >= checkpoint_ceiling:
            continue
        merged.setdefault(step, {}).update(row)
    return [merged[step] for step in sorted(merged)]


def guard_experiment(run_dir: str | Path, config: dict, git_sha: str) -> None:
    """Reject accidental reuse of a name for another config or source commit."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "experiment.json"
    signature = json.loads(json.dumps({"git_sha": git_sha, "config": config}, default=str))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != signature:
            raise ValueError("Run configuration or Git SHA changed; choose a new run_name.")
        return
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class JsonlLogger:
    """Mirror real offline W&B log calls after its init, preserving RMS hooks."""

    def __init__(self, path: str | Path, wandb_module=None):
        self.path = Path(path)
        self.wandb = wandb_module
        self.handle = self.original = self.wrapper = None

    def setup(self, context):
        self.close()
        self.wandb = self.wandb or importlib.import_module("wandb")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [row for row in _rows(self.path)
                if int(row.get("optimizer_step", -1)) < context.step]
        temporary = self.path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        temporary.replace(self.path)
        self.handle = self.path.open("a", encoding="utf-8")
        self.original = self.wandb.log

        def log(data, *args, **kwargs):
            scalar = {}
            for key, value in data.items():
                if hasattr(value, "item"):
                    value = value.item()
                if value is None or isinstance(value, (str, bool, int, float)):
                    scalar[key] = value
            self.handle.write(json.dumps(scalar) + "\n")
            if "val_loss" in scalar or (int(scalar.get("optimizer_step", -1)) + 1) % 25 == 0:
                self.handle.flush()
            return self.original(data, *args, **kwargs)

        self.wrapper = log
        self.wandb.log = log

    def __call__(self, _context):
        return {}

    def close(self):
        if self.original is not None and self.wandb.log is self.wrapper:
            self.wandb.log = self.original
        if self.handle is not None:
            self.handle.close()
        self.handle = self.original = self.wrapper = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ["WANDB_MODE"] = "offline"

    import torch
    from data import TokenDatasetConfig
    from metric_logging import AFTER_TRAIN_STEP, MetricLogger
    from model_config import LMConfig
    from train import TrainConfig, checked_train_config, train, training_run_name

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available; select a GPU runtime in Colab.")
    if "metric_loggers" in payload:
        raise ValueError("Keep the default RMS loggers; metric_loggers is managed by this runner.")
    for key in ("wandb_tags", "keep_checkpoint_steps", "ema_decays", "lr_dip"):
        if payload.get(key) is not None:
            payload[key] = tuple(payload[key])
    if isinstance(payload.get("model_config"), dict):
        payload["model_config"] = LMConfig(**payload["model_config"])
    for key in ("train_dataset", "val_dataset"):
        if isinstance(payload.get(key), dict):
            payload[key] = TokenDatasetConfig(**payload[key])
    payload.update(wandb_online=True, wandb_entity="local", wandb_project="cs312-a1")
    config = checked_train_config(TrainConfig(**payload))
    run_dir = Path(config.model_dir) / training_run_name(config)
    git_sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    signature = asdict(replace(config, metric_loggers=()))
    signature.pop("metric_loggers")
    signature["data_root"] = os.environ.get("DL_ALCHEMY_DATA_DIR")
    signature["data_revision"] = os.environ.get("DL_ALCHEMY_DATA_REVISION", "main")
    guard_experiment(run_dir, signature, git_sha)
    wandb_dir = run_dir / "wandb"
    wandb_dir.mkdir(exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    session = {
        "started_at": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    with (run_dir / "sessions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(session) + "\n")
    logger = JsonlLogger(run_dir / "history.jsonl")
    config = replace(config, metric_loggers=config.metric_loggers + (
        MetricLogger(AFTER_TRAIN_STEP, logger),))
    try:
        train(config)
    finally:
        logger.close()
        import wandb
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
