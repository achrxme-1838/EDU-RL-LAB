"""Resolve saved policies without depending on Isaac Lab task utilities."""

from pathlib import Path


def resolve_checkpoint(log_root: Path, load_run: str = "*", checkpoint: str | None = None) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path
    for run in sorted(log_root.glob(load_run), reverse=True):
        if not run.is_dir():
            continue
        candidates = [p for p in run.glob("model_*.pt") if p.stem.removeprefix("model_").isdigit()]
        if candidates:
            return max(candidates, key=lambda p: int(p.stem.removeprefix("model_"))).resolve()
    raise FileNotFoundError(f"No model_*.pt checkpoint under {log_root / load_run}. Train a policy first.")
