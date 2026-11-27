"""Export the loaded feed-forward play policy without importing the simulator."""

import copy
import hashlib
import os
import tempfile
from pathlib import Path

import onnx
import torch


def export_policy_as_onnx(policy, actor_obs: torch.Tensor, checkpoint: Path, *, task: str) -> Path:
    """Save a batch-one actor, including its observation normalizer, next to the checkpoint.

    Inputs are the concatenated policy observations; outputs are raw actions.
    Environment observation/history construction and action scaling stay outside ONNX.
    """
    if getattr(policy, "is_recurrent", False):
        raise ValueError("ONNX export currently supports feed-forward policies only")
    checkpoint = Path(checkpoint).resolve()
    output_dir = checkpoint.parent / "exported" / checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "policy.onnx"
    actor = torch.nn.Sequential(
        copy.deepcopy(policy.actor_obs_normalizer), copy.deepcopy(policy.actor)
    ).cpu().float().eval()
    sample = actor_obs[:1].detach().to(device="cpu", dtype=torch.float32)
    if sample.ndim != 2 or sample.shape[0] != 1:
        raise ValueError("Actor observations must have shape [batch, features] with a nonempty batch")

    # Replace only after successful export/checking; a failed export keeps the last good file.
    with tempfile.TemporaryDirectory(prefix=".onnx-", dir=output_dir) as temp:
        staging = Path(temp) / "policy.onnx"
        torch.onnx.export(
            actor, sample, str(staging), input_names=["obs"], output_names=["actions"],
            opset_version=17, dynamo=False,
        )
        model = onnx.load(str(staging))
        onnx.helper.set_model_props(model, {
            "checkpoint_file": checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "task": task,
            "actor_obs_normalization": str(policy.actor_obs_normalization).lower(),
        })
        onnx.checker.check_model(model)
        onnx.save(model, str(staging))
        os.replace(staging, output)
    return output
