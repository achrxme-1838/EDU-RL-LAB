"""ONNX export checks without Isaac Sim or a GPU."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator

from edu_rl_lab.utils.export_policy import export_policy_as_onnx
from rsl_rl.modules import ActorCritic


class PolicyExportTest(unittest.TestCase):
    def test_inference_matches_with_and_without_observation_normalization(self):
        torch.manual_seed(42)
        obs = {"policy": torch.randn(8, 225) * 2 + 3, "critic": torch.randn(8, 247)}
        for normalize in (False, True):
            with self.subTest(normalize=normalize), tempfile.TemporaryDirectory() as temp:
                policy = ActorCritic(
                    obs, {"policy": ["policy"], "critic": ["critic"]}, 12,
                    actor_obs_normalization=normalize,
                    actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
                )
                policy.update_normalization(obs)
                checkpoint = Path(temp) / "model_10001.pt"
                torch.save({"model_state_dict": policy.state_dict()}, checkpoint)
                output = export_policy_as_onnx(policy, policy.get_actor_obs(obs), checkpoint, task="test")
                self.assertEqual(output, Path(temp) / "exported/model_10001/policy.onnx")
                self.assertTrue(policy.training)
                self.assertTrue(policy.actor.training)
                policy.eval()
                model = onnx.load(str(output))
                metadata = {entry.key: entry.value for entry in model.metadata_props}
                self.assertEqual(metadata["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                self.assertEqual(metadata["actor_obs_normalization"], str(normalize).lower())
                self.assertEqual([d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim], [1, 225])
                self.assertEqual([d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim], [1, 12])
                evaluator = ReferenceEvaluator(model)
                for row in obs["policy"]:
                    expected = policy.act_inference({"policy": row[None]}).detach().numpy()
                    actual = evaluator.run(None, {"obs": row[None].numpy()})[0]
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

                original = output.read_bytes()
                with patch("torch.onnx.export", side_effect=RuntimeError("export failed")):
                    with self.assertRaisesRegex(RuntimeError, "export failed"):
                        export_policy_as_onnx(policy, obs["policy"], checkpoint, task="test")
                self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
