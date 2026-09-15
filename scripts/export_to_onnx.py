#!/usr/bin/env python3
"""Export the deterministic SAC actor to ONNX."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from dog_path_planning.policy.sac.actor import DiagGaussianActor


ROOT = Path(__file__).resolve().parents[1]


class DeterministicActor(torch.nn.Module):
    """Expose the mean SAC action expected by deployment runtimes."""

    def __init__(self, actor: DiagGaussianActor):
        super().__init__()
        self.actor = actor

    def forward(self, observation):
        mean, _ = self.actor.trunk(observation).chunk(2, dim=-1)
        return torch.tanh(mean)


def load_actor(model_path: Path, config_path: Path) -> DeterministicActor:
    """Reconstruct an actor from checkpoint tensor shapes and load its weights."""
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    linear_indices = sorted(
        int(key.split(".")[1])
        for key, value in state.items()
        if key.startswith("trunk.") and key.endswith(".weight") and value.ndim == 2
    )
    if len(linear_indices) < 2:
        raise ValueError(f"Could not infer actor architecture from {model_path}")

    input_dim = state[f"trunk.{linear_indices[0]}.weight"].shape[1]
    output_dim = state[f"trunk.{linear_indices[-1]}.weight"].shape[0]
    if output_dim % 2:
        raise ValueError(f"Expected paired mean/std outputs, got {output_dim}")

    hidden_dims = [
        state[f"trunk.{index}.weight"].shape[0]
        for index in linear_indices[:-1]
    ]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bounds = tuple(config["sac"]["actor"].get("log_std_bounds", [-2, 2]))
    actor = DiagGaussianActor(
        obs_dim=input_dim,
        action_dim=output_dim // 2,
        hidden_dim=hidden_dims,
        hidden_depth=len(hidden_dims),
        log_std_bounds=bounds,
    )
    actor.load_state_dict(state)
    actor.eval()
    return DeterministicActor(actor).eval()


def export_actor(
    model_path: Path,
    config_path: Path,
    output_path: Path,
    opset: int = 17,
    verify: bool = False,
) -> Path:
    model = load_actor(model_path, config_path)
    input_dim = model.actor.trunk[0].in_features
    sample = torch.zeros(1, input_dim)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    if verify:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install the onnx extra to verify the export") from error
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        observation = np.zeros((1, input_dim), dtype=np.float32)
        actual = session.run(None, {session.get_inputs()[0].name: observation})[0]
        with torch.no_grad():
            expected = model(torch.from_numpy(observation)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "checkpoints/navigation_sac/sac_actor.pth",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=ROOT / "configs/a1.yaml",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "outputs/sac_actor.onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    output = export_actor(
        args.model_path,
        args.config_path,
        args.output_path,
        args.opset,
        args.verify,
    )
    print(f"Exported ONNX actor to {output}")


if __name__ == "__main__":
    main()
