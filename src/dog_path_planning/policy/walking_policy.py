"""Load the bundled low-level locomotion actor."""

from pathlib import Path

import torch
from torch import nn


class WalkingPolicy(nn.Module):
    """Deterministic MLP reconstructed from an rsl_rl actor checkpoint."""

    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        layers = []
        for index, (input_size, output_size) in enumerate(
            zip(layer_sizes, layer_sizes[1:])
        ):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(layer_sizes) - 2:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)

    def act_inference(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)


def load_walking_policy_from_checkpoint(
    checkpoint_path: str | Path,
    obs_dim: int = 45,
    action_dim: int = 12,
) -> WalkingPolicy:
    """Load actor weights while ignoring training-only critic and optimizer state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    linear_indices = sorted(
        int(key.split(".")[1])
        for key, value in state.items()
        if key.startswith("actor.") and key.endswith(".weight") and value.ndim == 2
    )
    if not linear_indices:
        raise ValueError("Checkpoint does not contain actor weights")

    layer_sizes = [state[f"actor.{linear_indices[0]}.weight"].shape[1]]
    layer_sizes.extend(
        state[f"actor.{index}.weight"].shape[0] for index in linear_indices
    )
    if layer_sizes[0] != obs_dim or layer_sizes[-1] != action_dim:
        raise ValueError(
            "Locomotion checkpoint shape mismatch: "
            f"expected {obs_dim}->{action_dim}, got {layer_sizes[0]}->{layer_sizes[-1]}"
        )

    policy = WalkingPolicy(layer_sizes)
    actor_state = {
        key: value for key, value in state.items() if key.startswith("actor.")
    }
    policy.load_state_dict(actor_state)
    policy.eval()
    print(f"Loaded locomotion actor from {checkpoint_path}")
    return policy
