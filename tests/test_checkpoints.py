from pathlib import Path

import torch

from dog_path_planning.policy.walking_policy import load_walking_policy_from_checkpoint
from scripts.export_to_onnx import load_actor


ROOT = Path(__file__).resolve().parents[1]


def test_navigation_checkpoint_shape():
    state = torch.load(
        ROOT / "checkpoints/navigation_sac/sac_actor.pth",
        map_location="cpu",
        weights_only=True,
    )

    assert state["trunk.0.weight"].shape == (512, 56)
    assert state["trunk.6.weight"].shape == (6, 256)

    actor = load_actor(
        ROOT / "checkpoints/navigation_sac/sac_actor.pth",
        ROOT / "configs/a1.yaml",
    )
    with torch.no_grad():
        action = actor(torch.zeros(2, 56))
    assert action.shape == (2, 3)
    assert torch.all(action >= -1) and torch.all(action <= 1)


def test_locomotion_checkpoint_loads_for_inference():
    policy = load_walking_policy_from_checkpoint(
        ROOT / "checkpoints/locomotion/model_4999.pt"
    )

    with torch.no_grad():
        action = policy.act_inference(torch.zeros(1, 45))

    assert action.shape == (1, 12)
    assert torch.isfinite(action).all()
