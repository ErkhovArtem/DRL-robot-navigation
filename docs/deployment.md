# Deployment contract

The bundled SAC actor is the high-level part of a hierarchical controller. It maps
LiDAR, target geometry, angular velocity, and recent commands to normalized body
velocity commands. A separate locomotion policy maps those commands to 12 joint
actions.

## Actor input

The published actor expects a tensor shaped `[batch, 56]`:

| Slice | Size | Value |
| --- | ---: | --- |
| `0:40` | 40 | LiDAR sector ranges normalized to `[-1, 1]` |
| `40` | 1 | normalized yaw rate |
| `41:43` | 2 | sine and cosine of the target bearing |
| `43` | 1 | target distance normalized to `[-1, 1]` |
| `44:47` | 3 | previous normalized `[vx, vy, yaw_rate]` command |
| `47:56` | 9 | three-command normalized action history, oldest first |

LiDAR values are clipped to `[0, max_lidar_range]` and transformed with
`2 * range / max_lidar_range - 1`. Invalid or missing returns represent maximum
range. The remaining normalization constants and command scales are authoritative in
`configs/a1.yaml`.

## Actor output

The output tensor is `[batch, 3]` in `[-1, 1]`. Multiply it elementwise by
`cmd_scale` from `configs/a1.yaml` before sending the velocity command to the
locomotion controller.

## ONNX export

Install the optional dependencies and export the bundled actor:

```bash
python -m pip install -e ".[onnx]"
python scripts/export_to_onnx.py --verify
```

The default output is `outputs/sac_actor.onnx`. The exporter verifies ONNX Runtime
against PyTorch when `--verify` is supplied.

## Real robot

ROS 1 nodes, Docker configuration, visualization, and the Unitree message definitions
live in the separate
[`unitree-a1-drl-navigation`](https://github.com/ErkhovArtem/unitree-a1-drl-navigation)
repository. Keeping deployment separate prevents hardware-specific dependencies and
large generated ONNX files from entering the training repository.

Test emergency-stop handling, command limits, coordinate frames, and LiDAR ordering
in a controlled environment before enabling motors. Simulation success does not by
itself establish real-world safety.
