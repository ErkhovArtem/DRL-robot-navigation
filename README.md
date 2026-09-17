# Unitree A1 SAC Navigation

Hierarchical reinforcement-learning navigation for a Unitree A1 quadruped in MuJoCo.
A Soft Actor-Critic (SAC) policy turns a 2D LiDAR scan and relative goal into body
velocity commands; a pretrained locomotion policy turns those commands into joint
actions.

```text
LiDAR + relative goal + command history
                  │
                  ▼
          SAC navigation actor
                  │  [vx, vy, yaw rate]
                  ▼
        locomotion policy (PPO)
                  │  12 joint targets
                  ▼
              Unitree A1
```

## Demo

![Unitree A1 navigation demo](docs/media/demo.gif)

## Included checkpoint

The repository contains one navigation checkpoint in `checkpoints/navigation_sac` and
one locomotion checkpoint in `checkpoints/locomotion`. The navigation model was trained for 10,000 episodes on the baseline SAC
development line.

| Evaluation | Episodes | Success | Collision | Timeout |
| --- | ---: | ---: | ---: | ---: |
| Maximum curriculum difficulty, seed 42 | 100 | 74% | 24% | 2% |

The benchmark uses deterministic SAC actions and sequential headless MuJoCo. It is a
100-episode sample, not a real-robot safety guarantee.

## Installation

Python 3.10–3.12 is supported. Python 3.11 is used in CI.

```bash
git clone https://github.com/ErkhovArtem/DRL-robot-navigation.git
cd Dog_PathPlanning
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For batched MJX simulation, install the optional extra and then install the JAX build
that matches your CUDA runtime if GPU acceleration is required:

```bash
python -m pip install -e ".[mjx]"
```

## Evaluate the bundled policy

Run a quick, deterministic headless evaluation:

```bash
python scripts/train.py configs/a1.yaml --headless --episodes 10 --seed 42
```

Reproduce the table above with:

```bash
python scripts/train.py configs/a1.yaml --headless --episodes 100 --seed 42
```

Omit `--headless` to open the MuJoCo viewer. The script evaluates the bundled
checkpoint by default; use `--checkpoint-dir PATH` to select another checkpoint.
Generated scenes and evaluation outputs are ignored by Git.

## Train

Sequential MuJoCo:

```bash
python scripts/train.py configs/a1.yaml \
  --train --headless --episodes 10000 --seed 42
```

Batched MJX:

```bash
python scripts/train.py configs/a1.yaml \
  --train --headless --use_mjx --batch-size 32 --episodes 10000 --seed 42
```

Checkpoints, replay buffers, and TensorBoard events are written under `outputs/`.
Resume the latest generated run with `--load-pretrained`. To initialize a new run from
specific weights without modifying them, add
`--fine-tune --checkpoint-dir checkpoints/navigation_sac`.

## Export for deployment

```bash
python -m pip install -e ".[onnx]"
python scripts/export_to_onnx.py --verify
```

See [`docs/deployment.md`](docs/deployment.md) for the exact 56-value observation
layout and action scaling.

Inference code for running the exported policy on the real Unitree A1 is maintained
in the separate
[`unitree-a1-drl-navigation`](https://github.com/ErkhovArtem/unitree-a1-drl-navigation.git)
repository.

## Repository layout

```text
assets/                     MuJoCo A1 model and meshes
checkpoints/locomotion/     low-level locomotion weights
checkpoints/navigation_sac/ bundled SAC training state
configs/                    robot and curriculum configuration
scripts/                    evaluation, training, visualization, and ONNX export
src/dog_path_planning/      reusable policy and simulation modules
tests/                      checkpoint and scene-generation smoke tests
```

## Notes

- Training and evaluation regenerate obstacles from `assets/unitree_a1/scene.xml`
  without modifying that tracked template.
- The Unitree model assets retain their upstream license in
  `assets/unitree_a1/LICENSE`.
- Project source code is released under the MIT License; see [`LICENSE`](LICENSE).
