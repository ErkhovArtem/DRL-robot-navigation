# Dog Path Planning

Path planning for quadruped robot using SAC (Soft Actor-Critic) reinforcement learning.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Set up conda environment

```bash
conda create --name unitree-rl python=3.8
conda activate unitree-rl
```

### 2. Install the package in development mode

This is **required** for the project to work correctly. The package must be installed so that imports work properly.

```bash

cd Dog_PathPlanning
pip install -e .
```

This will install the package in editable mode, so changes to the code will be immediately available without reinstalling.

### 3. Install optional dependencies

For MJX support (GPU acceleration):
```bash
pip install -e .[mjx]
```

For ROS2 support (for lidar_2d_processor):
```bash
pip install -e .[ros2]
```

### Troubleshooting

If you see import errors like `ModuleNotFoundError: No module named 'utils'`, make sure you've run `pip install -e .` first.

## Project Structure

```
Dog_PathPlanning/
├── src/                    # Source code (Python package)
│   ├── policy/            # RL algorithms (SAC)
│   └── utils/             # Utilities (reward, observation, etc.)
├── scripts/               # Executable scripts
│   ├── train.py          # Training script
│   └── export_to_onnx.py # ONNX export
├── configs/               # Configuration files
├── data/                  # Models, logs, buffers
├── assets/                # Static resources
└── docs/                  # Documentation
```

## Usage

### Training

```bash
python scripts/train.py configs/a1.yaml --train --headless --episodes 10000 # add --load_pretrained to continue from last checkpoint
```

### Inference trained policy

```bash
python scripts/train.py configs/a1.yaml
```

### Export to ONNX

```bash
python scripts/export_to_onnx.py --model_path data/models/sac_actor.pth --output_path sac_actor.onnx
```

## Development

After installing in editable mode (`pip install -e .`), you can import modules directly:

```python
from utils.reward import compute_reward
from policy.SAC.SAC import SAC
```
