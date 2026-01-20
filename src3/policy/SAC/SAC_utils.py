import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import os
from collections import deque
import random
import math


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    """
    Create MLP network.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension(s). Can be int (same size for all layers) or list (different sizes)
        output_dim: Output dimension
        hidden_depth: Number of hidden layers (used only if hidden_dim is int)
        output_mod: Optional output modification (e.g., activation function)
    """
    if isinstance(hidden_dim, (list, tuple)):
        # Custom layer sizes specified - ignore hidden_depth or use it as validation
        hidden_dims = list(hidden_dim)
        # First layer: input -> first hidden
        mods = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(inplace=True)]
        # Middle layers: hidden -> hidden
        for i in range(len(hidden_dims) - 1):
            mods += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU(inplace=True)]
        # Last layer: last hidden -> output
        mods.append(nn.Linear(hidden_dims[-1], output_dim))
    elif hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        # Uniform layer sizes (original behavior)
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
