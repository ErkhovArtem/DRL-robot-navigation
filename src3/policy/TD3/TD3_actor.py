import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, max_action):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        
        # Build MLP
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * hidden_depth
        else:
            hidden_dims = hidden_dim
            
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
            
        layers.append(nn.Linear(in_dim, action_dim))
        self.trunk = nn.Sequential(*layers)
        
        self.max_action = max_action
        self.apply(weight_init)

    def forward(self, x):
        return self.max_action * torch.tanh(self.trunk(x))

