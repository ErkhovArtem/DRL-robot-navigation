import torch
from torch import nn
import torch.nn.functional as F

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        
        # Build MLP for Q1
        layers1 = []
        in_dim = obs_dim + action_dim
        
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * hidden_depth
        else:
            hidden_dims = hidden_dim
            
        for h in hidden_dims:
            layers1.append(nn.Linear(in_dim, h))
            layers1.append(nn.ReLU())
            in_dim = h
        layers1.append(nn.Linear(in_dim, 1))
        self.Q1 = nn.Sequential(*layers1)
        
        # Build MLP for Q2
        layers2 = []
        in_dim = obs_dim + action_dim
        for h in hidden_dims:
            layers2.append(nn.Linear(in_dim, h))
            layers2.append(nn.ReLU())
            in_dim = h
        layers2.append(nn.Linear(in_dim, 1))
        self.Q2 = nn.Sequential(*layers2)
        
        self.apply(weight_init)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2

