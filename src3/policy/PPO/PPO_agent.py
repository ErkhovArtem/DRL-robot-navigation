import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import deque
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOActor(nn.Module):
    """Actor network for PPO - outputs mean and log_std for action distribution"""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, max_action):
        super().__init__()
        self.max_action = max_action
        
        layers = []
        in_dim = obs_dim
        
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * hidden_depth
        else:
            hidden_dims = hidden_dim
        
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh())
            in_dim = h
        
        self.actor_mean = nn.Sequential(*layers)
        self.actor_mean.add_module("mean_output", layer_init(
            nn.Linear(in_dim, action_dim), std=0.01
        ))
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
    
    def forward(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        # Clamp logstd to prevent extreme values that could cause NaN
        action_logstd = torch.clamp(action_logstd, -10, 2)
        action_std = torch.exp(action_logstd)
        
        # Check for NaN/Inf in both mean and std, replace with safe values
        if torch.isnan(action_mean).any() or torch.isinf(action_mean).any():
            print("WARNING: NaN/Inf detected in action_mean, replacing with zeros")
            action_mean = torch.where(torch.isnan(action_mean) | torch.isinf(action_mean), 
                                     torch.zeros_like(action_mean), action_mean)
        
        if torch.isnan(action_std).any() or torch.isinf(action_std).any():
            print("WARNING: NaN/Inf detected in action_std, replacing with safe value (0.1)")
            action_std = torch.where(torch.isnan(action_std) | torch.isinf(action_std), 
                                    torch.ones_like(action_std) * 0.1, action_std)
        
        # Ensure std is positive (safety check)
        action_std = torch.clamp(action_std, min=1e-6)
        
        dist = Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        # Note: action is expected to be in unbounded space (from Normal distribution)
        # We clip it to [-max_action, max_action] range (matching CleanRL's ClipAction wrapper)
        
        # Compute log_prob in unbounded space
        # Use sum(1) instead of sum(-1, keepdim=True) to match CleanRL format
        log_prob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        
        # Clip action to [-max_action, max_action] range (like CleanRL's ClipAction wrapper)
        action_output = torch.clamp(action, -self.max_action, self.max_action)
        
        return action_output, log_prob, entropy


class PPOCritic(nn.Module):
    """Critic network for PPO - outputs value estimate"""
    def __init__(self, obs_dim, hidden_dim, hidden_depth):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * hidden_depth
        else:
            hidden_dims = hidden_dim
        
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh())
            in_dim = h
        
        layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.critic(x)


class PPO:
    """
    PPO (Proximal Policy Optimization) agent.
    Adapted from CleanRL's PPO implementation to work with the current training framework.
    
    Note: PPO is an on-policy algorithm, but we adapt it to work with the replay buffer
    interface by collecting experiences and updating periodically.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_action,
        discount=0.99,
        learning_rate=3e-4,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_vloss=True,
        norm_adv=True,
        update_epochs=10,
        num_minibatches=32,
        history_length=0,
        actor_hidden_dim=[256, 256],
        actor_hidden_depth=2,
        critic_hidden_dim=[256, 256],
        critic_hidden_depth=2,
        critic_state_dim=None,
        writer=None,
        model_name="PPO",
        save_directory=Path("models/PPO")
    ):
        self.device = torch.device(device)
        self.max_action = max_action
        self.gamma = discount
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.clip_vloss = clip_vloss
        self.norm_adv = norm_adv
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        
        self.base_actor_state_dim = state_dim
        self.base_critic_state_dim = critic_state_dim if critic_state_dim is not None else state_dim
        self.history_length = history_length
        
        self.actor_state_dim = self.base_actor_state_dim * (history_length + 1) if history_length > 0 else self.base_actor_state_dim
        self.critic_state_dim = self.base_critic_state_dim * (history_length + 1) if history_length > 0 else self.base_critic_state_dim
        
        # Create actor and critic networks
        self.actor = PPOActor(
            self.actor_state_dim, action_dim, actor_hidden_dim, actor_hidden_depth, max_action
        ).to(self.device)
        self.critic = PPOCritic(
            self.critic_state_dim, critic_hidden_dim, critic_hidden_depth
        ).to(self.device)
        
        # Single optimizer for both networks
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
            eps=1e-5
        )
        
        self.actor_obs_history = deque(maxlen=self.history_length + 1)
        self.critic_obs_history = deque(maxlen=self.history_length + 1)
        
        self.total_it = 0
        self.writer = writer if writer is not None else SummaryWriter()
        self.model_name = model_name
        self.save_directory = Path(save_directory)
    
    def process_observation(self, obs, is_critic=False, history_buffer=None):
        """Process observation with optional history stacking"""
        if self.history_length == 0:
            return obs.copy()
        
        if history_buffer is not None:
            history = history_buffer
        else:
            history = self.critic_obs_history if is_critic else self.actor_obs_history
        
        base_dim = self.base_critic_state_dim if is_critic else self.base_actor_state_dim
        history.append(obs.copy())
        
        while len(history) < self.history_length + 1:
            history.appendleft(obs.copy())
        
        stacked_obs = np.concatenate(list(history))
        return stacked_obs
    
    def reset_history(self):
        """Reset observation history"""
        self.actor_obs_history.clear()
        self.critic_obs_history.clear()
    
    def get_action(self, obs, add_noise=True):
        """
        Get action from policy.
        Note: PPO uses stochastic policy, so 'add_noise' is ignored - policy is always stochastic.
        """
        # Check for NaN/Inf in observations
        if np.isnan(obs).any() or np.isinf(obs).any():
            print(f"WARNING: Invalid observation detected (NaN/Inf), returning zero action")
            return np.zeros(3, dtype=np.float32)
        
        state = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor(state)
        
        # Check for NaN in action
        action_np = action.cpu().data.numpy().flatten()
        if np.isnan(action_np).any() or np.isinf(action_np).any():
            print(f"WARNING: Invalid action detected (NaN/Inf), returning zero action")
            return np.zeros(3, dtype=np.float32)
        
        return action_np
    
    def get_action_and_value(self, obs, action=None):
        """Get action, log_prob, entropy, and value from current policy"""
        state = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        if action is not None:
            action_tensor = torch.FloatTensor(action).to(self.device)
            if len(action_tensor.shape) == 1:
                action_tensor = action_tensor.unsqueeze(0)
        else:
            action_tensor = None
        
        action_out, log_prob, entropy = self.actor(state, action_tensor)
        value = self.critic(state)
        
        return action_out, log_prob, entropy, value
    
    def train(self, replay_buffer, iterations, batch_size, success_weight=2.0, collision_weight=0.5):
        """
        Train PPO on samples from replay buffer.
        
        Note: This is an approximation - true PPO requires on-policy rollouts.
        We sample from replay buffer and compute advantages using GAE.
        """
        if replay_buffer.size() < batch_size:
            return
        
        # Sample batch from replay buffer
        s_batch, a_batch, r_batch, t_batch, s2_batch, _ = replay_buffer.sample_batch(
            batch_size, success_weight=success_weight, collision_weight=collision_weight
        )
        
        # Convert to tensors
        states = torch.FloatTensor(s_batch).to(self.device)
        actions = torch.FloatTensor(a_batch).to(self.device)
        rewards = torch.FloatTensor(r_batch).to(self.device).flatten()  # Flatten to (batch_size,)
        dones = torch.FloatTensor(t_batch).to(self.device).flatten()    # Flatten to (batch_size,)
        next_states = torch.FloatTensor(s2_batch).to(self.device)
        
        # Compute values for current and next states
        with torch.no_grad():
            current_values = self.critic(states).flatten()
            next_values = self.critic(next_states).flatten()
            
            # Compute returns using simple TD(0) bootstrap (approximation for on-policy GAE)
            returns = rewards + (1 - dones) * self.gamma * next_values
            advantages = returns - current_values
            
            # Normalize advantages if requested
            if self.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old log probs (for computing ratio)
        with torch.no_grad():
            _, old_log_probs, _ = self.actor(states, actions)
            old_log_probs = old_log_probs.flatten()
        
        # Compute minibatch size
        minibatch_size = batch_size // self.num_minibatches
        
        # Training loop over epochs
        clipfracs = []
        for epoch in range(self.update_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]
                
                # Get new log probs and values
                _, new_log_probs, entropy = self.actor(states[mb_inds], actions[mb_inds])
                new_values = self.critic(states[mb_inds]).flatten()
                
                # new_log_probs already has shape (minibatch_size,) from sum(1)
                # old_log_probs[mb_inds] should have shape (minibatch_size,)
                old_log_probs_mb = old_log_probs[mb_inds]
                logratio = new_log_probs - old_log_probs_mb
                ratio = logratio.exp()
                
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                
                # Policy loss (clipped surrogate)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                if self.clip_vloss:
                    v_loss_unclipped = (new_values - mb_returns) ** 2
                    v_clipped = current_values[mb_inds] + torch.clamp(
                        new_values - current_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                
                # Check for NaN/Inf in loss before backward
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"WARNING: NaN/Inf loss detected, skipping this minibatch")
                    continue
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN gradients before clipping
                has_nan_grad = False
                for param in list(self.actor.parameters()) + list(self.critic.parameters()):
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            print(f"WARNING: NaN/Inf gradient detected, skipping this update step")
                            param.grad.zero_()
                
                if not has_nan_grad:
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                else:
                    # Skip optimizer step if gradients are invalid
                    self.optimizer.zero_grad()
                
                # Track clip fraction
                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())
        
        self.total_it += 1
        
        # Logging
        if self.total_it % 100 == 0:
            self.writer.add_scalar("train_policy/loss", pg_loss.item(), self.total_it)
            self.writer.add_scalar("train_value/loss", v_loss.item(), self.total_it)
            self.writer.add_scalar("train_entropy/loss", entropy_loss.item(), self.total_it)
            self.writer.add_scalar("train/clipfrac", np.mean(clipfracs), self.total_it)
    
    def save(self, filename=None, directory=None, metadata=None):
        """Save actor and critic networks"""
        filename = filename or self.model_name
        directory = Path(directory) if directory else self.save_directory
        directory.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.actor.state_dict(), directory / f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), directory / f"{filename}_critic.pth")
        
        if metadata is not None:
            import json
            with open(directory / f"{filename}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load(self, filename=None, directory=None):
        """Load actor and critic networks"""
        filename = filename or self.model_name
        directory = Path(directory) if directory else self.save_directory
        
        self.actor.load_state_dict(torch.load(directory / f"{filename}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(directory / f"{filename}_critic.pth", map_location=self.device))
        
        metadata_path = directory / f"{filename}_metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None

