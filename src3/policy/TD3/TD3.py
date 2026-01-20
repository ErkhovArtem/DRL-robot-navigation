import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import deque
from .TD3_actor import Actor
from .TD3_critic import DoubleQCritic
from torch.utils.tensorboard import SummaryWriter

class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_action,
        discount=0.99,
        tau=0.005,
        learning_rate=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
        exploration_noise=0.1,
        policy_frequency=2,
        history_length=0,
        actor_hidden_dim=[256, 256],
        actor_hidden_depth=2,
        critic_hidden_dim=[256, 256],
        critic_hidden_depth=2,
        critic_state_dim=None,
        writer=None,
        model_name="TD3",
        save_directory=Path("models/TD3")
    ):
        self.device = torch.device(device)
        self.max_action = max_action
        self.gamma = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_frequency = policy_frequency
        
        self.base_actor_state_dim = state_dim
        self.base_critic_state_dim = critic_state_dim if critic_state_dim is not None else state_dim
        self.history_length = history_length
        
        self.actor_state_dim = self.base_actor_state_dim * (history_length + 1) if history_length > 0 else self.base_actor_state_dim
        self.critic_state_dim = self.base_critic_state_dim * (history_length + 1) if history_length > 0 else self.base_critic_state_dim
        
        self.actor = Actor(self.actor_state_dim, action_dim, actor_hidden_dim, actor_hidden_depth, max_action).to(self.device)
        self.actor_target = Actor(self.actor_state_dim, action_dim, actor_hidden_dim, actor_hidden_depth, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        self.critic = DoubleQCritic(self.critic_state_dim, action_dim, critic_hidden_dim, critic_hidden_depth).to(self.device)
        self.critic_target = DoubleQCritic(self.critic_state_dim, action_dim, critic_hidden_dim, critic_hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        self.actor_obs_history = deque(maxlen=self.history_length + 1)
        self.critic_obs_history = deque(maxlen=self.history_length + 1)
        
        self.total_it = 0
        self.writer = writer if writer is not None else SummaryWriter()
        self.model_name = model_name
        self.save_directory = Path(save_directory)
        
    def process_observation(self, obs, is_critic=False, history_buffer=None):
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
        self.actor_obs_history.clear()
        self.critic_obs_history.clear()

    def get_action(self, obs, add_noise):
        state = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.max_action * self.exploration_noise, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)
            
        return action

    def train(self, replay_buffer, iterations, batch_size, success_weight=2.0, collision_weight=0.5):
        for it in range(iterations):
            self.total_it += 1
            
            # Sample batch
            s_batch, a_batch, r_batch, t_batch, s2_batch, _ = replay_buffer.sample_batch(
                batch_size, success_weight=success_weight, collision_weight=collision_weight
            )
            
            state = torch.FloatTensor(s_batch).to(self.device)
            action = torch.FloatTensor(a_batch).to(self.device)
            next_state = torch.FloatTensor(s2_batch).to(self.device)
            reward = torch.FloatTensor(r_batch).to(self.device)
            not_done = torch.FloatTensor(1 - t_batch).to(self.device)
            
            with torch.no_grad():
                # Select action according to target policy and add clipped noise
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                
                # Check if next_state needs slicing for target actor
                target_actor_next_state = next_state
                if next_state.shape[-1] > self.actor_state_dim:
                    target_actor_next_state = next_state[..., :self.actor_state_dim]
                
                next_action = (self.actor_target(target_actor_next_state) + noise).clamp(-self.max_action, self.max_action)
                
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.gamma * target_Q
                
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            if self.total_it % 100 == 0:
                self.writer.add_scalar("train_critic/loss", critic_loss.item(), self.total_it)
            
            # Delayed policy updates
            if self.total_it % self.policy_frequency == 0:
                # Compute actor loss
                # Actor only uses its own state dimension
                actor_state = state
                if state.shape[-1] > self.actor_state_dim:
                    actor_state = state[..., :self.actor_state_dim]
                
                # Get Q values using critic.forward method (returns Q1, Q2 tuple)
                q1, q2 = self.critic(actor_state, self.actor(actor_state))
                actor_loss = -q1.mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                if self.total_it % 100 == 0:
                    self.writer.add_scalar("train_actor/loss", actor_loss.item(), self.total_it)
                
                # Update target networks
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename=None, directory=None, metadata=None):
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
        filename = filename or self.model_name
        directory = Path(directory) if directory else self.save_directory
        
        self.actor.load_state_dict(torch.load(directory / f"{filename}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(directory / f"{filename}_critic.pth", map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        metadata_path = directory / f"{filename}_metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None

