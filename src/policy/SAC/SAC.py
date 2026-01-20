from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from statistics import mean
from .SAC_utils import soft_update_params, to_np
from .SAC_critic import DoubleQCritic as critic_model
from .SAC_actor import DiagGaussianActor as actor_model
from torch.utils.tensorboard import SummaryWriter

from collections import deque


class SAC(object):
    """SAC algorithm."""

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_action,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_betas=(0.9, 0.999),
        actor_lr=1e-4,
        actor_betas=(0.9, 0.999),
        actor_update_frequency=1,
        critic_lr=1e-4,
        critic_betas=(0.9, 0.999),
        critic_tau=0.005,
        critic_target_update_frequency=2,
        learnable_temperature=True,
        save_every=0,
        load_model=False,
        log_dist_and_hist = False,
        save_directory=Path("models/SAC"),
        model_name="SAC",
        load_directory=None,
        writer=None,
        history_length=0,
        critic_history_length=None,
        actor_hidden_dim=None,
        actor_hidden_depth=2,
        actor_log_std_bounds=[-5, 2],
        critic_hidden_dim=None,
        critic_hidden_depth=2,
        critic_state_dim=None,
    ):
        super().__init__()

        # History lengths
        self.actor_history_length = history_length if history_length is not None else 0  # История actions для Actor
        self.critic_history_length = critic_history_length if critic_history_length is not None else 0  # История наблюдений для Critic
        
        # Store base state dimensions
        # state_dim - это УЖЕ stacked dimension для Actor (base + action_history)
        # Вычисляем base_actor_state_dim: убираем историю actions
        action_history_size = self.actor_history_length * 3  # Каждый шаг истории = 3 action (vx, vy, w)
        self.base_actor_state_dim = state_dim - action_history_size  # Базовое наблюдение без истории actions
        
        self.base_critic_state_dim = critic_state_dim if critic_state_dim is not None else self.base_actor_state_dim + 2  # Actor + vx + vy
        
        # Stacked state dimensions
        # Actor: base + action_history (уже в state_dim)
        self.actor_state_dim = state_dim
        # Critic: base * (history_length + 1) для наблюдений
        self.critic_state_dim = self.base_critic_state_dim * (self.critic_history_length + 1) if self.critic_history_length > 0 else self.base_critic_state_dim
        
        self.action_dim = action_dim
        self.action_range = (-max_action, max_action)
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = Path(save_directory) if save_directory else Path("models/SAC")
        self.load_directory = Path(load_directory) if load_directory else self.save_directory
        self.log_dist_and_hist = log_dist_and_hist

        # Store optimizer parameters for potential reset during fine-tuning
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.actor_betas = actor_betas
        self.critic_betas = critic_betas
        self.alpha_betas = alpha_betas

        # History buffers
        # Actor: история actions (не observations!)
        self.actor_action_history = deque(maxlen=self.actor_history_length) if self.actor_history_length > 0 else None
        # Critic: история наблюдений
        self.critic_obs_history = deque(maxlen=self.critic_history_length + 1) if self.critic_history_length > 0 else None

        self.train_metrics_dict = { "train_critic/loss_av": [],
                                    "train_actor/loss_av": [],
                                    "train_actor/target_entropy_av": [],
                                    "train_actor/entropy_av": [],
                                    "train_alpha/loss_av": [],
                                    "train_alpha/value_av": [],
                                    "train/batch_reward_av": []
        }

        # Network architecture parameters (reference defaults to 1024)
        actor_hdim = actor_hidden_dim if actor_hidden_dim is not None else 1024
        critic_hdim = critic_hidden_dim if critic_hidden_dim is not None else 1024
        
        # Create networks with respective stacked state dimensions
        # Double Q-learning: critic contains two independent Q-networks (Q1, Q2)
        self.critic = critic_model(
            obs_dim=self.critic_state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hdim,
            hidden_depth=critic_hidden_depth,
        ).to(self.device)
        # Target critic also has two Q-networks (Q1_target, Q2_target)
        self.critic_target = critic_model(
            obs_dim=self.critic_state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hdim,
            hidden_depth=critic_hidden_depth,
        ).to(self.device)
        # Initialize target network with same weights as main critic
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = actor_model(
            obs_dim=self.actor_state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hdim,
            hidden_depth=actor_hidden_depth,
            log_std_bounds=actor_log_std_bounds,
        ).to(self.device)

        if load_model:
            self.load(filename=model_name, directory=load_directory)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=critic_betas
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.critic_target.train()

        self.actor.train(True)
        self.critic.train(True)
        self.step = 0
        self.alpha_min = 0.0  # Default alpha_min (may be set from config)
        self.alpha_update_frequency = 1 # Default (not used, alpha updates every time like src2)
        # Use provided writer or create new one
        self.writer = writer if writer is not None else SummaryWriter()
        
    def reset_history(self):
        """Сброс истории наблюдений и действий."""
        if hasattr(self, 'critic_obs_history') and self.critic_obs_history is not None:
            self.critic_obs_history.clear()
        if hasattr(self, 'actor_action_history') and self.actor_action_history is not None:
            self.actor_action_history.clear()

    def process_observation(self, obs, is_critic=False, history_buffer=None, action_history_buffer=None):
        """
        Добавляет наблюдение в историю и возвращает stacked наблюдение.
        
        Args:
            obs: текущее наблюдение (base observation, БЕЗ истории actions)
            is_critic: является ли это наблюдением для критика
            history_buffer: опциональный список/deque для хранения истории наблюдений (для Critic, MJX mode)
            action_history_buffer: опциональный список/deque для хранения истории actions (для Actor, MJX mode)
        
        Returns:
            stacked_obs: наблюдение с историей (для критика - history obs, для актора - obs + history actions)
        """
        # Actor: добавляем историю actions к наблюдению
        if not is_critic:
            if self.actor_history_length == 0:
                return obs.copy()
            
            # Используем переданный буфер или внутренний
            if action_history_buffer is not None:
                action_history = action_history_buffer
            else:
                action_history = self.actor_action_history
            
            # Если истории нет или недостаточно - заполняем нулями
            if action_history is None or len(action_history) == 0:
                # Заполняем историю нулевыми actions
                padded_actions = np.zeros(3 * self.actor_history_length)
                return np.concatenate([obs, padded_actions])
            
            # Конкатенируем историю actions (хронологический порядок: oldest to newest)
            action_history_flat = np.concatenate(list(action_history))
            
            # Если история не полная - дополняем нулями спереди
            if len(action_history) < self.actor_history_length:
                missing = self.actor_history_length - len(action_history)
                action_history_flat = np.concatenate([np.zeros(3 * missing), action_history_flat])
            
            # Объединяем наблюдение с историей actions
            return np.concatenate([obs, action_history_flat])
        
        # Critic: can have history
        if self.critic_history_length == 0:
            return obs.copy()
        
        # Select correct history and base dim
        if history_buffer is not None:
            history = history_buffer
        else:
            history = self.critic_obs_history
            
        base_dim = self.base_critic_state_dim
        
        # Append current observation to history
        history.append(obs.copy())
        
        # If history is not full yet, pad with first observation
        while len(history) < self.critic_history_length + 1:
            history.appendleft(obs.copy())
        
        # Concatenate all observations in chronological order (oldest to newest)
        stacked_obs = np.concatenate(list(history))
        
        # Verify size matches expected
        expected_size = base_dim * (self.critic_history_length + 1)
        assert stacked_obs.shape[0] == expected_size, \
            f"Stacked observation size mismatch: got {stacked_obs.shape[0]}, expected {expected_size}"
        
        return stacked_obs

    def build_observation_from_history(self, obs, is_critic=False, history_buffer=None, action_history_buffer=None):
        """
        Строит stacked наблюдение из текущего наблюдения и существующей истории
        БЕЗ обновления внутренней истории (используется для коллизий/терминальных состояний).
        
        Args:
            obs: текущее наблюдение (base observation, БЕЗ истории actions)
            is_critic: является ли это наблюдением для критика
            history_buffer: опциональный список/deque для хранения истории наблюдений (для Critic, MJX mode)
            action_history_buffer: опциональный список/deque для хранения истории actions (для Actor, MJX mode)
        
        Returns:
            stacked_obs: наблюдение с историей (без обновления внутренней истории)
        """
        # Actor: добавляем историю actions БЕЗ обновления истории
        if not is_critic:
            if self.actor_history_length == 0:
                return obs.copy()
            
            # Используем переданный буфер или внутренний
            if action_history_buffer is not None:
                action_history = action_history_buffer
            else:
                action_history = self.actor_action_history
            
            # Если истории нет или недостаточно - заполняем нулями
            if action_history is None or len(action_history) == 0:
                padded_actions = np.zeros(3 * self.actor_history_length)
                return np.concatenate([obs, padded_actions])
            
            # Конкатенируем существующую историю actions
            action_history_flat = np.concatenate(list(action_history))
            
            # Если история не полная - дополняем нулями спереди
            if len(action_history) < self.actor_history_length:
                missing = self.actor_history_length - len(action_history)
                action_history_flat = np.concatenate([np.zeros(3 * missing), action_history_flat])
            
            return np.concatenate([obs, action_history_flat])
            return obs.copy()
        
        # Critic: can have history
        if self.critic_history_length == 0:
            return obs.copy()
        
        # Select correct history and base dim
        if history_buffer is not None:
            history = history_buffer
        else:
            history = self.critic_obs_history
            
        base_dim = self.base_critic_state_dim
        
        # Create temporary history by copying current history and appending new observation
        # WITHOUT modifying the actual history
        temp_history = deque(history, maxlen=self.critic_history_length + 1)
        temp_history.append(obs.copy())
        
        # If history is not full yet, pad with current observation
        while len(temp_history) < self.critic_history_length + 1:
            temp_history.appendleft(obs.copy())
        
        # Concatenate all observations in chronological order (oldest to newest)
        stacked_obs = np.concatenate(list(temp_history))
        
        # Verify size matches expected
        expected_size = base_dim * (self.critic_history_length + 1)
        assert stacked_obs.shape[0] == expected_size, \
            f"Stacked observation size mismatch: got {stacked_obs.shape[0]}, expected {expected_size}"
        
        return stacked_obs

    def save(self, filename=None, directory=None, metadata=None):
        """
        Save model weights and optional metadata.
        
        Args:
            filename: Base filename for saved files
            directory: Directory to save files
            metadata: Optional dict with metadata (e.g., {'episode': 1000})
        """
        filename = filename or self.model_name
        directory = Path(directory) if directory else self.save_directory
        directory.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.actor.state_dict(), directory / f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), directory / f"{filename}_critic.pth")
        torch.save(
            self.critic_target.state_dict(),
            directory / f"{filename}_critic_target.pth",
        )
        
        # Save log_alpha if learnable_temperature is enabled
        if self.learnable_temperature:
            torch.save(self.log_alpha, directory / f"{filename}_log_alpha.pth")
        
        # Save metadata if provided
        if metadata is not None:
            import json
            metadata_path = directory / f"{filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def load(self, filename=None, directory=None, fine_tune=False):
        """
        Load model weights and optional metadata.
        
        Args:
            filename: Base filename
            directory: Directory path
            fine_tune: If True, resets entropy (log_alpha) to initial value and doesn't load it.
                      Also resets the training step counter and optimizers.
        
        Returns:
            dict: Metadata if available, None otherwise
        """
        filename = filename or self.model_name
        directory = Path(directory) if directory else self.load_directory
        
        self.actor.load_state_dict(
            torch.load(directory / f"{filename}_actor.pth")
        )
        self.critic.load_state_dict(
            torch.load(directory / f"{filename}_critic.pth")
        )
        self.critic_target.load_state_dict(
            torch.load(directory / f"{filename}_critic_target.pth")
        )
        
        # Load log_alpha unless fine_tuning
        if self.learnable_temperature:
            if fine_tune:
                print(f"Fine-tuning mode: resetting log_alpha to initial temperature (alpha={self.alpha.item():.4f})")
                # log_alpha is already initialized in __init__ from init_temperature
                # but let's re-initialize optimizer for alpha just in case
                self.log_alpha_optimizer = torch.optim.Adam(
                    [self.log_alpha], lr=self.alpha_lr, betas=self.alpha_betas
                )
            else:
                log_alpha_path = directory / f"{filename}_log_alpha.pth"
                if log_alpha_path.exists():
                    self.log_alpha = torch.load(log_alpha_path)
                    # Ensure it requires grad
                    self.log_alpha.requires_grad = True
                    print(f"Loaded log_alpha: {self.log_alpha.item():.4f} (alpha: {self.alpha.item():.4f})")
                else:
                    print(f"Warning: log_alpha file not found at {log_alpha_path}, using current value")
        
        if fine_tune:
            self.step = 0
            # Re-initialize optimizers to clear momentum/moments
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=self.actor_lr, betas=self.actor_betas
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=self.critic_lr, betas=self.critic_betas
            )
            print("Fine-tuning mode: optimizers and step counter have been reset.")
        
        print(f"Loaded weights from: {directory}")
        
        # Try to load metadata
        metadata_path = directory / f"{filename}_metadata.json"
        if metadata_path.exists():
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    print(f"Loaded metadata: {metadata}")
                    return metadata
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
        
        return None

    def train(self, replay_buffer, iterations, batch_size, success_weight=2.0, collision_weight=0.5):
        """
        Reference style aggressive training with your vectorized weights feature.
        """
        for _ in range(iterations):
            self.update(
                replay_buffer=replay_buffer, 
                step=self.step, 
                batch_size=batch_size,
                success_weight=success_weight,
                collision_weight=collision_weight
            )

        for key, value in self.train_metrics_dict.items():
            if len(value):
                self.writer.add_scalar(key, mean(value), self.step)
            self.train_metrics_dict[key] = []
        self.step += 1

        if self.save_every > 0 and self.step % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, obs, add_noise):
        """
        Get action from policy. 
        If add_noise is True, samples from the distribution (standard SAC exploration).
        If False, returns the mean (deterministic).
        """
        return self.act(obs, sample=add_noise)

    def act(self, obs, sample=False, update_action_history=True):
        """
        Get action from policy.
        
        Args:
            obs: наблюдение (может включать историю actions для Actor)
            sample: использовать ли sampling (True) или mean (False)
            update_action_history: обновлять ли историю actions для Actor (по умолчанию True)
        
        Returns:
            action: [vx, vy, w] в [-1, 1]
        """
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        action_np = to_np(action[0])
        
        # Обновляем историю actions для Actor (если история включена)
        if update_action_history and self.actor_action_history is not None:
            self.actor_action_history.append(action_np.copy())
        
        return action_np

    def update_critic(self, obs, action, reward, next_obs, done, step):
        # next_obs из replay buffer - это critic observation (torch tensor, может быть с историей наблюдений)
        # Для Actor нужен actor observation (base + action_history)
        # Извлекаем base actor observation из critic observation и добавляем текущую историю actions
        
        # next_obs может быть батчем [batch_size, features] или одиночным наблюдением [features]
        is_batch = next_obs.ndim == 2
        
        # Извлекаем base actor observation из critic observation
        if self.critic_history_length > 0 and next_obs.shape[-1] > self.base_critic_state_dim:
            # Critic observation с историей наблюдений: [obs_t-history, ..., obs_t-1, obs_t]
            frame_size = self.base_critic_state_dim
            last_frame = next_obs[..., -frame_size:]
            actor_next_base = last_frame[..., :self.base_actor_state_dim]  # Actor base (47)
        elif next_obs.shape[-1] > self.base_actor_state_dim:
            # Critic observation без истории наблюдений
            actor_next_base = next_obs[..., :self.base_actor_state_dim]  # Actor base (47)
        else:
            # Уже actor observation (может быть с историей actions)
            actor_next_obs = next_obs
        
        # Добавляем историю actions к base observation (если еще не добавлена)
        if 'actor_next_obs' not in locals():
            if self.actor_history_length > 0 and self.actor_action_history is not None and len(self.actor_action_history) > 0:
                # Конвертируем в numpy для конкатенации
                actor_next_base_np = to_np(actor_next_base)
                
                # Строим историю actions
                action_history_flat = np.concatenate(list(self.actor_action_history))
                if len(self.actor_action_history) < self.actor_history_length:
                    missing = self.actor_history_length - len(self.actor_action_history)
                    action_history_flat = np.concatenate([np.zeros(3 * missing), action_history_flat])
                
                # Если это батч, повторяем историю actions для каждого элемента батча
                if is_batch:
                    batch_size = actor_next_base_np.shape[0]
                    action_history_batch = np.tile(action_history_flat, (batch_size, 1))  # [batch_size, history_size]
                    actor_next_obs_np = np.concatenate([actor_next_base_np, action_history_batch], axis=1)
                else:
                    actor_next_obs_np = np.concatenate([actor_next_base_np, action_history_flat])
                
                actor_next_obs = torch.FloatTensor(actor_next_obs_np).to(self.device)
            else:
                actor_next_obs = actor_next_base
        
        # Compute target Q without gradients (target networks should not be trained)
        # SAC uses Double Q-learning: two Q-networks (Q1, Q2) to reduce overestimation bias
        with torch.no_grad():
            dist = self.actor(actor_next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            # Get both Q-values from target critic (Double Q-learning)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            # Use minimum of two Q-values to reduce overestimation (key SAC feature)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob
            target_Q = reward + ((1 - done) * self.discount * target_V)

        # Get current Q estimates from both Q-networks
        current_Q1, current_Q2 = self.critic(obs, action)
        # Both Q-networks are trained to minimize MSE with the same target_Q
        # This is correct: both networks learn the same target, but independently
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        self.train_metrics_dict["train_critic/loss_av"].append(critic_loss.item())
        self.writer.add_scalar("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.log_dist_and_hist:
            self.critic.log(self.writer, step)

    def update_actor_and_alpha(self, obs, step):
        # obs из replay buffer - это critic observation (torch tensor, может быть с историей наблюдений)
        # Для Actor нужен actor observation (base + action_history)
        # Извлекаем base actor observation из critic observation и добавляем текущую историю actions
        
        # obs может быть батчем [batch_size, features] или одиночным наблюдением [features]
        is_batch = obs.ndim == 2
        
        # Извлекаем base actor observation из critic observation
        if self.critic_history_length > 0 and obs.shape[-1] > self.base_critic_state_dim:
            # Critic observation с историей наблюдений: [obs_t-history, ..., obs_t-1, obs_t]
            frame_size = self.base_critic_state_dim
            last_frame = obs[..., -frame_size:]
            actor_obs_base = last_frame[..., :self.base_actor_state_dim]  # Actor base (47)
        elif obs.shape[-1] > self.base_actor_state_dim:
            # Critic observation без истории наблюдений
            actor_obs_base = obs[..., :self.base_actor_state_dim]  # Actor base (47)
        else:
            # Уже actor observation (может быть с историей actions)
            actor_obs = obs
        
        # Добавляем историю actions к base observation (если еще не добавлена)
        if 'actor_obs' not in locals():
            if self.actor_history_length > 0 and self.actor_action_history is not None and len(self.actor_action_history) > 0:
                # Конвертируем в numpy для конкатенации
                actor_obs_base_np = to_np(actor_obs_base)
                
                # Строим историю actions
                action_history_flat = np.concatenate(list(self.actor_action_history))
                if len(self.actor_action_history) < self.actor_history_length:
                    missing = self.actor_history_length - len(self.actor_action_history)
                    action_history_flat = np.concatenate([np.zeros(3 * missing), action_history_flat])
                
                # Если это батч, повторяем историю actions для каждого элемента батча
                if is_batch:
                    batch_size = actor_obs_base_np.shape[0]
                    action_history_batch = np.tile(action_history_flat, (batch_size, 1))  # [batch_size, history_size]
                    actor_obs_np = np.concatenate([actor_obs_base_np, action_history_batch], axis=1)
                else:
                    actor_obs_np = np.concatenate([actor_obs_base_np, action_history_flat])
                
                actor_obs = torch.FloatTensor(actor_obs_np).to(self.device)
            else:
                actor_obs = actor_obs_base
        
        dist = self.actor(actor_obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # Critic always uses the full observation (with history and extra features if any)
        # Get both Q-values from critic (Double Q-learning)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        # Use minimum of two Q-values for actor loss (reduces overestimation bias)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        self.train_metrics_dict["train_actor/loss_av"].append(actor_loss.item())
        self.train_metrics_dict["train_actor/target_entropy_av"].append(self.target_entropy)
        self.train_metrics_dict["train_actor/entropy_av"].append(-log_prob.mean().item())
        self.writer.add_scalar("train_actor/loss", actor_loss, step)
        self.writer.add_scalar("train_actor/target_entropy", self.target_entropy, step)
        self.writer.add_scalar("train_actor/entropy", -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.log_dist_and_hist:
            self.actor.log(self.writer, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            # Correct SAC alpha loss: alpha * (-log_pi - target_entropy)
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step, batch_size, success_weight=2.0, collision_weight=0.5):
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_dones,
            batch_next_states,
            batch_successes,  # May be unused but included for compatibility
        ) = replay_buffer.sample_batch(batch_size, success_weight=success_weight, collision_weight=collision_weight)

        state = torch.Tensor(batch_states).to(self.device)
        next_state = torch.Tensor(batch_next_states).to(self.device)
        action = torch.Tensor(batch_actions).to(self.device)
        reward = torch.Tensor(batch_rewards).to(self.device)
        done = torch.Tensor(batch_dones).to(self.device)
        self.train_metrics_dict["train/batch_reward_av"].append(batch_rewards.mean().item())
        self.writer.add_scalar("train/batch_reward", batch_rewards.mean(), step)

        self.update_critic(state, action, reward, next_state, done, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(state, step)

        if step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        # update the returned data from ROS into a form used for learning in the current model
        latest_scan = np.array(latest_scan)

        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0

        max_bins = self.state_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))

        # Initialize the list to store the minimum values of each bin
        min_values = []

        # Loop through the data and create bins
        for i in range(0, len(latest_scan), bin_size):
            # Get the current bin
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin))
        state = min_values + [distance, cos, sin] + [action[0], action[1]]

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        return state, terminal
