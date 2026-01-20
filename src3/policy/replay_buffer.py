import random
from collections import deque
from pathlib import Path

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)
        # OPTIMIZATION: Cache weights to avoid recomputing on every sample
        self._weights_cache = None
        self._weights_dirty = True
        self._last_success_weight = None
        self._last_collision_weight = None

    def add(self, s, a, r, t, s2, success=False):
        """
        Add experience to replay buffer.
        
        Args:
            s: current state
            a: action taken
            r: reward received
            t: terminal flag (done)
            s2: next state
            success: success flag (True if episode ended successfully, False otherwise)
        """
        experience = (s, a, r, t, s2, success)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        # OPTIMIZATION: Mark weights as dirty when buffer changes
        self._weights_dirty = True

    def size(self):
        return self.count

    def sample_batch(self, batch_size, success_weight=2.0, collision_weight=0.5):
        """
        Sample batch from replay buffer with weighted sampling.
        Optimized to avoid recomputing weights for the entire buffer unless necessary.
        """
        if self.count < batch_size:
            return self.sample_uniform(self.count) # Fallback if buffer is small
        
        # OPTIMIZATION: Recompute weights only if buffer changed significantly or weights changed
        # If buffer is huge, we only update the cache, not re-scan everything if only few items added
        if (self._weights_dirty or 
            self._last_success_weight != success_weight or 
            self._last_collision_weight != collision_weight):
            
            # For efficiency with large buffers, we'll use a faster approach
            # Pre-calculate boolean masks
            rewards = np.array([exp[2] for exp in self.buffer])
            dones = np.array([exp[3] for exp in self.buffer])
            successes = np.array([exp[5] for exp in self.buffer])
            
            weights = np.ones(len(self.buffer), dtype=np.float32)
            weights[successes == True] = success_weight
            weights[(dones == 1.0) & (rewards <= -100)] = collision_weight * 2.0
            
            # Normalize
            self._weights_cache = weights / weights.sum()
            self._weights_dirty = False
            self._last_success_weight = success_weight
            self._last_collision_weight = collision_weight
        
        indices = np.random.choice(len(self.buffer), size=batch_size, p=self._weights_cache, replace=True)
        
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch]).reshape(-1, 1)
        dones = np.array([exp[3] for exp in batch]).reshape(-1, 1)
        next_states = np.array([exp[4] for exp in batch])
        successes = np.array([exp[5] for exp in batch]).reshape(-1, 1)
        
        return states, actions, rewards, dones, next_states, successes

    def sample_uniform(self, batch_size):
        """
        Sample batch from replay buffer using uniform random sampling.
        """
        if self.count < batch_size:
            batch_size = self.count
        
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch]).reshape(-1, 1)
        dones = np.array([exp[3] for exp in batch]).reshape(-1, 1)
        next_states = np.array([exp[4] for exp in batch])
        successes = np.array([exp[5] for exp in batch]).reshape(-1, 1)

        return states, actions, rewards, dones, next_states, successes

    def return_buffer(self):
        s = np.array([_[0] for _ in self.buffer])
        a = np.array([_[1] for _ in self.buffer])
        r = np.array([_[2] for _ in self.buffer]).reshape(-1, 1)
        t = np.array([_[3] for _ in self.buffer]).reshape(-1, 1)
        s2 = np.array([_[4] for _ in self.buffer])
        success = np.array([_[5] for _ in self.buffer]).reshape(-1, 1)

        return s, a, r, t, s2, success

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def save(self, filepath):
        """
        Save replay buffer to file.
        
        Args:
            filepath: Path to save the buffer (can be str or Path object)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert buffer to numpy arrays for efficient saving
        if self.count == 0:
            # Empty buffer - save metadata only
            np.savez(
                filepath,
                buffer_size=self.buffer_size,
                count=0,
                states=np.array([]),
                actions=np.array([]),
                rewards=np.array([]),
                terminals=np.array([]),
                next_states=np.array([]),
                successes=np.array([])
            )
        else:
            # Get all experiences from buffer
            s_list = []
            a_list = []
            r_list = []
            t_list = []
            s2_list = []
            success_list = []
            
            for experience in self.buffer:
                s_list.append(experience[0])
                a_list.append(experience[1])
                r_list.append(experience[2])
                t_list.append(experience[3])
                s2_list.append(experience[4])
                # Handle backward compatibility: old buffers don't have success flag
                success_list.append(experience[5] if len(experience) > 5 else False)
            
            # Convert to numpy arrays
            states = np.array(s_list)
            actions = np.array(a_list)
            rewards = np.array(r_list)
            terminals = np.array(t_list)
            next_states = np.array(s2_list)
            successes = np.array(success_list)
            
            # Save with compression
            np.savez_compressed(
                filepath,
                buffer_size=self.buffer_size,
                count=self.count,
                states=states,
                actions=actions,
                rewards=rewards,
                terminals=terminals,
                next_states=next_states,
                successes=successes
            )
    
    def load(self, filepath):
        """
        Load replay buffer from file.
        
        Args:
            filepath: Path to load the buffer from (can be str or Path object)
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"Buffer file not found: {filepath}")
            return False
        
        try:
            # Load data
            data = np.load(filepath, allow_pickle=False)
            
            # Restore metadata
            self.buffer_size = int(data['buffer_size'])
            count = int(data['count'])
            
            # Clear current buffer
            self.buffer.clear()
            self.count = 0
            
            # If buffer is empty, return early
            if count == 0:
                print(f"Loaded empty buffer from {filepath}")
                return True
            
            # Restore experiences
            states = data['states']
            actions = data['actions']
            rewards = data['rewards']
            terminals = data['terminals']
            next_states = data['next_states']
            
            # Handle backward compatibility: old buffers don't have successes
            if 'successes' in data:
                successes = data['successes']
            else:
                # Default to False for old buffers
                successes = np.zeros(count, dtype=bool)
            
            # Add experiences back to buffer
            for i in range(count):
                experience = (
                    states[i],
                    actions[i],
                    float(rewards[i]),
                    float(terminals[i]),
                    next_states[i],
                    bool(successes[i]) if successes.ndim > 0 else False
                )
                self.buffer.append(experience)
                self.count += 1
            
            print(f"Loaded {self.count} experiences from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading buffer from {filepath}: {e}")
            return False
