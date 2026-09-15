"""
Curriculum Learning Manager for progressive training difficulty.
"""
class CurriculumManager:
    """
    Manages curriculum learning by tracking success rates and rewards,
    and updating environment parameters (obstacles, rewards) accordingly.
    """
    def __init__(self, curriculum_config, base_config):
        self.config = curriculum_config.get('curriculum', {})
        self.levels = self.config.get('levels', [])
        self.settings = self.config.get('settings', {})
        self.current_level_idx = 0
        self.base_config = base_config
        
        self.metric_window = self.settings.get('metric_window', 100)
        self.min_episodes_per_level = self.settings.get('min_episodes_per_level', 200)
        
        self.history_success = []
        self.history_reward = []
        self.episodes_on_current_level = 0
        
    def update(self, success, reward):
        """
        Update metrics with results from a new episode and check for level transition.
        
        Args:
            success (bool): Whether the episode was successful
            reward (float): Total reward for the episode
            
        Returns:
            bool: True if a level transition occurred, False otherwise
        """
        self.history_success.append(float(success))
        self.history_reward.append(float(reward))
        self.episodes_on_current_level += 1
        
        # Keep only the last N episodes for metrics
        if len(self.history_success) > self.metric_window:
            self.history_success.pop(0)
            self.history_reward.pop(0)
            
        # Check if we can transition to the next level
        if self.current_level_idx < len(self.levels) - 1:
            current_level_data = self.levels[self.current_level_idx]
            target_metrics = current_level_data.get('target_metrics', {})
            
            # Calculate current mean metrics
            mean_success = sum(self.history_success) / len(self.history_success) if self.history_success else 0
            mean_reward = sum(self.history_reward) / len(self.history_reward) if self.history_reward else 0
            
            # Transition criteria
            target_success = target_metrics.get('success_rate', 1.0)
            target_reward = target_metrics.get('mean_reward', -float('inf'))
            
            met_success = mean_success >= target_success
            met_reward = mean_reward >= target_reward
            enough_episodes = self.episodes_on_current_level >= self.min_episodes_per_level
            
            if met_success and met_reward and enough_episodes:
                self.current_level_idx += 1
                self.episodes_on_current_level = 0
                return True # Level changed
                
        return False

    def get_current_level_name(self):
        """Get the name of the current level."""
        if not self.levels:
            return "Base"
        return self.levels[self.current_level_idx].get('name', f"Level {self.current_level_idx + 1}")

    def apply_current_level(self, reward_weights, obstacle_params, policy_config=None, replay_buffer_config=None, episode_params=None):
        """
        Apply current level parameters to reward weights, obstacle generator, SAC, replay buffer, and episode settings.
        Incremental approach: applies all overrides from Level 1 up to current level.
        
        Args:
            reward_weights (dict): Reward weights to update in-place
            obstacle_params (dict): Obstacle parameters to update in-place
            policy_config (dict): SAC policy config to update in-place (optional)
            replay_buffer_config (dict): Replay buffer config to update in-place (optional)
            episode_params (dict): Episode parameters (max_steps, etc.) to update in-place (optional)
        """
        # Apply overrides from Level 1 up to current level
        for i in range(self.current_level_idx + 1):
            level_data = self.levels[i]
            
            # Update reward weights
            if 'reward_weights' in level_data:
                reward_weights.update(level_data['reward_weights'])
            
            # Update obstacle generator
            if 'obstacle_generator' in level_data:
                og = level_data['obstacle_generator']
                if 'cubes' in og:
                    # Update cube count, sizes, etc.
                    if 'cubes' not in obstacle_params:
                        obstacle_params['cubes'] = {}
                    obstacle_params['cubes'].update(og['cubes'])
                if 'position' in og:
                    # Update position margins, etc.
                    if 'position' not in obstacle_params:
                        obstacle_params['position'] = {}
                    obstacle_params['position'].update(og['position'])
            
            # Update episode parameters (max_steps, etc.)
            if episode_params is not None and 'max_steps' in level_data:
                episode_params['max_steps'] = level_data['max_steps']
            
            # Update SAC parameters
            if policy_config is not None:
                if 'sac' in level_data:
                    policy_config.update(level_data['sac'])
            
            # Update replay buffer parameters
            if replay_buffer_config is not None and 'replay_buffer' in level_data:
                replay_buffer_config.update(level_data['replay_buffer'])
