"""
MJX (MuJoCo XLA) utilities for parallel batched simulation.
"""
import numpy as np
from .observation import (
    fix_negative_lidar_values, build_critic_observation, build_actor_observation,
    process_lidar_to_sectors, transform_lidar_to_center_frame
)
from .target_generator import get_target_info, ROOM_X_MAX, ROOM_X_MIN, ROOM_Y_MAX, ROOM_Y_MIN
from .reward import compute_reward_reference_vectorized

try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    MJX_AVAILABLE = True
except ImportError:
    MJX_AVAILABLE = False


def create_mjx_batched_step_fn(mjx_model):
    """
    Create JIT-compiled batched step function for MJX using vmap.
    Optimized for GPU execution with JIT compilation.
    
    Args:
        mjx_model: MJX model (from mjx.put_model)
    
    Returns:
        JIT-compiled function that steps a batch of simulations (runs on GPU)
    """
    if not MJX_AVAILABLE:
        raise ImportError("MJX is not available. Install with: pip install mujoco-mjx")
    
    # Сначала создаем vmap функцию, затем компилируем JIT для ускорения на GPU
    @jax.vmap
    def step_single(mjx_data_single):
        """Step a single simulation forward"""
        return mjx.step(mjx_model, mjx_data_single)
    
    # JIT компиляция для GPU ускорения
    @jax.jit
    def batched_step(mjx_data):
        """Step a batch of simulations forward (JIT compiled for GPU)"""
        return step_single(mjx_data)
    
    return batched_step


def initialize_batch_episodes_mjx(mjx_model, batch_size, spawn_generator, m, 
                                  target_body_id, target_mocap_id, z_height=0.793):
    """
    Initialize a batch of parallel episodes with random spawn positions.
    
    Uses vmap to create batched initial states following MJX documentation.
    
    Args:
        mjx_model: MJX model
        batch_size: number of parallel environments
        spawn_generator: spawn point generator
        m: MuJoCo model (for reference)
        target_body_id: target body ID
        target_mocap_id: target mocap ID
        z_height: robot spawn height
    
    Returns:
        mjx_data: batched MJX data with initialized episodes (tree structure with batch dimension)
        target_positions: numpy array of target positions for each episode [batch_size, 3]
    """
    if not MJX_AVAILABLE:
        raise ImportError("MJX is not available. Install with: pip install mujoco-mjx")
    
    # Generate spawn positions and targets for all episodes
    target_positions = np.zeros((batch_size, 3))
    spawn_positions = np.zeros((batch_size, 3))
    spawn_quats = np.zeros((batch_size, 4))
    
    for i in range(batch_size):
        # Generate random spawn position
        robot_spawn_pos = spawn_generator.sample_spawn_point(z_height=z_height)
        spawn_positions[i] = robot_spawn_pos
        
        # Generate random yaw orientation
        random_yaw = np.random.uniform(0, 2 * np.pi)
        spawn_quats[i] = [np.cos(random_yaw / 2), 0, 0, np.sin(random_yaw / 2)]
        
        # Generate target position
        target_pos = spawn_generator.sample_target_point(
            robot_pos=robot_spawn_pos,
            min_distance=1.3,
            z_height=0.1
        )
        target_positions[i] = target_pos
    
    # Create single data structure first
    data_template = mjx.make_data(mjx_model)
    
    # Check if mocap_pos is available (has elements)
    # This is a Python boolean, checked before vmap
    has_mocap = data_template.mocap_pos.shape[0] > 0 and target_mocap_id >= 0 and target_mocap_id < data_template.mocap_pos.shape[0]
    
    # Create batched data by replicating and setting positions
    # MJX supports batched operations through tree structures
    # We'll use jax.vmap for batched operations
    
    if has_mocap:
        # Version with mocap update
        @jax.vmap
        def init_single_episode(spawn_pos, spawn_quat, target_pos):
            data = mjx.make_data(mjx_model)
            # Set robot position and orientation using functional updates
            # MJX Data is a frozen dataclass, so we need to use dataclasses.replace()
            from dataclasses import replace
            
            # Update qpos: first 3 elements are position, next 4 are quaternion
            new_qpos = data.qpos.at[0:3].set(spawn_pos)
            new_qpos = new_qpos.at[3:7].set(spawn_quat)
            data = replace(data, qpos=new_qpos)
            
            # Set target mocap position
            new_mocap_pos = data.mocap_pos.at[target_mocap_id].set(target_pos)
            new_mocap_quat = data.mocap_quat.at[target_mocap_id].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
            data = replace(data, mocap_pos=new_mocap_pos, mocap_quat=new_mocap_quat)
            
            # Forward kinematics for this single episode
            data = mjx.forward(mjx_model, data)
            
            return data
    else:
        # Version without mocap update (mocap_pos is empty)
        @jax.vmap
        def init_single_episode(spawn_pos, spawn_quat, target_pos):
            data = mjx.make_data(mjx_model)
            # Set robot position and orientation using functional updates
            # MJX Data is a frozen dataclass, so we need to use dataclasses.replace()
            from dataclasses import replace
            
            # Update qpos: first 3 elements are position, next 4 are quaternion
            new_qpos = data.qpos.at[0:3].set(spawn_pos)
            new_qpos = new_qpos.at[3:7].set(spawn_quat)
            data = replace(data, qpos=new_qpos)
            
            # No mocap update needed (mocap_pos is empty)
            
            # Forward kinematics for this single episode
            data = mjx.forward(mjx_model, data)
            
            return data
    
    # Create batched data (vmap automatically handles batching)
    mjx_data = init_single_episode(
        jnp.array(spawn_positions),
        jnp.array(spawn_quats),
        jnp.array(target_positions)
    )
    
    return mjx_data, target_positions

def extract_observations_from_batch(mjx_data, lidar_sensor_ids, lidar_sensor_angles, m, batch_size, 
                                   target_positions, prev_actions, max_lidar_range, max_vx, max_vy,
                                   max_angular_vel, max_distance, lidar_downsample_bins,
                                   critic_critical_topk=0,
                                   lidar_offset_x=0.12, lidar_offset_y=0.0,
                                   obs_noise_distance_std=0.0, obs_noise_angle_std=0.0,
                                   obs_noise_angular_vel_std=0.0, obs_noise_lidar_std=0.0,
                                   add_actor_noise=False):
    """
    Extract observations from batched MJX data.
    Returns (critic_observations, actor_observations, robot_positions, robot_quats, distances).
    Critic obs: clean. Actor obs: with noise when add_actor_noise and obs_noise_* > 0.
    """
    critic_observations = []
    actor_observations = []
    
    robot_positions = np.array(mjx_data.qpos[:, 0:3])
    robot_quats = np.array(mjx_data.qpos[:, 3:7])
    sensordata_array = np.array(mjx_data.sensordata)
    qvel_array = np.array(mjx_data.qvel)
    
    for i in range(batch_size):
        lidar_data_raw = sensordata_array[i, lidar_sensor_ids]
        lidar_data_raw = fix_negative_lidar_values(lidar_data_raw)
        # Lidar stays clean - noise only in build_actor_observation when add_actor_noise
        
        vx = float(qvel_array[i, 0])
        vy = float(qvel_array[i, 1])
        angular_vel = float(qvel_array[i, 5])
        
        robot_pos = robot_positions[i]
        target_pos = target_positions[i]
        robot_quat = robot_quats[i]
        distance, sin_angle, cos_angle = get_target_info(robot_pos, target_pos, robot_quat)
        
        # Critic obs: always clean
        obs_critic = build_critic_observation(
            lidar_data_raw, lidar_sensor_angles, vx, vy, angular_vel, distance,
            sin_angle, cos_angle, max_lidar_range, max_vx, max_vy,
            max_angular_vel, max_distance, prev_actions[i],
            lidar_downsample_bins,
            use_sector_processing=True,
            use_extended_features=False,
            critical_topk=critic_critical_topk,
            lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y
        )
        critic_observations.append(obs_critic)
        
        # Actor obs: with noise when add_actor_noise and training
        if add_actor_noise and (obs_noise_distance_std > 0 or obs_noise_angle_std > 0 or
                                obs_noise_angular_vel_std > 0 or obs_noise_lidar_std > 0):
            obs_actor = build_actor_observation(
                lidar_data_raw, lidar_sensor_angles, angular_vel, distance,
                sin_angle, cos_angle, max_lidar_range, max_angular_vel,
                max_distance, prev_actions[i],
                lidar_downsample_bins, use_sector_processing=True,
                lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y,
                obs_noise_distance_std=obs_noise_distance_std,
                obs_noise_angle_std=obs_noise_angle_std,
                obs_noise_angular_vel_std=obs_noise_angular_vel_std,
                obs_noise_lidar_std=obs_noise_lidar_std
            )
        else:
            obs_actor = obs_critic[:47]  # First 47 = actor base (clean)
        actor_observations.append(obs_actor)
    
    critic_observations = np.array(critic_observations)
    actor_observations = np.array(actor_observations)
    distances = np.array([get_target_info(robot_positions[i], target_positions[i], robot_quats[i])[0] 
                          for i in range(batch_size)])
    
    return critic_observations, actor_observations, robot_positions, robot_quats, distances


def run_batched_episodes_mjx_full(mjx_model, mjx_step_fn, batch_size, spawn_generator,
                                  agent, replay_buffer, m, lidar_sensor_ids, lidar_sensor_angles,
                                  target_body_id, target_mocap_id, reward_weights,
                                  config, args, max_steps=2000, train=True,
                                  critic_critical_topk=0, critic_history_length=0):
    """
    Run a full batch of parallel episodes using MJX and collect all experiences.
    
    Args:
        mjx_model: MJX model
        mjx_step_fn: JIT-compiled step function
        batch_size: number of parallel environments
        spawn_generator: spawn point generator
        agent: SAC agent
        replay_buffer: replay buffer
        m: MuJoCo model
        lidar_sensor_ids: lidar sensor IDs
        lidar_sensor_angles: numpy array углов сенсоров в радианах [n_sensors]
        target_body_id: target body ID
        target_mocap_id: target mocap ID
        reward_weights: reward weights dict
        config: configuration dict
        args: command line arguments
        max_steps: maximum steps per episode
        train: whether in training mode
    
    Returns:
        episode_results: list of dicts with episode statistics
    """
    # Normalization parameters
    max_lidar_range = 3.0
    max_vx = config["cmd_scale"][0]
    max_vy = config["cmd_scale"][1]
    max_angular_vel = config["cmd_scale"][2]
    lidar_config = config.get("lidar", {})
    lidar_offset_x = lidar_config.get("offset_x", 0.12)
    lidar_offset_y = lidar_config.get("offset_y", 0.0)
    room_width = ROOM_X_MAX - ROOM_X_MIN
    room_height = ROOM_Y_MAX - ROOM_Y_MIN
    max_distance = np.sqrt(room_width**2 + room_height**2)
    lidar_downsample_bins = 40
    
    # Initialize batch of episodes
    # Try to create MJX data - this may fail if CUDA compilation is not available
    try:
        mjx_data, target_positions = initialize_batch_episodes_mjx(
            mjx_model, batch_size, spawn_generator, m,
            target_body_id, target_mocap_id
        )
    except Exception as e:
        error_msg = str(e)
        if "ptxas" in error_msg.lower() or "FAILED_PRECONDITION" in error_msg:
            raise RuntimeError(
                "MJX CUDA compilation failed: CUDA toolkit (ptxas) not found.\n"
                "Solutions:\n"
                "  1. Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit\n"
                "  2. Or use CPU mode: remove --use_mjx flag\n"
                "  3. Or set JAX to use CPU: export JAX_PLATFORMS=cpu"
            ) from e
        raise
    
    # Stabilize all episodes (a few steps)
    for _ in range(10):
        mjx_data = mjx_step_fn(mjx_data)
    
    # Track episode states
    episode_dones = np.zeros(batch_size, dtype=bool)
    episode_rewards = np.zeros(batch_size)
    episode_step_counts = np.zeros(batch_size, dtype=int)
    prev_distances = np.zeros(batch_size)
    
    # Store current transition data
    prev_actions = np.zeros((batch_size, 3))
    prev_rewards = np.zeros(batch_size)
    prev_dones = np.zeros(batch_size, dtype=bool)
    prev_successes = np.zeros(batch_size, dtype=bool)
    
    obs_noise = config.get("observation_noise", {})
    obs_noise_dist = obs_noise.get("distance_std", 0.0)
    obs_noise_angle = obs_noise.get("angle_std", 0.0)
    obs_noise_angvel = obs_noise.get("angular_vel_std", 0.0)
    obs_noise_lidar = obs_noise.get("lidar_std", 0.0)
    
    # Initialize prev_distances (extended observations for critic)
    critic_obs_init, actor_obs_init, robot_positions_init, robot_quats_init, distances_init = extract_observations_from_batch(
        mjx_data, lidar_sensor_ids, lidar_sensor_angles, m, batch_size, target_positions,
        prev_actions, max_lidar_range, max_vx, max_vy, max_angular_vel, max_distance, lidar_downsample_bins,
        critic_critical_topk=critic_critical_topk,
        lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y,
        obs_noise_distance_std=obs_noise_dist, obs_noise_angle_std=obs_noise_angle,
        obs_noise_angular_vel_std=obs_noise_angvel, obs_noise_lidar_std=obs_noise_lidar,
        add_actor_noise=train
    )
    prev_distances = distances_init.copy()
    # Replace any NaN/Inf with default value
    prev_distances = np.where(np.isfinite(prev_distances), prev_distances, 5.0)
    
    # Storage for observations, actions and histories
    prev_observations = [None] * batch_size
    from collections import deque
    # Actor: история actions (не observations!)
    actor_history_length = config.get('sac', {}).get('history_length', 0)  # История actions для Actor
    actor_action_histories = [deque(maxlen=actor_history_length) for _ in range(batch_size)]  # История actions
    # Critic: история наблюдений
    critic_histories = [deque(maxlen=critic_history_length + 1) for _ in range(batch_size)]
    # Stacked observations for transition storage
    prev_stacked_critic_obs = [None] * batch_size
    
    episode_experiences = [[] for _ in range(batch_size)]
    
    # Main batched simulation loop
    for step in range(max_steps):
        # Check if all episodes are done
        if np.all(episode_dones):
            break
        
        # Reset histories for done episodes (новый эпизод начинается при инициализации)
        # История сбрасывается при initialize_batch_episodes_mjx, здесь только для done эпизодов
        for i in range(batch_size):
            if episode_dones[i]:
                # Сброс истории actions для Actor (эпизод завершен)
                if actor_history_length > 0:
                    actor_action_histories[i].clear()
                # Сброс истории наблюдений для Critic (эпизод завершен)
                if critic_history_length > 0:
                    critic_histories[i].clear()
        
        # Extract observations: critic (clean) for replay, actor (noisy when train) for get_action
        critic_observations, actor_observations, robot_positions, robot_quats, distances = extract_observations_from_batch(
            mjx_data, lidar_sensor_ids, lidar_sensor_angles, m, batch_size, target_positions,
            prev_actions, max_lidar_range, max_vx, max_vy, max_angular_vel, max_distance, lidar_downsample_bins,
            critic_critical_topk=critic_critical_topk,
            lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y,
            obs_noise_distance_std=obs_noise_dist, obs_noise_angle_std=obs_noise_angle,
            obs_noise_angular_vel_std=obs_noise_angvel, obs_noise_lidar_std=obs_noise_lidar,
            add_actor_noise=train
        )
        
        actor_state_dim = 47
        
        current_stacked_actor_obs = [None] * batch_size
        current_stacked_critic_obs = [None] * batch_size
        
        for i in range(batch_size):
            if not episode_dones[i]:
                actor_obs_single = actor_observations[i]
                # Добавляем историю actions к наблюдению Actor
                current_stacked_actor_obs[i] = agent.process_observation(
                    actor_obs_single, is_critic=False, action_history_buffer=actor_action_histories[i]
                )
                
                current_stacked_critic_obs[i] = agent.process_observation(
                    critic_observations[i], is_critic=True, history_buffer=critic_histories[i]
                )
        
        # 2. Get actions from policy for all active episodes using stacked actor observations
        actions = np.zeros((batch_size, 3))
        for i in range(batch_size):
            if not episode_dones[i]:
                if train:
                    action = agent.get_action(current_stacked_actor_obs[i], add_noise=True)
                else:
                    action = agent.get_action(current_stacked_actor_obs[i], add_noise=False)
                actions[i] = action
                # Обновляем историю actions для Actor (после получения action)
                if actor_history_length > 0:
                    actor_action_histories[i].append(action.copy())
        
        # ВЕКТОРИЗАЦИЯ: Вычислить награды для всех активных эпизодов за один вызов
        active_mask = ~episode_dones  # Маска активных эпизодов
        num_active = np.sum(active_mask)
        
        if num_active > 0:
            # Собрать данные для всех активных эпизодов в батчи
            active_indices = np.where(active_mask)[0]
            
            # Подготовить батчи для векторной функции наград
            batch_robot_pos = robot_positions[active_indices]  # [num_active, 3]
            batch_target_pos = target_positions[active_indices]  # [num_active, 3]
            batch_robot_quat = robot_quats[active_indices]  # [num_active, 4]
            batch_prev_dist = prev_distances[active_indices]  # [num_active]
            
            # Извлечь lidar данные для активных эпизодов
            sensordata_array = np.array(mjx_data.sensordata)  # [batch_size, n_sensors]
            batch_lidar_data_raw = sensordata_array[active_indices][:, lidar_sensor_ids]
            batch_lidar_data_raw = fix_negative_lidar_values(batch_lidar_data_raw)
            # Lidar stays clean for reward/collision. Noise only in Actor obs.
            
            # Извлечь глобальную скорость для активных эпизодов
            qvel_array = np.array(mjx_data.qvel)  # [batch_size, nq]
            batch_global_velocity = qvel_array[active_indices, 0:3]  # [num_active, 3]
            
            # Команды скорости для активных эпизодов
            batch_vx_cmd = actions[active_indices, 0] * config["cmd_scale"][0]  # [num_active]
            batch_vy_cmd = actions[active_indices, 1] * config["cmd_scale"][1]  # [num_active]
            
            # Transform lidar to center frame for collision and reward
            batch_lidar_sectors = process_lidar_to_sectors(
                batch_lidar_data_raw, lidar_sensor_angles,
                num_sectors=40, max_range=max_lidar_range, min_range=0.25
            )
            batch_lidar_center = transform_lidar_to_center_frame(
                batch_lidar_sectors, lidar_sensor_angles,
                lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y,
                max_range=max_lidar_range, min_range=0.25
            )
            
            # ВЫЗОВ ФУНКЦИИ НАГРАД (Vectorized Reference Style)
            # Check for goal and collision (using center-frame lidar)
            diff = batch_target_pos[:, :2] - batch_robot_pos[:, :2]
            dists = np.linalg.norm(diff, axis=1)
            
            goals = dists < 0.25
            collision_threshold = reward_weights.get('collision_threshold', 0.35)
            collisions = np.min(batch_lidar_center, axis=1) < collision_threshold
            
            # Use vectorized reference reward logic
            batch_rewards, batch_dones_from_reward, batch_reward_info = compute_reward_reference_vectorized(
                robot_pos=batch_robot_pos,
                target_pos=batch_target_pos,
                lidar_data=batch_lidar_center,
                actions=prev_actions[active_indices],
                goal=goals,
                collision=collisions,
                reward_weights=reward_weights,
                step_count=episode_step_counts[active_indices],
                max_steps=max_steps,
                prev_distance=batch_prev_dist
            )
            
            for idx, i in enumerate(active_indices):
                reward = float(batch_rewards[idx])
                done = bool(batch_dones_from_reward[idx])
                goal = bool(goals[idx])
                
                # Check timeout
                if episode_step_counts[i] >= max_steps:
                    done = True
                
                episode_rewards[i] += reward
                
                # Store experience (Correct transition: s_t, a_t, r_t, s_t+1, d_t)
                if prev_stacked_critic_obs[i] is not None and train:
                    replay_buffer.add(
                        prev_stacked_critic_obs[i], 
                        prev_actions[i], 
                        reward, 
                        float(done), 
                        current_stacked_critic_obs[i], 
                        success=goal
                    )
                
                # Save current for next transition
                prev_stacked_critic_obs[i] = current_stacked_critic_obs[i].copy()
                prev_actions[i] = actions[i].copy()
                prev_rewards[i] = reward
                prev_dones[i] = done
                prev_successes[i] = goal
                
                if done:
                    episode_dones[i] = True
                    episode_step_counts[i] = step + 1
                    # Сброс истории для завершенного эпизода
                    if actor_history_length > 0:
                        actor_action_histories[i].clear()
                    if critic_history_length > 0:
                        critic_histories[i].clear()
                    prev_stacked_critic_obs[i] = None
            
            # Update prev_distances for all active episodes
            prev_distances[active_indices] = dists
        
        # Apply actions to control (simplified - in full version would use walking policy)
        # For now, just apply direct velocity control
        # In real implementation, this would involve the walking policy
        
        # Step all environments
        mjx_data = mjx_step_fn(mjx_data)
    
    # Compile results
    episode_results = []
    for i in range(batch_size):
        episode_results.append({
            'reward': float(episode_rewards[i]),
            'steps': int(episode_step_counts[i]),
            'done': bool(episode_dones[i])
        })
    
    # Очистка промежуточных переменных для освобождения памяти
    # (особенно важно для GPU, чтобы не накапливать данные)
    del mjx_data, actions, episode_rewards, episode_step_counts, episode_dones
    if 'prev_observations' in locals():
        del prev_observations
    
    return episode_results
