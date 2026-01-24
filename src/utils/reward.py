"""
Векторизованная функция наград для path planning.
Поддерживает как одиночные значения, так и батчи для параллельной обработки.
Все компоненты наград нормализованы для лучшей сбалансированности.
"""
import numpy as np
from utils.target_generator import get_target_info


def compute_reward_vectorized(robot_pos, target_pos, prev_distance, lidar_data, 
                              vx_cmd, vy_cmd, reward_weights, 
                              robot_quat=None, global_velocity=None, step_count=None, max_steps=None):
    """
    Векторизованная функция наград для path planning.
    Поддерживает батчи данных для параллельной обработки (MJX, JAX).
    Все компоненты нормализованы к диапазону [-1, 1] или [0, 1].
    
    Args:
        robot_pos: текущая позиция робота [batch_size, 3] или [3]
        target_pos: позиция цели [batch_size, 3] или [3]
        prev_distance: предыдущее расстояние [batch_size] или scalar
        lidar_data: данные лидара [batch_size, n_beams] или [n_beams]
        vx_cmd: команда скорости X [batch_size] или scalar
        vy_cmd: команда скорости Y [batch_size] или scalar
        reward_weights: dict с весами наград (используются для масштабирования)
        robot_quat: кватернион ориентации [batch_size, 4] или [4] (optional)
        global_velocity: глобальная скорость [batch_size, 3] или [3] (optional)
        step_count: текущее количество шагов [batch_size] или scalar (optional, для расчета time_penalty)
        max_steps: максимальное количество шагов на эпизод [batch_size] или scalar (optional, для нормализации time_penalty)
    
    Returns:
        rewards: вычисленные награды (нормализованные, сумма компонентов) [batch_size] или scalar
        dones: флаги завершения эпизода [batch_size] или bool
        distances: текущие расстояния [batch_size] or scalar
        reward_info: dict с нормализованными компонентами наград
    """
    # Определяем, работаем ли с батчем или одиночными значениями
    is_batch = isinstance(robot_pos, np.ndarray) and robot_pos.ndim > 1
    
    # Получаем batch_size заранее для использования в обработке step_count и max_steps
    if robot_pos.ndim > 1:
        batch_size = robot_pos.shape[0]
    else:
        batch_size = 1
    
    if not is_batch:
        # Конвертируем скаляры в массивы для единообразия
        robot_pos = np.array(robot_pos).reshape(1, -1) if not isinstance(robot_pos, np.ndarray) else robot_pos.reshape(1, -1)
        target_pos = np.array(target_pos).reshape(1, -1) if not isinstance(target_pos, np.ndarray) else target_pos.reshape(1, -1)
        prev_distance = np.array([prev_distance]) if not isinstance(prev_distance, np.ndarray) else prev_distance
        lidar_data = np.array(lidar_data).reshape(1, -1) if not isinstance(lidar_data, np.ndarray) else lidar_data.reshape(1, -1) if lidar_data.ndim == 1 else lidar_data
        vx_cmd = np.array([vx_cmd]) if not isinstance(vx_cmd, np.ndarray) else vx_cmd
        vy_cmd = np.array([vy_cmd]) if not isinstance(vy_cmd, np.ndarray) else vy_cmd
        if robot_quat is not None:
            robot_quat = np.array(robot_quat).reshape(1, -1) if not isinstance(robot_quat, np.ndarray) else robot_quat.reshape(1, -1) if robot_quat.ndim == 1 else robot_quat
        if global_velocity is not None:
            global_velocity = np.array(global_velocity).reshape(1, -1) if not isinstance(global_velocity, np.ndarray) else global_velocity.reshape(1, -1) if global_velocity.ndim == 1 else global_velocity
        if step_count is not None:
            step_count = np.array([step_count]) if not isinstance(step_count, np.ndarray) else step_count.reshape(-1) if step_count.ndim == 0 else step_count
        if max_steps is not None:
            max_steps = np.array([max_steps]) if not isinstance(max_steps, np.ndarray) else max_steps.reshape(-1) if max_steps.ndim == 0 else max_steps
    else:
        # Для batch режима также обрабатываем step_count и max_steps
        if step_count is not None:
            step_count = np.asarray(step_count)
            if step_count.ndim == 0:
                step_count = np.full(batch_size, step_count.item())
        if max_steps is not None:
            max_steps = np.asarray(max_steps)
            if max_steps.ndim == 0:
                max_steps = np.full(batch_size, max_steps.item())
    
    # Дополнительная обработка для единообразия
    if robot_pos.ndim == 1:
        robot_pos = robot_pos.reshape(1, -1)
        target_pos = target_pos.reshape(1, -1)
        prev_distance = np.array([prev_distance]) if np.isscalar(prev_distance) else prev_distance.reshape(-1)
        if lidar_data.ndim == 1:
            lidar_data = lidar_data.reshape(1, -1)
        if np.isscalar(vx_cmd):
            vx_cmd = np.array([vx_cmd])
        if np.isscalar(vy_cmd):
            vy_cmd = np.array([vy_cmd])
    
    # Убеждаемся, что все имеют правильную размерность
    if prev_distance.ndim == 0:
        prev_distance = np.array([prev_distance])
    if vx_cmd.ndim == 0:
        vx_cmd = np.array([vx_cmd])
    if vy_cmd.ndim == 0:
        vy_cmd = np.array([vy_cmd])
    
    # 1. Вычисляем расстояние до цели (векторизовано)
    # OPTIMIZATION: Vectorize distance calculation instead of loop
    diff = target_pos[:, :2] - robot_pos[:, :2]  # [batch_size, 2] - только x, y координаты
    distances = np.sqrt(np.sum(diff**2, axis=1))  # [batch_size]
    
    # Обработка NaN/Inf: заменяем на предыдущее расстояние или дефолт
    prev_distance_array = np.broadcast_to(prev_distance, distances.shape).copy()
    invalid_mask = np.isnan(distances) | np.isinf(distances)
    distances = np.where(invalid_mask, 
                        np.where(np.isfinite(prev_distance_array), prev_distance_array, 5.0),
                        distances)
    
    prev_distance_array = np.where(np.isnan(prev_distance_array) | np.isinf(prev_distance_array),
                                  distances, prev_distance_array)
    
    # 2. Награда за достижение цели - нормализована к [0, 1]
    reached_threshold = reward_weights['reached_threshold']
    reached_mask = distances < reached_threshold
    reached_rewards_normalized = np.where(reached_mask, 1.0, 0.0)  # [0, 1]
    dones = reached_mask
    
    # 3. Штраф за препятствия - ЭКСПОНЕНЦИАЛЬНАЯ ФУНКЦИЯ: exp(scale*(min_lidar - 1))
    # Векторизованное вычисление min_lidar для каждого батча
    if lidar_data.ndim > 1:
        min_lidar = np.min(lidar_data, axis=1)  # [batch_size]
    else:
        min_lidar = np.array([np.min(lidar_data)])
    
    # Обработка NaN/Inf в лидаре
    min_lidar = np.nan_to_num(min_lidar, nan=3.0, posinf=3.0, neginf=3.0)
    
    exponential_scale = reward_weights.get('obstacle_exponential_scale', 4.0)
    obstacle_threshold = reward_weights.get('obstacle_threshold', 1.2)
    
    # Используем логарифм для нормализации (избегаем переполнения)
    min_lidar_safe = np.maximum(min_lidar, 0.0)
    log_exp_raw = np.where(
        min_lidar < obstacle_threshold,
        exponential_scale * (obstacle_threshold - min_lidar_safe) / obstacle_threshold,
        0.0
    )
    
    # Граничные значения в логарифмическом пространстве
    log_exp_at_min = exponential_scale * (obstacle_threshold - 0.0) / obstacle_threshold
    log_exp_at_threshold = 0.0
    
    obstacle_exp_normalized = np.where(
        min_lidar < obstacle_threshold,
        (log_exp_raw - log_exp_at_threshold) / (log_exp_at_min - log_exp_at_threshold),
        0.0
    )
    
    obstacle_exp_normalized = np.clip(obstacle_exp_normalized, 0.0, 1.0)
    obstacle_penalties_normalized = -obstacle_exp_normalized  # [-1, 0]
    
    # 4. Награда за прогресс - нормализована к [-1, 1]
    progress = prev_distance_array - distances
    max_progress_per_step = 0.5
    progress_clipped = np.clip(progress, -max_progress_per_step, max_progress_per_step)
    progress_normalized = progress_clipped / max_progress_per_step
    
    distance_multiplier = np.where(
        distances > 0.01,
        1.0 + 1.5 / (1.0 + distances),
        1.0
    )
    
    progress_rewards_normalized = np.where(
        progress_normalized > 0,
        np.clip(progress_normalized * distance_multiplier, 0.0, 2.5),
        progress_normalized
    )
    
    # 5. Штраф за боковую скорость - нормализован к [-1, 0]
    max_vy = 1.3
    vy_normalized = np.clip(vy_cmd / max_vy, -1.0, 1.0)
    vy_penalties_normalized = -np.abs(vy_normalized)
    
    # 6. Штраф за движение назад - нормализован к [-1, 0]
    max_vx = 1.8
    vx_normalized = np.clip(vx_cmd / max_vx, -1.0, 1.0)
    vx_backward_penalties_normalized = np.minimum(0.0, vx_normalized)
    
    # 7. Награда/штраф за выравнивание скорости - нормализована к [-1, 1]
    velocity_alignment_rewards_normalized = np.zeros(batch_size)
    if robot_quat is not None and global_velocity is not None:
        alignment_weight = reward_weights.get('velocity_alignment', 0.0)
        velocity_threshold = reward_weights.get('velocity_alignment_threshold', 0.1)
            
        qw = robot_quat[:, 0]
        qx = robot_quat[:, 1]
        qy = robot_quat[:, 2]
        qz = robot_quat[:, 3]
        robot_yaws = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        
        vx_global = global_velocity[:, 0]
        vy_global = global_velocity[:, 1]
        velocity_magnitudes = np.sqrt(vx_global**2 + vy_global**2)
        velocity_yaws = np.arctan2(vy_global, vx_global)
        
        angle_diffs = robot_yaws - velocity_yaws
        angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))
        abs_angle_diffs = np.abs(angle_diffs)
        alignment_factors = (np.cos(abs_angle_diffs) + 1.0) / 2.0
        
        max_velocity = 2.0
        velocity_normalized = np.clip(velocity_magnitudes / max_velocity, 0.0, 1.0)
        moving_mask = velocity_magnitudes > velocity_threshold
        
        if alignment_weight != 0.0:
            alignment_normalized = 2.0 * alignment_factors - 1.0
            velocity_alignment_rewards_normalized = np.where(
                moving_mask,
                alignment_normalized * velocity_normalized,
                0.0
            )
    
    # 8. Суммируем нормализованные компоненты с весами
    weight_reached = reward_weights.get('reached', 100.0) / 100.0
    weight_obstacle = abs(reward_weights.get('obstacle', -12.0))
    weight_progress = reward_weights.get('progress', 35.0)
    weight_vy = abs(reward_weights.get('vy_penalty', -0.5))
    weight_vx_backward = abs(reward_weights.get('vx_backward_penalty', -0.5))
    weight_alignment = abs(reward_weights.get('velocity_alignment', 2.0)) / 2.0
    weight_time = reward_weights.get('time_penalty', 0.0)
    
    if step_count is not None and max_steps is not None:
        step_count_arr = np.asarray(step_count)
        max_steps_arr = np.asarray(max_steps)
        if step_count_arr.ndim == 0:
            step_count_arr = np.full(batch_size, step_count_arr.item())
        if max_steps_arr.ndim == 0:
            max_steps_arr = np.full(batch_size, max_steps_arr.item())
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_steps = np.where(max_steps_arr > 0, step_count_arr / max_steps_arr, 0.0)
        time_penalty_value = abs(weight_time) * normalized_steps
    elif step_count is not None:
        step_count_arr = np.asarray(step_count)
        if step_count_arr.ndim == 0:
            step_count_arr = np.full(batch_size, step_count_arr.item())
        time_penalty_value = abs(weight_time) * step_count_arr / 10000.0
    else:
        time_penalty_value = np.full(batch_size, abs(weight_time)) if is_batch else abs(weight_time)
    
    total_rewards = (
        weight_reached * reached_rewards_normalized +
        weight_obstacle * obstacle_penalties_normalized +
        weight_progress * progress_rewards_normalized +
        weight_vy * vy_penalties_normalized +
        weight_vx_backward * vx_backward_penalties_normalized +
        weight_alignment * velocity_alignment_rewards_normalized -
        time_penalty_value
    )
    
    total_rewards = np.nan_to_num(total_rewards, nan=-0.1, posinf=-0.1, neginf=-0.1)
    
    reward_info = {
        'total': total_rewards,
        'reached': reached_rewards_normalized,
        'obstacle': obstacle_penalties_normalized,
        'progress': progress_rewards_normalized,
        'vy_penalty': vy_penalties_normalized,
        'vx_backward_penalty': vx_backward_penalties_normalized,
        'velocity_alignment': velocity_alignment_rewards_normalized,
        'distance_to_target': distances,
        'time_penalty': -time_penalty_value
    }
    
    if not is_batch:
        return (
            float(total_rewards[0]), 
            bool(dones[0]), 
            float(distances[0]),
            {k: float(v[0]) if isinstance(v, np.ndarray) else v for k, v in reward_info.items()}
        )
    else:
        return total_rewards, dones, distances, reward_info


def compute_reward(robot_pos, target_pos, prev_distance, lidar_data, vx_cmd, vy_cmd, 
                  reward_weights, robot_quat=None, global_velocity=None, step_count=None, max_steps=None):
    return compute_reward_vectorized(
        robot_pos, target_pos, prev_distance, lidar_data, vx_cmd, vy_cmd,
        reward_weights, robot_quat, global_velocity, step_count, max_steps
    )


def compute_reward_reference_vectorized(robot_pos, target_pos, lidar_data, actions, goal, collision, reward_weights, step_count=None, max_steps=None, prev_distance=None):
    """
    Vectorized version of strictly following the reference project's reward logic.
    Supports both batch and single values.
    
    Args:
        prev_distance: previous distance to target for progress calculation (optional)
    """
    is_batch = isinstance(robot_pos, np.ndarray) and robot_pos.ndim > 1
    
    if not is_batch:
        # Convert to batch of size 1 for uniform processing
        robot_pos = np.atleast_2d(robot_pos)
        target_pos = np.atleast_2d(target_pos)
        lidar_data = np.atleast_2d(lidar_data)
        actions = np.atleast_2d(actions)
        goal = np.atleast_1d(goal)
        collision = np.atleast_1d(collision)
        if step_count is not None:
            step_count = np.atleast_1d(step_count)
        if max_steps is not None:
            max_steps = np.atleast_1d(max_steps)
        if prev_distance is not None:
            prev_distance = np.atleast_1d(prev_distance)
            
    batch_size = robot_pos.shape[0]
    
    # Calculate current distance to target
    diff = target_pos[:, :2] - robot_pos[:, :2]  # only x, y coordinates
    current_distance = np.sqrt(np.sum(diff**2, axis=1))
    
    # Initialize reward components
    reward_info = {
        'total': np.zeros(batch_size),
        'reached': np.zeros(batch_size),
        'obstacle': np.zeros(batch_size),
        'progress': np.zeros(batch_size),
        'vy_penalty': np.zeros(batch_size),
        'vx_backward_penalty': np.zeros(batch_size),
        'velocity_alignment': np.zeros(batch_size),
        'time_penalty': np.zeros(batch_size)
    }
    
    # Calculate time penalty (increases with steps)
    time_penalty = np.zeros(batch_size)
    if step_count is not None and max_steps is not None:
        weight_time = reward_weights.get('time_penalty', -0.02)
        # Penalty grows linearly from 0 to weight_time
        time_penalty = abs(weight_time) * (step_count / max_steps)
        reward_info['time_penalty'] = -time_penalty
    
    # Reference logic components
    # Filter invalid lidar values (NaN, Inf, negative, too close)
    # IMPORTANT: Filter out values < 0.25m (robot body) to avoid false positives
    min_range = 0.25  # Minimum valid range (robot body radius)
    max_range = 3.0   # Maximum lidar range
    
    # Clean lidar data: filter invalid values and values too close (robot body)
    lidar_data_clean = np.where(
        (lidar_data >= min_range) & (lidar_data <= max_range) & np.isfinite(lidar_data),
        lidar_data,
        max_range  # Replace invalid/too-close with max range (no obstacle detected)
    )
    
    # Get minimum distance (closest obstacle)
    min_dist = np.min(lidar_data_clean, axis=1)
    
    # Exponential obstacle penalty formula:
    # If min_dist >= obstacle_threshold: penalty = 0
    # If min_dist from obstacle_threshold to death_distance: 
    #   penalty = -exp(k * (obstacle_threshold - x) / (x - death_distance))
    #   where k = obstacle_exponential_scale, x = min_dist
    # If min_dist <= death_distance: penalty = MAXIMUM (not 0!)
    obstacle_threshold = reward_weights.get('obstacle_threshold', 1.5)
    death_distance = 0.35  # Distance where collision occurs
    exponential_scale = reward_weights.get('obstacle_exponential_scale', 4.0)
    
    # Calculate exponential penalty for distances between threshold and death distance
    # Formula: -exp(k * (threshold - x) / (x - death_distance))
    # This creates a sharp exponential increase as we approach death distance
    max_arg_value = 10.0  # exp(10) ≈ 22,000 - reasonable maximum penalty
    
    # Three zones:
    # 1. Safe zone (x >= threshold): penalty = 0
    # 2. Danger zone (death_distance < x < threshold): penalty grows exponentially
    # 3. Death zone (x <= death_distance): penalty = MAXIMUM
    
    mask_safe = min_dist >= obstacle_threshold
    mask_danger = (min_dist < obstacle_threshold) & (min_dist > death_distance)
    mask_death = min_dist <= death_distance
    
    # Calculate argument for exponential in danger zone
    arg_exp = np.where(
        mask_danger,
        exponential_scale * (obstacle_threshold - min_dist) / np.maximum(min_dist - death_distance, 1e-6),
        0.0
    )
    # Clip argument to prevent overflow
    arg_exp = np.clip(arg_exp, 0.0, max_arg_value)
    
    # Calculate normalized exponential
    max_penalty_value = np.exp(max_arg_value)
    normalized_exp = np.where(
        mask_safe,
        0.0,  # Safe: no penalty
        np.where(
            mask_danger,
            np.exp(arg_exp) / max_penalty_value,  # Danger: exponential penalty
            1.0  # Death: maximum penalty (NOT zero!)
        )
    )
    
    # Apply weight and make negative (this is a penalty!)
    obs_weight = reward_weights.get('obs_penalty_weight', 2.5)
    obstacle_penalty = -normalized_exp * obs_weight
    
    vx = actions[:, 0]
    vy = actions[:, 1]
    w = actions[:, 2]
    
    # w_penalty_weight уже содержит знак (отрицательный = штраф за повороты)
    # Используем его напрямую: отрицательное значение = штраф, положительное = награда
    w_weight = reward_weights.get('w_penalty_weight', -0.5)
    
    # Progress reward: distance improvement
    progress_reward = np.zeros(batch_size)
    if prev_distance is not None:
        progress_diff = prev_distance - current_distance
        progress_weight = reward_weights.get('progress', 10.0)
        progress_reward = progress_diff * progress_weight
        reward_info['progress'] = progress_reward
    
    # Backward penalty: penalize negative vx
    vx_backward_penalty = np.zeros(batch_size)
    vx_backward_weight = abs(reward_weights.get('vx_backward_penalty', -0.5))
    vx_backward_penalty = np.where(vx < 0, vx * vx_backward_weight, 0.0)
    reward_info['vx_backward_penalty'] = vx_backward_penalty
    
    # W penalty: w_weight уже содержит знак (отрицательный = штраф)
    # Формула: w_weight * |w|, где w_weight отрицательный = штраф за повороты
    # TEMPORARILY DISABLED: w_penalty disabled for training experiment
    w_penalty = w_weight * np.abs(w)
    reward_info['w_penalty'] = w_penalty
    
    # Main formula: vx reward + progress - penalties
    # TEMPORARILY DISABLED: w_penalty removed from reward calculation
    rewards = vx + progress_reward + obstacle_penalty + vx_backward_penalty - time_penalty  # w_penalty removed
    
    # Override for goal and collision
    rewards = np.where(goal, 100.0 - time_penalty, rewards)
    rewards = np.where(collision, -100.0 - time_penalty, rewards)
    
    # Fill info
    reward_info['total'] = rewards
    reward_info['reached'] = np.where(goal, 1.0, 0.0)
    # obstacle_penalty is already negative, so use it directly (not -obstacle_penalty)
    reward_info['obstacle'] = np.where(collision, -1.0, obstacle_penalty)
    reward_info['vy_penalty'] = -np.abs(vy)
    
    if not is_batch:
        return (
            float(rewards[0]),
            bool(goal[0] or collision[0]),
            {k: float(v[0]) for k, v in reward_info.items()}
        )
    
    return rewards, (goal | collision), reward_info


def compute_reward_reference(robot_pos, target_pos, lidar_data, actions, goal, collision, reward_weights, step_count=None, max_steps=None, prev_distance=None):
    """
    Backwards compatible wrapper for vectorized reference reward.
    """
    return compute_reward_reference_vectorized(
        robot_pos, target_pos, lidar_data, actions, goal, collision, reward_weights, step_count, max_steps, prev_distance
    )
