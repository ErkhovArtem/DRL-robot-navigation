"""
Observation building functions for Actor and Critic.
"""
import numpy as np


def get_gravity_orientation(quaternion):
    """Calculate gravity orientation from quaternion."""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def build_walking_policy_observation(
    base_ang_vel,      # [3] angular velocity in base frame
    projected_gravity, # [3] projected gravity vector
    velocity_commands, # [3] velocity commands (vx, vy, w)
    joint_pos,         # [num_actions] joint positions (relative to default)
    joint_vel,         # [num_actions] joint velocities
    last_action,       # [num_actions] last action from walking policy
    base_ang_vel_scale=0.25,
    joint_pos_scale=1.0,
    joint_vel_scale=0.05,
):
    """
    Build observation for walking policy according to PolicyCfg structure.
    
    Observation order (matching PolicyCfg):
    2. base_ang_vel (3) - angular velocity in base frame  
    3. projected_gravity (3) - projected gravity vector
    4. velocity_commands (3) - velocity commands (vx, vy, w)
    5. joint_pos (num_actions) - joint positions (relative to default)
    6. joint_vel (num_actions) - joint velocities
    7. actions (num_actions) - last action from walking policy
    8. height_scan (N) - height scan data (optional)
    
    Returns:
        observation array with proper scaling and clipping
    """
    obs_parts = []
    
    # 2. base_ang_vel (3) - clip and scale
    base_ang_vel_scaled = np.clip(base_ang_vel * base_ang_vel_scale, -100.0, 100.0)
    obs_parts.append(base_ang_vel_scaled)
    
    # 3. projected_gravity (3) - already normalized, just clip
    projected_gravity_clipped = np.clip(projected_gravity, -100.0, 100.0)
    obs_parts.append(projected_gravity_clipped)
    
    # 4. velocity_commands (3) - clip
    velocity_commands_clipped = np.clip(velocity_commands, -100.0, 100.0)
    obs_parts.append(velocity_commands_clipped)
    
    # 5. joint_pos (num_actions) - clip and scale
    joint_pos_scaled = np.clip(joint_pos * joint_pos_scale, -100.0, 100.0)
    obs_parts.append(joint_pos_scaled)
    
    # 6. joint_vel (num_actions) - clip and scale
    joint_vel_scaled = np.clip(joint_vel * joint_vel_scale, -100.0, 100.0)
    obs_parts.append(joint_vel_scaled)
    
    # 7. actions (num_actions) - last action, clip
    last_action_clipped = np.clip(last_action, -100.0, 100.0)
    obs_parts.append(last_action_clipped)
    
    # Concatenate all parts
    observation = np.concatenate(obs_parts, dtype=np.float32)
    
    return observation


def downsample_lidar(lidar_data, max_bins):
    """
    Downsample lidar data by binning and taking minimum value in each bin.
    Similar to prepare_state function.
    
    Args:
        lidar_data: numpy array of lidar readings
        max_bins: maximum number of bins for downsampling
    
    Returns:
        Downsampled lidar data array
    """
    # Handle infinite values
    inf_mask = np.isinf(lidar_data)
    lidar_data = lidar_data.copy()
    lidar_data[inf_mask] = 3.0  # Replace inf with max range (3m visibility)
    
    if len(lidar_data) <= max_bins:
        return lidar_data
    
    bin_size = int(np.ceil(len(lidar_data) / max_bins))
    min_values = []
    
    # Loop through the data and create bins
    for i in range(0, len(lidar_data), bin_size):
        # Get the current bin
        bin_data = lidar_data[i : i + min(bin_size, len(lidar_data) - i)]
        # Find the minimum value in the current bin and append it to the min_values list
        min_values.append(np.min(bin_data))
    
    return np.array(min_values)


def fix_negative_lidar_values(lidar_data, default_value=3.0, max_search_distance=5, max_range=3.0):
    """
    Fix negative, NaN, and Inf lidar values by replacing them with average of neighboring values.
    Searches for valid neighbors within max_search_distance.
    Supports both single arrays and batches (2D arrays).
    
    Args:
        lidar_data: numpy array of lidar readings [n_beams] or [batch_size, n_beams]
        default_value: default value to use if no valid neighbors found (should be max_range)
        max_search_distance: maximum distance to search for valid neighbors
        max_range: maximum valid lidar range (used for clipping)
    
    Returns:
        Fixed lidar data array (same shape as input) - all values in [0, max_range]
    """
    lidar_data = lidar_data.copy()
    
    # Поддержка батчей: если 2D массив, обрабатываем каждую строку отдельно
    if lidar_data.ndim == 2:
        # Батч: применяем функцию к каждой строке
        for i in range(lidar_data.shape[0]):
            lidar_data[i] = fix_negative_lidar_values(lidar_data[i], default_value, max_search_distance, max_range)
        return lidar_data
    
    # Фильтруем NaN и Inf значения (заменяем на default_value)
    invalid_mask = np.isnan(lidar_data) | np.isinf(lidar_data) | (lidar_data < 0)
    invalid_indices = np.where(invalid_mask)[0]
    
    for idx in invalid_indices:
        # Find valid neighbors (positive, finite values) within search distance
        neighbors = []
        
        # Search for previous neighbor (closest valid one)
        for offset in range(1, max_search_distance + 1):
            prev_idx = idx - offset
            if prev_idx >= 0:
                val = lidar_data[prev_idx]
                if val >= 0 and np.isfinite(val):
                    neighbors.append(val)
                break  # Use closest valid neighbor
        
        # Search for next neighbor (closest valid one)
        for offset in range(1, max_search_distance + 1):
            next_idx = idx + offset
            if next_idx < len(lidar_data):
                val = lidar_data[next_idx]
                if val >= 0 and np.isfinite(val):
                    neighbors.append(val)
                break  # Use closest valid neighbor
        
        # If we found neighbors, use their average; otherwise use default
        if len(neighbors) > 0:
            lidar_data[idx] = np.mean(neighbors)
        else:
            lidar_data[idx] = default_value
    
    # Финальная проверка: обрезаем все значения до [0, max_range]
    lidar_data = np.clip(lidar_data, 0.0, max_range)
    
    return lidar_data


def compute_lidar_sensor_angles(m, lidar_sensor_ids, lidar_beam_index_mapping):
    """
    Вычисляет углы для каждого сенсора лидара.
    Углы в радианах, в диапазоне [-π, π] (как в lidar_2d_processor).
    
    Использует вычисление по индексу луча (0-39), так как они равномерно 
    распределены по кругу с шагом 9 градусов.
    
    Args:
        m: MuJoCo model
        lidar_sensor_ids: список ID сенсоров
        lidar_beam_index_mapping: маппинг индекса данных к индексу луча
    
    Returns:
        numpy array углов в радианах [n_sensors]
    """
    angles = []
    sector_angle = 2.0 * np.pi / 40.0  # 9 градусов на сектор
    
    for i, beam_idx in enumerate(lidar_beam_index_mapping):
        # Вычисляем угол луча: сектор 0 начинается с -π
        # Сектор 0: -π, сектор 1: -π + Δ, ..., сектор 39: π - Δ
        # Центр сектора: angle_min + sector_angle/2
        angle_min = -np.pi + beam_idx * sector_angle
        angle_center = angle_min + sector_angle / 2.0
        
        # Нормализуем в диапазон [-π, π] (на всякий случай)
        normalized_angle = np.arctan2(np.sin(angle_center), np.cos(angle_center))
        angles.append(normalized_angle)
    
    return np.array(angles)


def process_lidar_to_sectors(lidar_data, sensor_angles, num_sectors=40, max_range=3.0, min_range=0.25):
    """
    Обрабатывает данные лидара в секторный формат (как в lidar_2d_processor).
    Полностью совместимо с форматом ROS2 lidar_2d_processor.
    Оптимизировано для батчей с векторизацией.
    
    Args:
        lidar_data: numpy array расстояний [n_sensors] или [batch_size, n_sensors]
        sensor_angles: numpy array углов сенсоров в радианах [n_sensors]
        num_sectors: количество секторов (40, как в lidar_2d_processor)
        max_range: максимальная дальность лидара (3.0 м)
        min_range: минимальная дальность лидара (0.25 м, фильтруем ближе)
    
    Returns:
        sector_min_distances: numpy array минимальных расстояний по секторам
            [num_sectors] или [batch_size, num_sectors]
    """
    # Поддержка батчей
    is_batch = lidar_data.ndim == 2
    if is_batch:
        batch_size = lidar_data.shape[0]
        n_sensors = lidar_data.shape[1]
    else:
        lidar_data = lidar_data.reshape(1, -1)
        batch_size = 1
        n_sensors = lidar_data.shape[1]
    
    # Фильтруем значения вне диапазона (как в lidar_2d_processor)
    # Заменяем inf на max_range
    lidar_data = np.where(np.isinf(lidar_data), max_range, lidar_data)
    lidar_data = np.clip(lidar_data, 0, max_range)
    
    # Фильтруем точки слишком близко (как в lidar_2d_processor: min_range=0.25)
    valid_mask = lidar_data >= min_range
    
    # Нормализуем углы в [-π, π] (как в lidar_2d_processor)
    normalized_angles = np.arctan2(np.sin(sensor_angles), np.cos(sensor_angles))
    
    # Угол каждого сектора
    sector_angle = 2 * np.pi / num_sectors
    
    # Вычисляем индексы секторов для всех сенсоров
    # Сектор 0: [-π, -π+Δ), сектор 1: [-π+Δ, -π+2Δ), ..., сектор 39: [π-Δ, π]
    sector_indices = ((normalized_angles + np.pi) / sector_angle).astype(int)
    sector_indices = np.clip(sector_indices, 0, num_sectors - 1)
    
    # Вычисляем границы секторов
    angle_mins = -np.pi + np.arange(num_sectors) * sector_angle
    angle_maxs = angle_mins + sector_angle
    
    # Инициализируем массив минимальных расстояний
    sector_distances = np.full((batch_size, num_sectors), max_range, dtype=lidar_data.dtype)
    
    # Обрабатываем каждый батч
    for batch_idx in range(batch_size):
        for sensor_idx in range(min(n_sensors, len(sensor_angles))):
            if not valid_mask[batch_idx, sensor_idx]:
                continue
            
            distance = lidar_data[batch_idx, sensor_idx]
            angle = normalized_angles[sensor_idx]
            sector_idx = sector_indices[sensor_idx]
            
            # Обработка границ (как в lidar_2d_processor, строки 204-208)
            # Для последнего сектора (39) включаем правую границу π
            if sector_idx == num_sectors - 1:
                # Последний сектор: [π-Δ, π] (включает π)
                if angle >= angle_mins[sector_idx] and angle <= angle_maxs[sector_idx]:
                    if distance < sector_distances[batch_idx, sector_idx]:
                        sector_distances[batch_idx, sector_idx] = distance
            else:
                # Обычные секторы: [angle_min, angle_max)
                if angle >= angle_mins[sector_idx] and angle < angle_maxs[sector_idx]:
                    if distance < sector_distances[batch_idx, sector_idx]:
                        sector_distances[batch_idx, sector_idx] = distance
    
    # Возвращаем в исходном формате (убираем batch размерность если не было)
    if not is_batch and batch_size == 1:
        return sector_distances[0]
    
    return sector_distances


def transform_lidar_to_center_frame(lidar_sectors, sensor_angles, lidar_offset_x=0.12, lidar_offset_y=0.0,
                                    max_range=3.0, min_range=0.25):
    """
    Пересчитывает расстояния лидара так, как будто лидар расположен в центре робота.
    
    Для каждого луча: точка препятствия P = (d*cos(α), d*sin(α)) в frame лидара.
    Центр робота в frame лидара: C = (-lidar_offset_x, -lidar_offset_y).
    Расстояние от центра до препятствия: d_center = sqrt((d*cos(α)+Lx)² + (d*sin(α)+Ly)²).
    
    Args:
        lidar_sectors: numpy array [n_sectors] или [batch_size, n_sectors]
        sensor_angles: numpy array углов в радианах [n_sectors]
        lidar_offset_x: смещение лидара вперёд от центра (м)
        lidar_offset_y: смещение лидара вбок от центра (м)
        max_range: максимальная дальность
        min_range: минимальная валидная дальность
    
    Returns:
        lidar_sectors_center: расстояния от центра робота (тот же shape)
    """
    if lidar_offset_x == 0 and lidar_offset_y == 0:
        return np.array(lidar_sectors, copy=True)
    
    is_batch = lidar_sectors.ndim == 2
    if not is_batch:
        lidar_sectors = np.atleast_2d(lidar_sectors)
    
    n_sectors = lidar_sectors.shape[1]
    angles = np.asarray(sensor_angles)
    if len(angles) != n_sectors:
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        # Extend or truncate if needed
        if len(angles) < n_sectors:
            sector_angle = 2 * np.pi / 40
            angles = np.array([-np.pi + (i + 0.5) * sector_angle for i in range(n_sectors)])
        else:
            angles = angles[:n_sectors]
    
    # Vectorized: for each sector, d_center = sqrt((d*cos(α)+Lx)² + (d*sin(α)+Ly)²)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    # lidar_sectors: [batch, n_sectors]
    dx = lidar_sectors * cos_a + lidar_offset_x  # x from center to obstacle
    dy = lidar_sectors * sin_a + lidar_offset_y  # y from center to obstacle
    d_center = np.sqrt(dx**2 + dy**2)
    
    # Invalid/saturated values stay as-is (max_range)
    valid = (lidar_sectors >= min_range) & (lidar_sectors <= max_range) & np.isfinite(lidar_sectors)
    d_center = np.where(valid, d_center, lidar_sectors)
    
    # Clamp to valid range
    d_center = np.clip(d_center, 0, max_range)
    
    if not is_batch:
        return d_center[0]
    return d_center


def build_actor_observation(lidar_data_raw, sensor_angles, angular_vel, distance,
                           sin_angle, cos_angle, max_lidar_range, max_angular_vel,
                           max_distance, prev_action,
                           lidar_downsample_bins=40, use_sector_processing=True,
                           lidar_offset_x=0.12, lidar_offset_y=0.0,
                           obs_noise_distance_std=0.0, obs_noise_angle_std=0.0, obs_noise_angular_vel_std=0.0,
                           obs_noise_lidar_std=0.0):
    """
    Build normalized observation for ACTOR.
    Total size: lidar(40) + w(1) + sin(1) + cos(1) + dist(1) + prev_actions(3) = 47.
    Policy receives RAW lidar (no center-frame transform). Transform used only for collision/reward.
    Optional observation noise for sim-to-real (only when std > 0). Critic always gets clean data.
    """
    # Apply observation noise (sim-to-real) when std > 0 - only for Actor
    if obs_noise_distance_std > 0:
        distance = distance + np.random.normal(0, obs_noise_distance_std)
        distance = np.maximum(distance, 0.0)
    if obs_noise_angular_vel_std > 0:
        angular_vel = angular_vel + np.random.normal(0, obs_noise_angular_vel_std)
    if obs_noise_angle_std > 0:
        angle = np.arctan2(sin_angle, cos_angle)
        angle = angle + np.random.normal(0, obs_noise_angle_std)
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
    
    if use_sector_processing and sensor_angles is not None:
        lidar_sectors = process_lidar_to_sectors(
            lidar_data_raw, sensor_angles,
            num_sectors=lidar_downsample_bins,
            max_range=max_lidar_range,
            min_range=0.25
        )
    else:
        lidar_sectors = downsample_lidar(lidar_data_raw, lidar_downsample_bins)
    
    # Add lidar noise for Actor only (sim-to-real)
    if obs_noise_lidar_std > 0:
        noise = np.random.normal(0, obs_noise_lidar_std, size=lidar_sectors.shape).astype(np.float32)
        lidar_sectors = lidar_sectors + noise
        lidar_sectors = np.clip(lidar_sectors, 0.0, max_lidar_range)
    
    # Normalize lidar
    lidar_sectors = np.where(
        (lidar_sectors >= 0) & (lidar_sectors <= max_lidar_range) & np.isfinite(lidar_sectors),
        lidar_sectors,
        max_lidar_range
    )
    lidar_data = np.clip(lidar_sectors, 0, max_lidar_range) / max_lidar_range
    lidar_data = lidar_data * 2.0 - 1.0
    
    # Нормализация W (угловая скорость) и distance
    angular_vel_norm = np.clip(angular_vel, -max_angular_vel, max_angular_vel) / max_angular_vel
    distance_norm = np.clip(distance, 0, max_distance) / max_distance
    distance_norm = distance_norm * 2.0 - 1.0
    
    # Safety check for prev_action
    if prev_action is None:
        prev_action = np.zeros(3)
    
    # Concatenate all (47 features): lidar(40) + w(1) + sin(1) + cos(1) + dist(1) + prev(3)
    actor_obs = np.concatenate([
        lidar_data,                        # 40
        [angular_vel_norm],                # 1 - W (угловая скорость) - ЕСТЬ!
        [np.clip(sin_angle, -1, 1)],      # 1
        [np.clip(cos_angle, -1, 1)],      # 1
        [distance_norm],                   # 1
        [np.clip(prev_action[0], -1, 1)], # 1 - prev vx command
        [np.clip(prev_action[1], -1, 1)], # 1 - prev vy command
        [np.clip(prev_action[2], -1, 1)]  # 1 - prev w command
    ])
    
    return actor_obs


def build_critic_base_observation(lidar_data_raw, sensor_angles, vx, vy, angular_vel, distance,
                                  sin_angle, cos_angle, max_lidar_range, max_vx, max_vy,
                                  max_angular_vel, max_distance, prev_action,
                                  lidar_downsample_bins=40, use_sector_processing=True,
                                  lidar_offset_x=0.12, lidar_offset_y=0.0):
    """
    Build normalized observation FOR CRITIC.
    Total size: Actor(47) + vx(1) + vy(1) = 49.
    Critic ALWAYS gets clean data (no observation noise).
    """
    actor_obs = build_actor_observation(
        lidar_data_raw, sensor_angles, angular_vel, distance,
        sin_angle, cos_angle, max_lidar_range, max_angular_vel,
        max_distance, prev_action,
        lidar_downsample_bins, use_sector_processing,
        lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y,
        obs_noise_distance_std=0.0, obs_noise_angle_std=0.0, obs_noise_angular_vel_std=0.0,
        obs_noise_lidar_std=0.0
    )
    
    # Добавляем Vx и Vy для Critic
    vx_norm = np.clip(vx, -max_vx, max_vx) / max_vx
    vy_norm = np.clip(vy, -max_vy if max_vy else max_vx, max_vy if max_vy else max_vx) / (max_vy if max_vy else max_vx)
    
    # Concatenate: Actor(47) + vx(1) + vy(1) = 49 features
    critic_obs = np.concatenate([
        actor_obs,  # 47
        [vx_norm],  # 1
        [vy_norm]   # 1
    ])
    
    return critic_obs


def build_critic_observation(lidar_data_raw, sensor_angles, vx, vy, angular_vel, distance,
                             sin_angle, cos_angle, max_lidar_range, max_vx, max_vy,
                             max_angular_vel, max_distance, prev_action,
                             lidar_downsample_bins=40, use_sector_processing=True, 
                             use_extended_features=False, critical_topk=0,
                             lidar_offset_x=0.12, lidar_offset_y=0.0):
    """
    Build observation for critic (Asymmetric).
    Base: Actor(47) + vx(1) + vy(1) = 49 features, plus optional extra features.
    """
    # Safety check
    if prev_action is None:
        prev_action = np.zeros(3)
        
    base_obs = build_critic_base_observation(
        lidar_data_raw, sensor_angles, vx, vy, angular_vel, distance,
        sin_angle, cos_angle, max_lidar_range, max_vx, max_vy,
        max_angular_vel, max_distance, prev_action,
        lidar_downsample_bins, use_sector_processing,
        lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y
    )
    
    if critical_topk > 0:
        # Extra features: top-k nearest lidar beams (raw distance normalized)
        # Supports both single arrays and batches (if we add batch support later)
        if lidar_data_raw.ndim == 1:
            sorted_lidar = np.sort(lidar_data_raw)
            topk_lidar = sorted_lidar[:critical_topk]
        else:
            sorted_lidar = np.sort(lidar_data_raw, axis=1)
            topk_lidar = sorted_lidar[:, :critical_topk]
            
        topk_norm = np.clip(topk_lidar, 0, max_lidar_range) / max_lidar_range
        topk_norm = topk_norm * 2.0 - 1.0
        
        if lidar_data_raw.ndim == 1:
            return np.concatenate([base_obs, topk_norm])
        else:
            return np.concatenate([base_obs, topk_norm], axis=1)
    else:
        return base_obs
