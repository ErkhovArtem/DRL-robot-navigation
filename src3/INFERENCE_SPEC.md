# SAC Policy Inference Specification

## Model Architecture

**Network Type:** SAC Actor (Deterministic for inference)
**Input Dimension:** `base_obs_dim` (47 or 47 + history_length * 3)
**Output Dimension:** 3 (vx, vy, w commands in [-1, 1])
**Architecture:** MLP with configurable hidden layers

**Forward Pass:**
1. Input: `obs` [batch_size, obs_dim]
2. Trunk: Linear layers with ReLU activations
3. Output branch: `[mu, log_std] = chunk(trunk_output, 2)`
4. Apply log_std bounds: `log_std = clip(tanh(log_std), log_std_min, log_std_max)`
5. Deterministic output: `action = tanh(mu)` → [-1, 1]

## Input Observation Format

**Base Actor Observation (47 features, NO action history):**
```
[0:40]    - Lidar sectors (normalized to [-1, 1])
           - Normalization: (distance / max_range) * 2.0 - 1.0
           - max_range: 3.0 m (from config)
           - min_range: 0.25 m (filtered out)
           - -1 = close obstacle, +1 = far/no obstacle

[40]      - Angular velocity W (normalized to [-1, 1])
           - Normalization: (w / max_w) → [-1, 1]
           - max_w: from config (typically 0.35 rad/s)

[41]      - sin(angle_to_target) (already in [-1, 1])

[42]      - cos(angle_to_target) (already in [-1, 1])

[43]      - Distance to target (normalized to [-1, 1])
           - Normalization: (dist / max_dist) * 2.0 - 1.0
           - max_dist: from config (typically 10.75 m)

[44]      - Previous vx command (already in [-1, 1])

[45]      - Previous vy command (already in [-1, 1])

[46]      - Previous w command (already in [-1, 1])
```

**Actor Observation WITH Action History (if history_length > 0):**
```
[0:47]    - Base observation (as above)
[47:47+history_length*3] - Action history (flattened)
           - Each step: [vx, vy, w] (3 values)
           - Chronological order: oldest to newest
           - Example: history_length=3 → [vx_t-3, vy_t-3, w_t-3, vx_t-2, vy_t-2, w_t-2, vx_t-1, vy_t-1, w_t-1]
           - Total: 47 + 3*history_length features
```
!IMPORTANT!
Now we used history_length: 3
!IMPORTANT!

## Output Action Format

**Raw Output:** `[vx_cmd, vy_cmd, w_cmd]` in [-1, 1]

**Scaled Output (apply from config):**
- `vx_scaled = vx_cmd * cmd_scale[0]` (typically 1.7 m/s)
- `vy_scaled = vy_cmd * cmd_scale[1]` (typically 1.5 m/s)
- `w_scaled = w_cmd * cmd_scale[2]` (typically 0.35 rad/s)

## Lidar Data Processing

**Input:** Point cloud from ROS2 (PointCloud2 message)
**Output:** 40 sector distances in meters (NOT normalized)

**Processing Pipeline (see lidar_2d_processor.py):**
1. Filter floor points: `z < min_z + floor_threshold` (typically 1.2 m)
2. Project to 2D: set z = 0
3. Filter by range: `min_range <= distance <= max_range` (0.25-3.0 m)
4. Divide into 40 sectors: `angle_range = [-π, π]`, uniform distribution
5. For each sector: find minimum distance
6. If no points in sector: use `max_range` (3.0 m)

**Sector Calculation:**
- `num_sectors = 40`
- `sector_angle = 2π / 40 = 0.157 rad (9°)`
- Sector i: `[angle_min, angle_max]` where:
  - `angle_min = -π + i * sector_angle`
  - `angle_max = angle_min + sector_angle`
  - Last sector (i=39): includes π (closed interval)

**Normalization (for model input):**
```python
# Replace invalid values
lidar_sectors = np.where(
    (lidar_sectors >= 0) & (lidar_sectors <= max_range) & np.isfinite(lidar_sectors),
    lidar_sectors,
    max_range  # Replace invalid with max_range
)
# Clip and normalize to [-1, 1]
lidar_data = np.clip(lidar_sectors, 0, max_range) / max_range
lidar_data = lidar_data * 2.0 - 1.0  # -1 = close, +1 = far
```

## Observation Building

**Required Inputs:**
- `lidar_sectors`: 40 distances in meters (from lidar_2d_processor)
- `angular_vel`: Current angular velocity W (rad/s)
- `distance`: Distance to target (m)
- `sin_angle`: sin(angle_to_target)
- `cos_angle`: cos(angle_to_target)
- `prev_action`: Previous action [vx, vy, w] in [-1, 1]
- `max_lidar_range`: 3.0 m
- `max_angular_vel`: From config (typically 0.35 rad/s)
- `max_distance`: From config (typically 10.75 m)

**Normalization:**
```python
# Lidar
lidar_data = normalize_lidar(lidar_sectors, max_range=3.0)  # → [-1, 1]

# Angular velocity
angular_vel_norm = np.clip(angular_vel, -max_w, max_w) / max_w  # → [-1, 1]

# Distance
distance_norm = (np.clip(distance, 0, max_dist) / max_dist) * 2.0 - 1.0  # → [-1, 1]

# Angle (already in range)
sin_angle = np.clip(sin_angle, -1, 1)
cos_angle = np.clip(cos_angle, -1, 1)

# Previous action (already in [-1, 1])
prev_action = np.clip(prev_action, -1, 1)
```

**Concatenation:**
```python
base_obs = np.concatenate([
    lidar_data,           # 40
    [angular_vel_norm],   # 1
    [sin_angle],          # 1
    [cos_angle],          # 1
    [distance_norm],      # 1
    [prev_action[0]],     # 1
    [prev_action[1]],     # 1
    [prev_action[2]]      # 1
])  # Total: 47
```

**With Action History (if enabled):**
```python
# Append action history
if history_length > 0:
    action_history_flat = np.concatenate(list(action_history))  # [history_length * 3]
    obs = np.concatenate([base_obs, action_history_flat])  # 47 + history_length * 3
else:
    obs = base_obs  # 47
```

## Model Loading (ONNX)

```python
import onnxruntime as ort

session = ort.InferenceSession("sac_actor.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
```

**Input Shape:** `[batch_size, obs_dim]` where obs_dim = 47 or 47 + history_length * 3
**Output Shape:** `[batch_size, 3]` → `[vx_cmd, vy_cmd, w_cmd]` in [-1, 1]

## Action History Management

**For models with history_length > 0:**
1. Initialize: `action_history = deque(maxlen=history_length)`
2. After each inference:
   - Store action: `action_history.append(current_action.copy())`
3. Before inference:
   - If history not full: pad with zeros at the beginning
   - Concatenate: `obs = [base_obs, flattened_history]`

## Configuration Parameters

From `g1.yaml`:
- `max_lidar_range`: 3.0 m
- `max_angular_vel`: 0.35 rad/s (typically)
- `max_distance`: 10.75 m (typically, from curriculum)
- `cmd_scale`: [1.7, 1.5, 0.35] → [vx_max, vy_max, w_max]
- `history_length`: 0 or >0 (action history for actor)
- `lidar_downsample_bins`: 40
- `min_range`: 0.25 m (lidar filtering)
