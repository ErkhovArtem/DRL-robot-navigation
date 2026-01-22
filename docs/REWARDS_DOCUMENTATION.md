# Reward System Documentation

This document describes the reward system used for training the SAC (Soft Actor-Critic) path planning policy for the Unitree A1 robot.

**Version:** 1.0  
**Date:** 2025-01-22  
**Algorithm:** SAC with reference-style reward function

---

## Overview

The reward system is designed to encourage the robot to:
1. **Reach the target** efficiently
2. **Avoid obstacles** and collisions
3. **Move forward** (minimize backward/lateral movement)
4. **Maintain velocity alignment** (move in the direction it's facing)
5. **Complete tasks quickly** (time penalty)

The reward function uses a **reference-style** implementation (`compute_reward_reference`) that combines multiple components with configurable weights.

---

## Reward Components

### 1. Goal Reached Reward

**Purpose:** Reward the agent for successfully reaching the target.

**Formula:**
```python
if distance_to_target < reached_threshold:
    reward = 100.0 - time_penalty
else:
    reward = 0.0
```

**Parameters:**
- `reached_threshold`: 0.2 meters (default)
- `reached`: 100.0 (weight, used for terminal reward)

**Behavior:**
- Binary reward: 100.0 if target reached, 0.0 otherwise
- Time penalty is subtracted from the reward (encourages faster completion)
- Episode terminates when target is reached

---

### 2. Collision Penalty

**Purpose:** Strongly penalize collisions to encourage obstacle avoidance.

**Formula:**
```python
if collision_detected:
    reward = -100.0 - time_penalty
```

**Parameters:**
- `collision`: -500.0 (weight in config, but actual penalty is -100.0)
- `collision_threshold`: 0.35 meters (distance where collision occurs)

**Behavior:**
- Catastrophic penalty when collision is detected
- Episode terminates immediately
- Overrides all other reward components

**Detection:**
- Collision is detected when minimum lidar distance < 0.35m
- Lidar values < 0.25m are filtered out (robot body radius)

---

### 3. Obstacle Penalty (Exponential)

**Purpose:** Penalize proximity to obstacles using an exponential function that grows sharply as the robot approaches obstacles.

**Formula:**
```
Three zones:
1. Safe zone (min_dist >= obstacle_threshold): penalty = 0
2. Danger zone (death_distance < min_dist < obstacle_threshold): 
   penalty = -obs_penalty_weight * exp(k * (threshold - min_dist) / (min_dist - death_distance))
   where k = obstacle_exponential_scale
3. Death zone (min_dist <= death_distance): penalty = -obs_penalty_weight (maximum)
```

**Parameters:**
- `obs_penalty_weight`: 5.0 (default, scales the penalty)
- `obstacle_threshold`: 1.5 meters (distance where penalty starts)
- `obstacle_exponential_scale`: 5.0 (controls exponential growth rate)
- `death_distance`: 0.35 meters (hardcoded, collision distance)

**Mathematical Details:**
```python
# Normalized exponential (0 to 1)
arg_exp = exponential_scale * (obstacle_threshold - min_dist) / (min_dist - death_distance)
arg_exp = clip(arg_exp, 0.0, 10.0)  # Prevent overflow

normalized_exp = exp(arg_exp) / exp(10.0)  # Normalize to [0, 1]
obstacle_penalty = -normalized_exp * obs_penalty_weight
```

**Behavior:**
- No penalty when far from obstacles (≥ 1.5m)
- Exponential growth as robot approaches obstacles
- Maximum penalty at collision distance (0.35m)
- Creates a "fear zone" that encourages early avoidance

**Visualization:**
```
Distance:  3.0m  |  1.5m  |  1.0m  |  0.5m  |  0.35m
Penalty:     0    |   0    |  -0.5  |  -3.0  |  -5.0
Zone:     Safe   |  Safe  | Danger | Danger | Death
```

---

### 4. Progress Reward

**Purpose:** Reward the agent for moving closer to the target.

**Formula:**
```python
progress_diff = prev_distance - current_distance
progress_reward = progress_diff * progress_weight
```

**Parameters:**
- `progress`: 20.0 (default weight)

**Behavior:**
- Positive reward when moving closer to target
- Negative reward when moving away from target
- Linear scaling with distance improvement
- Encourages efficient path planning

**Example:**
- If robot moves 0.1m closer: reward = 0.1 * 20.0 = 2.0
- If robot moves 0.05m away: reward = -0.05 * 20.0 = -1.0

---

### 5. Forward Velocity Reward

**Purpose:** Encourage forward movement (positive vx).

**Formula:**
```python
reward = vx  # Direct reward for forward velocity
```

**Behavior:**
- Positive reward for forward movement (vx > 0)
- Negative reward for backward movement (vx < 0)
- Linear with velocity magnitude
- Encourages efficient navigation

**Note:** This is part of the main reward formula, not a separate weighted component.

---

### 6. Backward Movement Penalty

**Purpose:** Strongly discourage backward movement.

**Formula:**
```python
if vx < 0:
    penalty = vx * vx_backward_weight
else:
    penalty = 0.0
```

**Parameters:**
- `vx_backward_penalty`: -1.0 (default weight, negative value)

**Behavior:**
- Only applies when moving backward (vx < 0)
- Penalty increases with backward velocity magnitude
- Encourages forward navigation

**Example:**
- vx = -0.5 m/s: penalty = -0.5 * 1.0 = -0.5
- vx = 0.0 m/s: penalty = 0.0
- vx = 0.5 m/s: penalty = 0.0

---

### 7. Lateral Movement Penalty

**Purpose:** Discourage excessive lateral movement (vy).

**Formula:**
```python
penalty = -abs(vy) * vy_penalty_weight
```

**Parameters:**
- `vy_penalty`: -0.3 (default weight, negative value)

**Behavior:**
- Penalizes both left and right lateral movement
- Proportional to lateral velocity magnitude
- Allows some lateral movement for navigation but discourages excessive sideways motion

**Example:**
- vy = 0.0 m/s: penalty = 0.0
- vy = 0.2 m/s: penalty = -0.2 * 0.3 = -0.06
- vy = -0.2 m/s: penalty = -0.2 * 0.3 = -0.06

---

### 8. Angular Velocity Penalty

**Purpose:** Discourage excessive turning (w).

**Formula:**
```python
penalty = w_penalty_weight * abs(w)
```

**Parameters:**
- `w_penalty_weight`: -0.5 (default, negative = penalty)

**Behavior:**
- Penalizes turning in either direction
- Proportional to angular velocity magnitude
- Allows navigation turns but discourages excessive rotation

**Example:**
- w = 0.0 rad/s: penalty = 0.0
- w = 0.2 rad/s: penalty = -0.5 * 0.2 = -0.1
- w = -0.2 rad/s: penalty = -0.5 * 0.2 = -0.1

---

### 9. Velocity Alignment Reward

**Purpose:** Reward the robot for moving in the direction it's facing (encourages straight-line movement).

**Formula:**
```python
if velocity_magnitude > velocity_alignment_threshold:
    angle_diff = abs(robot_yaw - velocity_yaw)
    alignment_factor = (cos(angle_diff) + 1.0) / 2.0  # [0, 1]
    reward = alignment_factor * velocity_normalized * velocity_alignment_weight
else:
    reward = 0.0
```

**Parameters:**
- `velocity_alignment`: 0.5 (default weight)
- `velocity_alignment_threshold`: 0.1 m/s (minimum velocity to apply reward)

**Behavior:**
- Only applies when robot is moving (velocity > threshold)
- Maximum reward when moving exactly forward (angle_diff = 0)
- Zero reward when moving perpendicular (angle_diff = 90°)
- Encourages "face-forward" movement style

**Mathematical Details:**
```python
robot_yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy² + qz²))
velocity_yaw = atan2(vy_global, vx_global)
angle_diff = abs(robot_yaw - velocity_yaw)
alignment_factor = (cos(angle_diff) + 1.0) / 2.0  # 1.0 when aligned, 0.0 when perpendicular
```

**Example:**
- Moving forward (aligned): reward = 1.0 * 0.5 = 0.5
- Moving sideways (90°): reward = 0.0 * 0.5 = 0.0
- Moving backward (180°): reward = 0.0 * 0.5 = 0.0

---

### 10. Time Penalty

**Purpose:** Encourage faster task completion.

**Formula:**
```python
normalized_steps = step_count / max_steps
time_penalty = abs(time_penalty_weight) * normalized_steps
```

**Parameters:**
- `time_penalty`: -0.02 (default weight, negative = penalty)

**Behavior:**
- Penalty grows linearly with episode duration
- Normalized to [0, 1] based on max_steps
- Subtracted from total reward
- Encourages efficient path planning

**Example:**
- At start (step 0): penalty = 0.0
- At halfway (step 2500 / 5000): penalty = 0.5 * 0.02 = 0.01
- At end (step 5000 / 5000): penalty = 1.0 * 0.02 = 0.02

---

## Main Reward Formula

The total reward is computed as:

```python
total_reward = (
    vx +                                    # Forward velocity reward
    progress_reward +                       # Progress toward target
    w_penalty +                            # Angular velocity penalty
    obstacle_penalty +                     # Obstacle proximity penalty
    vx_backward_penalty +                  # Backward movement penalty
    -time_penalty                          # Time penalty
)

# Terminal rewards override the formula:
if goal_reached:
    total_reward = 100.0 - time_penalty
elif collision:
    total_reward = -100.0 - time_penalty
```

---

## Default Configuration

From `configs/a1.yaml`:

```yaml
reward_weights:
  # Terminal rewards
  reached: 100.0
  collision: -500.0
  
  # Obstacle avoidance
  obs_penalty_weight: 5.0
  obstacle_threshold: 1.5
  obstacle_exponential_scale: 5.0
  
  # Thresholds
  reached_threshold: 0.2
  collision_threshold: 0.35
  
  # Progress and movement
  progress: 20.0
  time_penalty: -0.02
  
  # Penalties
  vy_penalty: -0.3
  vx_backward_penalty: -1.0
  w_penalty_weight: -0.5
  
  # Alignment
  velocity_alignment: 0.5
  velocity_alignment_threshold: 0.1
```

---

## Reward Function Implementation

### Primary Function: `compute_reward_reference`

This is the main reward function used during training.

**Signature:**
```python
def compute_reward_reference(
    robot_pos,        # [3] Robot position (x, y, z)
    target_pos,       # [3] Target position (x, y, z)
    lidar_data,       # [40] Lidar distances in meters
    actions,          # [3] Actions [vx, vy, w]
    goal,             # bool: Target reached
    collision,        # bool: Collision detected
    reward_weights,   # dict: Reward weights configuration
    step_count=None,  # int: Current step count
    max_steps=None,   # int: Maximum steps per episode
    prev_distance=None # float: Previous distance to target
) -> (reward, done, reward_info)
```

**Returns:**
- `reward`: Total reward (float)
- `done`: Episode termination flag (bool)
- `reward_info`: Dictionary with individual reward components

### Vectorized Version: `compute_reward_reference_vectorized`

Supports batch processing for MJX parallel simulation.

**Signature:**
```python
def compute_reward_reference_vectorized(
    robot_pos,        # [batch_size, 3] or [3]
    target_pos,       # [batch_size, 3] or [3]
    lidar_data,       # [batch_size, 40] or [40]
    actions,          # [batch_size, 3] or [3]
    goal,             # [batch_size] or bool
    collision,        # [batch_size] or bool
    reward_weights,   # dict
    step_count=None,  # [batch_size] or int
    max_steps=None,   # [batch_size] or int
    prev_distance=None # [batch_size] or float
) -> (rewards, dones, reward_info)
```

---

## Reward Component Summary

| Component | Type | Default Weight | Range | Purpose |
|-----------|------|----------------|-------|---------|
| **Goal Reached** | Terminal | 100.0 | 0 or 100 | Reward success |
| **Collision** | Terminal | -100.0 | -100 | Penalize failure |
| **Obstacle Penalty** | Continuous | -5.0 | [-5.0, 0] | Avoid obstacles |
| **Progress** | Continuous | 20.0 | [-∞, +∞] | Move toward target |
| **Forward Velocity** | Continuous | 1.0 | [-∞, +∞] | Encourage forward motion |
| **Backward Penalty** | Continuous | -1.0 | [-∞, 0] | Discourage backward motion |
| **Lateral Penalty** | Continuous | -0.3 | [-∞, 0] | Discourage sideways motion |
| **Angular Penalty** | Continuous | -0.5 | [-∞, 0] | Discourage excessive turning |
| **Velocity Alignment** | Continuous | 0.5 | [0, 0.5] | Encourage straight movement |
| **Time Penalty** | Continuous | -0.02 | [-0.02, 0] | Encourage speed |

---

## Design Principles

### 1. **Collision Avoidance Priority**
- Collision penalty (-100.0) is catastrophic and overrides all other rewards
- Exponential obstacle penalty creates a "danger zone" that encourages early avoidance
- High `obs_penalty_weight` (5.0) ensures obstacles are strongly avoided

### 2. **Forward Movement Bias**
- Direct reward for forward velocity (vx)
- Strong penalty for backward movement
- Moderate penalty for lateral/angular movement
- Encourages efficient, forward-focused navigation

### 3. **Progress Over Speed**
- Progress reward (20.0) is high relative to time penalty (-0.02)
- Agent is encouraged to make progress, not just move fast
- Time penalty is minimal to avoid rushing into obstacles

### 4. **Smooth Behavior**
- Velocity alignment reward encourages smooth, straight-line movement
- Angular velocity penalty prevents excessive spinning
- Lateral movement penalty prevents zigzagging

### 5. **Terminal Rewards**
- Large positive reward (100.0) for reaching target
- Large negative reward (-100.0) for collision
- Both override continuous rewards to provide clear learning signal

---

## Curriculum Learning Integration

Reward weights can be adjusted dynamically through curriculum learning:

- **Early levels:** Lower obstacle penalties, higher exploration (lower temperature)
- **Later levels:** Higher obstacle penalties, stricter movement constraints
- **Final levels:** Maximum penalties, emphasis on efficiency and precision

See `configs/curriculum.yaml` for level-specific reward weight configurations.

---

## Tips for Tuning

### Increasing Obstacle Avoidance
- Increase `obs_penalty_weight` (e.g., 5.0 → 7.0)
- Decrease `obstacle_threshold` (e.g., 1.5 → 1.2) for earlier warnings
- Increase `obstacle_exponential_scale` (e.g., 5.0 → 7.0) for sharper penalty

### Encouraging Speed
- Increase `time_penalty` magnitude (e.g., -0.02 → -0.05)
- Increase `progress` weight (e.g., 20.0 → 30.0)

### Encouraging Straight Movement
- Increase `velocity_alignment` weight (e.g., 0.5 → 1.0)
- Increase `w_penalty_weight` magnitude (e.g., -0.5 → -1.0)
- Increase `vy_penalty` magnitude (e.g., -0.3 → -0.5)

### Reducing Collisions
- Increase `collision` penalty (e.g., -500.0 → -1000.0 in config, but actual penalty is -100.0)
- Increase `obs_penalty_weight` (e.g., 5.0 → 8.0)
- Decrease `obstacle_threshold` (e.g., 1.5 → 1.2)

---

## Implementation Notes

### Lidar Data Processing
- Lidar values < 0.25m are filtered out (robot body radius)
- Lidar values > 3.0m are clipped to 3.0m (maximum range)
- NaN and Inf values are replaced with max range (no obstacle detected)

### Distance Calculation
- Only uses x, y coordinates (ignores z/height)
- Euclidean distance: `sqrt((target_x - robot_x)² + (target_y - robot_y)²)`
- NaN/Inf handling: falls back to previous distance or default (5.0m)

### Episode Termination
- Episode ends when:
  - Target reached (`distance < reached_threshold`)
  - Collision detected (`min_lidar < collision_threshold`)
  - Maximum steps reached (`step_count >= max_steps`)

---

## References

- Main reward implementation: `src/utils/reward.py`
- Configuration: `configs/a1.yaml`
- Curriculum learning: `configs/curriculum.yaml`
- Training script: `scripts/train.py`
