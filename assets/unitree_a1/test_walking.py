#!/usr/bin/env python3
"""
Скрипт для тестирования A1 робота с случайными командами через walking policy.
Генерирует случайные команды (vx, vy, w) на частоте 10 Гц и передает их в предобученную политику.
"""

import mujoco
import mujoco.viewer
import os
import numpy as np
import time
import torch
import torch.nn as nn
from pathlib import Path

# Default angles for A1 (from Isaac Lab UNITREE_A1_CFG)
# Order: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
# Isaac Lab: hip=0.0, thigh=0.8, calf=-1.5
DEFAULT_ANGLES = np.array([0, 0.8, -1.5, 0, 0.8, -1.5, 0, 0.8, -1.5, 0, 0.8, -1.5], dtype=np.float32)

# PD control gains (updated to match config: kps=2, kds=0.5)
KPS = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32)
KDS = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

# Scales from config
# Different scales for different joint types (from Isaac Lab):
# hip_joint: 0.125, other joints: 0.25
# Order: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
ACTION_SCALES = np.array([
    0.125,  # FR_hip
    0.25,   # FR_thigh
    0.25,   # FR_calf
    0.125,  # FL_hip
    0.25,   # FL_thigh
    0.25,   # FL_calf
    0.125,  # RR_hip
    0.25,   # RR_thigh
    0.25,   # RR_calf
    0.125,  # RL_hip
    0.25,   # RL_thigh
    0.25    # RL_calf
], dtype=np.float32)

DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ANG_VEL_SCALE = 0.25
CMD_SCALE = np.array([1.7, 1.5, 0.35])  # [vx, vy, w]

# Frequency settings (matching config)
SIMULATION_DT = 0.002  # 500 Hz simulation
CONTROL_DECIMATION = 10  # 50 Hz controller
# Walking policy runs at 50 Hz (same as controller frequency)
# Walking policy runs every: CONTROL_DECIMATION = 10 simulation steps

# Collision detection removed - only used in training (train.py)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_gravity_orientation(quaternion):
    """Compute projected gravity vector in base frame"""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

class SimpleWalkingPolicy(nn.Module):
    """Simple MLP walking policy network"""
    def __init__(self, obs_dim=45, action_dim=12, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.net(obs)

def build_walking_policy_observation(
    base_lin_vel,      # [3] linear velocity in base frame
    base_ang_vel,      # [3] angular velocity in base frame
    projected_gravity, # [3] projected gravity vector
    velocity_commands, # [3] velocity commands (vx, vy, w)
    joint_pos,         # [num_actions] joint positions (relative to default)
    joint_vel,         # [num_actions] joint velocities
    last_action,       # [num_actions] last action from walking policy
    height_scan=None,  # [N] height scan data (optional)
    base_lin_vel_scale=1.0,
    base_ang_vel_scale=0.25,
    joint_pos_scale=1.0,
    joint_vel_scale=0.05,
    height_scan_scale=1.0
):
    """
    Build observation for walking policy according to PolicyCfg structure.
    """
    obs_parts = []
    
    # 1. base_lin_vel (3) - clip and scale
    base_lin_vel_scaled = np.clip(base_lin_vel * base_lin_vel_scale, -100.0, 100.0)
    obs_parts.append(base_lin_vel_scaled)
    
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
    
    # 8. height_scan (N) - optional, clip and scale
    if height_scan is not None:
        height_scan_scaled = np.clip(height_scan * height_scan_scale, -1.0, 1.0)
        obs_parts.append(height_scan_scaled)
    else:
        # If no height scan, use zeros (default size of 56)
        height_scan_default = np.zeros(56, dtype=np.float32)
        obs_parts.append(height_scan_default)
    
    # Concatenate all parts
    observation = np.concatenate(obs_parts, dtype=np.float32)
    
    return observation

def main():
    # Путь к файлу сцены
    scene_path = os.path.join(os.path.dirname(__file__), "scene.xml")
    
    if not os.path.exists(scene_path):
        print(f"Ошибка: файл сцены не найден: {scene_path}")
        return
    
    # Путь к walking policy
    project_root = Path(__file__).resolve().parent.parent
    policy_path = project_root / "pre_train" / "a1" / "model_4999.pt"
    
    walking_policy = None
    if not policy_path.exists():
        print(f"Ошибка: файл walking policy не найден: {policy_path}")
        print("Продолжаем без walking policy (робот будет стоять)...")
    else:
        print(f"Загрузка walking policy: {policy_path}")
        try:
            # Загружаем checkpoint
            checkpoint = torch.load(str(policy_path), map_location='cpu')
            
            # Проверяем структуру
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'actor' in checkpoint:
                state_dict = checkpoint['actor']
            else:
                state_dict = checkpoint
            
            # Определяем размерность входа из весов
            if 'actor.0.weight' in state_dict:
                obs_dim = state_dict['actor.0.weight'].shape[1]
                action_dim = state_dict['actor.6.weight'].shape[0] if 'actor.6.weight' in state_dict else 12
                print(f"Обнаружена модель: вход={obs_dim}, выход={action_dim}")
                
                # Создаем модель с правильной архитектурой
                # Используем архитектуру из state_dict: [512, 256, 128]
                walking_policy = SimpleWalkingPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=[512, 256, 128])
                # Фильтруем state_dict и переименовываем ключи: actor.* -> net.*
                # SimpleWalkingPolicy использует net.0, net.2, net.4, net.6
                # Actor использует actor.0, actor.2, actor.4, actor.6
                actor_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('actor.'):
                        # Переименовываем: actor.0 -> net.0, actor.2 -> net.2, actor.4 -> net.4, actor.6 -> net.6
                        new_key = k.replace('actor.', 'net.')
                        actor_state_dict[new_key] = v
                
                missing_keys, unexpected_keys = walking_policy.load_state_dict(actor_state_dict, strict=False)
                if missing_keys:
                    print(f"Предупреждение: отсутствующие ключи: {missing_keys[:5]}")
                if unexpected_keys:
                    print(f"Предупреждение: неожиданные ключи: {unexpected_keys[:5]}")
                walking_policy.eval()
                print(f"✓ Walking policy загружен успешно! (вход: {obs_dim}, выход: {action_dim})")
            else:
                print("Не удалось определить структуру модели")
                walking_policy = None
        except Exception as e:
            print(f"Ошибка загрузки walking policy: {e}")
            import traceback
            traceback.print_exc()
            print("Продолжаем без walking policy...")
            walking_policy = None
    
    print(f"Загрузка сцены: {scene_path}")
    
    # Загружаем модель
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    # Устанавливаем временной шаг
    model.opt.timestep = SIMULATION_DT
    
    # Сбрасываем в начальное состояние
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Устанавливаем правильную начальную высоту (робот должен быть над полом)
    # Keyframe указывает z=0.27, но это может быть слишком низко, устанавливаем безопасную высоту
    data.qpos[2] = 0.4  # Высота базы робота над полом (пол находится на z=0)
    
    # Инициализируем forward kinematics
    mujoco.mj_forward(model, data)
    
    # Лидар сенсоры не используются в тестовом скрипте (только в обучении)
    
    print("\n=== Режим тестирования ===")
    print("Генерация случайных команд на частоте 10 Гц")
    print("Детекция препятствий отключена (используется только в обучении)")
    print("===========================\n")
    
    # Инициализация для walking policy
    num_actions = 12
    last_action = np.zeros(num_actions, dtype=np.float32)
    
    # Генератор случайных команд
    rng = np.random.RandomState(42)  # Фиксированный seed для воспроизводимости
    
    # Команды скорости (в нормализованном виде [-1, 1])
    # Будем генерировать новые команды на частоте 10 Гц
    cmd_vx = 0.0  # Вперед/назад
    cmd_vy = 0.0  # Влево/вправо
    cmd_w = 0.0   # Поворот
    
    # Запускаем визуализатор
    with mujoco.viewer.launch_passive(model, data) as viewer:
            step_count = 0
            last_print_time = time.time()
            policy_call_count = 0
            episode_count = 0
            
            # Кэш для последнего action от walking policy
            cached_action = np.zeros(num_actions, dtype=np.float32)
            
            # Интервал для обновления случайных команд (10 Гц)
            cmd_update_interval = CONTROL_DECIMATION * 5  # 50 шагов = 10 Гц
            cmd_update_step = 0
            
            # Walking policy вызывается на 50 Гц (каждые CONTROL_DECIMATION шагов)
            policy_update_interval = CONTROL_DECIMATION  # 10 шагов = 50 Гц
            
            print(f"\n=== Частоты ===")
            print(f"Симуляция: {1.0/SIMULATION_DT:.0f} Hz (dt={SIMULATION_DT*1000:.1f} ms)")
            print(f"Контроллер: {1.0/(SIMULATION_DT*CONTROL_DECIMATION):.0f} Hz")
            print(f"Walking policy: {1.0/(SIMULATION_DT*CONTROL_DECIMATION):.0f} Hz (50 Гц)")
            print(f"Walking policy вызывается каждые {CONTROL_DECIMATION} шагов симуляции")
            print(f"Случайные команды обновляются на частоте 10 Гц (каждые {cmd_update_interval} шагов)")
            print("===============\n")
            
            while viewer.is_running():
                # Генерация случайных команд на частоте 10 Гц
                if cmd_update_step % cmd_update_interval == 0:
                    # Генерируем новые случайные команды в диапазоне [-1, 1]
                    cmd_vx = rng.uniform(-1.0, 1.0)
                    cmd_vy = rng.uniform(-1.0, 1.0)
                    cmd_w = rng.uniform(-1.0, 1.0)
                
                cmd_update_step += 1
                
                # Walking policy вызывается на 50 Hz (каждые CONTROL_DECIMATION шагов = 10 шагов)
                # Проверяем ДО увеличения step_count
                should_call_policy = (step_count % policy_update_interval) == 0
                
                if should_call_policy and walking_policy is not None:
                    # Масштабируем команды согласно CMD_SCALE
                    velocity_commands = np.array([
                        cmd_vx * CMD_SCALE[0],
                        cmd_vy * CMD_SCALE[1],
                        cmd_w * CMD_SCALE[2]
                    ], dtype=np.float32)
                    
                    # Извлекаем данные из MuJoCo для наблюдения
                    # Base pose and velocities
                    quat = data.qpos[3:7]  # Base quaternion
                    qj = data.qpos[7:7+num_actions]  # Joint positions
                    
                    # Base velocities (in world frame)
                    base_lin_vel_world = data.qvel[0:3]  # Linear velocity in world frame
                    base_ang_vel_world = data.qvel[3:6]  # Angular velocity in world frame
                    dqj = data.qvel[6:6+num_actions]  # Joint velocities
                    
                    # For simplicity, use world frame velocities (common approximation)
                    base_lin_vel = base_lin_vel_world
                    base_ang_vel = base_ang_vel_world
                    
                    # Compute projected gravity
                    projected_gravity = get_gravity_orientation(quat)
                    
                    # Joint positions relative to default
                    joint_pos_rel = qj - DEFAULT_ANGLES
                    
                    # Extract height scan if available (for now, use None)
                    height_scan = None
                    
                    # Строим наблюдение в формате, который ожидает модель (45 dim):
                    # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + 
                    # joint_pos(12) + joint_vel(12) + last_action(12) = 45 dim
                    # БЕЗ velocity_commands и height_scan
                    obs_parts = []
                    
                    # 1. base_lin_vel (3) - clip and scale
                    obs_parts.append(np.clip(base_lin_vel * 1.0, -100.0, 100.0))
                    
                    # 2. base_ang_vel (3) - clip and scale
                    obs_parts.append(np.clip(base_ang_vel * ANG_VEL_SCALE, -100.0, 100.0))
                    
                    # 3. projected_gravity (3) - clip
                    obs_parts.append(np.clip(projected_gravity, -100.0, 100.0))
                    
                    # 4. joint_pos (12) - clip and scale (relative to default)
                    obs_parts.append(np.clip(joint_pos_rel * DOF_POS_SCALE, -100.0, 100.0))
                    
                    # 5. joint_vel (12) - clip and scale
                    obs_parts.append(np.clip(dqj * DOF_VEL_SCALE, -100.0, 100.0))
                    
                    # 6. last_action (12) - clip
                    obs_parts.append(np.clip(last_action, -100.0, 100.0))
                    
                    # Concatenate to get 45 dim observation
                    obs = np.concatenate(obs_parts, dtype=np.float32)
                    
                    # Проверяем размерность
                    obs_dim_expected = list(walking_policy.parameters())[0].shape[1]
                    if len(obs) != obs_dim_expected:
                        print(f"WARNING: Observation size mismatch! Expected {obs_dim_expected}, got {len(obs)}")
                        if len(obs) > obs_dim_expected:
                            obs = obs[:obs_dim_expected]
                        else:
                            obs = np.pad(obs, (0, obs_dim_expected - len(obs)), 'constant')
                    
                    # Get action from walking policy
                    try:
                        with torch.no_grad():
                            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()  # [1, obs_dim]
                            action_tensor = walking_policy(obs_tensor)
                            new_action = action_tensor.numpy().squeeze()  # [num_actions]
                            # Проверяем диапазон action (должен быть [-1, 1])
                            if np.any(np.abs(new_action) > 1.1):
                                print(f"WARNING: Action out of range! Max: {np.max(np.abs(new_action)):.3f}")
                            cached_action = new_action.copy()
                            last_action = cached_action.copy()
                            policy_call_count += 1
                    except Exception as e:
                        print(f"Ошибка walking policy: {e}")
                        import traceback
                        traceback.print_exc()
                        cached_action = np.zeros(num_actions, dtype=np.float32)
                
                # Используем кэшированное action (обновляется на 50 Гц)
                # Action от policy уже в правильном порядке: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
                # Это соответствует порядку актуаторов в ctrl[0:12]
                # НЕ используем интерполяцию - policy работает на 50 Гц, этого достаточно для плавности
                action = cached_action.copy()
                
                # Применяем action через PD контроллер (на каждом шаге симуляции)
                # Action от policy в диапазоне [-1, 1], масштабируем по-разному для разных суставов
                # hip_joint: 0.125, остальные: 0.25
                # Порядок: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
                target_angles = action * ACTION_SCALES + DEFAULT_ANGLES
                
                # Получаем текущие позиции и скорости суставов в правильном порядке
                # qpos[7:19] соответствует суставам в порядке: FR_hip(7), FR_thigh(8), FR_calf(9), FL_hip(10), FL_thigh(11), FL_calf(12), RR_hip(13), RR_thigh(14), RR_calf(15), RL_hip(16), RL_thigh(17), RL_calf(18)
                # qvel[6:18] соответствует скоростям суставов в том же порядке
                joint_pos = data.qpos[7:7+num_actions]  # [12] - правильный порядок
                joint_vel = data.qvel[6:6+num_actions]   # [12] - правильный порядок
                
                # PD контроллер
                target_vel = np.zeros(num_actions, dtype=np.float32)
                tau = pd_control(target_angles, joint_pos, KPS, target_vel, joint_vel, KDS)
                
                # Применяем управление напрямую к актуаторам
                # ctrl[0:12] соответствует актуаторам в порядке: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
                # Это совпадает с порядком action от policy: action[0]=FR_hip, action[1]=FR_thigh, ..., action[11]=RL_calf
                # Убеждаемся, что применяем только к нужным актуаторам
                if len(data.ctrl) >= num_actions:
                    data.ctrl[:num_actions] = tau
                else:
                    print(f"WARNING: ctrl size ({len(data.ctrl)}) < num_actions ({num_actions})")
                    data.ctrl[:] = tau[:len(data.ctrl)]
                
                # Шаг симуляции
                mujoco.mj_step(model, data)
                
                # Синхронизация визуализатора
                viewer.sync()
                
                # Увеличиваем счетчик ПОСЛЕ шага симуляции
                step_count += 1
                
                # Периодический вывод информации
                if time.time() - last_print_time > 1.0:
                    base_pos = data.qpos[0:3]
                    base_vel = data.qvel[0:3]
                    # Правильный подсчет частоты: policy должна вызываться каждые policy_update_interval шагов
                    # За 1 секунду при 500 Гц симуляции = 500 шагов, должно быть 500/policy_update_interval = 50 вызовов
                    elapsed_time = time.time() - last_print_time
                    expected_calls = int(elapsed_time * (1.0 / (SIMULATION_DT * policy_update_interval)))
                    actual_policy_freq = policy_call_count / elapsed_time if elapsed_time > 0 else 0
                    # Отладочная информация
                    if abs(actual_policy_freq - 50.0) > 5.0:  # Если частота сильно отличается от ожидаемой
                        print(f"DEBUG: step_count={step_count}, policy_call_count={policy_call_count}, expected={expected_calls}, actual_freq={actual_policy_freq:.1f} Hz")
                    # Показываем примеры action для отладки
                    action_sample = f"action[0:3]={action[0]:.3f},{action[1]:.3f},{action[2]:.3f}" if len(action) >= 3 else "N/A"
                    joint_sample = f"joint[0:3]={joint_pos[0]:.3f},{joint_pos[1]:.3f},{joint_pos[2]:.3f}" if len(joint_pos) >= 3 else "N/A"
                    target_sample = f"target[0:3]={target_angles[0]:.3f},{target_angles[1]:.3f},{target_angles[2]:.3f}" if len(target_angles) >= 3 else "N/A"
                    print(f"Эпизод: {episode_count} | "
                          f"Позиция: [{base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.2f}] | "
                          f"Скорость: [{base_vel[0]:.2f}, {base_vel[1]:.2f}, {base_vel[2]:.2f}] | "
                          f"Команды: vx={cmd_vx:.2f}, vy={cmd_vy:.2f}, w={cmd_w:.2f} | "
                          f"Walking policy: {'да' if walking_policy is not None else 'нет'} | "
                          f"Вызовов policy: {policy_call_count} ({actual_policy_freq:.1f} Hz) | "
                          f"{action_sample} | {joint_sample} | {target_sample}")
                    policy_call_count = 0
                    last_print_time = time.time()
    
    print("\nВизуализация завершена")

if __name__ == "__main__":
    main()
