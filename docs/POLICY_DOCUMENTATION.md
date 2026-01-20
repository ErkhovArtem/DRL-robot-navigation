# Документация SAC политики для навигации Unitree G1

## Обзор

Эта документация описывает обученную SAC (Soft Actor-Critic) политику для навигации робота Unitree G1 с избеганием препятствий. Политика обучена в симуляторе MuJoCo и экспортирована в ONNX формат для использования на реальном роботе.

**Версия:** 2.0  
**Дата:** 2025-01-17  
**Алгоритм:** SAC (Soft Actor-Critic) с Asymmetric Actor-Critic  
**Формат модели:** ONNX (sac_actor.onnx)

### Ключевые особенности архитектуры

- **Asymmetric Actor-Critic:** Actor и Critic имеют разные observation spaces
- **Actor:** 47 признаков (БЕЗ линейных скоростей Vx, Vy, но С угловой W)
- **Critic:** 53 признака (48 базовых + 5 critical_topk nearest lidar beams)
- **Обоснование:** Actor обучается принимать решения без знания текущих линейных скоростей, что улучшает обобщение и устойчивость

---

## 1. Observation Space (Пространство наблюдений)

### 1.1 Asymmetric Actor-Critic архитектура

**ВАЖНО:** Actor и Critic имеют **разные** observation spaces!

#### Actor Observation (для инференса)
Политика (Actor) принимает **47 значений** (float32), все нормализованы в диапазон **[-1, 1]**.

```
Actor Observation = [lidar(40) + angular_vel(1) + distance(1) + sin_angle(1) + cos_angle(1) + prev_action(3)]
                  = 40 + 1 + 1 + 1 + 1 + 3 = 47 значений

БЕЗ: Vx, Vy (линейные скорости)
С: W (угловая скорость)
```

#### Critic Observation (только при обучении)
Критик принимает **53 значения** (float32), все нормализованы в диапазон **[-1, 1]**.

```
Critic Observation = [lidar(40) + vx(1) + angular_vel(1) + distance(1) + sin_angle(1) + cos_angle(1) + prev_action(3) + critical_topk(5)]
                   = 40 + 1 + 1 + 1 + 1 + 1 + 3 + 5 = 53 значения

С: Vx, W (обе скорости)
+ 5 ближайших lidar beams для лучшей оценки опасности
```

**Обоснование Asymmetric подхода:**
- Actor не знает текущие линейные скорости → лучше обобщается, не переобучается на конкретные динамики
- Critic знает всё → может точнее оценить Q-value действия
- История: Critic видит историю (3 шага), Actor - нет

### 1.2 Детальное описание компонентов Actor (для инференса)

#### 1.2.1 Лидар (40 значений) - индексы [0:40]

**Одинаково для Actor и Critic**

**Формат данных:**
- **Сырые данные:** 40 значений расстояний в метрах (от 0.25 м до 3.0 м)
- **Источник:** `lidar_2d_processor.py` публикует в топик `/lidar_observations` (Float32MultiArray)
- **Обработка:** Секторная обработка (sector-based processing)

**Процесс обработки:**

1. **Сырые данные лидара** (из PointCloud2):
   - Фильтрация точек пола (floor_height_threshold = 1.2 м)
   - Проекция в 2D плоскость (Z = 0)
   - Фильтрация по дальности: [min_range=0.25 м, max_range=3.0 м]

2. **Секторная обработка:**
   - 40 секторов равномерно распределены по 360° (9° на сектор)
   - Сектор 0: [-π, -π+9°), сектор 1: [-π+9°, -π+18°), ..., сектор 39: [π-9°, π]
   - Для каждого сектора берется **минимальное расстояние** до препятствия
   - Если препятствий нет в секторе → значение = max_range (3.0 м)

3. **Нормализация:**
   ```python
   # Шаг 1: Фильтрация невалидных значений
   lidar_sectors = np.where(
       (lidar_sectors >= 0) & (lidar_sectors <= max_range) & np.isfinite(lidar_sectors),
       lidar_sectors,
       max_range  # Заменяем невалидные на max_range
   )
   
   # Шаг 2: Нормализация в [0, 1]
   lidar_normalized = np.clip(lidar_sectors, 0, max_range) / max_range
   
   # Шаг 3: Преобразование в [-1, 1]
   lidar_normalized = lidar_normalized * 2.0 - 1.0
   ```
   
   **Интерпретация:**
   - `-1.0` = препятствие очень близко (0.25 м)
   - `1.0` = препятствий нет (3.0 м или дальше)
   - `0.0` ≈ 1.5 м до препятствия

**Параметры:**
- `max_range = 3.0` м
- `min_range = 0.25` м
- `num_sectors = 40`

#### 1.2.2 Угловая скорость angular_vel (1 значение) - индекс [40]

**⚠️ ВАЖНО: Actor ИМЕЕТ угловую скорость (в отличие от Vx, Vy)!**

**Формат данных:**
- **Сырые данные:** Угловая скорость вокруг вертикальной оси в рад/с
- **Диапазон:** [-max_angular_vel, max_angular_vel] = [-0.35, 0.35] рад/с
- **Нормализация:**
  ```python
  angular_vel_norm = np.clip(angular_vel, -max_angular_vel, max_angular_vel) / max_angular_vel
  ```
- **Интерпретация:**
  - `-1.0` = поворот влево с максимальной скоростью (-0.35 рад/с)
  - `0.0` = без поворота
  - `1.0` = поворот вправо с максимальной скоростью (0.35 рад/с)

**Источник:** Odometry или оценка из TF (разность yaw)

**Почему Actor знает W, но не знает Vx, Vy?**
- W критична для контроля поворотов и избегания препятствий
- Vx, Vy могут вводить в заблуждение (например, скольжение) → Actor учится полагаться на визуальную информацию (lidar)

#### 1.2.3 Расстояние до цели distance (1 значение) - индекс [41]

**Формат данных:**
- **Сырые данные:** Евклидово расстояние до целевой точки в метрах
- **Диапазон:** [0, max_distance] = [0, 10.0] м
- **Нормализация:**
  ```python
  distance_norm = np.clip(distance, 0, max_distance) / max_distance
  distance_norm = distance_norm * 2.0 - 1.0  # [0, 1] -> [-1, 1]
  ```
- **Интерпретация:**
  - `-1.0` = цель очень близко (0 м)
  - `1.0` = цель далеко (10 м или дальше)
  - `0.0` ≈ 5 м до цели

**Вычисление:**
```python
dx = target_x - robot_x
dy = target_y - robot_y
distance = math.hypot(dx, dy)
```

#### 1.2.4 Синус угла до цели sin_angle (1 значение) - индекс [42]

**Формат данных:**
- **Сырые данные:** Синус угла до цели относительно направления робота
- **Диапазон:** [-1, 1] (уже нормализован)
- **Вычисление:**
  ```python
  angle = math.atan2(dy, dx) - robot_yaw  # Угол относительно направления робота
  # Нормализация угла в [-π, π]
  while angle > math.pi:
      angle -= 2 * math.pi
  while angle < -math.pi:
      angle += 2 * math.pi
  sin_angle = math.sin(angle)
  sin_angle = np.clip(sin_angle, -1, 1)
  ```
- **Интерпретация:**
  - `-1.0` = цель слева от направления робота (угол = -90°)
  - `0.0` = цель прямо по курсу или сзади (угол = 0° или 180°)
  - `1.0` = цель справа от направления робота (угол = +90°)

#### 1.2.5 Косинус угла до цели cos_angle (1 значение) - индекс [43]

**Формат данных:**
- **Сырые данные:** Косинус угла до цели относительно направления робота
- **Диапазон:** [-1, 1] (уже нормализован)
- **Вычисление:**
  ```python
  cos_angle = math.cos(angle)  # angle вычислен как для sin_angle
  cos_angle = np.clip(cos_angle, -1, 1)
  ```
- **Интерпретация:**
  - `1.0` = цель прямо по курсу (угол = 0°)
  - `0.0` = цель слева или справа (угол = ±90°)
  - `-1.0` = цель сзади (угол = ±180°)

**Примечание:** sin_angle и cos_angle вместе определяют полное направление до цели.

#### 1.2.6 Предыдущие действия prev_action (3 значения) - индексы [44:47]

**Формат данных:**
- **Сырые данные:** Предыдущие команды управления в нормализованном виде [-1, 1]
- **Структура:** [prev_vx, prev_vy, prev_w]
- **Нормализация:** Уже в диапазоне [-1, 1] (не требуют дополнительной обработки)
- **Инициализация:** При первом шаге = [0.0, 0.0, 0.0]

**Интерпретация:**
- `prev_action[0]` (prev_vx): предыдущая команда скорости вперед [-1, 1]
- `prev_action[1]` (prev_vy): предыдущая команда боковой скорости [-1, 1]
- `prev_action[2]` (prev_w): предыдущая команда угловой скорости [-1, 1]

**Важно:** Это нормализованные значения, а не реальные команды в м/с или рад/с!

### 1.3 Дополнительные признаки Critic (только при обучении)

#### 1.3.1 Скорость вперед Vx (1 значение) - у Critic

**⚠️ Только у Critic! Actor НЕ имеет этого признака!**

- **Диапазон:** [-1.7, 1.7] м/с
- **Нормализация:** `vx_norm = np.clip(vx, -max_vx, max_vx) / max_vx`
- **Зачем:** Critic использует для точной оценки Q-value, особенно для прогресса к цели

#### 1.3.2 Critical Top-K Lidar (5 значений) - у Critic

**⚠️ Только у Critic! Actor НЕ имеет этих признаков!**

- **Что это:** 5 ближайших расстояний от лидара (sorted)
- **Зачем:** Critic видит самые опасные препятствия явно → лучше оценивает риск столкновения
- **Нормализация:** Такая же как основной lidar (в [-1, 1])

---

## 2. Action Space (Пространство действий)

### 2.1 Общая структура

Политика выдает **3 значения** (float32), все в диапазоне **[-1, 1]**.

```
Action = [vx, vy, w]
```

### 2.2 Детальное описание компонентов

#### 2.2.1 vx - скорость вперед/назад

**Формат выхода политики:**
- Диапазон: [-1, 1]
- Интерпретация:
  - `-1.0` = максимальная скорость назад
  - `0.0` = остановка
  - `1.0` = максимальная скорость вперед

**Преобразование в реальную команду:**
```python
vx_cmd = action[0] * cmd_scale[0]
# где cmd_scale[0] = 1.7 м/с
# Реальный диапазон: [-1.7, 1.7] м/с
```

#### 2.2.2 vy - боковая скорость

**Формат выхода политики:**
- Диапазон: [-1, 1]
- Интерпретация:
  - `-1.0` = максимальная скорость влево
  - `0.0` = без бокового движения
  - `1.0` = максимальная скорость вправо

**Преобразование в реальную команду:**
```python
vy_cmd = action[1] * cmd_scale[1]
# где cmd_scale[1] = 1.5 м/с
# Реальный диапазон: [-1.5, 1.5] м/с
```

#### 2.2.3 w - угловая скорость

**Формат выхода политики:**
- Диапазон: [-1, 1]
- Интерпретация:
  - `-1.0` = максимальный поворот влево
  - `0.0` = без поворота
  - `1.0` = максимальный поворот вправо

**Преобразование в реальную команду:**
```python
w_cmd = action[2] * cmd_scale[2]
# где cmd_scale[2] = 0.35 рад/с
# Реальный диапазон: [-0.35, 0.35] рад/с
```

### 2.3 Параметры масштабирования

Из конфига `g1.yaml`:
```yaml
cmd_scale: [1.7, 1.5, 0.35]  # [vx, vy, w]
```

**Полное преобразование:**
```python
vx_cmd = action[0] * 1.7   # м/с
vy_cmd = action[1] * 1.5   # м/с
w_cmd = action[2] * 0.35   # рад/с
```

---

## 3. Процесс обучения

### 3.1 Симулятор

**Используемый симулятор:** MuJoCo (Physics Engine)

**Версия:** MuJoCo 3.x  
**Оптимизация:** MJX (MuJoCo XLA) для параллельного батч-симулирования на GPU

**Модель робота:**
- Файл: `g1_description/scene.xml`
- Робот: Unitree G1 (12 DOF)
- Сцена: Плоскость с препятствиями (кубы, цилиндры)

### 3.2 Частоты работы

**Симуляция:**
- `simulation_dt = 0.002` с (шаг симуляции)
- Частота симуляции: 500 Hz (1 / 0.002)

**Контроллер (Walking Policy):**
- `control_decimation = 10`
- Частота контроллера: 50 Hz (500 Hz / 10)
- Обновление команд каждые 0.02 с

**SAC политика:**
- `sac_decimation = 5`
- Частота политики: **10 Hz** (50 Hz / 5)
- Обновление действий каждые 0.1 с

**Важно для инференса:** Политика должна запускаться с частотой **10 Hz**!

### 3.3 Архитектура сетей (Asymmetric Actor-Critic)

#### 3.3.1 Actor Network (для инференса)

```
Входной слой:     47 признаков (БЕЗ Vx, Vy)
                  ↓
Скрытый слой 1:   512 нейронов (ReLU)
                  ↓
Скрытый слой 2:   512 нейронов (ReLU)
                  ↓
Скрытый слой 3:   256 нейронов (ReLU)
                  ↓
Выходной слой:    6 выходов (mu и log_std для каждого из 3 действий)
                  ├─ mu[0], log_std[0] → vx action
                  ├─ mu[1], log_std[1] → vy action
                  └─ mu[2], log_std[2] → w action
                  ↓
Sampling:         action = tanh(mu + std * noise) ∈ [-1, 1]³
```

**Параметры:**
- `log_std_bounds: [-2, 2]` → std ∈ [0.135, 7.39]
- История: НЕТ (всегда только текущее наблюдение)
- Actor hidden: [512, 512, 256]

**Детальная последовательность:**

1. **Вход:** observation[47] ∈ [-1, 1]
   ```
   [lidar(40), w(1), dist(1), sin(1), cos(1), prev_vx(1), prev_vy(1), prev_w(1)]
   ```

2. **Forward pass:**
   ```python
   h1 = ReLU(Linear_1(obs))      # [47] → [512]
   h2 = ReLU(Linear_2(h1))        # [512] → [512]
   h3 = ReLU(Linear_3(h2))        # [512] → [256]
   out = Linear_4(h3)             # [256] → [6]
   ```

3. **Разделение на mu и log_std:**
   ```python
   mu = out[:3]                    # [3] - средние значения
   log_std = out[3:]               # [3] - логарифмы std
   log_std = tanh(log_std)         # Нормализуем в [-1, 1]
   log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
   # Итого: log_std ∈ [-2, 2]
   ```

4. **Sampling (при обучении) или mean (при инференсе):**
   ```python
   # Training mode:
   std = exp(log_std)
   action_raw = mu + std * Normal(0, 1).sample()
   
   # Inference mode (deterministic):
   action_raw = mu
   
   # Применяем tanh для ограничения
   action = tanh(action_raw)  # ∈ [-1, 1]³
   ```

5. **Выход:** action[3] ∈ [-1, 1]

#### 3.3.2 Critic Network (только при обучении)

```
Входной слой:     53 признака (С Vx + critical_topk)
                  С историей: 53 * (1 + history_length) = 53 * 4 = 212
                  ↓
Скрытый слой 1:   512 нейронов (ReLU)
                  ↓
Скрытый слой 2:   512 нейронов (ReLU)
                  ↓
Скрытый слой 3:   256 нейронов (ReLU)
                  ↓
Выходной слой:    1 выход (Q-value)

Два независимых Critic: Q1 и Q2 (Double Q-learning)
Target Critic: медленно обновляемая копия (tau=0.005)
```

**Параметры:**
- История: 3 предыдущих наблюдения (history_length=3)
- Critic hidden: [512, 512, 256]
- Critical_topk: 5 ближайших lidar beams

**Детальная последовательность:**

1. **Вход:** observation[53] ∈ [-1, 1]
   ```
   [lidar(40), vx(1), w(1), dist(1), sin(1), cos(1), 
    prev_vx(1), prev_vy(1), prev_w(1), topk(5)]
   ```

2. **Добавление истории:**
   ```python
   # Concatenate текущее + 3 предыдущих
   obs_with_history = concat([obs_t, obs_t-1, obs_t-2, obs_t-3])
   # Размер: 53 * 4 = 212
   ```

3. **Forward pass (Q1 и Q2 параллельно):**
   ```python
   # Q1 network
   h1_q1 = ReLU(Linear_1_q1(obs_with_history))  # [212] → [512]
   h2_q1 = ReLU(Linear_2_q1(h1_q1))             # [512] → [512]
   h3_q1 = ReLU(Linear_3_q1(h2_q1))             # [512] → [256]
   q1 = Linear_4_q1(h3_q1)                       # [256] → [1]
   
   # Q2 network (та же структура, другие веса)
   q2 = ...  # аналогично
   ```

4. **Выход:** Q1, Q2 - два Q-value для action

#### 3.3.3 Схема взаимодействия Actor-Critic

```
┌──────────────────────────────────────────────────────┐
│                    ENVIRONMENT                        │
│  State: [robot_pose, target, lidar_raw, velocities]  │
└────────────┬─────────────────────────┬────────────────┘
             │                         │
             ▼                         ▼
    ┌─────────────────┐      ┌──────────────────────┐
    │ build_actor_obs │      │ build_critic_obs     │
    │   (47 features) │      │   (53 features)      │
    │  БЕЗ Vx, Vy     │      │  С Vx + topk         │
    └────────┬────────┘      └──────────┬───────────┘
             │                           │
             ▼                           ▼
    ┌─────────────────┐      ┌──────────────────────┐
    │  ACTOR NETWORK  │      │  CRITIC NETWORK      │
    │  [47]→512→512   │      │  [212]→512→512       │
    │  →256→[6]       │      │  →256→[1]            │
    │  ↓ sample       │      │  (Q1 и Q2)           │
    │  action[3]      │      │                      │
    └────────┬────────┘      └──────────┬───────────┘
             │                           │
             ▼                           ▼
    ┌─────────────────┐      ┌──────────────────────┐
    │  Scale actions  │      │  Compute TD target   │
    │  [1.7,1.5,0.35] │      │  Update networks     │
    └────────┬────────┘      └──────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  Execute action │
    └─────────────────┘
```

### 3.4 Гиперпараметры обучения

**Из конфига `g1.yaml` (базовые) и `curriculum.yaml` (по уровням):**

```yaml
sac:
  actor_lr: 0.0001
  critic_lr: 0.0003
  alpha_lr: 0.0003              # Увеличен для быстрой адаптации temperature
  discount: 0.99
  init_temperature: 0.2         # Низкий для exploration на старте
  target_entropy: -1.5          # Высокий для поощрения exploration
  critic_tau: 0.005
  actor_update_frequency: 1
  critic_target_update_frequency: 2
  learnable_temperature: true   # ВАЖНО! Alpha адаптируется динамически
  max_action: 1.0
  train_every_n: 1
  training_iterations: 4
  batch_size: 256
  min_buffer_size: 5000
  log_std_bounds: [-2, 2]       # Предотвращает std collapse
  alpha_betas: [0.9, 0.999]     # Стабильная оптимизация alpha
```

**Curriculum Learning (7 уровней):**
- `init_temperature` растёт с 0.1 до 0.8 по мере усложнения
- `alpha_lr` падает с 0.0003 до 0.00005
- Награды за success/collision балансируются на каждом уровне

**Replay Buffer:**
- Размер: 1,000,000 переходов
- Weighted sampling: приоритет успешных и коллизионных переходов

### 3.5 Функция награды (Reward Function)

**Компоненты награды:**

1. **reached:** +100.0 (достижение цели, порог 0.2 м)
2. **collision:** -500.0 (столкновение, порог 0.3 м)
3. **obstacle_penalty:** -5.0 * exp_penalty (близость к препятствию, порог 1.5 м)
4. **progress:** +20.0 * (prev_distance - distance) (прогресс к цели)
5. **vy_penalty:** -0.3 * |vy| (штраф за боковое движение)
6. **vx_backward_penalty:** -2.0 * max(0, -vx) (штраф за движение назад)
7. **time_penalty:** -0.02 (штраф за время)

**Фокус обучения:** Избегание препятствий и достижение цели.

### 3.6 Генерация препятствий

**Кубы:**
- Количество: 7-9
- Размеры: X=[0.15, 0.35] м, Y=[0.15, 0.35] м, Z=1.0 м

**Регенерация:** Каждые 2 эпизода

### 3.7 Эпизоды

- Максимальная длительность: 60 секунд
- Максимальное количество шагов: 12,000 (при 10 Hz = 20 минут симуляции)
- Условия завершения:
  - Достижение цели (distance < 0.2 м)
  - Столкновение (min_lidar < 0.3 м)
  - Таймаут (60 с)

---

## 4. Использование ONNX модели

### 4.1 Загрузка модели

```python
import onnxruntime as ort

onnx_session = ort.InferenceSession('sac_actor.onnx')
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

print(f"Input shape: {onnx_session.get_inputs()[0].shape}")   # [batch_size, 47]
print(f"Output shape: {onnx_session.get_outputs()[0].shape}") # [batch_size, 3]
```

**Формат входных данных:**
- Размерность: [batch_size, 47] ← **ИЗМЕНИЛОСЬ! Было 48**
- Тип: float32
- Диапазон значений: [-1, 1]
- **БЕЗ Vx, Vy** (линейные скорости)
- **С W** (угловая скорость)

**Формат выходных данных:**
- Размерность: [batch_size, 3]
- Тип: float32
- Диапазон значений: [-1, 1]

### 4.2 Инференс

```python
# Подготовка наблюдения (47 значений) ← ИЗМЕНИЛОСЬ!
observation = np.array([...], dtype=np.float32)  # 47 значений в [-1, 1]
obs_batch = observation.reshape(1, -1)  # [1, 47]

# Запуск инференса
outputs = onnx_session.run([output_name], {input_name: obs_batch})
action = outputs[0][0]  # [3] - [vx, vy, w] в [-1, 1]

# Применение масштабирования
vx_cmd = action[0] * 1.7   # м/с
vy_cmd = action[1] * 1.5   # м/с
w_cmd = action[2] * 0.35   # рад/с
```

### 4.3 Детальная последовательность инференса

```python
def inference_pipeline():
    """Полная последовательность от сенсоров до команд."""
    
    # ШАГ 1: Получение сырых данных
    lidar_raw = get_lidar_data()           # [40] float, метры [0.25, 3.0]
    robot_x, robot_y, yaw = get_robot_pose()
    target_x, target_y = get_target_pose()
    angular_vel = get_angular_velocity()   # float, рад/с
    prev_action = get_prev_action()        # [3] float, [-1, 1]
    
    # ШАГ 2: Обработка лидара
    lidar_clipped = np.clip(lidar_raw, 0, 3.0)
    lidar_norm = (lidar_clipped / 3.0) * 2.0 - 1.0  # → [-1, 1]
    
    # ШАГ 3: Вычисление target info
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx) - yaw
    # Normalize angle to [-π, π]
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    
    # ШАГ 4: Нормализация скаляров
    w_norm = np.clip(angular_vel, -0.35, 0.35) / 0.35
    dist_norm = (np.clip(distance, 0, 10.0) / 10.0) * 2.0 - 1.0
    
    # ШАГ 5: Сборка observation для Actor (47 features)
    observation = np.concatenate([
        lidar_norm,                         # 40
        [w_norm],                           # 1  ← W есть
        [dist_norm],                        # 1
        [np.clip(sin_angle, -1, 1)],       # 1
        [np.clip(cos_angle, -1, 1)],       # 1
        np.clip(prev_action, -1, 1)        # 3
    ]).astype(np.float32)                  # Итого: 47
    
    # ⚠️ ВАЖНО: НЕТ Vx и Vy!
    
    # ШАГ 6: ONNX инференс
    obs_batch = observation.reshape(1, -1)  # [1, 47]
    action_norm = onnx_session.run(
        [output_name], 
        {input_name: obs_batch}
    )[0][0]  # [3] в [-1, 1]
    
    # ШАГ 7: Масштабирование команд
    cmd_vx = action_norm[0] * 1.7   # м/с
    cmd_vy = action_norm[1] * 1.5   # м/с
    cmd_w = action_norm[2] * 0.35   # рад/с
    
    # ШАГ 8: Публикация и сохранение
    publish_cmd(cmd_vx, cmd_vy, cmd_w)
    prev_action = action_norm.copy()  # Сохраняем нормализованное!
    
    return cmd_vx, cmd_vy, cmd_w
```

### 4.4 Частота инференса

**Критически важно:** Запускать инференс с частотой **10 Hz** (каждые 0.1 с).

Это соответствует частоте обучения и обеспечивает правильную работу политики.

---

## 5. Нейросетевая архитектура: детальная схема

### 5.1 Actor Network: пошаговый forward pass

```
Observation (47 features):
┌────────────────────────────────────────────────┐
│ lidar[40]: [-1, 1]                             │
│ angular_vel[1]: [-1, 1]                        │
│ distance[1]: [-1, 1]                           │
│ sin_angle[1]: [-1, 1]                          │
│ cos_angle[1]: [-1, 1]                          │
│ prev_vx[1]: [-1, 1]                            │
│ prev_vy[1]: [-1, 1]                            │
│ prev_w[1]: [-1, 1]                             │
└────────────────┬───────────────────────────────┘
                 ▼
        Linear(47 → 512)
                 ▼
            ReLU()
                 ▼
        Linear(512 → 512)
                 ▼
            ReLU()
                 ▼
        Linear(512 → 256)
                 ▼
            ReLU()
                 ▼
        Linear(256 → 6)
                 ▼
┌────────────────────────────────────────────────┐
│ Split into mu[3] and log_std[3]               │
├────────────────────────────────────────────────┤
│ mu[0]: vx mean                                 │
│ mu[1]: vy mean                                 │
│ mu[2]: w mean                                  │
│ log_std[0]: vx log_std (raw)                   │
│ log_std[1]: vy log_std (raw)                   │
│ log_std[2]: w log_std (raw)                    │
└────────────────┬───────────────────────────────┘
                 ▼
    tanh(log_std) → [-1, 1]
                 ▼
    Scale to [-2, 2]: 
    log_std = -2 + 2 * (tanh_log_std + 1)
                 ▼
    std = exp(log_std) ∈ [0.135, 7.39]
                 ▼
┌────────────────────────────────────────────────┐
│            Sampling (Training)                 │
│   action_raw = mu + std * ε, ε ~ N(0,1)       │
│                    OR                          │
│         Deterministic (Inference)              │
│   action_raw = mu                              │
└────────────────┬───────────────────────────────┘
                 ▼
         action = tanh(action_raw)
                 ▼
┌────────────────────────────────────────────────┐
│        Action Output (3 values)                │
│ action[0]: vx ∈ [-1, 1]                        │
│ action[1]: vy ∈ [-1, 1]                        │
│ action[2]: w ∈ [-1, 1]                         │
└────────────────────────────────────────────────┘
```

### 5.2 Critic Network: пошаговый forward pass

```
Observation (53 features):
┌────────────────────────────────────────────────┐
│ lidar[40]: [-1, 1]                             │
│ vx[1]: [-1, 1]          ← Есть у Critic        │
│ angular_vel[1]: [-1, 1]                        │
│ distance[1]: [-1, 1]                           │
│ sin_angle[1]: [-1, 1]                          │
│ cos_angle[1]: [-1, 1]                          │
│ prev_vx[1]: [-1, 1]                            │
│ prev_vy[1]: [-1, 1]                            │
│ prev_w[1]: [-1, 1]                             │
│ critical_topk[5]: [-1, 1] ← Только у Critic   │
└────────────────┬───────────────────────────────┘
                 ▼
    Add History (3 previous obs)
    obs_stacked = concat([obs_t, obs_t-1, obs_t-2, obs_t-3])
                 ▼
        Size: 53 * 4 = 212
                 ▼
┌────────────────────────────────────────────────┐
│              Q1 Network                        │
├────────────────────────────────────────────────┤
│         Linear(212 → 512)                      │
│               ▼                                │
│           ReLU()                               │
│               ▼                                │
│         Linear(512 → 512)                      │
│               ▼                                │
│           ReLU()                               │
│               ▼                                │
│         Linear(512 → 256)                      │
│               ▼                                │
│           ReLU()                               │
│               ▼                                │
│         Linear(256 → 1)                        │
│               ▼                                │
│          Q1_value                              │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│              Q2 Network                        │
│         (Same structure, different weights)    │
│               ▼                                │
│          Q2_value                              │
└────────────────────────────────────────────────┘
                 ▼
    Q_min = min(Q1_value, Q2_value)
         (Used for training)
```

## 6. Интеграция с ROS2

### 6.1 lidar_2d_processor.py

**Назначение:** Обработка PointCloud2 от лидара и публикация данных для политики.

**Подписки:**
- `/livox/lidar` (PointCloud2) - сырые данные лидара

**Публикации:**
- `/lidar_observations` (Float32MultiArray) - **40 значений в метрах** (не нормализованные!)
- `/floor_points_2d` (PointCloud2) - визуализация
- `/obstacles_sectors` (MarkerArray) - визуализация

**Параметры:**
- `max_range = 3.0` м
- `min_range = 0.25` м
- `num_sectors = 40`

**Важно:** Процессор публикует **сырые значения в метрах**, нормализация должна выполняться в ноде инференса!

### 6.2 Нода инференса политики

**Требуемые подписки:**
1. `/lidar_observations` (Float32MultiArray) - 40 значений в метрах
2. TF: `map -> base_link` - позиция робота
3. `/clicked_point` (PointStamped) - целевая точка (опционально)
4. `/odom` (Odometry) - для vx и angular_vel (опционально, можно оценить из TF)

**Публикации:**
1. `/policy/cmd` (Twist) - команды управления (vx, vy, w в м/с и рад/с)

**Алгоритм работы (ОБНОВЛЕНО для 47 features):**

```python
# 1. Чтение лидара (40 значений в метрах)
lidar_raw = msg.data  # [40] float32, значения в [0.25, 3.0] м

# 2. Нормализация лидара
lidar_normalized = (np.clip(lidar_raw, 0, 3.0) / 3.0) * 2.0 - 1.0

# 3. Получение позиции робота из TF
robot_x, robot_y, robot_yaw = get_pose_from_tf()

# 4. Вычисление расстояния и угла до цели
dx = target_x - robot_x
dy = target_y - robot_y
distance = math.hypot(dx, dy)
angle = math.atan2(dy, dx) - robot_yaw
# Нормализация угла в [-π, π]
angle = np.arctan2(np.sin(angle), np.cos(angle))

# 5. Получение ТОЛЬКО angular_vel (НЕ vx!)
# ⚠️ ВАЖНО: Actor НЕ использует Vx, Vy!
angular_vel = get_angular_vel_from_odom()

# 6. Нормализация скалярных значений
angular_vel_norm = np.clip(angular_vel, -0.35, 0.35) / 0.35
distance_norm = (np.clip(distance, 0, 10.0) / 10.0) * 2.0 - 1.0
sin_angle = np.clip(math.sin(angle), -1, 1)
cos_angle = np.clip(math.cos(angle), -1, 1)

# 7. Предыдущие действия (нормализованные [-1, 1])
prev_action = [prev_vx_norm, prev_vy_norm, prev_w_norm]

# 8. Сборка наблюдения (47 features, БЕЗ Vx, Vy)
observation = np.concatenate([
    lidar_normalized,         # 40
    [angular_vel_norm],       # 1  ← ТОЛЬКО W, БЕЗ Vx!
    [distance_norm],          # 1
    [sin_angle],              # 1
    [cos_angle],              # 1
    prev_action               # 3
])  # Итого: 47 значений (было 48!)

# 9. Инференс ONNX
obs_batch = observation.reshape(1, -1)  # [1, 47]
action = onnx_session.run([output_name], {input_name: obs_batch})[0][0]

# 10. Масштабирование команд
vx_cmd = action[0] * 1.7
vy_cmd = action[1] * 1.5
w_cmd = action[2] * 0.35

# 11. Публикация команды
cmd_msg.linear.x = vx_cmd
cmd_msg.linear.y = vy_cmd
cmd_msg.angular.z = w_cmd
pub_cmd.publish(cmd_msg)

# 12. Сохранение предыдущего действия (нормализованного!)
prev_action = action.copy()
```

### 6.3 Частота работы

**lidar_2d_processor:**
- Частота: определяется частотой входящего PointCloud2 (обычно 10-20 Hz)
- Публикует данные при каждом новом сообщении

**policy_inference_node:**
- Частота: **10 Hz** (каждые 0.1 с) - критически важно!
- Использует последние доступные данные лидара

**locomotion (применение команд):**
- Частота: 50 Hz (как контроллер)
- Интерполирует команды от политики (10 Hz) для плавного управления

---

## 7. Критические моменты реализации

### 7.1 Размерность observation (КРИТИЧНО!)

**ОШИБКА (неправильно):**
```python
# НЕ ДЕЛАЙТЕ ТАК - старая размерность!
observation = np.concatenate([
    lidar_normalized,  # 40
    [vx_norm],         # 1  ← НЕТ! Vx больше не используется!
    [angular_vel_norm],# 1
    [distance_norm],   # 1
    [sin_angle],       # 1
    [cos_angle],       # 1
    prev_action        # 3
])  # 48 значений - УСТАРЕЛО!
```

**ПРАВИЛЬНО:**
```python
# Новая архитектура: 47 features, БЕЗ Vx, Vy
observation = np.concatenate([
    lidar_normalized,     # 40
    [angular_vel_norm],   # 1  ← Только W, БЕЗ Vx!
    [distance_norm],      # 1
    [sin_angle],          # 1
    [cos_angle],          # 1
    prev_action           # 3
])  # 47 значений - ПРАВИЛЬНО!

# ⚠️ КРИТИЧНО: НЕ добавляйте Vx и Vy!
```

### 7.2 Нормализация лидара

**ОШИБКА (неправильно):**
```python
# НЕ ДЕЛАЙТЕ ТАК - lidar_2d_processor уже нормализует!
lidar_normalized = lidar_data  # НЕТ! Это сырые метры!
```

**ПРАВИЛЬНО:**
```python
# lidar_2d_processor публикует СЫРЫЕ значения в метрах
lidar_raw = msg.data  # [40] float32, метры

# Нормализация должна быть в ноде инференса
lidar_normalized = (np.clip(lidar_raw, 0, 3.0) / 3.0) * 2.0 - 1.0
```

### 7.3 Предыдущие действия

**ОШИБКА (неправильно):**
```python
# НЕ ДЕЛАЙТЕ ТАК - это реальные команды, а не нормализованные!
prev_action = [vx_cmd, vy_cmd, w_cmd]  # НЕТ! Это в м/с и рад/с!
```

**ПРАВИЛЬНО:**
```python
# Сохраняйте нормализованные значения [-1, 1]
prev_action = action.copy()  # action уже в [-1, 1] из ONNX
```

### 7.4 Частота инференса

**ОШИБКА (неправильно):**
```python
# НЕ ДЕЛАЙТЕ ТАК - слишком часто или слишком редко
timer = create_timer(0.05, callback)  # 20 Hz - НЕТ!
timer = create_timer(0.5, callback)   # 2 Hz - НЕТ!
```

**ПРАВИЛЬНО:**
```python
# Частота должна быть 10 Hz (как при обучении)
timer = create_timer(0.1, callback)  # 10 Hz - ДА!
```

### 6.4 Масштабирование команд

**ОШИБКА (неправильно):**
```python
# НЕ ДЕЛАЙТЕ ТАК - забыли применить cmd_scale
vx_cmd = action[0]  # НЕТ! action в [-1, 1], нужно масштабировать!
```

**ПРАВИЛЬНО:**
```python
# Всегда применяйте cmd_scale
vx_cmd = action[0] * 1.7
vy_cmd = action[1] * 1.5
w_cmd = action[2] * 0.35
```

### 6.5 Обработка отсутствия данных

**Важно:** Всегда проверяйте наличие данных перед инференсом:

```python
if lidar_data is None or target_point is None or not have_pose:
    publish_stop_cmd()  # Остановить робота
    return
```

### 6.6 Достижение цели

**Порог достижения:** 0.2 м (из конфига `reached_threshold`)

```python
if distance < 0.2:
    publish_stop_cmd()
    target_point = None  # Сбросить цель
```

---

## 7. Проверочный список для интеграции

- [ ] ONNX модель загружена корректно
- [ ] **Проверка размерности: input shape = [batch, 47] (НЕ 48!)**
- [ ] lidar_2d_processor публикует `/lidar_observations` (40 значений в метрах)
- [ ] Нормализация лидара выполняется правильно: `(lidar / 3.0) * 2.0 - 1.0`
- [ ] TF `map -> base_link` доступен
- [ ] Целевая точка устанавливается (clicked_point или параметр)
- [ ] **ТОЛЬКО angular_vel** получается из odometry (БЕЗ Vx, Vy!)
- [ ] Предыдущие действия сохраняются в нормализованном виде [-1, 1]
- [ ] Частота инференса = 10 Hz (каждые 0.1 с)
- [ ] Масштабирование команд применяется: `cmd_scale = [1.7, 1.5, 0.35]`
- [ ] Обработка отсутствия данных (таймауты, остановка робота)
- [ ] Проверка достижения цели (порог 0.2 м)

---

## 8. Примеры кода

### 8.1 Полная функция построения наблюдения (ОБНОВЛЕНО!)

```python
def build_actor_observation(lidar_raw, robot_x, robot_y, robot_yaw, 
                           target_x, target_y, angular_vel, prev_action):
    """
    Строит наблюдение для Actor политики (47 значений).
    
    ⚠️ ВАЖНО: БЕЗ Vx, Vy (линейные скорости)!
    
    Args:
        lidar_raw: [40] float32, значения в метрах [0.25, 3.0]
        robot_x, robot_y, robot_yaw: позиция и ориентация робота
        target_x, target_y: целевая точка
        angular_vel: угловая скорость в рад/с (НЕ vx!)
        prev_action: [3] float32, предыдущие действия в [-1, 1]
    
    Returns:
        observation: [47] float32, все значения в [-1, 1]
    """
    # 1. Нормализация лидара
    lidar_normalized = (np.clip(lidar_raw, 0, 3.0) / 3.0) * 2.0 - 1.0
    
    # 2. Расстояние и угол до цели
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) - robot_yaw
    
    # Нормализация угла в [-π, π]
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    
    # 3. Нормализация скалярных значений (ТОЛЬКО angular_vel, БЕЗ vx!)
    angular_vel_norm = np.clip(angular_vel, -0.35, 0.35) / 0.35
    distance_norm = (np.clip(distance, 0, 10.0) / 10.0) * 2.0 - 1.0
    
    # 4. Сборка наблюдения (47 features, БЕЗ Vx, Vy)
    observation = np.concatenate([
        lidar_normalized,               # 40
        [angular_vel_norm],             # 1  ← ТОЛЬКО W!
        [distance_norm],                # 1
        [np.clip(sin_angle, -1, 1)],   # 1
        [np.clip(cos_angle, -1, 1)],   # 1
        np.clip(prev_action, -1, 1)    # 3
    ]).astype(np.float32)              # Итого: 47
    
    return observation
```

### 8.2 Полная функция инференса (ОБНОВЛЕНО!)

```python
def inference_step(onnx_session, observation, cmd_scale=[1.7, 1.5, 0.35]):
    """
    Выполняет инференс политики.
    
    Args:
        onnx_session: ONNX Runtime сессия
        observation: [47] float32, наблюдение в [-1, 1] (БЕЗ Vx, Vy!)
        cmd_scale: [3] float, масштабирование команд
    
    Returns:
        vx_cmd, vy_cmd, w_cmd: команды в м/с и рад/с
        action_norm: [3] float32, нормализованное действие [-1, 1]
    """
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    
    # Проверка размерности
    assert observation.shape[0] == 47, f"Expected 47 features, got {observation.shape[0]}"
    
    # Подготовка батча
    obs_batch = observation.reshape(1, -1)  # [1, 47]
    
    # Инференс
    outputs = onnx_session.run([output_name], {input_name: obs_batch})
    action_norm = outputs[0][0]  # [3] в [-1, 1]
    
    # Масштабирование
    vx_cmd = action_norm[0] * cmd_scale[0]
    vy_cmd = action_norm[1] * cmd_scale[1]
    w_cmd = action_norm[2] * cmd_scale[2]
    
    return vx_cmd, vy_cmd, w_cmd, action_norm
```

---

## 9. Экспорт модели в ONNX

### 9.1 Скрипт экспорта

Для экспорта обученной модели используется скрипт `export_to_onnx.py`.

#### Простое использование:

```bash
# Если файлы в текущей директории
python export_to_onnx.py --model_path sac_actor.pth --config_path g1.yaml --output_path sac_actor.onnx
```

#### С проверкой модели:

```bash
# Требует: pip install onnxruntime
python export_to_onnx.py --model_path sac_actor.pth --config_path g1.yaml --output_path sac_actor.onnx --verify
```

#### С полными путями:

```bash
python export_to_onnx.py \
    --model_path data/models/sac_actor.pth \
    --config_path configs/a1.yaml \
    --output_path sac_actor.onnx
```

#### Просмотр всех опций:

```bash
python export_to_onnx.py --help
```

**Параметры:**
- `--model_path`: Путь к файлу весов actor (`sac_actor.pth`)
- `--config_path`: Путь к конфигурационному файлу (`g1.yaml`)
- `--output_path`: Путь для сохранения ONNX модели
- `--opset_version`: Версия ONNX opset (рекомендуется 13)
- `--verify`: Проверить экспортированную модель с помощью onnxruntime
- `--no_verify`: Пропустить проверку
- `--quiet`: Не выводить подробную информацию

### 9.2 Требования для экспорта

#### Основные зависимости (уже в среде обучения):
- `torch >= 1.9.0` (PyTorch)
- `yaml` (PyYAML)
- `numpy`

#### Дополнительные зависимости (для проверки):
- `onnxruntime >= 1.8.0` (опционально, для проверки экспортированной модели)

#### Новая conda среда для экспорта (если несовместимы с unitree-rl):

Если среда обучения несовместима с unitree-rl, создайте новую среду:

```bash
# Создание новой среды
conda create -n onnx_export python=3.9
conda activate onnx_export

# Установка PyTorch (CPU версия достаточна для экспорта)
pip install torch>=1.9.0

# Установка остальных зависимостей
pip install pyyaml numpy

# Опционально: для проверки экспортированной модели
pip install onnxruntime
```

**Важно:** Экспорт работает на CPU, GPU не требуется.

### 9.3 Как работает экспорт

Скрипт экспорта:

1. **Загружает конфигурацию** из `g1.yaml` для получения параметров архитектуры
2. **Создает модель Actor** с правильными размерностями:
   - Вход: 47 признаков (без Vx, Vy, но с W)
   - Выход: 3 действия [vx, vy, w]
   - Архитектура: [512, 512, 256] нейронов
3. **Загружает веса** из `.pth` файла
4. **Создает обертку DeterministicActorWrapper**:
   - Возвращает `tanh(mu)` вместо полного распределения
   - Это соответствует `deterministic=True` режиму при инференсе
   - ONNX не поддерживает `torch.distributions` напрямую
5. **Экспортирует в ONNX** с динамическим batch размером
6. **Проверяет модель** (если `--verify`):
   - Загружает через `onnxruntime`
   - Выполняет тестовый инференс
   - Сравнивает с PyTorch выводом

### 9.4 Структура экспортированной модели

```
Input:  observation [batch_size, 47] float32 ∈ [-1, 1]
        ├─ lidar[40]: лидарные сектора
        ├─ angular_vel[1]: угловая скорость W
        ├─ distance[1]: расстояние до цели
        ├─ sin_angle[1]: синус угла к цели
        ├─ cos_angle[1]: косинус угла к цели
        └─ prev_action[3]: предыдущие действия
        
Output: action [batch_size, 3] float32 ∈ [-1, 1]
        ├─ action[0]: vx команда (масштаб: 1.7 м/с)
        ├─ action[1]: vy команда (масштаб: 1.5 м/с)
        └─ action[2]: w команда (масштаб: 0.35 рад/с)
```

### 9.5 Проверка экспортированной модели

После экспорта модель автоматически проверяется:

1. **Структура**: Проверяется что входы/выходы имеют правильные размерности
2. **Диапазон выхода**: Проверяется что выход в [-1, 1]
3. **Совместимость**: Сравнивается вывод ONNX с PyTorch выводом
4. **Производительность**: Выполняется тестовый инференс

**Ожидаемый результат:**
```
✓ Модель проверена и работает корректно!
  Максимальная разница с PyTorch: < 1e-5
  ✓ ONNX и PyTorch выводы идентичны
```

---

## 10. Процесс обучения: полная история проекта

### 10.1 Общая архитектура системы

Система состоит из нескольких компонентов:

1. **SAC Policy (Actor)** - генератор команд высокого уровня
   - Частота: 10 Hz
   - Вход: 47 признаков (lidar + состояние + предыдущие действия)
   - Выход: 3 команды [vx, vy, w] в [-1, 1]

2. **Walking Policy** - низкоуровневый контроллер
   - Частота: 50 Hz
   - Вход: 47 признаков (omega + gravity + cmd + qj + dqj + action + phase)
   - Выход: 12 DOF позиций для PD контроллера

3. **PD Controller** - физический контроллер
   - Частота: 500 Hz (simulation_dt = 0.002s)
   - Реализует позиционный контроль для 12 DOF робота

4. **Reward Function** - функция награды для обучения
   - Компоненты: success, collision, progress, obstacle_penalty, penalties

### 10.2 Ключевые решения в архитектуре

#### 10.2.1 Asymmetric Actor-Critic

**Проблема:** Стандартный SAC использует одинаковые observation spaces для Actor и Critic.

**Решение:** Раздельные observation spaces:
- **Actor**: 47 признаков (БЕЗ Vx, Vy линейных скоростей)
- **Critic**: 53 признака (С Vx + 5 critical_topk lidar beams)

**Обоснование:**
- Actor не знает линейные скорости → лучше обобщается на разные динамики
- Critic видит всё → точнее оценивает Q-value
- Actor знает W (угловую скорость) → может контролировать повороты

**Результат:** Улучшенное обобщение и стабильность обучения.

#### 10.2.2 Curriculum Learning

**Проблема:** Прямое обучение в сложной среде приводит к плохой сходимости.

**Решение:** 7 уровней curriculum с постепенным усложнением:

1. **Warm_Start** (Уровень 1): Низкая сложность, высокие награды
2. **Basic_Navigation** (Уровень 2): Средняя сложность
3. **Obstacle_Avoidance** (Уровень 3): Фокус на избегании препятствий
4. **Complex_Scenarios** (Уровень 4): Сложные сценарии
5. **High_Density** (Уровень 5): Высокая плотность препятствий
6. **Fast_Maneuvering** (Уровень 6): Быстрые маневры
7. **Polish_And_Perfection** (Уровень 7): Финальная полировка

**Адаптация параметров:**
- `init_temperature`: 0.1 → 0.8 (инвертирован для лучшего exploration)
- `alpha_lr`: 0.0003 → 0.00005 (снижается по мере обучения)
- `reward_weights`: Балансируются для каждого уровня

**Результат:** Стабильная сходимость с успешностью >95% на финальном уровне.

#### 10.2.3 Exploration vs Exploitation

**Проблема:** Высокий `init_temperature` приводил к "walking straight" поведению.

**Диагностика:**
- `log_std_bounds: [-5, 2]` позволяли std collapse (std → 0.0067)
- Даже с высоким alpha, обученный actor выдавал очень низкий std
- Это приводило к детерминистическим действиям

**Решение:**
1. Изменен `log_std_bounds: [-2, 2]` → предотвращает std collapse
2. Инвертирован `init_temperature` в curriculum:
   - Низкий на старте (0.1) → alpha растет при низкой энтропии
   - Высокий в конце (0.8) → alpha стабилизируется
3. Увеличена `target_entropy: -1.5` (было -3.0)
4. Улучшены `alpha_betas: [0.9, 0.999]` для стабильной оптимизации

**Результат:** Правильный баланс exploration/exploitation на всех стадиях обучения.

#### 10.2.4 Spawn Point Generation

**Проблема:** Робот и цель часто спавнились внутри препятствий.

**Диагностика:**
- `spawn_clearance: 0.5m` был недостаточным
- Robot radius: ~0.25m
- Collision threshold: 0.3m
- При clearance 0.5m: edge робота был на расстоянии 0.25m от препятствия
- Это меньше collision threshold 0.3m → немедленное столкновение

**Решение:**
- Увеличен `spawn_clearance: 0.7m` (0.3m threshold + 0.25m radius + 0.15m margin)
- Добавлен в конфиг `g1.yaml` под `obstacle_generator.spawn.clearance`
- Улучшен grid-based spawn generation с правильным clearance

**Результат:** Чистые эпизоды без немедленных столкновений.

### 10.3 История изменений архитектуры

#### Версия 1.0 (Исходная)
- Actor: 48 признаков (с Vx, W)
- Critic: 48 признаков (с Vx, W)
- Стандартный SAC без асимметрии

#### Версия 2.0 (Текущая)
- **Actor: 47 признаков** (БЕЗ Vx, Vy, но С W)
- **Critic: 53 признака** (С Vx + critical_topk)
- **Asymmetric Actor-Critic** архитектура
- **Curriculum Learning** с 7 уровнями
- **Улучшенная spawn generation** с clearance 0.7m
- **Исправлена статистика** (Success + Collision + Timeout = 100%)

### 10.4 Функция награды: детальная структура

```python
# Компоненты награды (из reward.py)

1. reached: +100.0
   - Условие: distance < 0.2m
   - Terminal reward

2. collision: -500.0
   - Условие: min_lidar < 0.3m
   - Terminal reward (очень высокий штраф)

3. obstacle_penalty: -5.0 * exp_penalty
   - Условие: min_lidar < 1.5m
   - Exponential: exp(-5.0 * (min_lidar - 0.3))
   - Поощряет держаться подальше от препятствий

4. progress: +20.0 * (prev_distance - distance)
   - Reward за движение к цели
   - Поощряет постоянный прогресс

5. vy_penalty: -0.3 * |vy|
   - Штраф за боковое движение
   - Поощряет движение вперед

6. vx_backward_penalty: -2.0 * max(0, -vx)
   - Штраф за движение назад
   - Поощряет движение вперед

7. time_penalty: -0.02
   - Небольшой штраф за каждый шаг
   - Поощряет эффективность

8. velocity_alignment: +0.5 * alignment_score
   - Reward за движение "face forward"
   - Опционально (weight может быть 0.0)
```

**Веса из конфига:**
```yaml
reached: 100.0
collision: -500.0
obs_penalty_weight: 5.0
progress: 20.0
time_penalty: -0.02
vy_penalty: -0.3
vx_backward_penalty: -1.0
velocity_alignment: 0.5
```

### 10.5 Гиперпараметры обучения

**Базовые параметры (g1.yaml):**
```yaml
actor_lr: 0.0001
critic_lr: 0.0003
alpha_lr: 0.0003
discount: 0.99
init_temperature: 0.2
target_entropy: -1.5
critic_tau: 0.005
log_std_bounds: [-2, 2]
alpha_betas: [0.9, 0.999]
batch_size: 256
training_iterations: 4
train_every_n: 1
```

**Curriculum адаптация:**
- `init_temperature`: 0.1 → 0.8 (по уровням)
- `alpha_lr`: 0.0003 → 0.00005 (снижается)
- `reward_weights`: Адаптируются для каждого уровня

### 10.6 Результаты обучения

**Типичные метрики на финальном уровне (Polish_And_Perfection):**
- Success Rate: 96-98%
- Collision Rate: 1-2%
- Timeout Rate: 1-2%
- Excluded (errors): <1% (битые эпизоды не учитываются)

**Поведение:**
- Плавное движение к цели
- Эффективное избегание препятствий
- Минимальное боковое движение
- Предпочтение прямого движения вперед

---

## 11. Резюме ключевых изменений версии 2.0

### Архитектура
- ✅ Asymmetric Actor-Critic (Actor: 47, Critic: 53)
- ✅ Actor БЕЗ Vx, Vy, но С W
- ✅ Critic С Vx + critical_topk (5 features)

### Обучение
- ✅ Curriculum Learning (7 уровней)
- ✅ Инвертированный init_temperature (0.1 → 0.8)
- ✅ Исправлен log_std_bounds: [-2, 2]
- ✅ Улучшенная spawn generation (clearance: 0.7m)

### Статистика
- ✅ Исправлена синхронизация списков
- ✅ Исключение битых эпизодов из статистики
- ✅ Гарантия Success + Collision + Timeout = 100%

### Экспорт
- ✅ Скрипт export_to_onnx.py
- ✅ DeterministicActorWrapper для ONNX
- ✅ Автоматическая проверка экспортированной модели

---

## 12. Структура проекта

```
Dog_PathPlanning/
├── src/                          # Исходный код
│   ├── policy/                   # RL алгоритмы
│   │   └── SAC/
│   │       ├── SAC_actor.py     # Actor архитектура
│   │       ├── SAC_critic.py    # Critic архитектура
│   │       ├── SAC.py           # SAC алгоритм
│   │       └── SAC_utils.py     # Утилиты
│   └── utils/                    # Утилиты
│       ├── reward.py            # Функция награды
│       ├── target_generator.py  # Генерация целей
│       ├── scene_generator.py   # Генерация препятствий
│       ├── curriculum.py        # Curriculum learning manager
│       ├── observation.py       # Функции построения наблюдений
│       └── mjx_utils.py         # MJX утилиты для параллельной симуляции
├── scripts/                      # Исполняемые скрипты
│   ├── train.py                 # Основной скрипт обучения
│   ├── export_to_onnx.py        # Экспорт в ONNX
│   └── inference_onnx.py        # Инференс ONNX модели
├── configs/                      # Конфигурационные файлы
│   ├── a1.yaml                  # Конфигурация для A1 робота
│   └── curriculum.yaml          # Curriculum learning (7 уровней)
├── data/                         # Данные (модели, логи, буферы)
│   ├── models/                  # Обученные модели
│   │   ├── sac_actor.pth        # Веса Actor
│   │   ├── sac_critic.pth       # Веса Critic
│   │   └── sac_metadata.json    # Метаданные (episode number)
│   ├── buffer/                  # Replay buffer данные
│   └── runs/                    # TensorBoard логи
├── assets/                       # Статические ресурсы
│   └── unitree_a1/              # Модель робота Unitree A1
├── docs/                         # Документация
│   └── POLICY_DOCUMENTATION.md  # Эта документация
└── requirements.txt              # Зависимости Python
```

---

## 13. Чеклист для интеграции на реальном роботе

### 13.1 Подготовка модели

- [ ] Экспортировать модель в ONNX: `python export_to_onnx.py`
- [ ] Проверить размерность входов (47 признаков)
- [ ] Проверить размерность выходов (3 действия в [-1, 1])
- [ ] Проверить диапазон выходов (должен быть в [-1, 1])
- [ ] Скопировать ONNX модель в репозиторий инференса

### 13.2 Подготовка ROS2 ноды

- [ ] Установить `onnxruntime` в среде ROS2
- [ ] Создать ноду подписки на `/lidar_observations`
- [ ] Интегрировать TF listener для позиции робота
- [ ] Реализовать `build_actor_observation()` функцию
- [ ] Убедиться что **НЕ добавляются Vx, Vy** в observation!
- [ ] Реализовать инференс ONNX модели
- [ ] Применить масштабирование команд: [1.7, 1.5, 0.35]
- [ ] Публиковать команды на `/policy/cmd` с частотой 10 Hz

### 13.3 Тестирование

- [ ] Проверить нормализацию лидара: `(lidar / 3.0) * 2.0 - 1.0`
- [ ] Проверить нормализацию angular_vel: `clip(w, -0.35, 0.35) / 0.35`
- [ ] Проверить нормализацию distance: `(clip(dist, 0, 10.0) / 10.0) * 2.0 - 1.0`
- [ ] Проверить что prev_action сохраняется в нормализованном виде [-1, 1]
- [ ] Проверить частоту инференса (должна быть 10 Hz)
- [ ] Проверить масштабирование команд перед публикацией
- [ ] Протестировать на простом сценарии (одна цель, нет препятствий)
- [ ] Протестировать на сценарии с препятствиями

### 13.4 Проверка совместимости

- [ ] Убедиться что `lidar_2d_processor` публикует 40 значений в метрах
- [ ] Проверить формат сообщения `/lidar_observations` (Float32MultiArray)
- [ ] Проверить доступность TF `map -> base_link`
- [ ] Проверить формат целевой точки (PointStamped или параметр)
- [ ] Убедиться что odometry доступна для angular_vel (или оценить из TF)

---

## 14. Часто задаваемые вопросы (FAQ)

### Q1: Почему Actor не имеет Vx, Vy?

**A:** Это решение для улучшения обобщения. Actor, не зная текущие линейные скорости, учится принимать решения на основе:
- Визуальной информации (lidar)
- Положения цели
- Предыдущих команд
- Угловой скорости (W) для контроля поворотов

Это позволяет политике работать в разных динамических условиях (скольжение, разные поверхности и т.д.).

### Q2: Почему нужно применять масштабирование команд?

**A:** Модель выдает действия в диапазоне [-1, 1] для нормализации. Эти значения нужно масштабировать на реальные единицы:
- `vx_cmd = action[0] * 1.7` м/с
- `vy_cmd = action[1] * 1.5` м/с
- `w_cmd = action[2] * 0.35` рад/с

### Q3: Что делать если Success + Collision + Timeout ≠ 100%?

**A:** Это означает что есть "битые" эпизоды с ошибками (NaN, exceptions). В текущей версии эти эпизоды исключаются из статистики. Проверьте логи на наличие сообщений `"Episode X has errors - EXCLUDED from statistics"`.

### Q4: Можно ли использовать другую частоту инференса?

**A:** НЕТ! Частота 10 Hz критична, так как модель обучалась с этой частотой. Изменение частоты может привести к нестабильному поведению.

### Q5: Почему используется ONNX, а не PyTorch напрямую?

**A:** ONNX - стандартный формат для межплатформенного инференса. Это позволяет:
- Использовать модель на разных платформах (CPU, GPU, TPU)
- Интегрировать с различными языками (C++, Python, Rust и т.д.)
- Оптимизировать для мобильных устройств
- Использовать специализированные ускорители (TensorRT, CoreML и т.д.)

### Q6: Что делать если модель выдает действия вне [-1, 1]?

**A:** Это серьезная проблема. Проверьте:
1. Корректность экспорта (запустите `export_to_onnx.py --verify`)
2. Нормализацию входных данных (все должны быть в [-1, 1])
3. Версию PyTorch и ONNX opset (рекомендуется opset 13)
4. Если проблема сохраняется, переэкспортируйте модель

### Q7: Как обрабатывать отсутствие данных (лидар, TF, цель)?

**A:** Всегда проверяйте наличие данных перед инференсом:
```python
if lidar_data is None or not have_robot_pose or target_point is None:
    publish_stop_cmd()  # Остановить робота
    return  # Пропустить инференс
```

### Q8: Нужно ли обновлять prev_action при каждом шаге?

**A:** Да! `prev_action` - это предыдущая **нормализованная** команда (в [-1, 1]), а не реальная команда в м/с. Сохраняйте выход модели (до масштабирования) как `prev_action` для следующего шага.

---

## 15. Техническая поддержка

При возникновении проблем:

1. **Проверьте чеклист** (раздел 13)
2. **Проверьте логи** на наличие ошибок и предупреждений
3. **Проверьте форматы данных** (размерности, диапазоны)
4. **Проверьте частоту инференса** (должна быть 10 Hz)
5. **Проверьте масштабирование команд**

### Критические моменты для отладки:

- Размерность observation: **ДОЛЖНА быть 47** (не 48!)
- Отсутствие Vx, Vy в observation для Actor
- Наличие W (angular_vel) в observation
- Частота инференса: **10 Hz** (каждые 0.1 секунды)
- Масштабирование команд: `[1.7, 1.5, 0.35]`
- Нормализация всех компонентов в [-1, 1]

---

**Версия документа:** 2.0  
**Последнее обновление:** 2025-01-17  
**Автор:** G1 Path Planning Team  
**Лицензия:** Внутреннее использование  

---

## Приложение A: Зависимости для экспорта в ONNX

### Текущая среда обучения

Если среда обучения совместима с unitree-rl, используйте её напрямую:

```bash
# Экспорт в текущей среде
python scripts/export_to_onnx.py --model_path data/models/sac_actor.pth --output_path sac_actor.onnx
```

### Новая conda среда (если несовместимы)

Если среда обучения несовместима с unitree-rl (например, использует mujoco-mjx, jax и т.д.), создайте новую среду:

```bash
# Создание новой среды
conda create -n onnx_export python=3.9
conda activate onnx_export

# Установка зависимостей
pip install torch>=1.9.0
pip install pyyaml numpy
pip install onnxruntime  # Опционально, для проверки модели

# Экспорт
python scripts/export_to_onnx.py --model_path data/models/sac_actor.pth --output_path sac_actor.onnx
```

**Важно:**
- GPU не требуется для экспорта (работает на CPU)
- PyTorch CPU версия достаточна
- `onnxruntime` нужен только для проверки (`--verify`)

---

## Приложение A: Полный список изменений версии 2.0

### Архитектура
- ✅ Asymmetric Actor-Critic (Actor: 47, Critic: 53)
- ✅ Actor БЕЗ Vx, Vy, но С W
- ✅ Critic С Vx + critical_topk (5 features)

### Обучение
- ✅ Curriculum Learning (7 уровней)
- ✅ Инвертированный init_temperature (0.1 → 0.8)
- ✅ Исправлен log_std_bounds: [-2, 2]
- ✅ Улучшенная spawn generation (clearance: 0.7m)

### Статистика
- ✅ Исправлена синхронизация списков
- ✅ Исключение битых эпизодов из статистики
- ✅ Гарантия Success + Collision + Timeout = 100%

### Экспорт
- ✅ Скрипт export_to_onnx.py
- ✅ DeterministicActorWrapper для ONNX
- ✅ Автоматическая проверка экспортированной модели

---

## Приложение B: Структура проекта

См. раздел 12 выше для актуальной структуры проекта.

---

**Конец документации.**
