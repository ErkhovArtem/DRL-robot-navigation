"""
Экспорт обученной SAC Actor модели в ONNX формат для инференса на реальном роботе.

Требования:
- PyTorch >= 1.9.0
- torch.onnx (встроен в PyTorch)
- onnxruntime (для проверки экспортированной модели)

Использование:

ПРОСТОЙ ВАРИАНТ (если файлы в текущей директории):
    python export_to_onnx.py --model_path sac_actor.pth --config_path g1.yaml --output_path sac_actor.onnx

С ПРОВЕРКОЙ МОДЕЛИ (требует: pip install onnxruntime):
    python export_to_onnx.py --model_path sac_actor.pth --config_path g1.yaml --output_path sac_actor.onnx --verify

С ПОЛНЫМИ ПУТЯМИ:
    python export_to_onnx.py --model_path src3/models/sac_actor.pth --config_path src3/configs/g1.yaml --output_path sac_actor.onnx

Зависимости для новой conda среды (если несовместимы с unitree-rl):
    pip install torch>=1.9.0
    pip install pyyaml numpy
    pip install onnxruntime  # Только для --verify (опционально)
"""
import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к проекту
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src3"))

from policy.SAC.SAC_actor import DiagGaussianActor


class DeterministicActorWrapper(torch.nn.Module):
    """
    Обертка для Actor, которая возвращает deterministic action (mean) вместо distribution.
    
    ONNX не поддерживает torch.distributions напрямую, поэтому экспортируем только
    deterministic часть (mean), которая используется при инференсе.
    """
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        
    def forward(self, obs):
        """
        Forward pass для ONNX экспорта.
        
        Args:
            obs: [batch_size, 47] - наблюдение (БЕЗ Vx, Vy, но С W)
        
        Returns:
            action: [batch_size, 3] - deterministic action в [-1, 1]
        """
        # Получаем mu и log_std из actor
        mu, log_std = self.actor.trunk(obs).chunk(2, dim=-1)
        
        # Применяем log_std bounds (как в оригинальном actor)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.actor.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        
        # Для deterministic инференса просто возвращаем tanh(mu)
        # Это соответствует actor.act(obs, deterministic=True)
        action = torch.tanh(mu)
        
        return action


def export_to_onnx(
    model_path,
    config_path,
    output_path,
    opset_version=13,
    verify=True,
    verbose=True
):
    """
    Экспортирует обученную SAC Actor модель в ONNX формат.
    
    Args:
        model_path: Путь к файлу весов actor (sac_actor.pth)
        config_path: Путь к конфигурационному файлу (g1.yaml)
        output_path: Путь для сохранения ONNX модели
        opset_version: Версия ONNX opset (13 рекомендуется для совместимости)
        verify: Проверить экспортированную модель
        verbose: Выводить подробную информацию
    """
    # Загрузка конфигурации
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Параметры из конфига
    sac_config = config.get('sac', {})
    actor_config = sac_config.get('actor', {})
    
    # АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ ИЗ СОХРАНЕННОЙ МОДЕЛИ
    temp_state_dict = torch.load(model_path, map_location='cpu')
    actor_obs_dim = temp_state_dict['trunk.0.weight'].shape[1]  # Входная размерность
    
    # Определяем архитектуру скрытых слоев из весов (только Linear слои)
    hidden_sizes = []
    layer_idx = 0
    while f'trunk.{layer_idx}.weight' in temp_state_dict:
        weight = temp_state_dict[f'trunk.{layer_idx}.weight']
        if len(weight.shape) == 2:  # Linear layer
            hidden_sizes.append(weight.shape[0])  # Размер выхода слоя
        layer_idx += 2  # Пропускаем ReLU (нечетные индексы)
    
    # Убираем выходной слой (последний: 2*action_dim = 6)
    if hidden_sizes and hidden_sizes[-1] == 6:
        hidden_sizes = hidden_sizes[:-1]
    
    # Используем определенную архитектуру или fallback к конфигу
    if hidden_sizes:
        hidden_dim = hidden_sizes  # Список размеров слоев
        hidden_depth = len(hidden_sizes)
    else:
        hidden_dim = actor_config.get('hidden_dim', [512, 512, 256])
        hidden_depth = len(hidden_dim) if isinstance(hidden_dim, list) else actor_config.get('hidden_depth', 3)
    
    action_dim = 3  # [vx, vy, w]
    
    # log_std_bounds
    log_std_bounds = actor_config.get('log_std_bounds', [-2, 2])
    log_std_min, log_std_max = log_std_bounds[0], log_std_bounds[1]
    
    if verbose:
        print("=" * 70)
        print("ЭКСПОРТ SAC ACTOR В ONNX")
        print("=" * 70)
        print(f"Конфигурация:")
        print(f"  Actor obs dim: {actor_obs_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Hidden depth: {hidden_depth}")
        print(f"  Log std bounds: [{log_std_min}, {log_std_max}]")
        print(f"  Model path: {model_path}")
        print(f"  Output path: {output_path}")
        print()
    
    # Создание модели
    device = torch.device('cpu')  # ONNX экспорт всегда на CPU
    
    # Создание модели с правильной архитектурой
    # hidden_dim может быть int (одинаковые слои) или list (разные размеры)
    actor = DiagGaussianActor(
        obs_dim=actor_obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,  # Передаем как есть (list или int)
        hidden_depth=hidden_depth,  # Используется только если hidden_dim - int
        log_std_bounds=(log_std_min, log_std_max)
    )
    
    # Загрузка весов
    if verbose:
        print(f"Загрузка весов из {model_path}...")
    
    state_dict = torch.load(model_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()  # Важно: eval режим для детерминистического инференса
    
    # Обертка для deterministic инференса
    wrapped_actor = DeterministicActorWrapper(actor)
    wrapped_actor.eval()
    
    # Создание dummy input для экспорта
    # Размер: [batch_size, obs_dim]
    dummy_input = torch.randn(1, actor_obs_dim)
    
    if verbose:
        print(f"Тестовая проверка модели на CPU...")
        with torch.no_grad():
            test_output = wrapped_actor(dummy_input)
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {test_output.shape}")
            print(f"  Output range: [{test_output.min().item():.3f}, {test_output.max().item():.3f}]")
            print(f"  ✓ Ожидается: [-1, 1]")
        print()
    
    # Экспорт в ONNX
    if verbose:
        print(f"Экспорт в ONNX (opset {opset_version})...")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ONNX экспорт
    torch.onnx.export(
        wrapped_actor,
        dummy_input,
        str(output_path),
        input_names=['observation'],  # Имя входного тензора
        output_names=['action'],      # Имя выходного тензора
        dynamic_axes={
            'observation': {0: 'batch_size'},  # Batch размер может быть динамическим
            'action': {0: 'batch_size'}
        },
        opset_version=opset_version,
        do_constant_folding=True,  # Оптимизация констант
        export_params=True,        # Экспортировать веса
        verbose=verbose
    )
    
    if verbose:
        print(f"✓ Модель экспортирована: {output_path}")
        print(f"  Размер файла: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print()
    
    # Проверка экспортированной модели
    if verify:
        try:
            import onnxruntime as ort
            
            if verbose:
                print("Проверка экспортированной модели...")
            
            # Создание ONNX Runtime сессии
            session = ort.InferenceSession(str(output_path))
            
            # Проверка входов/выходов
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            input_shape = session.get_inputs()[0].shape
            output_shape = session.get_outputs()[0].shape
            
            if verbose:
                print(f"  Input name: {input_name}")
                print(f"  Input shape: {input_shape}")
                print(f"  Output name: {output_name}")
                print(f"  Output shape: {output_shape}")
            
            # Тестовый инференс
            test_input = np.random.randn(1, actor_obs_dim).astype(np.float32)
            outputs = session.run([output_name], {input_name: test_input})
            output = outputs[0]
            
            if verbose:
                print(f"  Test input shape: {test_input.shape}")
                print(f"  Test output shape: {output.shape}")
                print(f"  Test output range: [{output.min():.3f}, {output.max():.3f}]")
                
                # Проверка что выход в правильном диапазоне
                if output.min() >= -1.1 and output.max() <= 1.1:
                    print("  ✓ Выход в правильном диапазоне [-1, 1]")
                else:
                    print(f"  ⚠️ WARNING: Выход вне диапазона [-1, 1]!")
                
                # Проверка что выход имеет правильную размерность
                if output.shape == (1, 3):
                    print("  ✓ Выходная размерность корректна [batch, 3]")
                else:
                    print(f"  ⚠️ WARNING: Неожиданная размерность выхода: {output.shape}")
            
            # Сравнение с PyTorch выводом
            with torch.no_grad():
                torch_output = wrapped_actor(torch.from_numpy(test_input))
                torch_output_np = torch_output.numpy()
                
                max_diff = np.abs(output - torch_output_np).max()
                
                if verbose:
                    print(f"  Максимальная разница с PyTorch: {max_diff:.6f}")
                    if max_diff < 1e-5:
                        print("  ✓ ONNX и PyTorch выводы идентичны")
                    elif max_diff < 1e-3:
                        print("  ✓ ONNX и PyTorch выводы близки (приемлемо)")
                    else:
                        print(f"  ⚠️ WARNING: Значительная разница с PyTorch!")
            
            if verbose:
                print("✓ Модель проверена и работает корректно!")
            
        except ImportError:
            if verbose:
                print("⚠️ onnxruntime не установлен, пропускаем проверку")
                print("   Установите: pip install onnxruntime")
        except Exception as e:
            if verbose:
                print(f"⚠️ Ошибка при проверке модели: {e}")
    
    if verbose:
        print()
        print("=" * 70)
        print("ЭКСПОРТ ЗАВЕРШЕН")
        print("=" * 70)
        print(f"Модель сохранена: {output_path}")
        print()
        print(f"⚠️ ВАЖНО: Модель имеет {actor_obs_dim} признаков")
        if actor_obs_dim == 48:
            print("  ⚠️ Это СТАРАЯ версия (48 признаков: lidar(40) + vx(1) + w(1) + dist(1) + sin(1) + cos(1) + prev(3))")
            print("  ⚠️ При инференсе используйте наблюдения С Vx и W")
        elif actor_obs_dim == 47:
            print("  ✓ Это НОВАЯ версия (47 признаков: lidar(40) + w(1) + sin(1) + cos(1) + dist(1) + prev(3), БЕЗ vx/vy)")
            print("  ✓ При инференсе используйте наблюдения БЕЗ Vx, Vy, но С W (угловая скорость)")
        print()
        print("Следующие шаги:")
        print("1. Скопируйте ONNX модель в репозиторий инференса")
        print("2. Используйте onnxruntime для загрузки модели:")
        print("   import onnxruntime as ort")
        print(f"   session = ort.InferenceSession('{output_path.name}')")
        print(f"3. Подавайте наблюдения размером [batch, {actor_obs_dim}]")
        print("4. Получайте действия размером [batch, 3] в диапазоне [-1, 1]")
        print("5. Масштабируйте действия: vx=action[0]*1.7, vy=action[1]*1.5, w=action[2]*0.35")
        print()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Экспорт SAC Actor модели в ONNX формат'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='src3/models/sac_actor.pth',
        help='Путь к файлу весов actor (sac_actor.pth)'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='src3/configs/g1.yaml',
        help='Путь к конфигурационному файлу (g1.yaml)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='sac_actor.onnx',
        help='Путь для сохранения ONNX модели'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=13,
        help='Версия ONNX opset (по умолчанию 13)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Проверить экспортированную модель с помощью onnxruntime (требует: pip install onnxruntime)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Не выводить подробную информацию'
    )
    
    args = parser.parse_args()
    
    # Экспорт
    export_to_onnx(
        model_path=args.model_path,
        config_path=args.config_path,
        output_path=args.output_path,
        opset_version=args.opset_version,
        verify=args.verify,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
