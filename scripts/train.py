"""
Training script for SAC cmd generation with FULL MJX (MuJoCo XLA) parallelization.
Uses deploy_mujoco.py as environment - SAC generates cmd, which is fed to walking policy.

MJX Integration:
- FULL batched parallel simulation implemented (--use_mjx flag)
- Parallel simulation of multiple episodes on GPU/TPU using JAX vmap
- JIT-compiled step functions for maximum performance
- Automatic batch dimension handling in MJX data structures
- Compatible with existing SAC training pipeline

Paths:
- All paths (models, logs, configs, buffers) default to src/ directory

Installation:
- For MJX support: pip install mujoco-mjx
- MJX enables GPU/TPU acceleration for parallel simulation (10-100x speedup)

Usage:
- Sequential mode: python train.py configs/g1.yaml --train --headless
- MJX mode: python train.py configs/g1.yaml --train --headless --use_mjx --batch_size 128
"""
import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import argparse
from pathlib import Path
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MJX imports (if available)
try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
except ImportError:
    jax = None
    jnp = None
    mjx = None

# Import from package (after pip install -e .)
from utils.reward import compute_reward, compute_reward_vectorized, compute_reward_reference, compute_reward_reference_vectorized
from utils.target_generator import SpawnPointGenerator, get_target_info, ROOM_X_MIN, ROOM_X_MAX, ROOM_Y_MIN, ROOM_Y_MAX
from utils.scene_generator import regenerate_scene_obstacles
from policy.SAC.SAC import SAC
from policy.replay_buffer import ReplayBuffer
from utils.curriculum import CurriculumManager
from utils.observation import (
    get_gravity_orientation, build_walking_policy_observation,
    downsample_lidar, fix_negative_lidar_values, compute_lidar_sensor_angles,
    process_lidar_to_sectors, transform_lidar_to_center_frame,
    build_actor_observation, build_critic_base_observation,
    build_critic_observation
)
from utils.mjx_utils import create_mjx_batched_step_fn, run_batched_episodes_mjx_full, MJX_AVAILABLE
from policy.walking_policy import load_walking_policy_from_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name")
    parser.add_argument("--train", action="store_true", help="train mode (if not specified, runs in test/evaluation mode)")
    parser.add_argument("--load_pretrained", action="store_true", help="in training mode: load last checkpoint and continue training; ignored in test mode (test mode always loads latest checkpoint)")
    parser.add_argument("--fine_tune", action="store_true", help="fine-tuning mode: loads weights but resets entropy/buffer/optimizers")
    parser.add_argument("--log_dir", type=str, default="runs", help="tensorboard log dir")
    parser.add_argument("--sac_decimation", type=int, default=5, help="run SAC policy every N control cycles (on top of control_decimation)")
    parser.add_argument("--episodes", type=int, default=1000, help="number of training episodes")
    parser.add_argument("--headless", action="store_true", help="run without visualization (faster training)")
    parser.add_argument("--fresh_buffer", action="store_true", help="start with empty replay buffer even if loading pretrained model")
    parser.add_argument("--spawn_clearance", type=float, default=0.7, help="clearance from obstacles for spawn points (meters, accounts for robot radius)")
    parser.add_argument("--grid_step", type=float, default=0.1, help="grid step size for free space generation (meters)")
    parser.add_argument("--save_every_n", type=int, default=100, help="save model every N episodes")
    parser.add_argument("--use_mjx", action="store_true", help="use MJX for parallel batched simulation (requires mujoco-mjx)")
    parser.add_argument("--batch_size", type=int, default=128, help="number of parallel environments for MJX (only used with --use_mjx)")
    args = parser.parse_args()
   
    # Load config file - try multiple possible locations
    config_path = None
    config_file_path = Path(args.config_file)
    
    # Build potential paths to try (prioritize configs/)
    potential_paths = [
        config_file_path,  # Relative to current directory
        Path.cwd() / config_file_path,  # Explicitly from current directory
        PROJECT_ROOT / "configs" / config_file_path.name,  # configs/filename.yaml
        PROJECT_ROOT / "configs" / config_file_path,  # configs/g1.yaml (if relative path)
    ]
    
    # Try each path until one exists
    for path in potential_paths:
        path_resolved = path.resolve()
        if path_resolved.exists() and path_resolved.is_file():
            config_path = path_resolved
            break
    
    if config_path is None:
        raise FileNotFoundError(
            f"Config file not found: {args.config_file}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"PROJECT_ROOT: {PROJECT_ROOT}\n"
            f"Tried paths:\n" + "\n".join(f"  - {p.resolve() if p.resolve().exists() else p} (exists: {p.resolve().exists()})" for p in potential_paths)
        )
    
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Load curriculum config if exists
    curriculum_config = {}
    curriculum_path = PROJECT_ROOT / "configs" / "curriculum.yaml"
    if curriculum_path.exists():
        print(f"Loading curriculum config from: {curriculum_path}")
        with open(curriculum_path, "r") as f:
            curriculum_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print(f"Curriculum config not found at {curriculum_path}. Curriculum learning disabled.")
    
    # Resolve paths relative to project root
    walking_policy_path = str(PROJECT_ROOT / config.get("walking_policy_path", ""))
    xml_path = str(PROJECT_ROOT / config["xml_path"])
    
    # Get spawn parameters from config (with command-line override)
    spawn_config = config.get("obstacle_generator", {}).get("spawn", {})
    spawn_clearance = spawn_config.get("clearance", args.spawn_clearance)
    spawn_grid_step = spawn_config.get("grid_step", args.grid_step)
    
    # Initialize spawn point generator (parses obstacles from XML and generates free points grid)
    spawn_generator = SpawnPointGenerator(
        xml_path=xml_path,
        clearance=spawn_clearance,
        grid_step=spawn_grid_step
    )
    
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    # Get sac_decimation from config, but allow command line argument to override
    # Priority: command line argument > config > default (5)
    if "sac_decimation" in config:
        sac_decimation = config["sac_decimation"]
        # If user explicitly set it via command line, override config value
        if args.sac_decimation != 5:  # Default is 5, so != 5 means user set it
            sac_decimation = args.sac_decimation
            print(f"Using sac_decimation={sac_decimation} from command line (overrides config)")
        else:
            print(f"Using sac_decimation={sac_decimation} from config")
    else:
        sac_decimation = args.sac_decimation
        print(f"Using sac_decimation={sac_decimation} from command line (default)")
    # Calculate and print frequency information
    sim_freq = 1.0 / simulation_dt
    control_freq = sim_freq / control_decimation
    policy_freq = control_freq / sac_decimation
    print(f"\n=== Frequency Configuration ===")
    print(f"Simulation: {sim_freq:.1f} Hz (dt={simulation_dt*1000:.1f} ms)")
    print(f"Control: {control_freq:.1f} Hz (decimation={control_decimation})")
    print(f"Policy (SAC): {policy_freq:.1f} Hz (decimation={sac_decimation})")
    print(f"Policy period: {simulation_dt*control_decimation*sac_decimation*1000:.1f} ms")
    print(f"Lidar data update: {policy_freq:.1f} Hz (synchronized with policy)")
    print("=" * 40 + "\n")
    
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    num_actions = config["num_actions"]
    # Use per-joint action scales if available, otherwise fall back to single action_scale
    if "action_scales" in config:
        action_scales = np.array(config["action_scales"], dtype=np.float32)
    else:
        action_scale = config.get("action_scale", 0.25)
        action_scales = np.full(num_actions, action_scale, dtype=np.float32)
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_obs = config["num_obs"]
    
    # Load reward weights from config
    reward_weights = config["reward_weights"]
    
    # Load lidar config
    lidar_config = config.get("lidar", {})
    lidar_offset_x = lidar_config.get("offset_x", 0.12)
    lidar_offset_y = lidar_config.get("offset_y", 0.0)
    # Observation noise for Actor only (sim-to-real); Critic gets clean data
    obs_noise_config = config.get("observation_noise", {})
    obs_noise_distance_std = obs_noise_config.get("distance_std", 0.0)
    obs_noise_angle_std = obs_noise_config.get("angle_std", 0.0)
    obs_noise_angular_vel_std = obs_noise_config.get("angular_vel_std", 0.0)
    obs_noise_lidar_std = obs_noise_config.get("lidar_std", 0.0)
    if any(x > 0 for x in [obs_noise_distance_std, obs_noise_angle_std, obs_noise_angular_vel_std, obs_noise_lidar_std]):
        print(f"Observation noise (Actor only): dist={obs_noise_distance_std}m, angle={obs_noise_angle_std}rad, "
              f"ang_vel={obs_noise_angular_vel_std}rad/s, lidar={obs_noise_lidar_std}m")
    print(f"Lidar offset: ({lidar_offset_x}m, {lidar_offset_y}m) - transform to center frame enabled")
    
    # Initialize Curriculum Manager
    curriculum_manager = None
    if curriculum_config:
        curriculum_manager = CurriculumManager(curriculum_config, config)
        print(f"Curriculum Manager initialized. Starting at: {curriculum_manager.get_current_level_name()}")
    
    # Obstacle parameters container for curriculum updates
    # Initialized from config, can be overridden by curriculum
    obstacle_params = {
        'regeneration_interval': config.get("obstacle_generator", {}).get("regeneration_interval", 10),
        'cubes': config.get("obstacle_generator", {}).get("cubes", {}).copy(),
        'position': config.get("obstacle_generator", {}).get("position", {}).copy()
    }
    
    # Episode parameters (max_steps, etc.) - can be overridden by curriculum
    episode_params = {'max_steps': config.get("max_steps", 5000)}
    
    # Apply initial curriculum level (if any)
    # Note: policy_config will be created later, so we'll apply curriculum after policy_config is ready

    # Extract policy configuration (always SAC)
    policy_config = config.get("sac", {}).copy()  # Make a copy to allow curriculum updates
    if not policy_config:
        print(f"Warning: Configuration for SAC not found in config. Using defaults.")
    
    # Apply initial curriculum level (if any) - updates policy_config and replay_buffer config
    replay_buffer_config = config.get("replay_buffer", {}).copy()
    if curriculum_manager:
        # Inference mode: use maximum difficulty (last level) from the start
        if not args.train and curriculum_manager.levels:
            curriculum_manager.current_level_idx = len(curriculum_manager.levels) - 1
            curriculum_manager.episodes_on_current_level = 0
            print(f"Inference mode: forcing maximum difficulty (Level {curriculum_manager.current_level_idx + 1})")
        curriculum_manager.apply_current_level(
            reward_weights, obstacle_params, 
            policy_config=policy_config, 
            replay_buffer_config=replay_buffer_config,
            episode_params=episode_params
        )
        print(f"Applied curriculum Level {curriculum_manager.current_level_idx + 1} settings.")
    
    # Shared parameters extracted from SAC configuration (after curriculum updates)
    history_length = policy_config.get("history_length", 0)  # Actor history (always 0 in new design)
    critic_history_length = policy_config.get("critic_history_length", 0)  # Critic history (can be > 0)
    
    # Network architecture from policy config
    actor_config = policy_config.get("actor", {})
    actor_hidden_dim = actor_config.get("hidden_dim", [256, 128])
    if isinstance(actor_hidden_dim, (list, tuple)):
        actor_hidden_depth = len(actor_hidden_dim)
    else:
        actor_hidden_depth = actor_config.get("hidden_depth", 2)
    
    critic_config = policy_config.get("critic", {})
    critic_hidden_dim = critic_config.get("hidden_dim", [256, 128])
    if isinstance(critic_hidden_dim, (list, tuple)):
        critic_hidden_depth = len(critic_hidden_dim)
    else:
        critic_hidden_depth = critic_config.get("hidden_depth", 2)
    
    # Critic extra "critical sectors" features (closest K sectors)
    # This allows asymmetric input for the critic
    critic_critical_topk = critic_config.get("critical_topk", 0)
    
    # Learning rates from policy config
    actor_lr = policy_config.get("actor_lr", 2.5e-4)
    critic_lr = policy_config.get("critic_lr", 2.5e-4)
    
    # Training hyperparameters from policy config
    discount = policy_config.get("discount", 0.99)
    critic_tau = policy_config.get("critic_tau", 0.005)
    max_action = policy_config.get("max_action", 1.0)
    
    # Training schedule parameters from policy config
    train_every_n = policy_config.get("train_every_n", 2)
    training_iterations = policy_config.get("training_iterations", 100)
    training_batch_size = policy_config.get("batch_size", 256)
    min_buffer_size = policy_config.get("min_buffer_size", 4096)
    
    # SAC-specific parameters
    sac_config = policy_config
    alpha_lr = sac_config.get("alpha_lr", 2.5e-4)
    init_temperature = sac_config.get("init_temperature", 0.2)
    learnable_temperature = sac_config.get("learnable_temperature", True)
    actor_update_frequency = sac_config.get("actor_update_frequency", 1)
    critic_target_update_frequency = sac_config.get("critic_target_update_frequency", 2)
    actor_log_std_bounds = sac_config.get("log_std_bounds", [-5, 2])
    actor_betas = tuple(sac_config.get("actor_betas", [0.9, 0.999]))
    critic_betas = tuple(sac_config.get("critic_betas", [0.9, 0.999]))
    alpha_betas = tuple(sac_config.get("alpha_betas", [0.9, 0.999]))
    
    # Compute normalization bounds based on room size and cmd_scale
    # Maximum distance in room = diagonal
    room_width = ROOM_X_MAX - ROOM_X_MIN
    room_height = ROOM_Y_MAX - ROOM_Y_MIN
    max_distance = np.sqrt(room_width**2 + room_height**2)
    
    # Maximum velocity components from cmd_scale
    max_vx = cmd_scale[0]  # Maximum forward/backward velocity
    max_vy = cmd_scale[1]  # Maximum lateral velocity
    
    # Maximum angular velocity from cmd_scale
    max_angular_vel = cmd_scale[2]
    
    # Maximum lidar range: 3 meters visibility radius
    max_lidar_range = 3.0
    
    print(f"Normalization bounds:")
    print(f"  Max distance: {max_distance:.2f}m")
    print(f"  Max vx: {max_vx:.2f}m/s")
    print(f"  Max vy: {max_vy:.2f}m/s")
    print(f"  Max angular velocity: {max_angular_vel:.2f}rad/s")
    print(f"  Max lidar range: {max_lidar_range:.2f}m")
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {device}")
    
    # Check MJX availability and user request
    # Auto-enable MJX in headless mode for faster training (if available)
    if args.headless and not args.use_mjx and MJX_AVAILABLE:
        print("⚠️  Headless mode detected: Auto-enabling MJX for faster training (use --use_mjx explicitly to disable)")
        args.use_mjx = True
    
    use_mjx = args.use_mjx and MJX_AVAILABLE
    if args.use_mjx and not MJX_AVAILABLE:
        print("Warning: --use_mjx requested but MJX is not available. Install with: pip install mujoco-mjx")
        print("Continuing with sequential CPU simulation...")
        use_mjx = False
    
    if args.headless and not use_mjx:
        print("⚠️  WARNING: Running in headless mode without MJX will be VERY SLOW!")
        print("   Install MJX for 10-100x speedup: pip install mujoco-mjx")
        print("   Or use: python train.py ... --headless --use_mjx --batch_size 128")
    
    if use_mjx:
        print(f"MJX parallelization enabled with batch_size={args.batch_size}")
        jax_devices = jax.devices()
        jax_backend = jax.default_backend()
        print(f"JAX backend: {jax_backend}")
        print(f"JAX devices: {jax_devices}")
        if jax_backend == 'gpu':
            print("✅ MJX will use GPU acceleration")
            # Проверка наличия CUDA toolkit для компиляции
            import shutil
            if not shutil.which('ptxas'):
                print("⚠️  WARNING: ptxas (CUDA toolkit) not found in PATH")
                print("   MJX GPU compilation may fail. Install with:")
                print("   sudo apt install nvidia-cuda-toolkit")
                print("   Or use CPU mode: export JAX_PLATFORMS=cpu")
        else:
            print("⚠️  MJX is using CPU (install JAX with CUDA support for GPU acceleration)")
    else:
        print("Using sequential MuJoCo simulation (CPU)")
    
    # Load model first to determine lidar dimension
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    # Initialize forward kinematics to ensure all data is ready
    mujoco.mj_forward(m, d)
    
    # Initialize MJX model if using MJX
    mjx_model = None
    mjx_step_fn = None
    if use_mjx:
        try:
            print("Initializing MJX model...")
            mjx_model = mjx.put_model(m)
            # Try to create a test data structure to catch compilation errors early
            test_data = mjx.make_data(mjx_model)
            mjx_step_fn = create_mjx_batched_step_fn(mjx_model)
            print("MJX model initialized successfully")
        except Exception as e:
            error_msg = str(e)
            print(f"Error initializing MJX: {error_msg}")
            
            # Проверка на ошибку компиляции CUDA (ptxas не найден)
            if "ptxas" in error_msg.lower() or "FAILED_PRECONDITION" in error_msg or "Couldn't invoke ptxas" in error_msg:
                print("\n" + "="*60)
                print("⚠️  CUDA COMPILATION ERROR: CUDA toolkit (ptxas) not found")
                print("="*60)
                print("MJX требует CUDA toolkit для компиляции кода на GPU.")
                print("\nРешения:")
                print("1. Установите CUDA toolkit (требует sudo):")
                print("   sudo apt update")
                print("   sudo apt install nvidia-cuda-toolkit")
                print("\n2. Временно используйте CPU для MJX:")
                print("   export JAX_PLATFORMS=cpu")
                print("   python train.py ... --use_mjx ...")
                print("\n3. Или используйте sequential режим (без --use_mjx)")
                print("="*60)
                print("\nFalling back to sequential MuJoCo simulation (CPU)")
            else:
                print("\nFalling back to sequential MuJoCo simulation (CPU)")
            
            use_mjx = False
            mjx_model = None
            mjx_step_fn = None
    
    # Find lidar sensors, excluding problematic indices that intersect with robot body
    # Problematic indices: [11, 12, 13, 14, 15, 35, 36, 37, 38, 39] (only for front hemisphere)
    # All 40 lidar beams are evenly spaced from 0 to 39
    lidar_sensor_ids = []
    lidar_beam_index_mapping = []  # Maps lidar_data index to actual beam index
    
    # Find all 40 lidar sensors (0-39)
    for i in range(40):
        sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
        if sensor_id >= 0:
            lidar_sensor_ids.append(sensor_id)
            lidar_beam_index_mapping.append(i)
    
    # Calculate raw lidar dimension (40 evenly spaced sensors)
    raw_lidar_dim = len(lidar_sensor_ids)
    print(f"Found {raw_lidar_dim} valid lidar sensors (40 evenly spaced beams, 9 degrees apart)")
    
    # Compute lidar sensor angles for sector-based processing (compatible with ROS2 lidar_2d_processor)
    lidar_sensor_angles = compute_lidar_sensor_angles(m, lidar_sensor_ids, lidar_beam_index_mapping)
    print(f"Computed {len(lidar_sensor_angles)} lidar sensor angles for sector-based processing")
    
    # Downsampling parameters
    lidar_downsample_bins = 40  # Number of sectors (compatible with ROS2 lidar_2d_processor)
    print(f"Lidar will be processed using {lidar_downsample_bins} sectors (sector-based processing, compatible with ROS2)")
    
    # Calculate SAC state dimensions (base, without history):
    # Actor: lidar (40) + w(1) + sin(1) + cos(1) + dist(1) + prev_actions(3) = 47
    # БЕЗ Vx, Vy (линейные скорости), но С W (угловая скорость)
    actor_state_dim = lidar_downsample_bins + 7  # 47 - базовое наблюдение для актора
    
    # Critic: то же самое что Actor (47) + vx(1) + vy(1) = 49
    critic_base_dim = actor_state_dim + 2  # 49 - базовое наблюдение для критика (47 + vx + vy)
    
    # Asymmetric critic: add critical top-k nearest beams to the observation
    # This adds extra features specifically for the critic
    critic_state_dim = critic_base_dim + critic_critical_topk  # 48 + 5 = 53
    
    # Для совместимости используем actor_state_dim как state_dim (актор всегда получает первые actor_state_dim признаков)
    state_dim = actor_state_dim

    # Base directory for models
    base_models_dir = PROJECT_ROOT / "data" / "models"
    base_models_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = "sac"
    buffer_dir = PROJECT_ROOT / "data" / "buffer"
    buffer_dir.mkdir(parents=True, exist_ok=True)
    
    # Function to find the latest checkpoint directory
    def find_latest_checkpoint_dir(base_dir):
        """Find the most recent checkpoint directory based on modification time."""
        if not base_dir.exists():
            return None
        
        # Get all subdirectories
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not subdirs:
            return None
        
        # Sort by modification time (most recent first)
        subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return subdirs[0]
    
    # Determine model directory based on mode
    # Note: Directory is NOT created here - it will be created only when saving models
    if args.train:
        # Training mode
        if args.load_pretrained:
            # Load from latest checkpoint and continue training in the same directory
            latest_dir = find_latest_checkpoint_dir(base_models_dir)
            if latest_dir:
                model_dir = latest_dir
                print(f"Resuming training in existing directory: {model_dir}")
            else:
                # No checkpoint found, will create new directory on first save
                model_dir = base_models_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"No existing checkpoint found. Will create directory on first save: {model_dir}")
        else:
            # Start fresh training, will create new directory on first save
            model_dir = base_models_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Starting new training. Will create directory on first save: {model_dir}")
    else:
        # Test mode: always try to load from latest checkpoint
        latest_dir = find_latest_checkpoint_dir(base_models_dir)
        if latest_dir:
            model_dir = latest_dir
            print(f"Test mode: Loading from latest checkpoint directory: {model_dir}")
        else:
            # No checkpoint found, don't create directory (test mode doesn't save models)
            model_dir = None
            print(f"Test mode: No checkpoint found. Will initialize new model (no directory created)")
    
    # Initialize TensorBoard for logging (with unique subdirectory for each run)
    base_log_dir = Path(args.log_dir) if args.log_dir.startswith("/") else PROJECT_ROOT / "data" / args.log_dir
    # Create unique subdirectory for each run
    try:
        hostname = os.uname().nodename
    except (AttributeError, OSError):
        import socket
        hostname = socket.gethostname()
    run_name = datetime.now().strftime("%b%d_%H-%M-%S") + "_" + hostname
    if args.fine_tune:
        run_name += "_finetune"
    log_dir = base_log_dir / run_name
    writer = SummaryWriter(log_dir=str(log_dir)) if args.train else None
    if writer:
        print(f"TensorBoard logs written to: {log_dir}")
        print(f"View logs with: tensorboard --logdir={base_log_dir}")
    
    # Calculate actual state dimension for networks (with history)
    # Note: state_dim is base dimension, SAC will compute stacked dimension internally
    print(f"Algorithm: SAC")
    print(f"Policy configuration:")
    print(f"  Actor base state dimension: {state_dim} (40 lidar + w(1) + sin(1) + cos(1) + dist(1) + prev(3) = 47, БЕЗ vx/vy)")
    print(f"  Actor action history length: {history_length} (история предыдущих actions: vx, vy, w)")
    if history_length > 0:
        actor_stacked_dim = state_dim + history_length * 3
        print(f"  Actor stacked state dimension: {actor_stacked_dim} (base {state_dim} + action_history {history_length * 3})")
    else:
        print(f"  Actor stacked state dimension: {state_dim} (no action history)")
    print(f"  Critic base state dimension: {critic_state_dim} ({critic_base_dim} base + {critic_critical_topk} critical_topk = {critic_state_dim})")
    print(f"    Critic base: Actor(47) + vx(1) + vy(1) = {critic_base_dim}")
    print(f"  Critic observation history length: {critic_history_length}")
    print(f"  Critic stacked state dimension: {critic_state_dim * (critic_history_length + 1) if critic_history_length > 0 else critic_state_dim}")
    print(f"  Actor hidden: {actor_hidden_dim}")
    print(f"  Critic hidden: {critic_hidden_dim}")
    
    # Initialize Agent (always SAC)
    # Calculate Actor stacked state dimension (with action history)
    # Base: 47 (lidar + w + sin + cos + dist + prev_action(3))
    # History: history_length * 3 (каждый шаг истории - 3 action: vx, vy, w)
    actor_stacked_dim = actor_state_dim + history_length * 3  # 47 + history_length * 3
    
    agent = SAC(
            state_dim=actor_stacked_dim,  # Actor stacked: base(47) + action_history(history_length * 3)
            critic_state_dim=critic_state_dim, # Base SAC observation for critic (may be wider than actor)
            action_dim=3,  # vx, vy, w (angular velocity)
            device=device,
            max_action=max_action,
            discount=discount,
            init_temperature=init_temperature,
            alpha_lr=alpha_lr,
            alpha_betas=alpha_betas,
            actor_lr=actor_lr,
            actor_betas=actor_betas,
            actor_update_frequency=actor_update_frequency,
            critic_lr=critic_lr,
            critic_betas=critic_betas,
            critic_tau=critic_tau,
            critic_target_update_frequency=critic_target_update_frequency,
            learnable_temperature=learnable_temperature,
            save_every=0,
            load_model=False,
            log_dist_and_hist=False,
            save_directory=model_dir,
            model_name=model_name,
            load_directory=model_dir,
            writer=writer,
            history_length=history_length,  # История actions для Actor
            critic_history_length=critic_history_length,  # История наблюдений для Critic
            actor_hidden_dim=actor_hidden_dim,  # Can be list or int
            actor_hidden_depth=actor_hidden_depth,
            actor_log_std_bounds=actor_log_std_bounds,
            critic_hidden_dim=critic_hidden_dim,  # Can be list or int
            critic_hidden_depth=critic_hidden_depth,
        )
    # Legacy parameters (not used in standard SAC, kept for compatibility)
    if hasattr(agent, 'alpha_min'):
        agent.alpha_min = sac_config.get("alpha_min", 0.0)
        agent.alpha_update_frequency = sac_config.get("alpha_update_frequency", 1)
        if agent.alpha_min > 0.0:
            print(f"SAC: alpha_min set to {agent.alpha_min} (legacy mode)")
    
    print("SAC: initialized")

    episode = 0
    agent.reset_history()
    
    # Load model logic based on mode
    should_load_model = False
    if args.train:
        # Training mode: only load if --load_pretrained is specified
        should_load_model = args.load_pretrained or args.fine_tune
    else:
        # Test mode: always try to load from latest checkpoint (ignore --load_pretrained flag)
        should_load_model = True
    
    if should_load_model:
        # ПРОВЕРКА: совпадает ли размерность загружаемой модели с текущей
        # Skip if model_dir is None (test mode without checkpoint)
        if model_dir is not None:
            model_path = model_dir / f"{model_name}_actor.pth"
            if model_path.exists():
                try:
                    import torch
                    loaded_state = torch.load(model_path, map_location='cpu')
                    loaded_obs_dim = loaded_state['trunk.0.weight'].shape[1]
                    
                    # Вычисляем ожидаемую stacked dimension
                    expected_stacked_dim = actor_state_dim + history_length * 3
                    
                    if loaded_obs_dim != expected_stacked_dim:
                        # Определяем что за модель загружается
                        print(f"\n⚠️ ВНИМАНИЕ: Размерность загружаемой модели ({loaded_obs_dim}) не совпадает с текущей ({expected_stacked_dim})!")
                        print(f"  Загружаемая модель: {loaded_obs_dim} признаков")
                        print(f"  Текущая конфигурация: {expected_stacked_dim} признаков (base {actor_state_dim} + history {history_length * 3})")
                        
                        # Определяем тип загружаемой модели
                        if loaded_obs_dim == 48:
                            print(f"  ⚠️ Загружается СТАРАЯ модель (48 признаков: lidar(40) + vx(1) + w(1) + sin(1) + cos(1) + dist(1) + prev(3))")
                            print(f"  ⚠️ Текущий код ожидает: base(47) + history({history_length * 3}) = {expected_stacked_dim}")
                        elif loaded_obs_dim == 47:
                            print(f"  ⚠️ Загружается модель без истории (47 признаков: base только)")
                            if history_length > 0:
                                print(f"  ⚠️ Текущий код ожидает модель С историей: base(47) + history({history_length * 3}) = {expected_stacked_dim}")
                        elif loaded_obs_dim > 47 and (loaded_obs_dim - 47) % 3 == 0:
                            # Модель с историей, но другой длины
                            loaded_history_length = (loaded_obs_dim - 47) // 3
                            print(f"  ⚠️ Загружается модель с историей действий (base 47 + history {loaded_history_length} * 3 = {loaded_obs_dim})")
                            print(f"  ⚠️ Текущий код ожидает: base(47) + history({history_length} * 3) = {expected_stacked_dim}")
                        else:
                            print(f"  ⚠️ Неожиданная размерность: {loaded_obs_dim}")
                        
                        print(f"  ⚠️ Модель будет загружена, но поведение может быть непредсказуемым!")
                        print()
                    else:
                        # Размерности совпадают - все хорошо
                        if history_length > 0:
                            print(f"\n✓ Загружается модель с историей действий: {loaded_obs_dim} признаков (base {actor_state_dim} + history {history_length * 3})")
                        else:
                            print(f"\n✓ Загружается модель без истории: {loaded_obs_dim} признаков (base только)")
                except Exception as e:
                    print(f"⚠️ Не удалось проверить размерность модели: {e}")
        
            metadata = agent.load(
                filename=model_name,
                directory=model_dir,
                fine_tune=args.fine_tune if args.train else False
            )
            # Restore episode number from metadata if available (unless fine-tuning or test mode)
            if args.train:
                if not args.fine_tune and metadata is not None and 'episode' in metadata:
                    episode = metadata['episode']
                    print(f"Resumed training from episode {episode}")
                elif args.fine_tune:
                    print("--- Fine-tuning mode active: Weights loaded, but resetting Alpha and Buffer ---")
            else:
                # Test mode
                if metadata is not None and 'episode' in metadata:
                    print(f"Loaded model from episode {metadata['episode']}")
                else:
                    print("No checkpoint found in directory, model initialized from scratch")
        else:
            # Test mode without checkpoint directory
            metadata = None
            print("Test mode: No checkpoint directory found, model initialized from scratch")
    max_episode = episode + args.episodes
    
    # Track success rate over last 100 episodes
    episode_successes = []  # List of booleans: True if episode was successful (reached target)
    episode_collisions = []  # List of booleans: True if episode ended due to collision
    episode_timeouts = []  # List of booleans: True if episode ended due to timeout (max_steps)
    success_rate_window = 100
    
    # For test mode, collect all statistics (not just last 100)
    test_episode_successes = [] if not args.train else None
    test_episode_collisions = [] if not args.train else None
    test_episode_timeouts = [] if not args.train else None
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_size=7e5, random_seed=69)
    
    # Try to load buffer if in training mode with --load_pretrained or --fine_tune
    if args.train and should_load_model:
        if args.fresh_buffer or args.fine_tune:
            print("--- Fine-tuning with a FRESH replay buffer as requested ---")
        else:
            buffer_path = buffer_dir / f"{model_name}_buffer.npz"
            if replay_buffer.load(buffer_path):
                print(f"Successfully loaded replay buffer with {replay_buffer.size()} experiences")
            else:
                print("Could not load replay buffer, starting with empty buffer")
    
    # Load walking policy (rsl_rl ActorCritic for Unitree A1 locomotion)
    # This is separate from the SAC path planning policy
    walking_policy_model = None
    if walking_policy_path and os.path.exists(walking_policy_path):
        try:
            print(f"Loading walking policy (locomotion) from checkpoint: {walking_policy_path}")
            walking_policy_model = load_walking_policy_from_checkpoint(
                checkpoint_path=walking_policy_path,
                obs_dim=num_obs,  # 45 for Unitree A1
                action_dim=num_actions  # 12 for Unitree A1
            )
            print("✓ Walking policy (locomotion) loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load walking policy from {walking_policy_path}: {e}")
            print("Continuing without walking policy (robot will use default angles)...")
            walking_policy_model = None
    else:
        if walking_policy_path:
            print(f"⚠️ Walking policy file not found at {walking_policy_path}")
        else:
            print("⚠️ No walking_policy_path specified in config")
        print("Continuing without walking policy (robot will use default angles)...")
        walking_policy_model = None
    
    # Find robot body IDs once at initialization (for collision detection)
    robot_body_ids = set()
    for i in range(m.nbody):
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            body_name_lower = body_name.lower()
            # Check if it's a robot body (pelvis, links, etc.)
            if any(word in body_name_lower for word in 
                   ["pelvis", "link", "foot", "ankle", "knee", "hip", 
                    "yaw", "pitch", "roll", "lidar"]):
                robot_body_ids.add(i)
    print(f"Found {len(robot_body_ids)} robot body IDs for collision detection")
    
    # Find target body ID (target_point is now a mocap body)
    target_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_body")
    target_mocap_id = -1
    if target_body_id < 0:
        print("Warning: target_body not found in scene.xml")
        target_body_id = None
    else:
        # Get mocap ID for this body
        target_mocap_id = m.body_mocapid[target_body_id]
        if target_mocap_id >= 0:
            print(f"Found target_body (ID: {target_body_id}, mocap_id: {target_mocap_id})")
        else:
            print("Warning: target_body is not a mocap body")
            target_body_id = None
            target_mocap_id = -1    
    
    # Run with or without visualization
    # IMPORTANT: Don't create viewer here if loading pretrained model with episode > 0,
    # because obstacles may be regenerated immediately, creating new m and d objects
    # Viewer will be created at the start of the first episode instead
    if args.headless:
        print("Running in headless mode (no visualization)")
        if use_mjx:
            print(f"✅ MJX acceleration enabled - training will be {args.batch_size}x faster with parallel simulation")
        else:
            print("⚠️  MJX not enabled - training will be slow. Use --use_mjx for 10-100x speedup")
        viewer = None  # No viewer in headless mode
    else:
        # Delay viewer creation until first episode starts to avoid hanging
        # if obstacles are regenerated immediately after loading pretrained model
        viewer = None
        print("Viewer will be created at start of first episode")

    
    # Load obstacle generator config from our parameters container (handles curriculum overrides)
    obstacle_regeneration_interval = obstacle_params.get("regeneration_interval", 10)
    
    # Cube configuration
    cube_config = obstacle_params.get("cubes", {})
    cube_count_min = cube_config.get("count_min", 3)
    cube_count_max = cube_config.get("count_max", 7)
    cube_size_x_min = cube_config.get("size_x_min", 0.1)
    cube_size_x_max = cube_config.get("size_x_max", 0.5)
    cube_size_y_min = cube_config.get("size_y_min", 0.1)
    cube_size_y_max = cube_config.get("size_y_max", 0.5)
    cube_size_z = cube_config.get("size_z", 1.0)
    
    # Position configuration
    position_config = obstacle_params.get("position", {})
    min_pos_margin = position_config.get("min_margin", 0.5)
    
    # Episode max steps (from curriculum or config)
    current_max_steps = episode_params.get("max_steps", config.get("max_steps", 5000))
    
    # Log loaded configuration
    print(f"\n=== Obstacle Generator Configuration ===")
    print(f"  Regeneration interval: {obstacle_regeneration_interval}")
    print(f"  Cubes: min={cube_count_min}, max={cube_count_max}")
    print(f"  Position margin: {min_pos_margin}")
    print(f"  Max steps per episode: {current_max_steps}")
    print()
    
    try:
        # Main episode loop
        if use_mjx:
            # Batched MJX training mode
            print("\n=== Starting MJX Batched Training Mode ===")
            print(f"Batch size: {args.batch_size}")
            print(f"Episodes to run: {max_episode - episode}")
            print("\n⚠️  WARNING: Complex models with many collision geometries require smaller batch sizes.")
            print("   If you encounter GPU memory errors, try reducing --batch_size (e.g., 32, 16, or 8).")
            print("   Recommended batch sizes for this model: 16-32")
            
            # Calculate number of batches needed
            episodes_remaining = max_episode - episode
            num_batches = (episodes_remaining + args.batch_size - 1) // args.batch_size
            
            # Dynamic batch size adjustment on OOM errors
            current_batch_size_target = args.batch_size
            
            for batch_idx in range(num_batches):
                current_batch_size = min(current_batch_size_target, episodes_remaining - batch_idx * args.batch_size)
                if current_batch_size <= 0:
                    break
                
                # Refresh max_steps from curriculum (may have changed on level up)
                mjx_max_steps = episode_params.get("max_steps", config.get("max_steps", 5000))
                
                print(f"\n=== Batch {batch_idx + 1}/{num_batches} (size: {current_batch_size}, max_steps={mjx_max_steps}) ===")
                
                # Run batched episodes with retry on OOM
                max_retries = 5
                retry_batch_size = current_batch_size
                success = False
                
                for retry in range(max_retries):
                    try:
                        batch_results = run_batched_episodes_mjx_full(
                            mjx_model=mjx_model,
                            mjx_step_fn=mjx_step_fn,
                            batch_size=retry_batch_size,
                            spawn_generator=spawn_generator,
                            agent=agent,
                            replay_buffer=replay_buffer,
                            m=m,
                            lidar_sensor_ids=lidar_sensor_ids,
                            lidar_sensor_angles=lidar_sensor_angles,
                            target_body_id=target_body_id,
                            target_mocap_id=target_mocap_id,
                            reward_weights=reward_weights,
                            config=config,
                            args=args,
                            max_steps=mjx_max_steps,
                            train=args.train,
                            critic_critical_topk=critic_critical_topk,
                            critic_history_length=critic_history_length
                        )
                        success = True
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        error_str = str(e)
                        # Check if it's an out-of-memory error
                        if "RESOURCE_EXHAUSTED" in error_str or "Out of memory" in error_str or "ran out of memory" in error_str:
                            if retry < max_retries - 1:
                                # Reduce batch size by half and retry
                                retry_batch_size = max(1, retry_batch_size // 2)
                                print(f"\n⚠️  GPU Out of Memory! Retrying with reduced batch size: {retry_batch_size} (attempt {retry + 2}/{max_retries})")
                                # Clear JAX cache to free memory
                                import jax
                                jax.clear_caches()
                                continue
                            else:
                                print(f"\n❌ Failed after {max_retries} retries with reduced batch sizes.")
                                print(f"   Last attempted batch size: {retry_batch_size}")
                                print(f"   This model is too complex for MJX batching with available GPU memory.")
                                print(f"   Recommendations:")
                                print(f"     1. Use smaller batch size: --batch_size 16 or --batch_size 8")
                                print(f"     2. Use sequential mode (without --use_mjx)")
                                print(f"     3. Use a GPU with more memory")
                                raise
                        else:
                            # Other error, re-raise immediately
                            raise
                
                if not success:
                    print(f"\n❌ Failed to run batch after {max_retries} retries.")
                    continue  # Skip this batch and try next
                
                # Process batch results
                for env_idx, result in enumerate(batch_results):
                    episode_idx = episode + batch_idx * args.batch_size + env_idx
                    
                    if args.train:
                        # Determine if episode was successful
                        episode_successful = result.get('done', False) and result.get('reward', 0) > 0
                        episode_successes.append(episode_successful)
                    if not args.train:
                        # For test mode, also collect statistics
                        episode_successful = result.get('done', False) and result.get('reward', 0) > 0
                        test_episode_successes.append(episode_successful)
                        if len(episode_successes) > success_rate_window:
                            episode_successes.pop(0)
                        
                        success_rate = sum(episode_successes) / len(episode_successes) if len(episode_successes) > 0 else 0.0
                        
                        # Log episode
                        if episode_idx % 1 == 0:
                            curr_level_str = ""
                            if curriculum_manager:
                                curr_level_str = f" [Level: {curriculum_manager.get_current_level_name()}]"
                                
                            print(f"\nEpisode {episode_idx}{curr_level_str}: Reward={result['reward']:.2f}, "
                                  f"Steps={result['steps']}, Success={'YES' if episode_successful else 'NO'}, "
                                  f"Success Rate={success_rate*100:.1f}%")
                        
                        # Update Curriculum Manager (in MJX mode)
                        if curriculum_manager:
                            level_changed = curriculum_manager.update(episode_successful, result['reward'])
                            if level_changed:
                                new_level_name = curriculum_manager.get_current_level_name()
                                print(f"\n🚀 CURRICULUM LEVEL UP: {new_level_name}!")
                                
                                # Apply new parameters (updates policy_config and replay_buffer_config)
                                curriculum_manager.apply_current_level(
                                    reward_weights, obstacle_params,
                                    policy_config=policy_config,
                                    replay_buffer_config=replay_buffer_config,
                                    episode_params=episode_params
                                )
                                
                                # Re-extract training parameters from updated policy_config
                                training_iterations = policy_config.get("training_iterations", training_iterations)
                                training_batch_size = policy_config.get("batch_size", training_batch_size)
                                # Update SAC temperature parameters if changed
                                new_init_temperature = policy_config.get("init_temperature", init_temperature)
                                new_alpha_lr = policy_config.get("alpha_lr", alpha_lr)
                                if new_init_temperature != init_temperature or new_alpha_lr != alpha_lr:
                                    init_temperature = new_init_temperature
                                    alpha_lr = new_alpha_lr
                                    # Update agent's temperature if needed
                                    if hasattr(agent, 'log_alpha'):
                                        new_log_alpha = np.log(init_temperature)
                                        agent.log_alpha.data.fill_(new_log_alpha)
                                        # Re-initialize alpha optimizer with new learning rate
                                        agent.log_alpha_optimizer = torch.optim.Adam(
                                            [agent.log_alpha], lr=alpha_lr, betas=agent.alpha_betas
                                        )
                                        print(f"Updated agent temperature to {init_temperature}, alpha_lr to {alpha_lr}")
                                
                                # Update local variables for next batches
                                obstacle_regeneration_interval = obstacle_params.get("regeneration_interval", obstacle_regeneration_interval)
                                cube_config = obstacle_params.get("cubes", {})
                                cube_count_min = cube_config.get("count_min", cube_count_min)
                                cube_count_max = cube_config.get("count_max", cube_count_max)
                                min_pos_margin = obstacle_params.get("position", {}).get("min_margin", min_pos_margin)
                                mjx_max_steps = episode_params.get("max_steps", config.get("max_steps", 5000))
                                
                                print(f"New parameters applied: Batch={training_batch_size}, Iterations={training_iterations}, "
                                      f"Cubes={cube_count_min}-{cube_count_max}, Margin={min_pos_margin}, MaxSteps={mjx_max_steps}")
                        
                        # Train SAC periodically
                        if episode_idx % train_every_n == 0 and replay_buffer.size() >= min_buffer_size:
                            # Get replay buffer sampling weights from config (updated by curriculum)
                            success_weight = replay_buffer_config.get("success_weight", 2.0)
                            collision_weight = replay_buffer_config.get("collision_weight", 0.5)
                            
                            agent.train(
                                replay_buffer=replay_buffer,
                                iterations=training_iterations,
                                batch_size=training_batch_size,
                                success_weight=success_weight,
                                collision_weight=collision_weight
                            )
                        
                        # Save periodically (only in training mode)
                        if args.train and episode_idx % args.save_every_n == 0 and episode_idx > 0:
                            # Create directory if it doesn't exist (lazy creation)
                            if model_dir is not None:
                                model_dir.mkdir(parents=True, exist_ok=True)
                            metadata = {'episode': episode_idx}
                            agent.save(
                                filename=model_name,
                                directory=model_dir,
                                metadata=metadata
                            )
                            buffer_path = buffer_dir / f"{model_name}_buffer.npz"
                            replay_buffer.save(buffer_path)
                            print(f"Model and buffer saved at episode {episode_idx}")
                        
                        # TensorBoard logging
                        if writer is not None:
                            # Log to TensorBoard
                            if writer:
                                writer.add_scalar("episode/reward", result['reward'], episode_idx)
                                writer.add_scalar("episode/success", 1.0 if episode_successful else 0.0, episode_idx)
                                writer.add_scalar("episode/success_rate", success_rate, episode_idx)
                                writer.add_scalar("episode/steps", result['steps'], episode_idx)
                
                # Update episode counter only if batch succeeded
                if success:
                    episode += retry_batch_size
                else:
                    print(f"⚠️  Skipping batch {batch_idx + 1} due to failures.")
                    break  # Exit batch loop if we can't proceed
            
            print("\n=== MJX Batched Training Complete ===")
        
        else:
            # Sequential training mode (original code)
            # Main episode loop
            # Disable extra lidar debug visualization to avoid viewer segfaults
            enable_lidar_debug_vis = False

            # VISUAL MODE: if regeneration interval is 0, generate obstacles once before episodes start
            # This keeps viewer stable (no runtime regeneration) but still adds cubes/cylinders initially.
            if not args.headless and obstacle_regeneration_interval == 0:
                print("\nVisual mode: obstacle_regeneration_interval=0, generating obstacles once before first episode")
                # Generate random number of obstacles within configured ranges
                # (reuse same params as regeneration block)
                if cube_count_min < 0:
                    cube_count_min = 0
                if cube_count_max < cube_count_min:
                    cube_count_max = cube_count_min
                num_cubes = np.random.randint(cube_count_min, cube_count_max + 1)
                print(f"Generating {num_cubes} cubes for initial scene (visual mode)")
                regenerate_scene_obstacles(
                    xml_path=xml_path,
                    num_cubes=num_cubes,
                    num_cylinders=0,  # cylinders disabled for MJX compatibility
                    cube_size_x_min=cube_size_x_min,
                    cube_size_x_max=cube_size_x_max,
                    cube_size_y_min=cube_size_y_min,
                    cube_size_y_max=cube_size_y_max,
                    cube_size_z=cube_size_z,
                    min_pos_margin=min_pos_margin
                )
                # Reload model/data after regeneration
                m = mujoco.MjModel.from_xml_path(xml_path)
                d = mujoco.MjData(m)
                m.opt.timestep = simulation_dt
                mujoco.mj_forward(m, d)
                # Recompute lidar sensors/angles
                lidar_sensor_ids = []
                lidar_beam_index_mapping = []
                for i in range(40):
                    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
                    if sensor_id >= 0:
                        lidar_sensor_ids.append(sensor_id)
                        lidar_beam_index_mapping.append(i)
                raw_lidar_dim = len(lidar_sensor_ids)
                lidar_sensor_angles = compute_lidar_sensor_angles(m, lidar_sensor_ids, lidar_beam_index_mapping)
                # Re-find target body/mocap
                target_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_body")
                target_mocap_id = -1
                if target_body_id >= 0:
                    target_mocap_id = m.body_mocapid[target_body_id]
                    if target_mocap_id < 0:
                        target_body_id = None
                        target_mocap_id = -1
                else:
                    target_body_id = None
                # Update spawn generator with new obstacles
                spawn_generator.update_obstacles(xml_path)
                print("Initial obstacles generated for visual mode.\n")

            while episode < max_episode:
                # Check if viewer is still running (in visual mode)
                # Use try-except to avoid segfault if viewer is in invalid state
                if viewer is not None:
                    try:
                        if hasattr(viewer, 'is_running') and not viewer.is_running():
                            break
                    except Exception as e:
                        # Viewer may be in invalid state, close it and continue without visualization
                        print(f"Warning: Viewer check failed: {e}. Continuing without visualization.")
                        try:
                            if hasattr(viewer, 'close'):
                                viewer.close()
                        except:
                            pass
                        viewer = None
                
                # Regenerate obstacles every N episodes (before episode starts)
                # Skip if interval <= 0 (disabled in visual mode)
                if obstacle_regeneration_interval > 0 and episode > 0 and episode % obstacle_regeneration_interval == 0:
                    print(f"\n=== Regenerating obstacles at episode {episode} ===")
                    
                    # Debug: log actual values being used
                    print(f"DEBUG: Using cube_count_min={cube_count_min}, cube_count_max={cube_count_max}")
                    
                    # Validate count ranges
                    if cube_count_min < 0:
                        print(f"ERROR: cube_count_min={cube_count_min} is negative! Fixing to 0")
                        cube_count_min = 0
                    if cube_count_max < cube_count_min:
                        print(f"ERROR: cube_count_max={cube_count_max} < cube_count_min={cube_count_min}! Cannot generate cubes.")
                        num_cubes = 0
                    else:
                        # Generate random number of obstacles within configured ranges
                        num_cubes = np.random.randint(cube_count_min, cube_count_max + 1)
                    
                    print(f"Result: Generating {num_cubes} cubes")
                    print(f"Cube size ranges: X=[{cube_size_x_min:.2f}, {cube_size_x_max:.2f}], "
                          f"Y=[{cube_size_y_min:.2f}, {cube_size_y_max:.2f}], "
                          f"Z={cube_size_z:.2f} (fixed)")
                    
                    # Regenerate obstacles in XML (cubes only, no cylinders)
                    regenerate_scene_obstacles(
                        xml_path=xml_path,
                        num_cubes=num_cubes,
                        num_cylinders=0,  # Always 0 - cylinders removed for MJX compatibility
                        cube_size_x_min=cube_size_x_min,
                        cube_size_x_max=cube_size_x_max,
                        cube_size_y_min=cube_size_y_min,
                        cube_size_y_max=cube_size_y_max,
                        cube_size_z=cube_size_z,
                        min_pos_margin=min_pos_margin
                    )

                    # Reload MuJoCo model with new obstacles
                    # IMPORTANT: Create new m and d, old ones may be referenced by viewer
                    m_new = mujoco.MjModel.from_xml_path(xml_path)
                    d_new = mujoco.MjData(m_new)
                    m_new.opt.timestep = simulation_dt
                    # Initialize forward kinematics to ensure all data is ready
                    mujoco.mj_forward(m_new, d_new)
                    # Replace old references
                    m = m_new
                    d = d_new
                    
                    # Re-find lidar sensors (should be the same, but just in case)
                    lidar_sensor_ids = []
                    lidar_beam_index_mapping = []
                    
                    # Find all 40 lidar sensors (0-39)
                    for i in range(40):
                        sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
                        if sensor_id >= 0:
                            lidar_sensor_ids.append(sensor_id)
                            lidar_beam_index_mapping.append(i)
                    
                    raw_lidar_dim = len(lidar_sensor_ids)
                    print(f"Re-found {raw_lidar_dim} valid lidar sensors after obstacle regeneration "
                          f"(40 evenly spaced beams, 9 degrees apart)")
                    
                    # Re-compute lidar sensor angles (after obstacle regeneration)
                    lidar_sensor_angles = compute_lidar_sensor_angles(m, lidar_sensor_ids, lidar_beam_index_mapping)
                    
                    # Re-find target body
                    target_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_body")
                    target_mocap_id = -1
                    if target_body_id >= 0:
                        target_mocap_id = m.body_mocapid[target_body_id]
                        if target_mocap_id < 0:
                            target_body_id = None
                    else:
                        target_body_id = None
                    
                    # Update spawn generator with new obstacles
                    spawn_generator.update_obstacles(xml_path)
                    
                    # Close viewer before regeneration; optionally allow recreation if user permits
                    if viewer is not None:
                        print("Closing viewer before obstacle regeneration...")
                        try:
                            if hasattr(viewer, 'close'):
                                viewer.close()
                        except Exception as e:
                            print(f"Warning: Could not close viewer: {e}")
                        viewer = None
                        import gc
                        gc.collect()
                        # Skip delay in headless mode
                        if not args.headless:
                            time.sleep(0.2)
                    
                    print(f"Scene regenerated! New obstacles parsed and free space recalculated.\n")
                
                # Reset episode first (before creating viewer)
                # Ensure m and d are valid before resetting
                if m is None or d is None:
                    raise RuntimeError("MuJoCo model or data is None before episode reset. This should not happen.")
                mujoco.mj_resetData(m, d)
                
                # Forced delay to ensure simulation state is fully cleared and stabilized
                # and to prevent commands from previous episode from leaking
                # Skip delay in headless mode for faster training
                if not args.headless:
                    print(f"Waiting 0.5s before starting episode {episode}...")
                    time.sleep(0.5)
                else:
                    # Minimal delay in headless mode (just for state clearing)
                    time.sleep(0.01)
                
                # Reset control variables to zero to prevent carry-over from previous episode
                vx_cmd = 0.0
                vy_cmd = 0.0
                yaw_rate_cmd = 0.0
                action_np = np.zeros(3, dtype=np.float32)
                target_dof_pos = default_angles.copy()
                
                # Initialize step counter BEFORE viewer creation (needed for warmup logic)
                step_count = 0
                
                # Create viewer at start of episode (after obstacle regeneration if needed)
                # This prevents hanging when loading pretrained model with episode > 0
                # Viewer is created AFTER mj_resetData to ensure it uses the reset state
                # Recreate viewer in visual mode
                if viewer is None and not args.headless:
                    print("Creating viewer at start of episode...")
                    # Initialize forward kinematics before creating viewer
                    mujoco.mj_forward(m, d)
                    try:
                        # Ensure m and d are valid before creating viewer
                        # Check that model and data are properly initialized
                        if m is None or d is None:
                            raise ValueError("MuJoCo model or data is None")
                        
                        # Try to create viewer with proper error handling
                        # Use launch_passive for non-blocking viewer
                        import sys
                        
                        # Check DISPLAY for Linux systems
                        if sys.platform.startswith('linux'):
                            if 'DISPLAY' not in os.environ:
                                print("WARNING: DISPLAY environment variable not set. Cannot create viewer.")
                                print("Running in headless mode. Set DISPLAY or use --headless flag.")
                                print("To set DISPLAY, use: export DISPLAY=:0")
                                viewer = None
                            else:
                                print(f"DISPLAY={os.environ.get('DISPLAY')}")
                                try:
                                    viewer = mujoco.viewer.launch_passive(m, d)
                                    # Give viewer time to initialize (skip in headless mode)
                                    if not args.headless:
                                        time.sleep(1.5)
                                    
                                    # Verify viewer was created successfully
                                    if viewer is None:
                                        raise RuntimeError("Viewer creation returned None")
                                    
                                    # Test if viewer is working
                                    try:
                                        if hasattr(viewer, 'is_running'):
                                            if not viewer.is_running():
                                                print("WARNING: Viewer created but not running. Continuing without visualization.")
                                                try:
                                                    if hasattr(viewer, 'close'):
                                                        viewer.close()
                                                except:
                                                    pass
                                                viewer = None
                                            else:
                                                print("✓ Viewer created successfully and running")
                                                # Update viewer creation step for warmup period
                                                if 'step_count' in locals():
                                                    viewer_creation_step = step_count
                                                else:
                                                    viewer_creation_step = 0
                                        else:
                                            print("✓ Viewer created successfully (legacy API)")
                                            # Update viewer creation step for warmup period
                                            if 'step_count' in locals():
                                                viewer_creation_step = step_count
                                            else:
                                                viewer_creation_step = 0
                                    except Exception as e:
                                        print(f"WARNING: Could not verify viewer status: {e}")
                                        print("Continuing with viewer, but may have issues...")
                                except Exception as viewer_error:
                                    print(f"Failed to create viewer: {viewer_error}")
                                    print("This might be due to:")
                                    print("  - Missing OpenGL/graphics drivers")
                                    print("  - X11 forwarding issues (if using SSH)")
                                    print("  - MuJoCo viewer compatibility")
                                    raise viewer_error
                        else:
                            # Non-Linux systems (Windows, macOS)
                            try:
                                viewer = mujoco.viewer.launch_passive(m, d)
                                # Give viewer time to initialize (skip in headless mode)
                                if not args.headless:
                                    time.sleep(2.0)
                                
                                if viewer is None:
                                    raise RuntimeError("Viewer creation returned None")
                                
                                print("✓ Viewer created successfully")
                                # Update viewer creation step for warmup period
                                if 'step_count' in locals():
                                    viewer_creation_step = step_count
                                else:
                                    viewer_creation_step = 0
                            except Exception as viewer_error:
                                print(f"Failed to create viewer: {viewer_error}")
                                raise viewer_error
                        
                    except Exception as e:
                        print(f"Error creating viewer: {e}")
                        print("This may be due to:")
                        print("  1. Missing DISPLAY environment variable (Linux)")
                        print("  2. Graphics driver issues")
                        print("  3. MuJoCo viewer compatibility issues")
                        print("Continuing without visualization...")
                        if viewer is not None:
                            try:
                                if hasattr(viewer, 'close'):
                                    viewer.close()
                            except:
                                pass
                        viewer = None
                
                # Reset observation history for new episode
                agent.reset_history()
                
                # Initialize step_count early (before viewer creation, which may reference it)
                step_count = 0
                # Initialize viewer_creation_step early to avoid NameError
                viewer_creation_step = -1
                
                # Generate random spawn position for robot from precomputed free points
                robot_spawn_pos = spawn_generator.sample_spawn_point(z_height=0.37)
                # Set robot position (x, y, z)
                d.qpos[0:3] = robot_spawn_pos
                
                # Generate random yaw orientation (rotation around z-axis)
                random_yaw = np.random.uniform(0, 2 * np.pi)
                # Convert yaw to quaternion: q = [cos(yaw/2), 0, 0, sin(yaw/2)]
                d.qpos[3:7] = [np.cos(random_yaw / 2), 0, 0, np.sin(random_yaw / 2)]
                
                # Generate target point far enough from robot
                target_pos = spawn_generator.sample_target_point(
                    robot_pos=robot_spawn_pos,
                    min_distance=1.8,  # Increased from 1.3 for more space
                    z_height=0.1
                )
                
                # Update target position in scene (mocap body)
                if target_body_id is not None and target_mocap_id >= 0:
                    d.mocap_pos[target_mocap_id] = target_pos
                    # Set quaternion to identity (no rotation)
                    d.mocap_quat[target_mocap_id] = [1, 0, 0, 0]
                
                # Forward kinematics to update all derived quantities
                mujoco.mj_forward(m, d)
                
                # CRITICAL: Do NOT update viewer.data here - viewer was created with correct model reference
                # Updating viewer.data can cause segfault if model was regenerated
                # Viewer already has correct data reference from launch_passive
                
                # Stabilize simulation (only in visual mode, or minimal steps in headless)
                if args.headless:
                    # Quick stabilization - just a few steps
                    for _ in range(10):
                        mujoco.mj_step(m, d)
                else:
                    # In headless mode, skip long delay - just do a few steps for stabilization
                    if args.headless:
                        for _ in range(10):
                            mujoco.mj_step(m, d)
                    else:
                        print(f"Episode {episode}: Waiting 2 seconds for simulation to stabilize...")
                        time.sleep(2.0)
                
                # Сброс истории для нового эпизода
                agent.reset_history()
                
                # Initialize prev_distance for SAC reward computation
                robot_pos_init = d.qpos[:3]
                robot_quat_init = d.qpos[3:7]
                distance_init, _, _ = get_target_info(robot_pos_init, target_pos, robot_quat_init)
                prev_distance = distance_init if not (np.isnan(distance_init) or np.isinf(distance_init)) else 5.0
                
                # Инициализация локальной истории actions для Actor (новый эпизод)
                from collections import deque
                actor_action_history = deque(maxlen=history_length) if history_length > 0 else None
                
                episode_reward = 0
                # step_count already initialized above (before viewer creation)
                max_steps = episode_params.get("max_steps", config.get("max_steps", 5000))  # From curriculum or config
                done = False  # Initialize done flag
                # Initialize planned commands (used if SAC not called yet)
                vx_cmd = 0.0
                vy_cmd = 0.0
                yaw_rate_cmd = 0.0
                
                # Track reward components for logging (use counters instead of lists)
                reward_components = {
                    'distance': 0.0,
                    'reached': 0.0,
                    'obstacle': 0.0,
                    'vy_penalty': 0.0,
                    'vx_backward_penalty': 0.0,
                    'velocity_alignment': 0.0,
                    'time_penalty': 0.0,
                    'total': 0.0,
                    'count': 0  # Track number of SAC steps
                }
                
                # Track episode termination type
                episode_ended_by_collision = False
                episode_ended_by_timeout = False
                
                # Initialize walking policy variables
                action = np.zeros(num_actions, dtype=np.float32)
                target_dof_pos = default_angles.copy()
                # obs will be built by build_walking_policy_observation function
                # Initialize with zeros for now, will be overwritten
                obs = np.zeros(num_obs, dtype=np.float32)
                cmd = np.array([0, 0, 0], dtype=np.float32)
                action_np = np.zeros(3, dtype=np.float32)  # SAC action placeholder (vx, vy, w)
                prev_action_np = np.zeros(3, dtype=np.float32)  # SAC prev action placeholder
                # OPTIMIZATION: Pre-allocate phase array to avoid repeated allocation
                phase_array = np.zeros(2, dtype=np.float32)
                
                # Transition storage for Sequential mode
                prev_critic_obs = None  # Store critic observation (with history) for replay buffer
                prev_reward = 0.0
                prev_done_flag = 0.0
                prev_success = False
                
                # История actions для Actor (Sequential mode, для одного эпизода)
                from collections import deque
                actor_action_history = deque(maxlen=history_length) if history_length > 0 else None
                
                # Track when viewer was created to avoid syncing too early
                # Initialize before viewer creation check
                viewer_creation_step = -1
                viewer_warmup_steps = 50  # Don't sync viewer for first N steps after creation
                
                # Lidar data cache - updated only at SAC frequency (10Hz)
                cached_lidar_data_raw = None
                
                # Main simulation loop
                while step_count < max_steps and not done:
                    # Check viewer in visual mode (with safe error handling)
                    if viewer is not None:
                        try:
                            if hasattr(viewer, 'is_running') and not viewer.is_running():
                                print("Viewer closed by user, stopping simulation")
                                break
                        except Exception as e:
                            # Viewer may be in invalid state, close it and continue
                            print(f"Warning: Viewer check failed: {e}. Continuing without visualization.")
                            try:
                                if hasattr(viewer, 'close'):
                                    viewer.close()
                            except:
                                pass
                            viewer = None
                    
                    step_start = time.time() if not args.headless else None
                    
                    # PD control for walking
                    # Use only robot DOF (exclude target body DOF)
                    # Add safety checks to avoid segfault
                    if m is None or d is None:
                        raise RuntimeError("MuJoCo model or data is None in simulation loop. This should not happen.")
                    
                    try:
                        d.ctrl[:] = target_dof_pos
                        mujoco.mj_step(m, d)
                    except Exception as e:
                        print(f"ERROR: MuJoCo step failed: {e}")
                        print("This may indicate model/data corruption. Ending episode.")
                        done = True
                        break
                    
                    step_count += 1
                    
                    # Update viewer less frequently to avoid segfault issues
                    # Sync every 5 steps instead of every step for better stability
                    # IMPORTANT: Do NOT update viewer.data - viewer was created with correct model reference
                    # Updating viewer.data can cause segfault if model was regenerated
                    # Also wait for viewer to fully initialize before first sync
                    # CRITICAL: If viewer was disabled after regeneration, don't sync at all
                    if (viewer is not None and step_count % 5 == 0):
                        steps_since_viewer_creation = step_count - viewer_creation_step if viewer_creation_step >= 0 else float('inf')
                        
                        # CRITICAL: Only sync viewer if it's been created and warmed up
                        # Skip sync entirely if viewer was just created (first 50 steps)
                        if steps_since_viewer_creation >= viewer_warmup_steps:  # Only after warmup
                            # Check if viewer is still running before syncing
                            try:
                                # CRITICAL: Verify m and d are valid before syncing
                                if m is None or d is None:
                                    # Skip sync if data is invalid
                                    pass
                                else:
                                    # First check if viewer is still valid
                                    if not hasattr(viewer, 'is_running'):
                                        # Old viewer API, try to sync anyway (but less frequently)
                                        # Use try-except to catch any segfaults
                                        try:
                                            viewer.sync()
                                        except:
                                            # If sync fails, viewer might be broken
                                            print("Viewer sync failed (old API), closing viewer")
                                            viewer = None
                                    elif viewer.is_running():
                                        # CRITICAL: Do not update viewer.data - this can cause segfault
                                        # Viewer was created with correct model, so it already has correct data reference
                                        # Only sync, don't update data reference
                                        try:
                                            viewer.sync()
                                        except:
                                            # If sync fails, viewer might be broken
                                            print("Viewer sync failed, closing viewer")
                                            try:
                                                if hasattr(viewer, 'close'):
                                                    viewer.close()
                                            except:
                                                pass
                                            viewer = None
                                    else:
                                        # Viewer is not running, close it
                                        print("Viewer is not running, closing...")
                                        try:
                                            if hasattr(viewer, 'close'):
                                                viewer.close()
                                        except:
                                            pass
                                        viewer = None
                            except (AttributeError, RuntimeError, OSError) as e:
                                # Viewer may have issues, but continue simulation
                                # If sync fails, viewer might be broken, but don't crash
                                # Close viewer if it's in invalid state
                                print(f"Warning: Viewer sync failed: {e}. Closing viewer.")
                                try:
                                    if hasattr(viewer, 'close'):
                                        viewer.close()
                                except:
                                    pass
                                viewer = None
                            except Exception as e:
                                # Other exceptions - log but don't close viewer (might be temporary)
                                # Only close on critical errors
                                error_str = str(e).lower()
                                if "segmentation" in error_str or "invalid" in error_str or "signal" in error_str:
                                    print(f"Critical viewer error: {e}. Closing viewer.")
                                    try:
                                        if hasattr(viewer, 'close'):
                                            viewer.close()
                                    except:
                                        pass
                                    viewer = None
                                # For other errors, just skip this sync
                                pass
                    
                    if step_count % control_decimation == 0:
                        # Add safety checks before accessing d
                        if d is None:
                            print("ERROR: d is None in control loop. Ending episode.")
                            done = True
                            break
                        
                        try:
                            robot_pos = d.qpos[:3]
                            robot_quat = d.qpos[3:7]
                        except Exception as e:
                            print(f"ERROR: Failed to access d.qpos: {e}. Ending episode.")
                            done = True
                            break
                        
                        # Get target info (needed for collision handling)
                        distance, sin_angle, cos_angle = get_target_info(robot_pos, target_pos, robot_quat)
                        
                        # Get lidar data (raw, before normalization) - read at control frequency (50Hz) for collision detection
                        try:
                            lidar_data_raw = d.sensordata[lidar_sensor_ids] if len(lidar_sensor_ids) > 0 else np.zeros(raw_lidar_dim)
                        except Exception as e:
                            print(f"ERROR: Failed to access d.sensordata: {e}. Ending episode.")
                            done = True
                            break
                        # Fix negative values that can cause false collision detection
                        lidar_data_raw = fix_negative_lidar_values(lidar_data_raw)
                        # Lidar stays clean for collision/reward. Noise added only in build_actor_observation.
                        
                        # Check for collision - use lidar transformed to center frame
                        collision_detected = False
                        spawn_grace_period = 100  # Don't check collisions in first N control steps (allow robot to stabilize)
                        
                        # Collision detection: transform lidar to center frame, then check min distance
                        lidar_sectors_raw = process_lidar_to_sectors(
                            lidar_data_raw, lidar_sensor_angles,
                            num_sectors=lidar_downsample_bins, max_range=max_lidar_range, min_range=0.25
                        )
                        lidar_sectors_center = transform_lidar_to_center_frame(
                            lidar_sectors_raw, lidar_sensor_angles,
                            lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y,
                            max_range=max_lidar_range, min_range=0.25
                        )
                        min_lidar = np.min(lidar_sectors_center) if len(lidar_sectors_center) > 0 else 10.0
                        
                        if step_count > spawn_grace_period:
                            collision_threshold = reward_weights.get('collision_threshold', 0.35)
                            if min_lidar < collision_threshold:
                                collision_detected = True
                        
                        # Update cached lidar data at SAC frequency for observations
                        if step_count % (control_decimation * sac_decimation) == 0:
                            cached_lidar_data_raw = lidar_data_raw.copy()

                        
                        # If collision detected, end episode
                        if collision_detected:
                            done = True
                            episode_ended_by_collision = True
                            # Use collision penalty from config
                            reward = reward_weights['collision']
                            episode_reward += reward
                            
                            # Update reward components (use counters)
                            reward_components['total'] += reward
                            reward_components['obstacle'] += 0.0
                            reward_components['distance'] += 0.0
                            reward_components['reached'] += 0.0
                            reward_components['vy_penalty'] += 0.0
                            reward_components['vx_backward_penalty'] += 0.0
                            reward_components['velocity_alignment'] += 0.0
                            
                            # Collision is NOT a success (episode ended with failure)
                            success = False
                            
                            # Store collision transition in replay buffer if we have previous observation and action
                            # IMPORTANT: Use build_observation_from_history to avoid corrupting history
                            # Check if action_np was set (not zero vector)
                            # Ensure action_np is defined (should be initialized at line 2098, but safety check)
                            try:
                                action_was_set = np.any(np.abs(action_np) > 1e-6)
                            except (NameError, UnboundLocalError):
                                # If action_np is not defined (shouldn't happen, but safety check)
                                action_np = np.zeros(3, dtype=np.float32)
                                action_was_set = False
                            
                            if args.train and prev_critic_obs is not None and action_was_set:
                                # Build observation for critic (Actor + vx + vy)
                                vx = d.qvel[0]  # X component of velocity
                                vy = d.qvel[1]  # Y component of velocity
                                angular_vel = d.qvel[5]
                                critic_obs_single = build_critic_observation(
                                    lidar_data_raw, lidar_sensor_angles, vx, vy, angular_vel, distance,
                                    sin_angle, cos_angle, max_lidar_range, max_vx, max_vx,  # max_vy = max_vx
                                    max_angular_vel, max_distance, prev_action_np,
                                    lidar_downsample_bins,
                                    use_sector_processing=True,
                                    use_extended_features=False,
                                    critical_topk=critic_critical_topk,
                                    lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y
                                )
                                # IMPORTANT: Use build_observation_from_history instead of process_observation
                                # to avoid corrupting history when collision happens between SAC steps
                                critic_obs = agent.build_observation_from_history(
                                    critic_obs_single, 
                                    is_critic=True, 
                                    history_buffer=None
                                )
                                replay_buffer.add(
                                    prev_critic_obs, action_np, reward, 1.0, critic_obs, success=success
                                )
                            break  # Exit inner loop to end episode
                        
                        # Throttle SAC inference: only every (control_decimation * sac_decimation) steps
                        if step_count % (control_decimation * sac_decimation) == 0:
                            # Use cached lidar data (updated at 10Hz, synchronized with SAC)
                            if cached_lidar_data_raw is None:
                                cached_lidar_data_raw = lidar_data_raw.copy()
                            
                            # Get velocity components (only when needed for SAC)
                            vx = d.qvel[0]  # X component of velocity (forward/backward)
                            vy = d.qvel[1]  # Y component of velocity (lateral)
                            angular_vel = d.qvel[5]  # Yaw angular velocity
                            
                            # Current observations for SAC (Actor and Critic)
                            # Actor: lidar(40) + w(1) + sin(1) + cos(1) + dist(1) + prev(3) = 47 признаков
                            # БЕЗ Vx, Vy (линейные скорости), но С W (угловая скорость)
                            actor_obs_single = build_actor_observation(
                                cached_lidar_data_raw, lidar_sensor_angles, angular_vel, distance,
                                sin_angle, cos_angle, max_lidar_range, max_angular_vel,
                                max_distance, prev_action_np,
                                lidar_downsample_bins,
                                use_sector_processing=True,
                                lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y,
                                obs_noise_distance_std=obs_noise_distance_std if args.train else 0.0,
                                obs_noise_angle_std=obs_noise_angle_std if args.train else 0.0,
                                obs_noise_angular_vel_std=obs_noise_angular_vel_std if args.train else 0.0,
                                obs_noise_lidar_std=obs_noise_lidar_std if args.train else 0.0
                            )
                            
                            # Critic: Actor(47) + vx(1) + vy(1) = 49, плюс история и critical_topk
                            critic_obs_single = build_critic_observation(
                                cached_lidar_data_raw, lidar_sensor_angles, vx, vy, angular_vel, distance,
                                sin_angle, cos_angle, max_lidar_range, max_vx, max_vx,  # max_vy = max_vx
                                max_angular_vel, max_distance, prev_action_np,
                                lidar_downsample_bins,
                                use_sector_processing=True,
                                use_extended_features=False,
                                critical_topk=critic_critical_topk,
                                lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y
                            )
                            
                            # Process observations separately:
                            # Actor: добавляем историю actions к наблюдению
                            # Critic: добавляем историю наблюдений
                            sac_obs = agent.process_observation(
                                actor_obs_single, 
                                is_critic=False, 
                                action_history_buffer=actor_action_history
                            )
                            critic_obs = agent.process_observation(
                                critic_obs_single, 
                                is_critic=True, 
                                history_buffer=None
                            )
                            
                            # Check for NaN/Inf
                            if np.any(np.isnan(sac_obs)) or np.any(np.isinf(sac_obs)):
                                continue
                            if np.any(np.isnan(critic_obs)) or np.any(np.isinf(critic_obs)):
                                continue
                            
                            # Get action from policy (uses actor observation)
                            # get_action автоматически обновит историю actions внутри agent
                            if args.train:
                                # Add noise for exploration during training
                                action_np = agent.get_action(sac_obs, add_noise=True)
                            else:
                                # Use deterministic action for evaluation
                                action_np = agent.get_action(sac_obs, add_noise=False)
                            
                            # Обновляем локальную историю actions для Sequential mode
                            if actor_action_history is not None:
                                actor_action_history.append(action_np.copy())

                            # Check for NaN
                            if np.any(np.isnan(action_np)):
                                continue
                            
                            # SAC outputs in [-1, 1], apply scaling from config
                            vx_cmd = action_np[0] * cmd_scale[0]
                            vy_cmd = action_np[1] * cmd_scale[1]
                            yaw_rate_cmd = action_np[2] * cmd_scale[2]
                            
                            cmd[0] = vx_cmd
                            cmd[1] = vy_cmd
                            cmd[2] = yaw_rate_cmd
                            
                            # REWARD CALCULATION (Reference Style)
                            # Use lidar transformed to center frame (same as collision detection)
                            cached_lidar_sectors = process_lidar_to_sectors(
                                cached_lidar_data_raw, lidar_sensor_angles,
                                num_sectors=lidar_downsample_bins, max_range=max_lidar_range, min_range=0.25
                            )
                            cached_lidar_sectors_center = transform_lidar_to_center_frame(
                                cached_lidar_sectors, lidar_sensor_angles,
                                lidar_offset_x=lidar_offset_x, lidar_offset_y=lidar_offset_y,
                                max_range=max_lidar_range, min_range=0.25
                            )
                            goal = distance < 0.25 # reached_threshold
                            collision_threshold = reward_weights.get('collision_threshold', 0.35)
                            collision = np.min(cached_lidar_sectors_center) < collision_threshold
                            
                            reward, done, reward_info = compute_reward_reference(
                                robot_pos=robot_pos,
                                target_pos=target_pos,
                                lidar_data=cached_lidar_sectors_center,  # Center-frame lidar (meters)
                                actions=prev_action_np,
                                goal=goal,
                                collision=collision,
                                reward_weights=reward_weights,
                                step_count=step_count,
                                max_steps=max_steps,
                                prev_distance=prev_distance
                            )
                            
                            episode_reward += reward
                            
                            # Update prev_distance for next step
                            prev_distance = distance
                            
                            # Store transition in replay buffer if training
                            # Correct transition: (obs_t, action_t, reward_t, obs_t+1, done_t)
                            # Both obs_t and obs_t+1 must be critic observations (with history) for consistency
                            if args.train and prev_critic_obs is not None:
                                replay_buffer.add(
                                    prev_critic_obs, prev_action_np, reward, float(done), critic_obs, success=goal
                                )
                            
                            # Save for next transition
                            # prev_critic_obs stores critic observation (with history) for next step
                            prev_critic_obs = critic_obs.copy()
                            prev_action_np = action_np.copy()
                            prev_reward = reward
                            prev_done_flag = 1.0 if done else 0.0
                            prev_success = goal
                            
                            # Очистка промежуточных переменных для предотвращения утечки памяти
                            del sac_obs, critic_obs, critic_obs_single, action_np
                            
                            # Update reward components (use counters instead of append)
                            reward_components['distance'] += reward_info.get('progress', 0.0)
                            reward_components['reached'] += reward_info.get('reached', 0.0)
                            reward_components['obstacle'] += reward_info.get('obstacle', 0.0)
                            reward_components['vy_penalty'] += reward_info.get('vy_penalty', 0.0)
                            reward_components['vx_backward_penalty'] += reward_info.get('vx_backward_penalty', 0.0)
                            reward_components['velocity_alignment'] += reward_info.get('velocity_alignment', 0.0)
                            reward_components['time_penalty'] += reward_info.get('time_penalty', 0.0)
                            reward_components['total'] += reward_info.get('total', 0.0)
                            reward_components['count'] += 1
                            
                            # Check if episode is done - if target reached, end episode
                            if done:
                                break  # Exit inner loop to end episode
                        
                        # Create observation for walking policy according to PolicyCfg structure
                        # Extract data from MuJoCo
                        try:
                            # Base pose and velocities
                            base_pos = d.qpos[0:3]  # Base position (not used in observation, but needed for transforms)
                            quat = d.qpos[3:7]  # Base quaternion
                            qj = d.qpos[7:7+num_actions]  # Joint positions
                            
                            # Base velocities (in world frame, need to transform to base frame)
                            base_lin_vel_world = d.qvel[0:3]  # Linear velocity in world frame
                            base_ang_vel_world = d.qvel[3:6]  # Angular velocity in world frame
                            dqj = d.qvel[6:6+num_actions]  # Joint velocities
                            
                            # Transform velocities to base frame
                            # For simplicity, we'll use world frame velocities (common approximation)
                            # In a full implementation, you'd rotate by the inverse of base orientation
                            base_lin_vel = base_lin_vel_world  # TODO: Transform to base frame if needed
                            base_ang_vel = base_ang_vel_world  # Angular velocity is already in base frame for most cases
                            
                        except Exception as e:
                            print(f"ERROR: Failed to access d.qpos/d.qvel for walking policy: {e}. Ending episode.")
                            done = True
                            break
                        
                        # Compute projected gravity (in base frame)
                        projected_gravity = get_gravity_orientation(quat)
                        
                        # Joint positions relative to default
                        joint_pos_rel = qj - default_angles
                        
                        # Velocity commands (from SAC planner, already scaled)
                        velocity_commands = cmd.copy()
                        
                        # Build observation using PolicyCfg structure
                        obs_new = build_walking_policy_observation(
                            base_ang_vel=base_ang_vel,
                            projected_gravity=projected_gravity,
                            velocity_commands=velocity_commands,
                            joint_pos=joint_pos_rel,
                            joint_vel=dqj,
                            last_action=action,
                        )
                        # Ensure observation size matches expected size
                        if len(obs_new) != num_obs:
                            print(f"WARNING: Observation size mismatch: got {len(obs_new)}, expected {num_obs}")
                            # Resize obs array if needed
                            if len(obs_new) > num_obs:
                                obs = obs_new[:num_obs]
                            else:
                                obs = np.zeros(num_obs, dtype=np.float32)
                                obs[:len(obs_new)] = obs_new
                        else:
                            obs = obs_new
                        
                        # Walking policy (locomotion) inference (rsl_rl ActorCritic)
                        # This generates joint angles for robot locomotion, separate from SAC path planning policy
                        if walking_policy_model is None:
                            # If no walking policy, use default angles (robot will stand still)
                            # This allows training to continue even without a walking policy
                            target_dof_pos = default_angles.copy()
                            # Keep action as zeros for consistency
                            action = np.zeros(num_actions, dtype=np.float32)
                        else:
                            try:
                                with torch.no_grad():  # Не строить граф автоградиента - исправляет утечку памяти
                                    # rsl_rl ActorCritic.act_inference() expects observations tensor directly
                                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()  # Shape: [1, 45]
                                    
                                    # Use act_inference() for deterministic inference (returns mean, not sampled)
                                    # act_inference() returns mean action directly
                                    action_tensor = walking_policy_model.act_inference(obs_tensor)  # Shape: [1, 12]
                                    action = action_tensor.numpy().squeeze()  # Shape: [12]

                                    if not args.headless:
                                        time.sleep(0.02)                                    
                                    # Явно удаляем тензоры для освобождения памяти
                                    del obs_tensor, action_tensor
                                target_dof_pos = action * action_scales + default_angles
                            except Exception as e:
                                print(f"ERROR: Walking policy (locomotion) inference failed: {e}")
                                print("This may indicate model corruption or memory issues. Ending episode.")
                                done = True
                                break
                        
                        # Update target point position in scene (mocap body)
                        if target_pos is not None and target_body_id is not None and target_mocap_id >= 0:
                            try:
                                # Check if position actually changed
                                current_pos = d.mocap_pos[target_mocap_id]
                                if np.linalg.norm(current_pos - target_pos) > 1e-6:
                                    d.mocap_pos[target_mocap_id] = target_pos
                                    d.mocap_quat[target_mocap_id] = [1, 0, 0, 0]
                            except Exception as e:
                                # Silently ignore mocap update errors - not critical
                                pass
                    
                    # Visualize lidar sectors only if explicitly enabled (default: off for stability)
                    if enable_lidar_debug_vis:
                        if (viewer is not None and d is not None and 
                            hasattr(viewer, 'user_scn') and viewer.user_scn is not None and 
                            step_count > 0):
                            try:
                                if hasattr(viewer, 'is_running') and not viewer.is_running():
                                    viewer = None
                                    raise RuntimeError("Viewer is not running")
                                viewer.user_scn.ngeom = 0
                                if len(lidar_sensor_ids) > 0:
                                    lidar_distances = d.sensordata[lidar_sensor_ids]
                                    lidar_distances = fix_negative_lidar_values(lidar_distances)
                                    # Visualization uses clean lidar (noise only in Actor obs)
                                    lidar_sectors = process_lidar_to_sectors(
                                        lidar_distances, lidar_sensor_angles,
                                        num_sectors=40, max_range=3.0, min_range=0.25
                                    )
                                    robot_pos = d.qpos[:3]
                                    robot_quat = d.qpos[3:7]
                                    w, x, y, z = robot_quat[0], robot_quat[1], robot_quat[2], robot_quat[3]
                                    R = np.array([
                                        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                                        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                                        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
                                    ])
                                    num_sectors = 40
                                    sector_angle = 2 * np.pi / num_sectors
                                    for sector_idx in range(num_sectors):
                                        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                                            break
                                        distance = lidar_sectors[sector_idx]
                                        angle_min = -np.pi + sector_idx * sector_angle
                                        angle_max = angle_min + sector_angle
                                        angle_center = (angle_min + angle_max) / 2
                                        sector_dir_robot = np.array([
                                            np.cos(angle_center),
                                            np.sin(angle_center),
                                            0.0
                                        ])
                                        sector_dir_global = R @ sector_dir_robot
                                        hit_point = robot_pos + sector_dir_global * distance
                                        sphere_id = viewer.user_scn.ngeom
                                        sphere = viewer.user_scn.geoms[sphere_id]
                                        sphere.type = mujoco.mjtGeom.mjGEOM_SPHERE
                                        sphere.size[0] = 0.02
                                        sphere.pos[:] = hit_point
                                        ratio = distance / 3.0
                                        sphere.rgba[:] = [1.0 - ratio, ratio, 0, 0.8]
                                        viewer.user_scn.ngeom += 1
                            except Exception:
                                # Silently ignore visualization errors to avoid segfault
                                if viewer is not None:
                                    try:
                                        if hasattr(viewer, 'close'):
                                            viewer.close()
                                    except Exception:
                                        pass
                                    viewer = None
                
                # Sync viewer only in visual mode (redundant, already synced in loop above)
                # But keep this for end-of-episode sync
                # IMPORTANT: Do NOT update viewer.data - viewer was created with correct model reference
                if viewer is not None:
                    try:
                        # Check if viewer is still running before syncing
                        if hasattr(viewer, 'is_running') and viewer.is_running():
                            try:
                                # CRITICAL: Do not update viewer.data - this can cause segfault
                                # Viewer was created with correct model, so it already has correct data reference
                                # Only sync, don't update data reference
                                viewer.sync()
                            except Exception as e:
                                # Viewer may have been closed or model/data changed
                                print(f"Warning: Could not sync viewer at end of episode: {e}")
                                # Close viewer if it's broken
                                try:
                                    if hasattr(viewer, 'close'):
                                        viewer.close()
                                except:
                                    pass
                                viewer = None
                        else:
                            print("Viewer is not running, skipping sync")
                    except Exception as e:
                        # Viewer may be in completely invalid state
                        print(f"Warning: Viewer end-of-episode check failed: {e}")
                        try:
                            if hasattr(viewer, 'close'):
                                viewer.close()
                        except:
                            pass
                        viewer = None
                    
                    # Real-time pacing only in visual mode
                    # Note: step_start is set inside the simulation loop, so it may be None
                    # if episode ended before entering the loop or if it's headless mode
                    if step_start is not None and not args.headless:
                        time_until_next_step = m.opt.timestep - (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)
                
                # Final update at end of episode (train agent on collected experiences)
                if args.train:
                    if episode % train_every_n == 0:
                        # Only train if buffer has enough samples
                        if replay_buffer.size() >= min_buffer_size:
                            # Get replay buffer sampling weights from config (updated by curriculum)
                            success_weight = replay_buffer_config.get("success_weight", 2.0)
                            collision_weight = replay_buffer_config.get("collision_weight", 0.5)
                            
                            agent.train(
                                replay_buffer=replay_buffer,
                                iterations=training_iterations,
                                batch_size=training_batch_size,
                                success_weight=success_weight,
                                collision_weight=collision_weight
                            )
                            
                            # Очистка GPU кеша после обучения для предотвращения утечки памяти
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                
                # Determine episode termination type
                episode_successful = reward_components['reached'] > 0
                # Check if episode ended due to timeout (reached max_steps without success or collision)
                if step_count >= max_steps and not episode_successful and not episode_ended_by_collision:
                    episode_ended_by_timeout = True
                
                # Update statistics
                episode_successes.append(episode_successful)
                episode_collisions.append(episode_ended_by_collision)
                episode_timeouts.append(episode_ended_by_timeout)
                
                # For test mode, also store in full statistics
                if not args.train:
                    test_episode_successes.append(episode_successful)
                    test_episode_collisions.append(episode_ended_by_collision)
                    test_episode_timeouts.append(episode_ended_by_timeout)
                
                # Keep only last 100 episodes (for training mode display)
                if len(episode_successes) > success_rate_window:
                    episode_successes.pop(0)
                if len(episode_collisions) > success_rate_window:
                    episode_collisions.pop(0)
                if len(episode_timeouts) > success_rate_window:
                    episode_timeouts.pop(0)
                
                # Calculate success rates over last 100 episodes
                success_rate = sum(episode_successes) / len(episode_successes) if len(episode_successes) > 0 else 0.0
                collision_rate = sum(episode_collisions) / len(episode_collisions) if len(episode_collisions) > 0 else 0.0
                timeout_rate = sum(episode_timeouts) / len(episode_timeouts) if len(episode_timeouts) > 0 else 0.0
                
                # Update Curriculum Manager
                if curriculum_manager:
                    level_changed = curriculum_manager.update(episode_successful, episode_reward)
                    if level_changed:
                        new_level_name = curriculum_manager.get_current_level_name()
                        print(f"\n🚀 CURRICULUM LEVEL UP: {new_level_name}!")
                        
                        # Apply new parameters (updates policy_config and replay_buffer_config)
                        curriculum_manager.apply_current_level(
                            reward_weights, obstacle_params,
                            policy_config=policy_config,
                            replay_buffer_config=replay_buffer_config,
                            episode_params=episode_params
                        )
                        
                        # Re-extract training parameters from updated policy_config
                        training_iterations = policy_config.get("training_iterations", training_iterations)
                        training_batch_size = policy_config.get("batch_size", training_batch_size)
                        # Update SAC temperature parameters if changed
                        new_init_temperature = policy_config.get("init_temperature", init_temperature)
                        new_alpha_lr = policy_config.get("alpha_lr", alpha_lr)
                        if new_init_temperature != init_temperature or new_alpha_lr != alpha_lr:
                            init_temperature = new_init_temperature
                            alpha_lr = new_alpha_lr
                            # Update agent's temperature if needed
                            if hasattr(agent, 'log_alpha'):
                                new_log_alpha = np.log(init_temperature)
                                agent.log_alpha.data.fill_(new_log_alpha)
                                # Re-initialize alpha optimizer with new learning rate
                                agent.log_alpha_optimizer = torch.optim.Adam(
                                    [agent.log_alpha], lr=alpha_lr, betas=agent.alpha_betas
                                )
                                print(f"Updated agent temperature to {init_temperature}, alpha_lr to {alpha_lr}")
                        
                        # Update local variables for obstacle generator
                        obstacle_regeneration_interval = obstacle_params.get("regeneration_interval", 10)
                        cube_config = obstacle_params.get("cubes", {})
                        cube_count_min = cube_config.get("count_min", cube_count_min)
                        cube_count_max = cube_config.get("count_max", cube_count_max)
                        cube_size_x_min = cube_config.get("size_x_min", cube_size_x_min)
                        cube_size_x_max = cube_config.get("size_x_max", cube_size_x_max)
                        cube_size_y_min = cube_config.get("size_y_min", cube_size_y_min)
                        cube_size_y_max = cube_config.get("size_y_max", cube_size_y_max)
                        min_pos_margin = obstacle_params.get("position", {}).get("min_margin", min_pos_margin)
                        max_steps = episode_params.get("max_steps", config.get("max_steps", 5000))
                        
                        print(f"New parameters applied: Cubes={cube_count_min}-{cube_count_max}, Margin={min_pos_margin}, MaxSteps={max_steps}")

                # Calculate average reward components for this episode
                if reward_components['count'] > 0:
                    count = reward_components['count']
                    avg_progress_reward = reward_components['distance'] / count  # 'distance' stores progress
                    avg_reached_reward = reward_components['reached'] / count
                    avg_obstacle_penalty = reward_components['obstacle'] / count
                    avg_vy_penalty = reward_components['vy_penalty'] / count
                    avg_vx_backward_penalty = reward_components['vx_backward_penalty'] / count
                    avg_velocity_alignment = reward_components['velocity_alignment'] / count
                    avg_time_penalty = reward_components['time_penalty'] / count
                    avg_total_reward = reward_components['total'] / count
                else:
                    avg_progress_reward = avg_reached_reward = avg_obstacle_penalty = avg_vy_penalty = avg_vx_backward_penalty = avg_velocity_alignment = avg_time_penalty = 0.0
                    avg_total_reward = 0.0
                
                # Log detailed reward information
                if episode % 1 == 0:  # Log every episode
                    curr_level_str = ""
                    if curriculum_manager:
                        curr_level_str = f" [Level: {curriculum_manager.get_current_level_name()}]"
                    
                    print(f"\n=== Episode {episode}{curr_level_str} Summary ===")
                    print(f"Total Reward: {episode_reward:.2f}")
                    print(f"Success: {'YES' if episode_successful else 'NO'}")
                    print(f"Termination: {'Success' if episode_successful else ('Collision' if episode_ended_by_collision else ('Timeout' if episode_ended_by_timeout else 'Unknown'))}")
                    print(f"Success Rates (last {len(episode_successes)} episodes):")
                    print(f"  Success: {success_rate*100:.1f}%")
                    print(f"  Collision: {collision_rate*100:.1f}%")
                    print(f"  Timeout: {timeout_rate*100:.1f}%")
                    print(f"Reward Components (avg per SAC step):")
                    print(f"  Progress: {avg_progress_reward:.3f}")
                    print(f"  Reached: {avg_reached_reward:.3f}")
                    print(f"  Obstacle: {avg_obstacle_penalty:.3f}")
                    print(f"  Vy penalty: {avg_vy_penalty:.3f}")
                    print(f"  Vx backward penalty: {avg_vx_backward_penalty:.3f}")
                    print(f"  Velocity alignment: {avg_velocity_alignment:.3f}")
                    print(f"  Time penalty: {avg_time_penalty:.3f}")
                    print(f"Steps: {step_count}")
                    print("=" * 40)
                    
                    if args.train and episode % args.save_every_n == 0 and episode > 0:
                        # Create directory if it doesn't exist (lazy creation)
                        if model_dir is not None:
                            model_dir.mkdir(parents=True, exist_ok=True)
                        # Save model with metadata (episode number)
                        metadata = {'episode': episode}
                        agent.save(
                            filename=model_name,
                            directory=model_dir,
                            metadata=metadata
                        )
                        print(f"Model saved to {model_dir} / {model_name} (episode {episode})")
                        
                        # Save replay buffer together with model
                        buffer_path = buffer_dir / f"{model_name}_buffer.npz"
                        replay_buffer.save(buffer_path)
                        print(f"Replay buffer saved to {buffer_path} ({replay_buffer.size()} experiences)")

                    # Logging
                    # Log to TensorBoard
                    if writer:
                        writer.add_scalar("episode/reward", episode_reward, episode)
                        writer.add_scalar("episode/avg_total_per_step", avg_total_reward, episode)
                        writer.add_scalar("episode/avg_progress", avg_progress_reward, episode)
                        writer.add_scalar("episode/avg_reached", avg_reached_reward, episode)
                        writer.add_scalar("episode/avg_obstacle", avg_obstacle_penalty, episode)
                        writer.add_scalar("episode/avg_vy_penalty", avg_vy_penalty, episode)
                        writer.add_scalar("episode/avg_vx_backward_penalty", avg_vx_backward_penalty, episode)
                        writer.add_scalar("episode/avg_velocity_alignment", avg_velocity_alignment, episode)
                        writer.add_scalar("episode/avg_time_penalty", avg_time_penalty, episode)
                        writer.add_scalar("episode/success", 1.0 if episode_successful else 0.0, episode)
                        writer.add_scalar("episode/success_rate", success_rate, episode)
                        writer.add_scalar("episode/collision_rate", collision_rate, episode)
                        writer.add_scalar("episode/timeout_rate", timeout_rate, episode)
                        writer.add_scalar("episode/steps", step_count, episode)
                
                episode += 1
    finally:
        # Clean up viewer if it was created (with safe error handling)
        if viewer is not None:
            try:
                if hasattr(viewer, 'close'):
                    viewer.close()
            except Exception as e:
                # Ignore errors during cleanup - viewer may already be closed
                print(f"Note: Viewer cleanup warning: {e}")
            viewer = None
    
    if writer is not None:
        writer.close()
    
    # Print test statistics if in test mode
    if not args.train:
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        
        # Use full test statistics if available
        if test_episode_successes is not None and len(test_episode_successes) > 0:
            total_episodes = len(test_episode_successes)
            successes = sum(test_episode_successes)
            collisions = sum(test_episode_collisions)
            timeouts = sum(test_episode_timeouts)
            success_rate = successes / total_episodes * 100
            collision_rate = collisions / total_episodes * 100
            timeout_rate = timeouts / total_episodes * 100
            
            print(f"Total Episodes: {total_episodes}")
            print(f"Success Rate: {success_rate:.2f}% ({successes} episodes)")
            print(f"Collision Rate: {collision_rate:.2f}% ({collisions} episodes)")
            print(f"Timeout Rate: {timeout_rate:.2f}% ({timeouts} episodes)")
        elif len(episode_successes) > 0:
            # Fallback to regular statistics
            total_episodes = len(episode_successes)
            successes = sum(episode_successes)
            collisions = sum(episode_collisions)
            timeouts = sum(episode_timeouts)
            success_rate = successes / total_episodes * 100
            collision_rate = collisions / total_episodes * 100
            timeout_rate = timeouts / total_episodes * 100
            
            print(f"Total Episodes: {total_episodes}")
            print(f"Success Rate: {success_rate:.2f}% ({successes} episodes)")
            print(f"Collision Rate: {collision_rate:.2f}% ({collisions} episodes)")
            print(f"Timeout Rate: {timeout_rate:.2f}% ({timeouts} episodes)")
        else:
            print("No episodes completed")
        
        print("=" * 60 + "\n")
    
    if args.train:
        # Create directory if it doesn't exist (lazy creation)
        if model_dir is not None:
            model_dir.mkdir(parents=True, exist_ok=True)
        # Save model with metadata (episode number) at the end
        metadata = {'episode': episode}
        agent.save(
            filename=model_name,
            directory=model_dir,
            metadata=metadata
        )
        print(f"Training completed. Model saved to {model_dir} / {model_name} (episode {episode})")
        
        # Close TensorBoard writer
        if writer:
            writer.close()
            print("TensorBoard writer closed.")
        
        # Save replay buffer together with model
        buffer_path = buffer_dir / f"{model_name}_buffer.npz"
        replay_buffer.save(buffer_path)
        print(f"Replay buffer saved to {buffer_path} ({replay_buffer.size()} experiences)")

