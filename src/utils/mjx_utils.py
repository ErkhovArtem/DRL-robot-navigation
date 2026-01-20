"""
MJX (MuJoCo XLA) utilities for parallel batched simulation.
"""
import numpy as np

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
