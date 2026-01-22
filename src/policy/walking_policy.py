import torch
from rsl_rl.modules import ActorCritic


def load_walking_policy_from_checkpoint(checkpoint_path, obs_dim=45, action_dim=12):
    """
    Load walking policy from PyTorch checkpoint using rsl_rl ActorCritic.
    
    Uses standard load_state_dict() method from rsl_rl, which handles
    weight mapping automatically.
    
    Architecture matches RslRlPpoActorCriticCfg:
    - actor_hidden_dims=[512, 256, 128]
    - activation="elu"
    - init_noise_std=1.0
    - actor_obs_normalization=False
    
    Args:
        checkpoint_path: Path to checkpoint file (model_4999.pt)
        obs_dim: Observation dimension (45 for Unitree A1)
        action_dim: Action dimension (12 for Unitree A1)
    
    Returns:
        torch.nn.Module: Walking policy model (ActorCritic) in eval mode
    """
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state_dict from checkpoint (rsl_rl typically saves as 'model_state_dict')
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume checkpoint is already a state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Determine architecture from state_dict
    if 'actor.0.weight' not in state_dict:
        raise ValueError(f"No actor weights found in checkpoint. Available keys: {list(state_dict.keys())[:10]}")
    
    actual_obs_dim = state_dict['actor.0.weight'].shape[1]
    
    # Find actor layer indices to determine architecture
    actor_keys = [k for k in state_dict.keys() if k.startswith('actor.') and '.weight' in k]
    actor_layer_indices = sorted([int(k.split('.')[1]) for k in actor_keys])
    
    if len(actor_layer_indices) == 0:
        raise ValueError("No actor layers found in checkpoint")
    
    # Get last layer to determine action_dim
    last_layer_idx = max(actor_layer_indices)
    last_layer_key = f'actor.{last_layer_idx}.weight'
    last_layer_output_dim = state_dict[last_layer_key].shape[0]
    
    # Actor outputs mean only (action_dim), std is a learnable parameter
    actual_action_dim = last_layer_output_dim
    
    # Determine critic obs dimension from critic weights (if available)
    critic_keys = [k for k in state_dict.keys() if k.startswith('critic.') and '.weight' in k]
    if len(critic_keys) > 0:
        critic_layer_indices = sorted([int(k.split('.')[1]) for k in critic_keys])
        first_critic_layer_key = f'critic.{critic_layer_indices[0]}.weight'
        actual_critic_obs_dim = state_dict[first_critic_layer_key].shape[1]
    else:
        # If no critic weights, assume same as actor obs
        actual_critic_obs_dim = actual_obs_dim
    
    print(f"Loading walking policy: actor_obs={actual_obs_dim}, critic_obs={actual_critic_obs_dim}, actions={actual_action_dim}")
    
    # Create ActorCritic from rsl_rl with correct architecture
    # Signature: ActorCritic(num_actor_obs, num_critic_obs, num_actions, ...)
    # Matches RslRlPpoActorCriticCfg configuration
    actor_critic = ActorCritic(
        num_actor_obs=actual_obs_dim,
        num_critic_obs=actual_critic_obs_dim,
        num_actions=actual_action_dim,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # Use standard load_state_dict() method from rsl_rl
    # This automatically handles weight mapping if structure matches
    actor_critic.load_state_dict(state_dict, strict=False)    
    actor_critic.eval()
    print(f"✓ Walking policy loaded successfully from {checkpoint_path}")
    
    return actor_critic