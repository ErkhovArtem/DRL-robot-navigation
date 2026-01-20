#!/usr/bin/env python3
"""
Test script for exponential obstacle penalty formula.
Verifies that the formula works correctly and produces reasonable values.
"""
import numpy as np
import matplotlib.pyplot as plt

def calculate_obstacle_penalty(min_dist, obstacle_threshold=1.5, death_distance=0.3, 
                               exponential_scale=4.0, obs_penalty_weight=2.5):
    """
    Calculate exponential obstacle penalty.
    
    Formula (3 zones):
    - If min_dist >= obstacle_threshold: penalty = 0 (SAFE)
    - If min_dist from death_distance to obstacle_threshold: 
      penalty = -(exp(k * (threshold - x) / (x - death_distance)) / exp(max_arg)) * obs_penalty_weight (DANGER)
      Normalized to range [0, -obs_penalty_weight]
    - If min_dist <= death_distance: penalty = -obs_penalty_weight (DEATH - MAXIMUM penalty, NOT zero!)
      This is a NEGATIVE penalty (penalty, not reward!)
    """
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
    obstacle_penalty = -normalized_exp * obs_penalty_weight
    
    # For debugging, also return raw penalty
    obstacle_penalty_raw = np.where(
        mask_safe,
        0.0,
        np.where(
            mask_danger,
            -np.exp(arg_exp),
            -max_penalty_value  # Death zone gets maximum raw penalty
        )
    )
    
    return obstacle_penalty, obstacle_penalty_raw, max_penalty_value

def test_formula():
    """Test the formula with various distances."""
    print("=" * 60)
    print("Testing Exponential Obstacle Penalty Formula")
    print("=" * 60)
    
    obstacle_threshold = 1.5
    death_distance = 0.3
    exponential_scale = 4.0
    obs_penalty_weight = 2.5
    
    # Test distances
    test_distances = [
        2.0,   # Far away (no penalty)
        1.5,   # At threshold (should be 0)
        1.2,   # Between threshold and death
        1.0,   # Closer
        0.8,   # Even closer
        0.5,   # Very close
        0.4,   # Almost at death distance
        0.31,  # Just above death distance
        0.3,   # At death distance (should be MAXIMUM)
        0.25,  # Below death distance (should be MAXIMUM, collision zone)
    ]
    
    print(f"\nParameters:")
    print(f"  obstacle_threshold: {obstacle_threshold} m")
    print(f"  death_distance: {death_distance} m")
    print(f"  exponential_scale: {exponential_scale}")
    print(f"  obs_penalty_weight: {obs_penalty_weight}")
    print(f"\n{'Distance (m)':<15} {'Penalty (final)':<20} {'Raw Penalty':<20} {'Status':<20}")
    print("-" * 75)
    
    for dist in test_distances:
        penalty_final, penalty_raw, max_penalty = calculate_obstacle_penalty(
            np.array([dist]), obstacle_threshold, death_distance, 
            exponential_scale, obs_penalty_weight
        )
        
        if dist >= obstacle_threshold:
            status = "No penalty"
        elif dist <= death_distance:
            status = "Collision zone"
        else:
            status = "Penalty active"
        
        print(f"{dist:<15.2f} {penalty_final[0]:<20.6f} {penalty_raw[0]:<20.6f} {status:<20}")
    
    print(f"\nMax penalty (at death_distance + epsilon): {max_penalty:.2e}")
    print("=" * 60)
    
    # Test edge cases
    print("\nEdge Case Tests:")
    print("-" * 60)
    
    # Test exactly at threshold
    dist = obstacle_threshold
    penalty_final, _, _ = calculate_obstacle_penalty(
        np.array([dist]), obstacle_threshold, death_distance, 
        exponential_scale, obs_penalty_weight
    )
    assert abs(penalty_final[0]) < 1e-10, f"Penalty at threshold should be 0, got {penalty_final[0]}"
    print(f"✓ At threshold ({dist}m): penalty = {penalty_final[0]:.6f} (should be ~0)")
    
    # Test just above death distance
    dist = death_distance + 0.01
    penalty_final, _, _ = calculate_obstacle_penalty(
        np.array([dist]), obstacle_threshold, death_distance, 
        exponential_scale, obs_penalty_weight
    )
    print(f"✓ Just above death ({dist}m): penalty = {penalty_final[0]:.6f} (should be large)")
    assert penalty_final[0] < 0, "Penalty should be negative"
    
    # Test far away
    dist = obstacle_threshold + 0.1
    penalty_final, _, _ = calculate_obstacle_penalty(
        np.array([dist]), obstacle_threshold, death_distance, 
        exponential_scale, obs_penalty_weight
    )
    assert abs(penalty_final[0]) < 1e-10, f"Penalty far away should be 0, got {penalty_final[0]}"
    print(f"✓ Far away ({dist}m): penalty = {penalty_final[0]:.6f} (should be 0)")
    
    print("\n✓ All edge case tests passed!")
    print("=" * 60)
    
    # Generate plot
    distances = np.linspace(0.25, 2.0, 1000)
    penalties, _, _ = calculate_obstacle_penalty(
        distances, obstacle_threshold, death_distance, 
        exponential_scale, obs_penalty_weight
    )
    
    # Filter out invalid values for plotting
    valid_mask = np.isfinite(penalties)
    distances_plot = distances[valid_mask]
    penalties_plot = penalties[valid_mask]
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances_plot, penalties_plot, 'b-', linewidth=2, label='Normalized Penalty')
    plt.axvline(x=obstacle_threshold, color='g', linestyle='--', label=f'Threshold ({obstacle_threshold}m)')
    plt.axvline(x=death_distance, color='r', linestyle='--', label=f'Death Distance ({death_distance}m)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Distance to Obstacle (m)', fontsize=12)
    plt.ylabel('Penalty', fontsize=12)
    plt.title('Exponential Obstacle Penalty Function', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.25, 2.0)
    if len(penalties_plot) > 0:
        y_min = np.min(penalties_plot[penalties_plot < 0]) if np.any(penalties_plot < 0) else -0.1
        plt.ylim(y_min * 1.1, 0.1)
    plt.tight_layout()
    plt.savefig('/home/griga/G1_PathPlanning/src3/obstacle_penalty_plot.png', dpi=150)
    print(f"\n✓ Plot saved to: obstacle_penalty_plot.png")
    
    return True

if __name__ == "__main__":
    try:
        test_formula()
        print("\n✅ All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

