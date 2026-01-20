#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX Inference Script for SAC Actor Policy

This script loads an ONNX model and performs inference for robot control.
Designed to work with lidar_2d_processor.py output.

Usage:
    python inference_onnx.py --model_path sac_actor.onnx --config_path configs/g1.yaml

Dependencies:
    pip install onnxruntime numpy pyyaml
"""

import argparse
import numpy as np
import yaml
from pathlib import Path
from collections import deque
from typing import Optional, Tuple


class SACInference:
    """SAC Actor inference with ONNX model."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        history_length: Optional[int] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to ONNX model file
            config_path: Path to config YAML file
            history_length: Action history length (None = auto-detect from model)
        """
        import onnxruntime as ort
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        sac_config = config.get('sac', {})
        
        # Normalization parameters
        self.max_lidar_range = sac_config.get('max_lidar_range', 3.0)
        self.max_angular_vel = sac_config.get('max_angular_vel', 0.35)
        self.max_distance = config.get('max_distance', 10.75)  # From curriculum or default
        self.cmd_scale = config.get('cmd_scale', [1.7, 1.5, 0.35])
        
        # Model parameters
        if history_length is None:
            history_length = sac_config.get('history_length', 0)
        self.history_length = history_length
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get actual input dimension from model
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 2:
            self.obs_dim = input_shape[1]
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
        
        # Calculate base observation dimension
        self.base_obs_dim = 47  # Base actor observation
        if self.obs_dim > self.base_obs_dim:
            # Model expects history
            self.expected_history_length = (self.obs_dim - self.base_obs_dim) // 3
            if self.history_length != self.expected_history_length:
                print(f"⚠️ WARNING: Model expects history_length={self.expected_history_length}, "
                      f"but config has {self.history_length}. Using model's expected length.")
                self.history_length = self.expected_history_length
        else:
            self.expected_history_length = 0
        
        # Action history buffer
        if self.history_length > 0:
            self.action_history = deque(maxlen=self.history_length)
        else:
            self.action_history = None
        
        print(f"✓ Model loaded: {model_path}")
        print(f"  Input dimension: {self.obs_dim}")
        print(f"  Base obs dimension: {self.base_obs_dim}")
        print(f"  Action history length: {self.history_length}")
        print(f"  Output dimension: 3 (vx, vy, w)")
    
    def normalize_lidar(self, lidar_sectors: np.ndarray) -> np.ndarray:
        """
        Normalize lidar sector distances to [-1, 1].
        
        Args:
            lidar_sectors: 40 sector distances in meters [40]
        
        Returns:
            Normalized lidar data [-1, 1] where -1=close, +1=far
        """
        # Replace invalid values
        lidar_sectors = np.where(
            (lidar_sectors >= 0) & 
            (lidar_sectors <= self.max_lidar_range) & 
            np.isfinite(lidar_sectors),
            lidar_sectors,
            self.max_lidar_range
        )
        # Clip and normalize to [-1, 1]
        lidar_data = np.clip(lidar_sectors, 0, self.max_lidar_range) / self.max_lidar_range
        lidar_data = lidar_data * 2.0 - 1.0
        return lidar_data
    
    def build_observation(
        self,
        lidar_sectors: np.ndarray,  # 40 distances in meters
        angular_vel: float,          # Current angular velocity W (rad/s)
        distance: float,             # Distance to target (m)
        sin_angle: float,            # sin(angle_to_target)
        cos_angle: float,            # cos(angle_to_target)
        prev_action: Optional[np.ndarray] = None  # Previous action [vx, vy, w] in [-1, 1]
    ) -> np.ndarray:
        """
        Build normalized observation for Actor.
        
        Args:
            lidar_sectors: 40 sector distances in meters [40]
            angular_vel: Current angular velocity W (rad/s)
            distance: Distance to target (m)
            sin_angle: sin(angle_to_target)
            cos_angle: cos(angle_to_target)
            prev_action: Previous action [vx, vy, w] in [-1, 1], or None
        
        Returns:
            Observation array [obs_dim] ready for model input
        """
        # Normalize inputs
        lidar_data = self.normalize_lidar(lidar_sectors)  # [40] → [-1, 1]
        
        angular_vel_norm = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel) / self.max_angular_vel
        distance_norm = (np.clip(distance, 0, self.max_distance) / self.max_distance) * 2.0 - 1.0
        
        sin_angle = np.clip(sin_angle, -1, 1)
        cos_angle = np.clip(cos_angle, -1, 1)
        
        if prev_action is None:
            prev_action = np.zeros(3)
        prev_action = np.clip(prev_action, -1, 1)
        
        # Build base observation [47]
        base_obs = np.concatenate([
            lidar_data,                # 40
            [angular_vel_norm],        # 1
            [sin_angle],               # 1
            [cos_angle],               # 1
            [distance_norm],           # 1
            [prev_action[0]],          # 1
            [prev_action[1]],          # 1
            [prev_action[2]]           # 1
        ], dtype=np.float32)
        
        # Add action history if needed
        if self.history_length > 0 and self.action_history is not None:
            if len(self.action_history) == 0:
                # Pad with zeros
                action_history_flat = np.zeros(self.history_length * 3, dtype=np.float32)
            else:
                action_history_flat = np.concatenate(list(self.action_history), dtype=np.float32)
                # Pad at beginning if not full
                if len(self.action_history) < self.history_length:
                    missing = self.history_length - len(self.action_history)
                    action_history_flat = np.concatenate([
                        np.zeros(missing * 3, dtype=np.float32),
                        action_history_flat
                    ])
            
            obs = np.concatenate([base_obs, action_history_flat], dtype=np.float32)
        else:
            obs = base_obs
        
        return obs
    
    def predict(
        self,
        lidar_sectors: np.ndarray,
        angular_vel: float,
        distance: float,
        sin_angle: float,
        cos_angle: float,
        prev_action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference and return action.
        
        Args:
            lidar_sectors: 40 sector distances in meters [40]
            angular_vel: Current angular velocity W (rad/s)
            distance: Distance to target (m)
            sin_angle: sin(angle_to_target)
            cos_angle: cos(angle_to_target)
            prev_action: Previous action [vx, vy, w] in [-1, 1], or None
        
        Returns:
            (raw_action, scaled_action) where:
            - raw_action: [vx, vy, w] in [-1, 1]
            - scaled_action: [vx, vy, w] scaled by cmd_scale (m/s, m/s, rad/s)
        """
        # Build observation
        obs = self.build_observation(
            lidar_sectors, angular_vel, distance, sin_angle, cos_angle, prev_action
        )
        
        # Reshape for batch input [1, obs_dim]
        obs_batch = obs.reshape(1, -1)
        
        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: obs_batch})
        raw_action = outputs[0][0]  # [3] in [-1, 1]
        
        # Scale action
        scaled_action = raw_action * np.array(self.cmd_scale)
        
        # Update action history
        if self.action_history is not None:
            self.action_history.append(raw_action.copy())
        
        return raw_action, scaled_action
    
    def reset_history(self):
        """Reset action history (call at episode start)."""
        if self.action_history is not None:
            self.action_history.clear()
    
    def reset(self):
        """Reset inference state (alias for reset_history)."""
        self.reset_history()


def main():
    """Example usage."""
    parser = argparse.ArgumentParser(description='SAC ONNX Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--config_path', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--history_length', type=int, default=None, 
                       help='Action history length (None = auto-detect)')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = SACInference(args.model_path, args.config_path, args.history_length)
    
    # Example: dummy inputs for testing
    print("\nTesting inference with dummy inputs...")
    
    lidar_sectors = np.full(40, 3.0, dtype=np.float32)  # All max range (no obstacles)
    angular_vel = 0.0
    distance = 5.0
    sin_angle = 0.0
    cos_angle = 1.0
    prev_action = np.zeros(3)
    
    raw_action, scaled_action = inference.predict(
        lidar_sectors, angular_vel, distance, sin_angle, cos_angle, prev_action
    )
    
    print(f"Raw action: {raw_action} (range should be [-1, 1])")
    print(f"Scaled action: {scaled_action} (vx={scaled_action[0]:.3f} m/s, "
          f"vy={scaled_action[1]:.3f} m/s, w={scaled_action[2]:.3f} rad/s)")
    print("\n✓ Inference test completed successfully!")


if __name__ == '__main__':
    main()
