"""
Target generator for path planning.
Generates spawn points for robot and targets using precomputed free space grid.
Obstacles are parsed from XML file.
"""
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

# Room bounds (safe area inside walls)
ROOM_X_MIN, ROOM_X_MAX = -3.8, 3.8
ROOM_Y_MIN, ROOM_Y_MAX = -3.8, 3.8


def parse_obstacles_from_xml(xml_path: str) -> dict:
    """
    Parse obstacles (walls, cubes, cylinders) from MuJoCo XML file.
    
    Returns:
        dict with keys: 'walls', 'cubes', 'cylinders'
        Each contains list of dicts with obstacle information
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    obstacles = {
        'walls': [],
        'cubes': [],
        'cylinders': []  # Kept for backward compatibility, but always empty
    }
    
    # Find worldbody
    worldbody = root.find('.//worldbody')
    if worldbody is None:
        return obstacles
    
    # Parse all geoms in worldbody
    for geom in worldbody.findall('geom'):
        name = geom.get('name', '')
        geom_type = geom.get('type', '')
        pos_str = geom.get('pos', '0 0 0')
        size_str = geom.get('size', '0 0 0')
        
        try:
            pos = [float(x) for x in pos_str.split()]
            size = [float(x) for x in size_str.split()]
        except (ValueError, AttributeError):
            continue
        
        # Skip floor and target
        if name in ['floor', 'target_point']:
            continue
        
        # Parse walls (walls have 'wall' in name)
        if 'wall' in name.lower():
            if geom_type == 'box' and len(size) >= 2:
                # size format: [half_x, half_y, half_z]
                obstacles['walls'].append({
                    'name': name,
                    'pos': pos[:2],  # x, y only
                    'size': size[:2]  # half_x, half_y
                })
        
        # Parse cubes (boxes that are not walls)
        elif geom_type == 'box' and 'wall' not in name.lower():
            if len(size) >= 2:
                obstacles['cubes'].append({
                    'name': name,
                    'pos': pos[:2],  # x, y only
                    'size': size[:2]  # half_x, half_y
                })
        
        # Cylinders parsing removed for MJX compatibility
        # MJX does not support (CYLINDER, BOX) collisions
        elif geom_type == 'cylinder':
            # Skip cylinders - they are not supported with MJX
            pass
    
    return obstacles


def is_point_free(x: float, y: float, obstacles: dict, clearance: float) -> bool:
    """
    Check if point (x, y) is free from all obstacles with given clearance.
    
    Args:
        x, y: Point coordinates
        obstacles: Dict from parse_obstacles_from_xml
        clearance: Safety margin from obstacles
    """
    # Check room bounds
    if not (ROOM_X_MIN <= x <= ROOM_X_MAX and ROOM_Y_MIN <= y <= ROOM_Y_MAX):
        return False
    
    # Check walls (AABB)
    for wall in obstacles['walls']:
        px, py = wall['pos'][:2]
        hx, hy = wall['size'][:2]
        xmin, xmax = px - hx - clearance, px + hx + clearance
        ymin, ymax = py - hy - clearance, py + hy + clearance
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return False
    
    # Check cubes (AABB)
    for cube in obstacles['cubes']:
        px, py = cube['pos'][:2]
        hx, hy = cube['size'][:2]
        xmin, xmax = px - hx - clearance, px + hx + clearance
        ymin, ymax = py - hy - clearance, py + hy + clearance
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return False
    
    # Cylinders check removed for MJX compatibility
    # MJX does not support (CYLINDER, BOX) collisions
    # (cylinders list is always empty now)
    for cyl in obstacles['cylinders']:  # This loop will never execute
        px, py = cyl['pos'][:2]
        r = cyl['radius'] + clearance
        dist_sq = (x - px)**2 + (y - py)**2
        if dist_sq <= r**2:
            return False
    
    return True


def generate_free_points_grid(obstacles: dict, clearance: float, step: float = 0.1) -> np.ndarray:
    """
    Generate grid of free points in the room.
    
    Args:
        obstacles: Dict from parse_obstacles_from_xml
        clearance: Safety margin from obstacles
        step: Grid step size
    
    Returns:
        Array of shape (N, 2) with free (x, y) points
    """
    xs = np.arange(ROOM_X_MIN, ROOM_X_MAX + 1e-6, step)
    ys = np.arange(ROOM_Y_MIN, ROOM_Y_MAX + 1e-6, step)
    
    free_points = []
    for x in xs:
        for y in ys:
            if is_point_free(x, y, obstacles, clearance):
                free_points.append([x, y])
    
    return np.array(free_points, dtype=np.float32) if free_points else np.empty((0, 2), dtype=np.float32)


class SpawnPointGenerator:
    """
    Generator for robot and target spawn points using precomputed free space grid.
    """
    def __init__(self, xml_path: str, clearance: float = 0.5, grid_step: float = 0.1):
        """
        Initialize generator by parsing obstacles and creating free points grid.
        
        Args:
            xml_path: Path to MuJoCo XML scene file
            clearance: Safety margin from obstacles (same for all obstacle types)
            grid_step: Step size for grid generation
        """
        self.clearance = clearance
        self.grid_step = grid_step
        
        # Parse obstacles from XML
        self.obstacles = parse_obstacles_from_xml(xml_path)
        print(f"Parsed obstacles: {len(self.obstacles['walls'])} walls, "
              f"{len(self.obstacles['cubes'])} cubes, {len(self.obstacles['cylinders'])} cylinders")
        
        # Generate free points grid
        self.free_points = generate_free_points_grid(self.obstacles, clearance, grid_step)
        print(f"Generated {len(self.free_points)} free spawn points (clearance={clearance:.2f}m, step={grid_step:.2f}m)")
        
        if len(self.free_points) == 0:
            raise ValueError("No free points found! Check clearance and room bounds.")
    
    def sample_spawn_point(self, z_height: float = 0.793) -> np.ndarray:
        """
        Sample a random spawn point from free points grid.
        
        Args:
            z_height: Z coordinate for spawn point
        
        Returns:
            (x, y, z) spawn position
        """
        if len(self.free_points) == 0:
            return np.array([0.0, 0.0, z_height])
        
        idx = np.random.randint(0, len(self.free_points))
        x, y = self.free_points[idx]
        return np.array([x, y, z_height], dtype=np.float32)
    
    def sample_target_point(self, robot_pos: np.ndarray, min_distance: float = 1.3, 
                           z_height: float = 0.1) -> np.ndarray:
        """
        Sample a target point that is at least min_distance away from robot.
        
        Args:
            robot_pos: Robot position (x, y, z)
            min_distance: Minimum distance from robot
            z_height: Z coordinate for target
        
        Returns:
            (x, y, z) target position
        """
        if len(self.free_points) == 0:
            return np.array([0.0, 0.0, z_height])
        
        robot_xy = robot_pos[:2]
        
        # Filter points by distance
        distances = np.linalg.norm(self.free_points - robot_xy, axis=1)
        valid_indices = np.where(distances >= min_distance)[0]
        
        if len(valid_indices) == 0:
            # Fallback: use any free point
            valid_indices = np.arange(len(self.free_points))
        
        idx = np.random.choice(valid_indices)
        x, y = self.free_points[idx]
        return np.array([x, y, z_height], dtype=np.float32)
    
    def update_obstacles(self, xml_path: str):
        """
        Re-parse obstacles from XML and regenerate free points grid.
        Call this after updating the scene XML file.
        
        Args:
            xml_path: Path to updated MuJoCo XML scene file
        """
        # Re-parse obstacles from XML
        self.obstacles = parse_obstacles_from_xml(xml_path)
        print(f"Updated obstacles: {len(self.obstacles['walls'])} walls, "
              f"{len(self.obstacles['cubes'])} cubes, {len(self.obstacles['cylinders'])} cylinders")
        
        # Regenerate free points grid
        self.free_points = generate_free_points_grid(self.obstacles, self.clearance, self.grid_step)
        print(f"Regenerated {len(self.free_points)} free spawn points (clearance={self.clearance:.2f}m, step={self.grid_step:.2f}m)")
        
        if len(self.free_points) == 0:
            raise ValueError("No free points found after obstacle update! Check clearance and room bounds.")


def get_target_info(robot_pos: np.ndarray, target_pos: np.ndarray, 
                    robot_quat: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    Calculate target information relative to robot position and orientation.
    
    Args:
        robot_pos: (x, y, z) robot position
        target_pos: (x, y, z) target position
        robot_quat: (qw, qx, qy, qz) robot orientation quaternion. If None, uses global coordinates.
    
    Returns:
        distance: distance to target
        sin_angle: sin of angle to target in robot's local frame (x is forward)
        cos_angle: cos of angle to target in robot's local frame (x is forward)
    """
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    if distance < 1e-6:
        sin_angle = 0.0
        cos_angle = 1.0
    else:
        if robot_quat is not None:
            # Transform direction vector to robot's local frame
            # Extract yaw angle from quaternion: q = [qw, qx, qy, qz]
            qw, qx, qy, qz = robot_quat[0], robot_quat[1], robot_quat[2], robot_quat[3]
            
            # Extract yaw angle (rotation around z-axis) using Euler angle formula
            # This works for any orientation, not just pure z-axis rotation
            yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
            
            # Rotate direction vector from global to local frame
            # In robot's local frame: x is forward, y is left
            # Rotation matrix for 2D: [cos(yaw)  sin(yaw)]  [dx]
            #                          [-sin(yaw) cos(yaw)] [dy]
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            # Transform to local coordinates
            dx_local = cos_yaw * dx + sin_yaw * dy  # Forward component
            dy_local = -sin_yaw * dx + cos_yaw * dy  # Left component
            
            # Normalize to get sin and cos
            sin_angle = dy_local / distance
            cos_angle = dx_local / distance
        else:
            # Fallback to global coordinates if no quaternion provided
            sin_angle = dy / distance
            cos_angle = dx / distance
    
    return distance, sin_angle, cos_angle