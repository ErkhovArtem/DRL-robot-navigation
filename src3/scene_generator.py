"""
Scene generator for dynamic obstacle placement.
Generates random obstacles in the scene and updates XML file.
"""
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

# Room bounds (safe area inside walls)
ROOM_X_MIN, ROOM_X_MAX = -3.8, 3.8
ROOM_Y_MIN, ROOM_Y_MAX = -3.8, 3.8


def generate_random_obstacles(
    num_cubes: int = 5,
    num_cylinders: int = 0,  # Kept for backward compatibility, but always ignored
    cube_size_x_min: float = 0.1,
    cube_size_x_max: float = 0.5,
    cube_size_y_min: float = 0.1,
    cube_size_y_max: float = 0.5,
    cube_size_z: float = 1.0,
    cylinder_radius_min: float = 0.1,  # Kept for backward compatibility, but ignored
    cylinder_radius_max: float = 0.5,  # Kept for backward compatibility, but ignored
    cylinder_height: float = 2.0,  # Kept for backward compatibility, but ignored
    min_pos_margin: float = 0.5
) -> Tuple[List[dict], List[dict]]:
    """
    Generate random obstacles (cubes only) with random positions and sizes.
    
    Args:
        num_cubes: Number of cubes to generate
        num_cylinders: Ignored (kept for backward compatibility, always 0 for MJX compatibility)
        cube_size_x_min/max: Minimum/maximum half-width (X dimension) for cubes
        cube_size_y_min/max: Minimum/maximum half-length (Y dimension) for cubes
        cube_size_z: Fixed half-height (Z dimension) for cubes
        cylinder_radius_min/max: Ignored (kept for backward compatibility)
        cylinder_height: Ignored (kept for backward compatibility)
        min_pos_margin: Minimum margin from room boundaries
    
    Returns:
        Tuple of (cubes_list, empty_cylinders_list)
        Each cube: {'name': str, 'pos': [x, y, z], 'size': [half_x, half_y, half_z], 'quat': [w, x, y, z] or None}
        Cylinders list is always empty (MJX compatibility)
    """
    cubes = []
    cylinders = []  # Always empty - cylinders removed for MJX compatibility
    
    # Generate random cubes
    for i in range(num_cubes):
        # Random size (half-extents) with individual ranges for X and Y, fixed Z
        size_x = np.random.uniform(cube_size_x_min, cube_size_x_max)
        size_y = np.random.uniform(cube_size_y_min, cube_size_y_max)
        size_z = cube_size_z  # Fixed half-height
        
        # Random position within room bounds (with margin)
        x = np.random.uniform(ROOM_X_MIN + min_pos_margin, ROOM_X_MAX - min_pos_margin)
        y = np.random.uniform(ROOM_Y_MIN + min_pos_margin, ROOM_Y_MAX - min_pos_margin)
        z = size_z  # Position at half height (size_z is already half-height)
        
        # Random orientation (optional, 50% chance)
        quat = None
        if np.random.random() > 0.5:
            # Random rotation around Z axis
            angle = np.random.uniform(0, 2 * np.pi)
            quat = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        
        cubes.append({
            'name': f'cube{i+1}',
            'pos': [x, y, z],
            'size': [size_x, size_y, size_z],
            'quat': quat,
            'material': 'cube1'  # Use cube1 material
        })
    
    # Cylinders generation removed for MJX compatibility
    # MJX does not support (CYLINDER, BOX) collisions
    
    return cubes, cylinders  # cylinders is always empty


def update_scene_xml(xml_path: str, cubes: List[dict], cylinders: List[dict], 
                     keep_walls: bool = True, keep_target: bool = True):
    """
    Update scene XML file with new obstacles (cubes only).
    
    Args:
        xml_path: Path to XML file
        cubes: List of cube dictionaries
        cylinders: Ignored (kept for backward compatibility, should be empty for MJX)
        keep_walls: Whether to keep existing walls
        keep_target: Whether to keep target_body
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find worldbody
    worldbody = root.find('.//worldbody')
    if worldbody is None:
        raise ValueError("worldbody not found in XML")
    
    # Remove existing obstacles (cubes and cylinders), but keep walls, floor, and target
    geoms_to_remove = []
    for geom in worldbody.findall('geom'):
        name = geom.get('name', '')
        geom_type = geom.get('type', '')
        
        # Keep floor and walls
        if name == 'floor' or 'wall' in name.lower():
            continue
        
        # Keep target (it's in a body, not a geom)
        if name == 'target_point':
            continue
        
        # Remove cubes and cylinders (but keep walls)
        if geom_type in ['box', 'cylinder'] and 'wall' not in name.lower():
            geoms_to_remove.append(geom)
    
    for geom in geoms_to_remove:
        worldbody.remove(geom)
    
    # Add new cubes
    for cube in cubes:
        geom_elem = ET.SubElement(worldbody, 'geom')
        geom_elem.set('name', cube['name'])
        geom_elem.set('type', 'box')
        geom_elem.set('size', f"{cube['size'][0]} {cube['size'][1]} {cube['size'][2]}")
        geom_elem.set('pos', f"{cube['pos'][0]} {cube['pos'][1]} {cube['pos'][2]}")
        geom_elem.set('material', cube['material'])
        
        # Add quaternion if provided
        if cube['quat'] is not None:
            quat = cube['quat']
            geom_elem.set('quat', f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
    
    # Cylinders removed for MJX compatibility
    # MJX does not support (CYLINDER, BOX) collisions
    
    # Write updated XML
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    print(f"Updated scene XML with {len(cubes)} cubes")


def regenerate_scene_obstacles(
    xml_path: str,
    num_cubes: int = 5,
    num_cylinders: int = 0,  # Ignored, kept for backward compatibility
    cube_size_x_min: float = 0.1,
    cube_size_x_max: float = 0.5,
    cube_size_y_min: float = 0.1,
    cube_size_y_max: float = 0.5,
    cube_size_z: float = 1.0,
    cylinder_radius_min: float = 0.1,  # Ignored, kept for backward compatibility
    cylinder_radius_max: float = 0.5,  # Ignored, kept for backward compatibility
    cylinder_height: float = 2.0,  # Ignored, kept for backward compatibility
    min_pos_margin: float = 0.5
):
    """
    Regenerate obstacles in scene XML file (cubes only, no cylinders for MJX compatibility).
    
    Args:
        xml_path: Path to XML file
        num_cubes: Number of cubes to generate
        num_cylinders: Ignored (always 0 for MJX compatibility)
        cube_size_x_min/max: Minimum/maximum half-width (X dimension) for cubes
        cube_size_y_min/max: Minimum/maximum half-length (Y dimension) for cubes
        cube_size_z: Fixed half-height (Z dimension) for cubes
        cylinder_radius_min/max: Ignored
        cylinder_height: Ignored
        min_pos_margin: Minimum margin from room boundaries
    
    Returns:
        Tuple of (cubes, empty_cylinders_list)
    """
    cubes, cylinders = generate_random_obstacles(
        num_cubes=num_cubes,
        num_cylinders=0,  # Always 0 for MJX compatibility
        cube_size_x_min=cube_size_x_min,
        cube_size_x_max=cube_size_x_max,
        cube_size_y_min=cube_size_y_min,
        cube_size_y_max=cube_size_y_max,
        cube_size_z=cube_size_z,
        min_pos_margin=min_pos_margin
    )
    
    update_scene_xml(xml_path, cubes, cylinders)
    
    return cubes, cylinders

