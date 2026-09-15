import hashlib
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from dog_path_planning.utils.scene_generator import regenerate_scene_obstacles


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_obstacle_generation_only_changes_runtime_copy(tmp_path):
    template = ROOT / "assets/unitree_a1/scene.xml"
    original_digest = digest(template)
    runtime_scene = tmp_path / "scene.xml"
    shutil.copyfile(template, runtime_scene)

    np.random.seed(42)
    regenerate_scene_obstacles(str(runtime_scene), num_cubes=3)

    root = ET.parse(runtime_scene).getroot()
    cubes = [
        geom
        for geom in root.findall(".//worldbody/geom")
        if geom.get("name", "").startswith("cube")
    ]
    assert len(cubes) == 3
    assert digest(template) == original_digest
