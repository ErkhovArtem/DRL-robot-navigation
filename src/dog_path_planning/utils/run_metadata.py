"""Helpers for tagging training runs with git info and config snapshots."""
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


RUN_METADATA_KEYS = (
    "git_commit",
    "git_branch",
    "git_dirty",
    "description",
    "config_file",
)


def get_git_info(project_root: Path) -> Dict[str, Any]:
    """Return git commit info for the project, or empty values if unavailable."""
    root = str(project_root)
    try:
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.call(
            ["git", "-C", root, "diff", "--quiet", "--ignore-submodules"],
            stderr=subprocess.DEVNULL,
        ) != 0 or subprocess.call(
            ["git", "-C", root, "diff", "--cached", "--quiet", "--ignore-submodules"],
            stderr=subprocess.DEVNULL,
        ) != 0
        return {
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
        }


def build_run_metadata(
    config_path: Path,
    description: str = "",
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build static metadata fields for a training run."""
    root = project_root or config_path.resolve().parent.parent
    git_info = get_git_info(root)
    return {
        **git_info,
        "description": description,
        "config_file": config_path.name,
    }


def load_existing_run_metadata(
    model_dir: Path,
    model_name: str = "sac",
) -> Optional[Dict[str, Any]]:
    """Load static run metadata from an existing checkpoint directory."""
    metadata_path = model_dir / f"{model_name}_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return {key: metadata[key] for key in RUN_METADATA_KEYS if key in metadata}


def copy_run_configs(
    model_dir: Path,
    config_path: Path,
    curriculum_path: Path,
) -> Path:
    """Copy config snapshots next to model weights. Returns configs directory."""
    configs_dir = model_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, configs_dir / config_path.name)
    if curriculum_path.exists():
        shutil.copy2(curriculum_path, configs_dir / curriculum_path.name)
    return configs_dir


def make_save_metadata(run_metadata: Dict[str, Any], episode: int) -> Dict[str, Any]:
    """Merge static run metadata with the current episode number."""
    return {**run_metadata, "episode": episode}
