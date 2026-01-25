"""
Setup script for Dog PathPlanning package.
Install in development mode with: pip install -e .
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README if exists
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="dog-pathplanning",
    version="0.1.0",
    description="Path planning for quadruped robot using SAC reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/Dog_PathPlanning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.24.3",
        "torch==2.4.1",
        "pyyaml>=5.4.1",
        "mujoco==3.2.3",
        "tensorboard>=2.7.0",
        "rsl-rl-lib==2.3.3",
    ],
    extras_require={
        "mjx": [
            "mujoco-mjx",
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
        ],
        "ros2": [
            "rclpy",
            "sensor_msgs",
            "visualization_msgs",
            "geometry_msgs",
            "std_msgs",
            "sensor_msgs_py",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
