# Setup is optional — the codebase is run directly as scripts.
# If you want to install dependencies in one shot:
#     pip install -r requirements.txt
#
# The scripts in src/paper3/ and src/paper4/ are standalone — no package install required.

from setuptools import setup

setup(
    name="fast-sampling-mode-collapse-3d",
    version="1.0.0",
    description="Code for: Mode Collapse in Fast-Sampling 3D Medical Synthesis (Y. Abdallah Ahmed, 2026)",
    python_requires=">=3.10",
)
