#!/usr/bin/env bash
# Source this before any dimOS blueprint run on this WSL2 box:
#   source scripts/dimos_env.sh && dimos --simulation --rerun-open none run unitree-go2
# Without the adapter override WSLg's Mesa d3d12 driver fails EGL context creation
# inside the MuJoCo child; the bundled dimos-viewer replaces the incompatible
# Rerun viewer on PATH.
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export MUJOCO_GL=egl
