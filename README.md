# TAMOLS-MPPI

Trajectory Adaptive Multi-Objective Locomotion Solver using Model Predictive Path Integral (MPPI) control.

This repository implements two MPPI approaches for quadruped locomotion:

1. **State-Space MPPI** (JAX): Samples spline coefficients and foot positions directly
2. **Control-Space MPPI** (PyTorch): Samples control inputs and propagates through dynamics (theoretically correct)

## Features

- **Dual MPPI Implementations**: Choose between state-space or control-space formulations
- **GPU Acceleration**: PyTorch-based control-space MPPI for efficient parallel computation
- **Trajectory Generation**: Outputs base and foot trajectories satisfying dynamics and constraints
- **Flexible Integration**: Seamless API for comparing both approaches
- **Visualization**: Built-in plotting for trajectories and optimization results

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### State-Space MPPI (Original)

```python
from tamols import TAMOLS
from tamols.tamols_dataclasses import Gait, Terrain, Robot

problem = TAMOLS(gait, terrain, robot)
sols, infos = problem.run_repeated_optimizations()
```

### Control-Space MPPI (New)

```python
problem = TAMOLS(
    gait, terrain, robot,
    use_control_space_mppi=True,  # Enable control-space
    control_horizon=20
)
sols, infos = problem.run_repeated_optimizations()
```

## Documentation

- [MPPI Implementation](MPPI_IMPLEMENTATION.md) - Original state-space MPPI
- [Control-Space MPPI](CONTROL_SPACE_MPPI.md) - New control-space implementation
- [Example Usage](tamols_control_space_example.py) - Comparison demo

## Examples

Run the comparison example:

```bash
python tamols_control_space_example.py
```

This demonstrates both state-space and control-space MPPI approaches.
