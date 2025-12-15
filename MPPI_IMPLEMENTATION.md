# MPPI Implementation Summary

This document describes the MPPI (Model Predictive Path Integral) implementation that replaces the cyipopt SQP solver in TAMOLS.

## Overview

The implementation replaces the gradient-based Sequential Quadratic Programming (SQP) solver (cyipopt) with a sampling-based Model Predictive Path Integral (MPPI) solver. This approach operates directly in the parameter space of splines and footholds.

## Key Changes

### 1. Removed Dependencies
- Removed `cyipopt==1.6.1` from `requirements.txt`
- Removed all cyipopt-related callback methods from the `TAMOLS` class

### 2. MPPI Solver Implementation

#### Constraint Handling
The MPPI solver uses soft penalties to handle constraints:

- **Equality constraints** (h(x) = 0): Penalized as `w_eq * ||h(x)||²`
- **Inequality constraints** (g(x) ≥ 0): Penalized as `w_ineq * ||ReLU(-g(x))||²`
- **Variable bounds** (lb ≤ x ≤ ub): Enforced by clamping samples

#### Core Algorithm
The MPPI algorithm works as follows:

1. Sample perturbations from a Gaussian distribution
2. Apply perturbations to the current mean and clamp to bounds
3. Evaluate the total cost (objective + constraint penalties) for all samples
4. Compute importance weights using softmax with temperature
5. Update mean as weighted average of samples
6. Repeat until convergence or maximum iterations

### 3. Configurable Parameters

The `TAMOLS` constructor now accepts the following MPPI-related parameters:

```python
TAMOLS(gait, terrain, robot,
       num_samples=1000,        # Number of samples per MPPI iteration
       num_iterations=100,      # Maximum number of MPPI iterations
       temperature=1.0,         # Temperature for importance weighting
       noise_sigma=0.1,         # Standard deviation of sampling noise
       w_eq=1000.0,            # Penalty weight for equality constraints
       w_ineq=1000.0,          # Penalty weight for inequality constraints
       convergence_tol=1e-6)    # Cost improvement threshold for convergence
```

### 4. Convergence Tracking

The solver tracks convergence based on cost improvement between iterations. If the cost improvement falls below `convergence_tol`, the optimization is considered converged.

The returned `info` dictionary includes:
- `status`: 0 if converged, 1 if max iterations reached
- `obj_val`: Original objective value (without constraint penalties)
- `message`: Description of the termination condition

## Usage Example

```python
import numpy as np
from tamols import TAMOLS
from tamols.tamols_dataclasses import Gait, Terrain, Robot
from tamols.manual_heightmaps import get_stairs_heightmap
from tamols.helpers import transform_inertia

# Define gait, terrain, and robot (as before)
# ...

# Create problem with custom MPPI parameters
problem = TAMOLS(gait, terrain, robot,
                num_samples=500,      # Reduce for faster computation
                num_iterations=50,     # Fewer iterations
                temperature=2.0,       # Higher temperature for more exploration
                noise_sigma=0.2)       # Larger noise for broader sampling

# Run optimization
sols, infos = problem.run_repeated_optimizations(warm_start=True)
```

## Performance Considerations

### Tuning Guidelines

1. **num_samples**: More samples improve solution quality but increase computation time. Start with 200-500 for quick tests, use 1000+ for final results.

2. **num_iterations**: Depends on problem difficulty. Monitor convergence - if status=1 (max iterations) consistently, increase this value.

3. **temperature**: Controls exploration vs exploitation
   - Lower values (0.1-0.5): More exploitation, faster convergence
   - Higher values (1.0-2.0): More exploration, better global search

4. **noise_sigma**: Standard deviation of sampling noise
   - Smaller values (0.01-0.05): Fine-tuning near local optimum
   - Larger values (0.1-0.5): Broader exploration

5. **Constraint penalties** (w_eq, w_ineq): Balance constraint satisfaction vs objective
   - Higher values enforce constraints more strictly
   - Lower values allow more constraint violation for better objective
   - Default 1000.0 works well for most cases

## Technical Details

### NumPy 64-bit Precision
The implementation uses NumPy's 64-bit precision for numerical stability. This is important for maintaining accuracy in the optimization process.

### Random Seed
By default, the solver uses a time-based random seed for each optimization. You can specify a fixed seed for reproducibility:

```python
x_sol, info = problem.run_single_optimization(seed=42)
```

### Performance
The MPPI implementation uses NumPy for computation. For better performance, consider using the control-space MPPI with PyTorch GPU acceleration.

## Comparison with cyipopt

| Aspect | cyipopt (SQP) | MPPI |
|--------|---------------|------|
| Approach | Gradient-based | Sampling-based |
| Derivatives | Required | Not required |
| Constraints | Hard constraints | Soft penalties |
| Parallelization | Limited | Easily parallelizable |
| Convergence | Fast near optimum | Slower but more robust |
| Global search | Local optimization | Better global exploration |

## Troubleshooting

### High constraint violations
- Increase `w_eq` and `w_ineq` penalty weights
- Increase `num_iterations` to allow more refinement
- Check that constraint functions are correctly defined

### Slow convergence
- Reduce `temperature` for more focused sampling
- Reduce `noise_sigma` if already near optimum
- Increase `num_samples` for better gradient estimation

### Poor solution quality
- Increase `num_samples` and `num_iterations`
- Adjust `temperature` for better exploration
- Check initialization quality (warm-start helps)

## References

Williams, G., Drews, P., Goldfain, B., Rehg, J. M., & Theodorou, E. A. (2017). 
"Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving." 
IEEE Transactions on Robotics, 34(6), 1603-1622.
