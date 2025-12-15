# Re-export selected symbols
from .problem_class import TAMOLS, build_initial_x0, update_state_from_solution
from .map_processing import (
    compute_heightmap_gradients,
    get_hs1,
    get_hs2,
    save_heightmap_to_png,
    show_map
)
from .helpers import (
    euler_xyz_to_matrix,
    bilinear_interp,
    evaluate_spline_position,
    evaluate_spline_velocity,
    evaluate_spline_acceleration,
    robust_norm,
)

from .tamols_dataclasses import *

from .manual_heightmaps import (
    get_flat_heightmap,
    get_rough_terrain_heightmap,
    get_stairs_heightmap,
    get_heightmap_ramp
)

from .plot_sol import plot_all_iterations, plot_base
from .trajectory import get_trajectory_function
from .mppi_control_space import (
    ControlSpaceMPPI, 
    QuadrupedControlSpaceMPPI,
    extract_trajectory_from_control_rollout
)

__all__ = [
    'TAMOLS', 'build_initial_x0', 'update_state_from_solution',
    'compute_heightmap_gradients', 'get_hs1', 'get_hs2', 'save_heightmap_to_png',
    'euler_xyz_to_matrix', 'bilinear_interp', 'evaluate_spline_position',
    'evaluate_spline_velocity', 'evaluate_spline_acceleration', 'robust_norm',
    'get_flat_heightmap', 'get_rough_terrain_heightmap', 'get_stairs_heightmap',
    'Gait', 'Terrain', 'Robot', 'CurrentState',
    'plot_all_iterations', 'get_trajectory_function',
    'show_map', 'save_heightmap_to_png', 'get_heightmap_ramp', 'plot_base',
    'ControlSpaceMPPI', 'QuadrupedControlSpaceMPPI', 'extract_trajectory_from_control_rollout'
]
