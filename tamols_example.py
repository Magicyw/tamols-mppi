import numpy as np
from tamols.tamols_dataclasses import Gait, Terrain, Robot
from tamols import TAMOLS, plot_all_iterations, get_trajectory_function, plot_base
from tamols.helpers import transform_inertia
from tamols.manual_heightmaps import get_flat_heightmap, get_rough_terrain_heightmap, get_stairs_heightmap, get_heightmap_ramp
from tamols.map_processing import save_heightmap_to_png, show_map

gait = Gait(
    n_steps=5,
    n_phases=2,
    spline_order=5,
    tau_k=np.array([1.0, 1.0]), # Time duration of phases [s]
    h_des=0.445,
    eps_min=0.1,
    weights = np.array([10.0e4, 0.001, 7.0, 100.0, 3.0, 0.01, 2.0, 0.001]),
    desired_base_velocity = np.array([0.3, 0.0, 0.0]),
    desired_base_angular_velocity = np.array([0.0, 0.0, 0.0]),
    apex_height=0.1,
    contact_schedule= np.array([
        [0, 1, 1, 0],  # Phase 0: FR, RL in contact; FL, RR swing
        [1, 0, 0, 1],  # Phase 1: FL, RR in contact; FR, RL swing
    ]),
    at_des_position= np.array([
        [1, 0, 0, 1],  # Phase 0: move FL and RR
        [1, 1, 1, 1],  # Phase 1: all at desired by the end
    ])
)

# h = np.array(get_flat_heightmap(a=150, b=150, height=0.0))  # Flat heightmap for testing
# h = np.array(get_rough_terrain_heightmap(a=400, b=400, sigma=0.03, platform_height=0.0, platform_size=5, smooth_sigma=3, seed=917)) # Heightmap with platforms
# h = np.array(get_stairs_heightmap(a=100, b=100, start_col=60, step_depth=2, step_height=0.01))
h = np.array(get_stairs_heightmap(a=100, b=200, start_col=110, step_depth=10, step_height=0.1))  # Heightmap with stairs



terrain = Terrain(
    heightmap = h,
    grid_cell_length = 0.04,  # Length of grid cells in heightmap [m/cell]
    mu = 0.6,
    gravity = np.array([0.0, 0.0, -9.81]),
)

I_A = np.diag(np.array([0.107027, 0.0980771, 0.0244531]))  # Inertia matrix of the robot base (diagonal)
r = np.array([0.021112, 0, -0.005366])
q = np.array([-0.000543471, 0.713435, -0.00173769, 0.700719])
mass = 6.921

robot = Robot(
    mass=mass,
    inertia=transform_inertia(I_A, mass, r, q),
    l_min=0.1,
    l_max=0.5,
    r_1=np.array([0.1934,0.0465+0.0955,0.0]), # Hip joint offset positions in base frame
    r_2=np.array([0.1934,-0.0465-0.0955,0.0]),
    r_3=np.array([-0.1934,0.0465+0.0955,0.0]),
    r_4=np.array([-0.1934,-0.0465-0.0955,0.0]),
    p_1_start=np.array([0.1934,0.0465+0.0955,0.0]),
    p_2_start=np.array([0.1934,-0.0465-0.0955,0.0]),
    p_3_start=np.array([-0.1934,0.0465+0.0955,0.0]),
    p_4_start=np.array([-0.1934,-0.0465-0.0955,0.0]),
    initial_base_pose=np.array([0.0, 0.0, 0.445, 0.0, 0.0, 0.0]),
    initial_base_velocity=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)

problem = TAMOLS(gait, terrain, robot)

sols, infos = problem.run_repeated_optimizations()

plot_base(sols, problem, dt=0.02)

plot_all_iterations(sols, problem, dt=0.02)

save_heightmap_to_png(terrain.heightmap, "outputs/heightmap.png")

trajectory_fn = get_trajectory_function(sols, problem)

time, feet_positions, base_positions, base_rotations = trajectory_fn(0.02)  # positions shape: (N, 4, 3)

