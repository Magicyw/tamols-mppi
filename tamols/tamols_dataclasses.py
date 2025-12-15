from dataclasses import dataclass
import numpy as np

@dataclass
class Gait:
    n_steps: int       # Number of steps to optimize
    n_phases: int      # Number of phases in each step
    spline_order: int  # Spline order for trajectory generation
    tau_k: np.ndarray  # Time duration of phases [s]
    h_des: float    # Desired height above ground
    eps_min: float  # Minimum distance between feet
    weights: np.ndarray  # Weights for the cost function
    desired_base_velocity: np.ndarray  # Desired base velocity in world frame
    desired_base_angular_velocity: np.ndarray  # Desired base angular velocity in world frame
    apex_height: float # Apex height for swing leg trajectory
    contact_schedule: np.ndarray = None  # Contact schedule (n_phases x 4), 1 if in contact, 0 if swing
    at_des_position: np.ndarray = None  # (n_phases x 4) 1 if foot should be at desired position at the end of the phase, 0 otherwise

@dataclass
class Terrain:
    heightmap: np.ndarray # Original heightmap
    grid_cell_length: float # Length of grid cells in heightmap
    mu: float # Friction coefficient
    gravity: np.ndarray # Gravity vector

@dataclass
class Robot:
    mass: float
    inertia: np.ndarray
    l_min: float # Minimum leg length
    l_max: float # Maximum leg length
    r_1: np.ndarray # Position of limb 1 center in base frame
    r_2: np.ndarray # Position of limb 2 center in base frame
    r_3: np.ndarray # Position of limb 3 center in base frame
    r_4: np.ndarray # Position of limb 4 center in base frame
    p_1_start: np.ndarray # Start foot position of limb 1
    p_2_start: np.ndarray # Start foot position of limb 2
    p_3_start: np.ndarray # Start foot position of limb 3
    p_4_start: np.ndarray # Start foot position of limb 4
    initial_base_pose: np.ndarray  # Initial base pose (x, y, z, roll, pitch, yaw)
    initial_base_velocity: np.ndarray  # Initial base velocity (vx, vy, vz, wx, wy, wz)

@dataclass
class CurrentState: 
    p_1_meas: np.ndarray  # Measured foot position of limb 1
    p_2_meas: np.ndarray  # Measured foot position of limb 2
    p_3_meas: np.ndarray  # Measured foot position of limb 3
    p_4_meas: np.ndarray  # Measured foot position of limb 4
    current_base_pose: np.ndarray  # Current base pose (x, y, z, roll, pitch, yaw)
    current_base_velocity: np.ndarray  # Current base velocity (vx, vy, vz, wx, wy, wz)
    # Use virtual floor (h_s2) for foothold costs when True; real heightmap when False
    use_virtual_floor: np.ndarray