import numpy as np
from .manual_heightmaps import *
from .helpers import *
from .map_processing import *


def foothold_on_ground_cost(p, heightmap, grid_cell_length):
    """
        p: foothold position vector (shape: (3,))
        heightmap: heightmap array or callable

        return: cost of foothold position
    """
    height = bilinear_interp(heightmap, p, grid_cell_length)
    return np.square(height - p[2])

def leg_collision_avoidance_cost(p_i, p_j, eps_min):
    """
        p_i: position vector of foot i
        p_j: position vector of foot j
        eps_min: minimum distance between feet

        return: cost of leg collision avoidance
    """
    z = robust_norm((p_i - p_j)[:2])
    return np.where(z < eps_min, np.square(eps_min - z), 0.0)

def nominal_kinematics_cost(p_B, p, h_des, r_B, R_B):
    """
        p_B: base position vector
        p: foot position vector
        h_des: desired height above ground
        r_B: limb center position vector in base frame
        R_B: rotation matrix of base frame

        return: cost of nominal kinematics
    """
    target = p_B + R_B @ r_B - np.array([0.0, 0.0, h_des])
    return np.square(robust_norm(target - p))

def base_pose_alignment_cost(p_B, h_des, r_B, R_B, h_s2, grid_cell_length):
    """
        p_B: base position vector
        h_des: desired height above ground
        r_B: limb center position vector in base frame
        R_B: rotation matrix of base frame
        h_s2: virtual floor

        return: cost of base pose alignment
    """
    l_des = np.array([0.0, 0.0, h_des])    
    h = bilinear_interp(h_s2, p_B + R_B @ r_B, grid_cell_length)
    # Align base with virtual floor
    return np.square((p_B + R_B @ r_B - l_des)[2] - h)

def edge_avoidance_cost(grad_h_x, grad_h_y, grad_h_s1_x, grad_h_s1_y, p, grid_cell_length):
    """
        Compute edge avoidance cost based on heightmap gradients.
        grad_h_x: Gradient in x direction of heightmap
        grad_h_y: Gradient in y direction of heightmap
        grad_h_s1_x: Gradient in x direction of smoothed heightmap
        grad_h_s1_y: Gradient in y direction of smoothed heightmap
        p: position vector (length-3 array-like, [x, y, z])

        return: edge avoidance cost
    """
    grad_h = np.array([
        bilinear_interp(grad_h_x, p, grid_cell_length),
        bilinear_interp(grad_h_y, p, grid_cell_length)
    ])
    grad_h_s1 = np.array([
        bilinear_interp(grad_h_s1_x, p, grid_cell_length),
        bilinear_interp(grad_h_s1_y, p, grid_cell_length)
    ])
    # Use squared L2 norm for both gradients
    return np.dot(grad_h, grad_h) + np.dot(grad_h_s1, grad_h_s1)

def previous_solution_cost(p, p_prev, eps=1.0e-8):
    """
        Compute cost based on previous solution (scalar).
        Returns ||p - p_prev||^2 + eps
    """
    diff = p - p_prev
    return np.sum(np.square(diff)) + eps

def tracking_cost(p_dot_B, phi_B, phi_B_dot, p_dot_desired, omega_desired, mass, inertia):
    """
        Compute tracking cost for base velocity and angular velocity.
        p_dot_B: current base velocity in world frame
        phi_B: euler angles of the base
        phi_B_dot: euler angle rates of the base
        p_dot_desired: desired base velocity in world frame
        omega_desired: desired base angular velocity in world frame
        mass: mass of the robot
        inertia: inertia matrix of the robot in base frame
        R: rotation matrix of the base frame

        return: total tracking costs
    """
    R = euler_xyz_to_matrix(phi_B)

    omega_b, _ = euler_xyz_rates_to_body_omega_alpha(phi_B, phi_B_dot, np.zeros(3))
    omega_W = R @ omega_b  # Angular velocity of the base in world frame

    P_B = mass * p_dot_B # Linear momentum of the base in world frame
    P_desired = mass * p_dot_desired # Desired linear momentum in world frame
    L_B = R @ inertia @ R.T @ omega_W # Angular momentum of the base in world frame
    L_desired = R @ inertia @ R.T @ omega_desired # Desired angular momentum in world frame

    return np.square(robust_norm(P_B - P_desired))/(mass**2) + np.square(robust_norm(L_B - L_desired))

def smoothness_cost(phi, phi_dot, phi_dotdot, inertia):
    """
        Compute smoothness cost based on angular velocity.
        omega_B_dot: angular acceleration of the base
        inertia: inertia matrix of the robot
        R: rotation matrix of the base frame

        return: smoothness cost
    """
    R = euler_xyz_to_matrix(phi)
    omega_b, alpha_b = euler_xyz_rates_to_body_omega_alpha(phi, phi_dot, phi_dotdot)
    hdot_b = inertia @ alpha_b + np.cross(omega_b, inertia @ omega_b)
    Ldot_w = R @ hdot_b
    return np.square(robust_norm(Ldot_w))

def slack_variables_cost(eps):
    """
        Compute cost for slack variables.
        eps: array of slack variables

        return: slack variable cost (sum of squares)
    """
    return np.sum(np.square(eps))




