"""
Hybrid Control-Space MPPI with Quintic Spline Planning.

This module implements a hybrid approach that combines:
1. Control-space MPPI sampling (theoretically correct)
2. Quintic spline trajectory representation (smooth and structured)

Instead of sampling raw accelerations, we sample spline coefficients
and evaluate constraints along the generated smooth trajectories.
"""

import torch
import numpy as np
from typing import Tuple, Callable, Optional, Dict


class SplineControlMPPI:
    """
    Hybrid MPPI optimizer that samples spline coefficients as control inputs
    and generates smooth quintic trajectories for constraint evaluation.
    
    This combines the benefits of:
    - Control-space sampling (theoretically correct MPPI)
    - Spline-based planning (smooth trajectories, boundary conditions)
    """
    
    def __init__(
        self,
        gait,
        terrain,
        robot,
        num_samples: int = 1000,
        num_iterations: int = 50,
        temperature: float = 1.0,
        noise_sigma: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize spline-based control-space MPPI optimizer.
        
        Args:
            gait: Gait object with locomotion parameters
            terrain: Terrain object with heightmap
            robot: Robot object with physical parameters
            num_samples: Number of trajectory samples per iteration
            num_iterations: Maximum MPPI iterations
            temperature: Temperature parameter for importance weighting
            noise_sigma: Standard deviation for spline coefficient noise
            device: Computation device ('cpu' or 'cuda')
        """
        self.gait = gait
        self.terrain = terrain
        self.robot = robot
        self.device = torch.device(device)
        
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        
        # Spline parameters
        self.n_phases = int(gait.n_phases)
        self.spline_order = int(gait.spline_order)
        self.n_coeffs = self.spline_order + 1  # e.g., 6 for quintic
        
        # Control dimension: spline coefficients for pos + rot + foot positions
        # a_pos: (n_phases, 3, n_coeffs) - position splines
        # a_rot: (n_phases, 3, n_coeffs) - rotation splines
        # feet: (4, 3) - foot positions
        self.control_dim = self.n_phases * 3 * self.n_coeffs * 2 + 4 * 3
        
        # Initialize control mean (spline coefficients + foot positions)
        self.control_mean = torch.zeros(self.control_dim, device=self.device, dtype=torch.float32)
        
        # Physics parameters
        self.gravity = torch.tensor(np.asarray(terrain.gravity), device=self.device, dtype=torch.float32)
        self.mu = float(terrain.mu)
        self.mass = float(robot.mass)
        self.inertia = torch.tensor(np.asarray(robot.inertia), device=self.device, dtype=torch.float32)
        self.l_min = float(robot.l_min)
        self.l_max = float(robot.l_max)
        
        # Limb centers in base frame
        self.limb_centers = torch.tensor(np.stack([
            np.asarray(robot.r_1),
            np.asarray(robot.r_2),
            np.asarray(robot.r_3),
            np.asarray(robot.r_4)
        ]), device=self.device, dtype=torch.float32)
        
        # Phase durations
        self.tau_k = torch.tensor(np.asarray(gait.tau_k), device=self.device, dtype=torch.float32)
        
        # Timesteps for evaluation (6 per phase)
        N = 6
        t_idx = torch.arange(1, N+1, dtype=torch.float32, device=self.device)
        T_k = self.tau_k / 6.0
        self.timesteps = T_k[:, None] * t_idx[None, :]  # (n_phases, 6)
        
        # Constraint weights
        self.w_friction = 1000.0
        self.w_leg_length = 1000.0
        self.w_vertical_accel = 1000.0
        self.w_dynamics = 100.0
        self.w_velocity = 1.0
        self.w_height = 10.0
        self.w_orientation = 0.1
        
    def unpack_control(self, control: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Unpack flat control vector into spline coefficients and foot positions.
        
        Args:
            control: Flat control vector (control_dim,) or (batch, control_dim)
        
        Returns:
            Dictionary with:
            - a_pos: (n_phases, 3, n_coeffs) position spline coefficients
            - a_rot: (n_phases, 3, n_coeffs) rotation spline coefficients
            - feet: (4, 3) foot positions
        """
        # Handle batched or single control
        if control.dim() == 1:
            control = control.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        batch_size = control.shape[0]
        idx = 0
        
        # Extract position splines: (batch, n_phases, 3, n_coeffs)
        pos_size = self.n_phases * 3 * self.n_coeffs
        a_pos = control[:, idx:idx+pos_size].view(batch_size, self.n_phases, 3, self.n_coeffs)
        idx += pos_size
        
        # Extract rotation splines: (batch, n_phases, 3, n_coeffs)
        rot_size = self.n_phases * 3 * self.n_coeffs
        a_rot = control[:, idx:idx+rot_size].view(batch_size, self.n_phases, 3, self.n_coeffs)
        idx += rot_size
        
        # Extract foot positions: (batch, 4, 3)
        feet = control[:, idx:idx+12].view(batch_size, 4, 3)
        
        result = {
            'a_pos': a_pos.squeeze(0) if squeeze else a_pos,
            'a_rot': a_rot.squeeze(0) if squeeze else a_rot,
            'feet': feet.squeeze(0) if squeeze else feet,
        }
        
        return result
    
    def evaluate_spline_position(self, coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spline position at time t.
        
        Args:
            coeffs: Spline coefficients (3, n_coeffs) or (batch, 3, n_coeffs)
            t: Time value (scalar or tensor)
        
        Returns:
            Position (3,) or (batch, 3)
        """
        # t_powers: [1, t, t^2, t^3, t^4, t^5, ...]
        t_powers = torch.stack([t**i for i in range(self.n_coeffs)], dim=-1)
        
        # coeffs: (..., 3, n_coeffs), t_powers: (n_coeffs,)
        # Result: (..., 3)
        return torch.sum(coeffs * t_powers, dim=-1)
    
    def evaluate_spline_velocity(self, coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate spline velocity at time t."""
        t_powers = torch.stack([i * t**(i-1) if i >= 1 else torch.zeros_like(t) 
                                for i in range(self.n_coeffs)], dim=-1)
        return torch.sum(coeffs * t_powers, dim=-1)
    
    def evaluate_spline_acceleration(self, coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate spline acceleration at time t."""
        t_powers = torch.stack([i * (i-1) * t**(i-2) if i >= 2 else torch.zeros_like(t)
                                for i in range(self.n_coeffs)], dim=-1)
        return torch.sum(coeffs * t_powers, dim=-1)
    
    def euler_xyz_to_matrix(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Convert Euler angles to rotation matrix.
        
        Args:
            phi: Euler angles (roll, pitch, yaw) - (..., 3)
        
        Returns:
            Rotation matrix (..., 3, 3)
        """
        roll, pitch, yaw = phi[..., 0], phi[..., 1], phi[..., 2]
        
        cx, cy, cz = torch.cos(roll), torch.cos(pitch), torch.cos(yaw)
        sx, sy, sz = torch.sin(roll), torch.sin(pitch), torch.sin(yaw)
        
        # Build rotation matrix
        R = torch.stack([
            torch.stack([cy*cz, -cy*sz, sy], dim=-1),
            torch.stack([sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy], dim=-1),
            torch.stack([-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy], dim=-1)
        ], dim=-2)
        
        return R
    
    def compute_cost(self, control: torch.Tensor, current_state: Dict) -> torch.Tensor:
        """
        Compute cost for a control (spline coefficients + feet) including all constraints.
        
        Args:
            control: Control vector (batch, control_dim) or (control_dim,)
            current_state: Dictionary with current base pose/velocity and foot positions
        
        Returns:
            Cost (batch,) or scalar
        """
        # Unpack control
        sv = self.unpack_control(control)
        a_pos = sv['a_pos']  # (batch, n_phases, 3, n_coeffs) or (n_phases, 3, n_coeffs)
        a_rot = sv['a_rot']
        feet = sv['feet']  # (batch, 4, 3) or (4, 3)
        
        is_batched = a_pos.dim() == 4
        if not is_batched:
            a_pos = a_pos.unsqueeze(0)
            a_rot = a_rot.unsqueeze(0)
            feet = feet.unsqueeze(0)
        
        batch_size = a_pos.shape[0]
        total_cost = torch.zeros(batch_size, device=self.device)
        
        # Evaluate constraints at multiple timesteps per phase
        for phase in range(self.n_phases):
            for t_idx in range(self.timesteps.shape[1]):
                t = self.timesteps[phase, t_idx]
                
                # Evaluate splines at time t
                p_B = self.evaluate_spline_position(a_pos[:, phase], t)  # (batch, 3)
                phi_B = self.evaluate_spline_position(a_rot[:, phase], t)  # (batch, 3)
                p_B_dd = self.evaluate_spline_acceleration(a_pos[:, phase], t)  # (batch, 3)
                v_B = self.evaluate_spline_velocity(a_pos[:, phase], t)  # (batch, 3)
                
                # 1. Friction cone constraint
                a_B = self.gravity - p_B_dd  # Base acceleration in world
                friction_cone = self.mu * (-a_B[:, 2]) - torch.norm(a_B[:, :2], dim=-1)
                friction_violation = torch.relu(-friction_cone)
                total_cost += self.w_friction * friction_violation ** 2
                
                # 2. Vertical acceleration constraint
                az_g = p_B_dd[:, 2] - self.gravity[2]
                az_violation = torch.relu(-az_g)
                total_cost += self.w_vertical_accel * az_violation ** 2
                
                # 3. Leg length constraints
                R_B = self.euler_xyz_to_matrix(phi_B)  # (batch, 3, 3)
                
                for leg_idx in range(4):
                    # Hip position in world frame
                    lc = self.limb_centers[leg_idx]  # (3,)
                    hip_world = p_B + torch.matmul(R_B, lc)  # (batch, 3)
                    
                    # Leg vector from foot to hip
                    leg_vec = hip_world - feet[:, leg_idx]  # (batch, 3)
                    leg_length_sq = torch.sum(leg_vec ** 2, dim=-1)
                    
                    # Bounds
                    min_violation = torch.relu(self.l_min**2 - leg_length_sq)
                    max_violation = torch.relu(leg_length_sq - self.l_max**2)
                    total_cost += self.w_leg_length * (min_violation + max_violation)
                
                # 4. Velocity tracking (objective term)
                target_vel = torch.tensor(np.asarray(self.gait.desired_base_velocity), 
                                         device=self.device, dtype=torch.float32)
                vel_error = torch.sum((v_B - target_vel) ** 2, dim=-1)
                total_cost += self.w_velocity * vel_error
                
                # 5. Height tracking
                desired_height = float(self.gait.h_des)
                height_error = (p_B[:, 2] - desired_height) ** 2
                total_cost += self.w_height * height_error
                
                # 6. Orientation regularization
                orient_error = torch.sum(phi_B ** 2, dim=-1)
                total_cost += self.w_orientation * orient_error
        
        return total_cost.squeeze() if not is_batched else total_cost
    
    def optimize(self, initial_state: Dict, convergence_tol: float = 1e-6) -> Tuple[np.ndarray, dict]:
        """
        Run MPPI optimization to find optimal spline coefficients.
        
        Args:
            initial_state: Dictionary with current state
            convergence_tol: Convergence tolerance
        
        Returns:
            optimal_control: Optimal spline coefficients and foot positions
            info: Dictionary with optimization info
        """
        prev_cost = float('inf')
        converged = False
        
        for iteration in range(self.num_iterations):
            # Sample perturbations
            noise = torch.randn(self.num_samples, self.control_dim, 
                               device=self.device, dtype=torch.float32) * self.noise_sigma
            
            # Generate control samples
            control_samples = self.control_mean.unsqueeze(0) + noise  # (num_samples, control_dim)
            
            # Evaluate costs
            costs = self.compute_cost(control_samples, initial_state)  # (num_samples,)
            
            # Compute importance weights
            min_cost = torch.min(costs)
            exp_weights = torch.exp(-(costs - min_cost) / self.temperature)
            weights = exp_weights / torch.sum(exp_weights)
            
            # Update control mean
            self.control_mean = torch.sum(weights.unsqueeze(-1) * control_samples, dim=0)
            
            # Check convergence
            avg_cost = torch.mean(costs).item()
            cost_improvement = abs(prev_cost - avg_cost)
            
            if cost_improvement < convergence_tol and iteration > 0:
                converged = True
                break
            
            prev_cost = avg_cost
        
        info = {
            'status': 0 if converged else 1,
            'final_cost': prev_cost,
            'iterations': iteration + 1,
            'message': 'Converged' if converged else 'Max iterations reached'
        }
        
        # Convert to numpy
        optimal_control = self.control_mean.cpu().detach().numpy()
        
        return optimal_control, info
    
    def reset_control_sequence(self):
        """Reset the control sequence to zero."""
        self.control_mean.zero_()
