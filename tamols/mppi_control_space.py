"""
Control-Space Model Predictive Path Integral (MPPI) implementation.

This module implements true MPPI control by:
1. Sampling control sequences (accelerations/velocities)
2. Rolling out trajectories through forward dynamics
3. Evaluating costs along rollouts
4. Computing optimal control via information-theoretic weighting

Uses PyTorch for GPU acceleration.
"""

import torch
import numpy as np
from typing import Tuple, Callable, Optional


class ControlSpaceMPPI:
    """
    Control-space MPPI optimizer that samples control inputs and propagates
    through dynamics, rather than directly sampling in state space.
    
    This is the theoretically correct MPPI formulation for control problems.
    """
    
    def __init__(
        self,
        control_dim: int,
        state_dim: int,
        horizon: int,
        num_samples: int,
        num_iterations: int = 50,
        temperature: float = 1.0,
        noise_sigma: float = 0.1,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        dt: float = 0.1,
    ):
        """
        Initialize control-space MPPI optimizer.
        
        Args:
            control_dim: Dimension of control input (e.g., 6 for base accel/angular accel)
            state_dim: Dimension of state (e.g., 12 for base pose + velocity)
            horizon: Planning horizon length (number of timesteps)
            num_samples: Number of trajectory samples per iteration
            num_iterations: Maximum MPPI iterations
            temperature: Temperature parameter for importance weighting
            noise_sigma: Standard deviation for control noise sampling
            control_bounds: Optional tuple of (lower, upper) control bounds
            device: Device to run computations on ('cpu' or 'cuda')
            dt: Timestep for integration
        """
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.dt = dt
        self.device = torch.device(device)
        
        # Control bounds
        if control_bounds is not None:
            self.control_lb = torch.tensor(control_bounds[0], device=self.device, dtype=torch.float32)
            self.control_ub = torch.tensor(control_bounds[1], device=self.device, dtype=torch.float32)
        else:
            self.control_lb = None
            self.control_ub = None
        
        # Initialize control sequence (mean of the distribution)
        self.control_mean = torch.zeros(
            (horizon, control_dim), device=self.device, dtype=torch.float32
        )
        
    def rollout_dynamics(
        self,
        initial_state: torch.Tensor,
        controls: torch.Tensor,
        dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Roll out trajectories given control sequences using forward dynamics.
        
        Args:
            initial_state: Initial state (state_dim,)
            controls: Control sequences (num_samples, horizon, control_dim)
            dynamics_fn: Function that computes next state given (state, control)
                        Should handle batched inputs
        
        Returns:
            states: Trajectory states (num_samples, horizon+1, state_dim)
        """
        num_samples = controls.shape[0]
        states = torch.zeros(
            (num_samples, self.horizon + 1, self.state_dim),
            device=self.device,
            dtype=torch.float32
        )
        states[:, 0] = initial_state.unsqueeze(0).expand(num_samples, -1)
        
        # Forward integrate dynamics
        for t in range(self.horizon):
            states[:, t + 1] = dynamics_fn(states[:, t], controls[:, t])
        
        return states
    
    def evaluate_trajectory_costs(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Evaluate costs along trajectories.
        
        Args:
            states: Trajectory states (num_samples, horizon+1, state_dim)
            controls: Control sequences (num_samples, horizon, control_dim)
            cost_fn: Function that computes cost given (state, control)
                    Should handle batched inputs and return (num_samples,) tensor
        
        Returns:
            costs: Total costs for each sample (num_samples,)
        """
        costs = torch.zeros(self.num_samples, device=self.device, dtype=torch.float32)
        
        # Accumulate costs over horizon
        for t in range(self.horizon):
            step_costs = cost_fn(states[:, t], controls[:, t])
            costs += step_costs
        
        # Terminal cost
        terminal_costs = cost_fn(states[:, -1], torch.zeros_like(controls[:, -1]))
        costs += terminal_costs
        
        return costs
    
    def compute_control_update(
        self,
        costs: torch.Tensor,
        control_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the weighted control update using MPPI information-theoretic weights.
        
        Args:
            costs: Costs for each sample (num_samples,)
            control_samples: Sampled controls (num_samples, horizon, control_dim)
        
        Returns:
            updated_control: New mean control sequence (horizon, control_dim)
        """
        # Compute importance weights with temperature
        min_cost = torch.min(costs)
        exp_weights = torch.exp(-(costs - min_cost) / self.temperature)
        weights = exp_weights / torch.sum(exp_weights)
        
        # Weighted average of control samples
        weights_expanded = weights.view(-1, 1, 1)  # (num_samples, 1, 1)
        updated_control = torch.sum(weights_expanded * control_samples, dim=0)
        
        return updated_control
    
    def optimize(
        self,
        initial_state: torch.Tensor,
        dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        convergence_tol: float = 1e-6,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Run MPPI optimization to find optimal control sequence.
        
        Args:
            initial_state: Initial state (state_dim,)
            dynamics_fn: Dynamics function for rollouts
            cost_fn: Cost function for trajectory evaluation
            convergence_tol: Convergence tolerance on cost improvement
        
        Returns:
            optimal_controls: Optimal control sequence (horizon, control_dim)
            info: Dictionary with optimization info
        """
        prev_cost = float('inf')
        converged = False
        
        for iteration in range(self.num_iterations):
            # Sample control perturbations
            noise = torch.randn(
                (self.num_samples, self.horizon, self.control_dim),
                device=self.device,
                dtype=torch.float32
            ) * self.noise_sigma
            
            # Generate control samples around current mean
            control_samples = self.control_mean.unsqueeze(0) + noise  # (num_samples, horizon, control_dim)
            
            # Clamp to bounds if specified
            if self.control_lb is not None:
                control_samples = torch.clamp(control_samples, self.control_lb, self.control_ub)
            
            # Rollout dynamics with sampled controls
            states = self.rollout_dynamics(initial_state, control_samples, dynamics_fn)
            
            # Evaluate costs
            costs = self.evaluate_trajectory_costs(states, control_samples, cost_fn)
            
            # Update control mean using MPPI weighting
            self.control_mean = self.compute_control_update(costs, control_samples)
            
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
        
        return self.control_mean, info
    
    def reset_control_sequence(self):
        """Reset the control sequence to zero (useful for new optimization problems)."""
        self.control_mean.zero_()


class QuadrupedControlSpaceMPPI:
    """
    Specialized control-space MPPI for quadruped locomotion in TAMOLS framework.
    
    Integrates control-space MPPI with the TAMOLS problem structure, enabling
    sampling of base accelerations and propagating them through kinematic/dynamic models.
    """
    
    def __init__(
        self,
        gait,
        terrain,
        robot,
        horizon: int = 20,
        num_samples: int = 1000,
        num_iterations: int = 50,
        temperature: float = 1.0,
        noise_sigma: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize quadruped control-space MPPI.
        
        Args:
            gait: Gait object with locomotion parameters
            terrain: Terrain object with heightmap
            robot: Robot object with physical parameters
            horizon: Control horizon
            num_samples: Number of samples per iteration
            num_iterations: Max MPPI iterations
            temperature: MPPI temperature parameter
            noise_sigma: Control noise standard deviation
            device: Computation device
        """
        self.gait = gait
        self.terrain = terrain
        self.robot = robot
        self.device = torch.device(device)
        
        # Control dimension: base linear + angular acceleration (6D)
        control_dim = 6
        
        # State dimension: base pose (6D) + base velocity (6D) + foot positions (12D) = 24D
        state_dim = 24
        
        # Timestep from gait
        self.dt = float(np.mean(gait.tau_k)) / 10.0  # Subdivide phase for finer control
        
        # Control bounds (reasonable acceleration limits)
        control_lb = np.array([-5.0, -5.0, -15.0, -10.0, -10.0, -10.0])  # [m/s^2, rad/s^2]
        control_ub = np.array([5.0, 5.0, 5.0, 10.0, 10.0, 10.0])
        
        # Initialize core MPPI optimizer
        self.mppi = ControlSpaceMPPI(
            control_dim=control_dim,
            state_dim=state_dim,
            horizon=horizon,
            num_samples=num_samples,
            num_iterations=num_iterations,
            temperature=temperature,
            noise_sigma=noise_sigma,
            control_bounds=(control_lb, control_ub),
            device=device,
            dt=self.dt,
        )
        
        # Convert terrain data to torch
        self.heightmap_torch = torch.tensor(
            np.asarray(terrain.heightmap), device=self.device, dtype=torch.float32
        )
        self.grid_cell_length = float(terrain.grid_cell_length)
    
    def state_to_torch(self, state) -> torch.Tensor:
        """Convert numpy state to PyTorch tensor."""
        if isinstance(state, np.ndarray):
            return torch.tensor(np.asarray(state), device=self.device, dtype=torch.float32)
        return state
    
    def torch_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        return tensor.cpu().detach().numpy()
    
    def create_dynamics_function(self) -> Callable:
        """
        Create dynamics function for forward integration.
        
        Returns function that computes next state given current state and control.
        Uses simple kinematic integration with gravity.
        """
        gravity = torch.tensor(
            np.asarray(self.terrain.gravity), device=self.device, dtype=torch.float32
        )
        dt = self.dt
        
        def dynamics_fn(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
            """
            Integrate state forward one timestep.
            
            State: [pos(3), rot(3), vel(3), ang_vel(3), feet(12)]
            Control: [acc(3), ang_acc(3)]
            """
            # Extract state components
            pos = state[..., 0:3]
            rot = state[..., 3:6]
            vel = state[..., 6:9]
            ang_vel = state[..., 9:12]
            feet = state[..., 12:24]
            
            # Extract control
            acc = control[..., 0:3]
            ang_acc = control[..., 3:6]
            
            # Add gravity to acceleration
            total_acc = acc + gravity
            
            # Integrate velocity and position (Euler integration)
            new_vel = vel + total_acc * dt
            new_pos = pos + vel * dt + 0.5 * total_acc * dt**2
            
            # Integrate angular velocity and orientation
            new_ang_vel = ang_vel + ang_acc * dt
            new_rot = rot + ang_vel * dt + 0.5 * ang_acc * dt**2
            
            # Feet positions remain constant (updated discretely in gait phases)
            new_feet = feet
            
            # Concatenate new state
            new_state = torch.cat([new_pos, new_rot, new_vel, new_ang_vel, new_feet], dim=-1)
            return new_state
        
        return dynamics_fn
    
    def create_cost_function(self, target_velocity: torch.Tensor) -> Callable:
        """
        Create cost function for trajectory evaluation.
        
        Args:
            target_velocity: Desired base velocity (6D: linear + angular)
        
        Returns:
            Cost function that evaluates state-control pair
        """
        target_vel = self.state_to_torch(target_velocity)
        heightmap = self.heightmap_torch
        grid_cell = self.grid_cell_length
        desired_height = float(self.gait.h_des)
        
        def cost_fn(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
            """
            Compute cost for state-control pair.
            
            Cost includes:
            - Velocity tracking error
            - Height tracking error
            - Control effort
            - Base orientation regularization
            """
            # Extract state components
            pos = state[..., 0:3]
            rot = state[..., 3:6]
            vel = state[..., 6:9]
            ang_vel = state[..., 9:12]
            
            # Velocity tracking cost
            current_vel = torch.cat([vel, ang_vel], dim=-1)
            vel_error = torch.sum((current_vel - target_vel) ** 2, dim=-1)
            
            # Height tracking cost (want to maintain desired height above terrain)
            # For simplicity, use fixed desired height (could sample terrain)
            z = pos[..., 2]
            height_error = (z - desired_height) ** 2
            
            # Orientation regularization (prefer upright orientation)
            orient_error = torch.sum(rot ** 2, dim=-1)
            
            # Control effort (penalize large accelerations)
            control_effort = torch.sum(control ** 2, dim=-1) * 0.01
            
            # Total cost
            cost = vel_error + 10.0 * height_error + 0.1 * orient_error + control_effort
            
            return cost
        
        return cost_fn
    
    def optimize_trajectory(
        self,
        initial_state,
        target_velocity,
    ) -> Tuple[np.ndarray, dict]:
        """
        Optimize control sequence for quadruped motion.
        
        Args:
            initial_state: Initial state (pose + velocity + feet)
            target_velocity: Desired base velocity
        
        Returns:
            optimal_controls: Optimal control sequence as numpy array
            info: Optimization information
        """
        # Convert initial state to torch
        state_torch = self.state_to_torch(initial_state)
        
        # Create dynamics and cost functions
        dynamics_fn = self.create_dynamics_function()
        cost_fn = self.create_cost_function(target_velocity)
        
        # Run MPPI optimization
        optimal_controls, info = self.mppi.optimize(
            state_torch, dynamics_fn, cost_fn
        )
        
        # Convert back to numpy
        controls_np = self.torch_to_numpy(optimal_controls)
        
        return controls_np, info
    
    def integrate_controls_to_trajectory(
        self,
        initial_state,
        controls: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate control sequence to generate full trajectory.
        
        Args:
            initial_state: Initial state
            controls: Control sequence (horizon, control_dim)
        
        Returns:
            states: State trajectory (horizon+1, state_dim)
            controls: Control sequence (horizon, control_dim)
        """
        state_torch = self.state_to_torch(initial_state)
        controls_torch = self.state_to_torch(controls)
        
        # Rollout with optimal controls
        dynamics_fn = self.create_dynamics_function()
        controls_expanded = controls_torch.unsqueeze(0)  # Add batch dimension
        states = self.mppi.rollout_dynamics(state_torch, controls_expanded, dynamics_fn)
        
        # Remove batch dimension and convert to numpy
        states_np = self.torch_to_numpy(states[0])
        
        return states_np, controls


def extract_trajectory_from_control_rollout(
    states: np.ndarray,
    gait,
    robot,
) -> dict:
    """
    Extract base and foot trajectories from control-space rollout.
    
    Args:
        states: State trajectory (T, state_dim) where state = [pose, vel, feet]
        gait: Gait parameters
        robot: Robot parameters
    
    Returns:
        Dictionary with trajectory components:
        - base_positions: (T, 3)
        - base_orientations: (T, 3) Euler angles
        - base_velocities: (T, 6)
        - foot_positions: (T, 4, 3)
    """
    T = states.shape[0]
    
    # Extract components
    base_positions = states[:, 0:3]
    base_orientations = states[:, 3:6]
    base_velocities = states[:, 6:12]
    feet_flat = states[:, 12:24]
    foot_positions = feet_flat.reshape(T, 4, 3)
    
    return {
        'base_positions': base_positions,
        'base_orientations': base_orientations,
        'base_velocities': base_velocities,
        'foot_positions': foot_positions,
    }
