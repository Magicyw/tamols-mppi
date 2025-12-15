"""
Simple test to verify control-space MPPI implementation works.
"""

import torch
import numpy as np
import jax.numpy as jnp
from tamols.tamols_dataclasses import Gait, Terrain, Robot
from tamols.mppi_control_space import ControlSpaceMPPI, QuadrupedControlSpaceMPPI
from tamols.manual_heightmaps import get_flat_heightmap
from tamols.helpers import transform_inertia


def test_control_space_mppi_basic():
    """Test basic ControlSpaceMPPI functionality."""
    print("Testing basic ControlSpaceMPPI...")
    
    # Simple dynamics: integrate velocity
    def simple_dynamics(state, control):
        # state: [pos(3), vel(3)]
        # control: [acc(3)]
        pos = state[..., 0:3]
        vel = state[..., 3:6]
        acc = control[..., 0:3]
        
        dt = 0.1
        new_vel = vel + acc * dt
        new_pos = pos + vel * dt
        
        return torch.cat([new_pos, new_vel], dim=-1)
    
    # Simple cost: minimize distance to target
    target = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    def simple_cost(state, control):
        return torch.sum((state - target) ** 2, dim=-1)
    
    # Create optimizer
    mppi = ControlSpaceMPPI(
        control_dim=3,
        state_dim=6,
        horizon=10,
        num_samples=50,
        num_iterations=5,
        device='cpu'
    )
    
    # Initial state
    initial_state = torch.zeros(6)
    
    # Optimize
    controls, info = mppi.optimize(initial_state, simple_dynamics, simple_cost)
    
    print(f"  Status: {info['message']}")
    print(f"  Final cost: {info['final_cost']:.6f}")
    print(f"  Controls shape: {controls.shape}")
    
    assert controls.shape == (10, 3), "Control shape mismatch"
    assert info['status'] in [0, 1], "Invalid status"
    
    print("✓ Basic ControlSpaceMPPI test passed")


def test_quadruped_control_space_mppi():
    """Test QuadrupedControlSpaceMPPI functionality."""
    print("\nTesting QuadrupedControlSpaceMPPI...")
    
    # Define minimal gait
    gait = Gait(
        n_steps=1,
        n_phases=1,
        spline_order=3,
        tau_k=jnp.array([0.5]),
        h_des=0.4,
        eps_min=0.1,
        weights=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        desired_base_velocity=jnp.array([0.2, 0.0, 0.0]),
        desired_base_angular_velocity=jnp.array([0.0, 0.0, 0.0]),
        apex_height=0.05,
        contact_schedule=jnp.array([[1, 1, 1, 1]]),
        at_des_position=jnp.array([[1, 1, 1, 1]])
    )
    
    # Flat terrain
    h = jnp.array(get_flat_heightmap(a=50, b=50, height=0.0))
    terrain = Terrain(
        heightmap=h,
        grid_cell_length=0.04,
        mu=0.6,
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )
    
    # Simple robot
    I_A = jnp.diag(jnp.array([0.1, 0.1, 0.02]))
    robot = Robot(
        mass=5.0,
        inertia=I_A,
        l_min=0.1,
        l_max=0.4,
        r_1=jnp.array([0.2, 0.1, 0.0]),
        r_2=jnp.array([0.2, -0.1, 0.0]),
        r_3=jnp.array([-0.2, 0.1, 0.0]),
        r_4=jnp.array([-0.2, -0.1, 0.0]),
        p_1_start=jnp.array([0.2, 0.1, 0.0]),
        p_2_start=jnp.array([0.2, -0.1, 0.0]),
        p_3_start=jnp.array([-0.2, 0.1, 0.0]),
        p_4_start=jnp.array([-0.2, -0.1, 0.0]),
        initial_base_pose=jnp.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0]),
        initial_base_velocity=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    
    # Create optimizer
    mppi = QuadrupedControlSpaceMPPI(
        gait=gait,
        terrain=terrain,
        robot=robot,
        horizon=10,
        num_samples=50,
        num_iterations=5,
        device='cpu'
    )
    
    # Initial state
    initial_state = np.concatenate([
        np.asarray(robot.initial_base_pose),
        np.asarray(robot.initial_base_velocity),
        np.asarray(robot.p_1_start),
        np.asarray(robot.p_2_start),
        np.asarray(robot.p_3_start),
        np.asarray(robot.p_4_start),
    ])
    
    target_velocity = np.concatenate([
        np.asarray(gait.desired_base_velocity),
        np.asarray(gait.desired_base_angular_velocity),
    ])
    
    # Optimize
    controls, info = mppi.optimize_trajectory(initial_state, target_velocity)
    
    print(f"  Status: {info['message']}")
    print(f"  Final cost: {info['final_cost']:.6f}")
    print(f"  Controls shape: {controls.shape}")
    
    assert controls.shape == (10, 6), "Control shape mismatch"
    
    # Test trajectory integration
    states, _ = mppi.integrate_controls_to_trajectory(initial_state, controls)
    
    print(f"  States shape: {states.shape}")
    print(f"  Initial position: {states[0, 0:3]}")
    print(f"  Final position: {states[-1, 0:3]}")
    
    assert states.shape == (11, 24), "States shape mismatch"
    
    print("✓ QuadrupedControlSpaceMPPI test passed")


def test_dynamics_function():
    """Test dynamics function creates valid trajectories."""
    print("\nTesting dynamics function...")
    
    # Minimal setup
    gait = Gait(
        n_steps=1, n_phases=1, spline_order=3, tau_k=jnp.array([0.5]),
        h_des=0.4, eps_min=0.1, weights=jnp.ones(8),
        desired_base_velocity=jnp.array([0.2, 0.0, 0.0]),
        desired_base_angular_velocity=jnp.zeros(3),
        apex_height=0.05,
        contact_schedule=jnp.ones((1, 4)),
        at_des_position=jnp.ones((1, 4))
    )
    
    h = jnp.array(get_flat_heightmap(a=50, b=50, height=0.0))
    terrain = Terrain(
        heightmap=h, grid_cell_length=0.04, mu=0.6,
        gravity=jnp.array([0.0, 0.0, -9.81])
    )
    
    robot = Robot(
        mass=5.0, inertia=jnp.diag(jnp.array([0.1, 0.1, 0.02])),
        l_min=0.1, l_max=0.4,
        r_1=jnp.array([0.2, 0.1, 0.0]), r_2=jnp.array([0.2, -0.1, 0.0]),
        r_3=jnp.array([-0.2, 0.1, 0.0]), r_4=jnp.array([-0.2, -0.1, 0.0]),
        p_1_start=jnp.array([0.2, 0.1, 0.0]), p_2_start=jnp.array([0.2, -0.1, 0.0]),
        p_3_start=jnp.array([-0.2, 0.1, 0.0]), p_4_start=jnp.array([-0.2, -0.1, 0.0]),
        initial_base_pose=jnp.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0]),
        initial_base_velocity=jnp.zeros(6)
    )
    
    mppi = QuadrupedControlSpaceMPPI(
        gait=gait, terrain=terrain, robot=robot,
        horizon=5, num_samples=10, num_iterations=2, device='cpu'
    )
    
    # Test dynamics function
    dynamics_fn = mppi.create_dynamics_function()
    
    state = torch.zeros(24)
    state[2] = 0.4  # z position
    control = torch.zeros(6)
    
    next_state = dynamics_fn(state, control)
    
    print(f"  Initial z: {state[2].item():.4f}")
    print(f"  Next z: {next_state[2].item():.4f}")
    print(f"  Expected gravity effect: {mppi.dt * mppi.dt * (-9.81) / 2:.4f}")
    
    # With zero control, should fall due to gravity
    assert next_state[2] < state[2], "Dynamics should include gravity"
    
    print("✓ Dynamics function test passed")


def test_cost_function():
    """Test cost function produces reasonable values."""
    print("\nTesting cost function...")
    
    gait = Gait(
        n_steps=1, n_phases=1, spline_order=3, tau_k=jnp.array([0.5]),
        h_des=0.4, eps_min=0.1, weights=jnp.ones(8),
        desired_base_velocity=jnp.array([0.2, 0.0, 0.0]),
        desired_base_angular_velocity=jnp.zeros(3),
        apex_height=0.05,
        contact_schedule=jnp.ones((1, 4)),
        at_des_position=jnp.ones((1, 4))
    )
    
    h = jnp.array(get_flat_heightmap(a=50, b=50, height=0.0))
    terrain = Terrain(
        heightmap=h, grid_cell_length=0.04, mu=0.6,
        gravity=jnp.array([0.0, 0.0, -9.81])
    )
    
    robot = Robot(
        mass=5.0, inertia=jnp.diag(jnp.array([0.1, 0.1, 0.02])),
        l_min=0.1, l_max=0.4,
        r_1=jnp.array([0.2, 0.1, 0.0]), r_2=jnp.array([0.2, -0.1, 0.0]),
        r_3=jnp.array([-0.2, 0.1, 0.0]), r_4=jnp.array([-0.2, -0.1, 0.0]),
        p_1_start=jnp.array([0.2, 0.1, 0.0]), p_2_start=jnp.array([0.2, -0.1, 0.0]),
        p_3_start=jnp.array([-0.2, 0.1, 0.0]), p_4_start=jnp.array([-0.2, -0.1, 0.0]),
        initial_base_pose=jnp.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0]),
        initial_base_velocity=jnp.zeros(6)
    )
    
    mppi = QuadrupedControlSpaceMPPI(
        gait=gait, terrain=terrain, robot=robot,
        horizon=5, num_samples=10, num_iterations=2, device='cpu'
    )
    
    # Test cost function
    target_vel = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    cost_fn = mppi.create_cost_function(target_vel)
    
    # State at target
    state_good = torch.zeros(24)
    state_good[2] = 0.4  # Correct height
    state_good[6:9] = torch.tensor([0.2, 0.0, 0.0])  # Correct velocity
    
    # State far from target
    state_bad = torch.zeros(24)
    state_bad[2] = 0.2  # Wrong height
    state_bad[6:9] = torch.tensor([0.0, 0.0, 0.0])  # Wrong velocity
    
    control = torch.zeros(6)
    
    cost_good = cost_fn(state_good, control)
    cost_bad = cost_fn(state_bad, control)
    
    print(f"  Cost at target: {cost_good.item():.4f}")
    print(f"  Cost away from target: {cost_bad.item():.4f}")
    
    assert cost_bad > cost_good, "Cost should be higher away from target"
    
    print("✓ Cost function test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Control-Space MPPI Tests")
    print("=" * 60)
    
    test_control_space_mppi_basic()
    test_quadruped_control_space_mppi()
    test_dynamics_function()
    test_cost_function()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
