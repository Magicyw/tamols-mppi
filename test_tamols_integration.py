"""
Test TAMOLS integration with control-space MPPI.
"""

import jax.numpy as jnp
import numpy as np
from tamols import TAMOLS
from tamols.tamols_dataclasses import Gait, Terrain, Robot
from tamols.manual_heightmaps import get_flat_heightmap
from tamols.helpers import transform_inertia


def test_tamols_state_space():
    """Test TAMOLS with state-space MPPI."""
    print("Testing TAMOLS with state-space MPPI...")
    
    gait = Gait(
        n_steps=1,
        n_phases=1,
        spline_order=3,
        tau_k=jnp.array([0.5]),
        h_des=0.4,
        eps_min=0.1,
        weights=jnp.array([10.0, 0.01, 1.0, 10.0, 1.0, 0.01, 1.0, 0.01]),
        desired_base_velocity=jnp.array([0.2, 0.0, 0.0]),
        desired_base_angular_velocity=jnp.array([0.0, 0.0, 0.0]),
        apex_height=0.05,
        contact_schedule=jnp.array([[1, 1, 1, 1]]),
        at_des_position=jnp.array([[1, 1, 1, 1]])
    )
    
    h = jnp.array(get_flat_heightmap(a=50, b=50, height=0.0))
    terrain = Terrain(
        heightmap=h,
        grid_cell_length=0.04,
        mu=0.6,
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )
    
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
    
    # State-space MPPI (default)
    problem = TAMOLS(
        gait, terrain, robot,
        num_samples=50,
        num_iterations=5,
        use_control_space_mppi=False
    )
    
    x_sol, info = problem.run_single_optimization()
    
    print(f"  Status: {info['status']}")
    print(f"  Objective: {info['obj_val']:.6f}")
    print(f"  Solution shape: {x_sol.shape}")
    
    assert x_sol.shape[0] == problem.n, "Solution dimension mismatch"
    assert 'obj_val' in info, "Missing objective value"
    
    print("✓ State-space MPPI integration test passed")


def test_tamols_control_space():
    """Test TAMOLS with control-space MPPI."""
    print("\nTesting TAMOLS with control-space MPPI...")
    
    gait = Gait(
        n_steps=1,
        n_phases=1,
        spline_order=3,
        tau_k=jnp.array([0.5]),
        h_des=0.4,
        eps_min=0.1,
        weights=jnp.array([10.0, 0.01, 1.0, 10.0, 1.0, 0.01, 1.0, 0.01]),
        desired_base_velocity=jnp.array([0.2, 0.0, 0.0]),
        desired_base_angular_velocity=jnp.array([0.0, 0.0, 0.0]),
        apex_height=0.05,
        contact_schedule=jnp.array([[1, 1, 1, 1]]),
        at_des_position=jnp.array([[1, 1, 1, 1]])
    )
    
    h = jnp.array(get_flat_heightmap(a=50, b=50, height=0.0))
    terrain = Terrain(
        heightmap=h,
        grid_cell_length=0.04,
        mu=0.6,
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )
    
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
    
    # Control-space MPPI
    problem = TAMOLS(
        gait, terrain, robot,
        num_samples=50,
        num_iterations=5,
        use_control_space_mppi=True,
        control_horizon=10
    )
    
    x_sol, info = problem.run_single_optimization()
    
    print(f"  Status: {info['status']}")
    print(f"  Objective: {info['obj_val']:.6f}")
    print(f"  Solution shape: {x_sol.shape}")
    
    assert x_sol.shape[0] == problem.n, "Solution dimension mismatch"
    assert 'obj_val' in info, "Missing objective value"
    
    print("✓ Control-space MPPI integration test passed")


def test_tamols_comparison():
    """Compare state-space and control-space MPPI."""
    print("\nComparing state-space vs control-space MPPI...")
    
    gait = Gait(
        n_steps=1,
        n_phases=1,
        spline_order=3,
        tau_k=jnp.array([0.5]),
        h_des=0.4,
        eps_min=0.1,
        weights=jnp.array([10.0, 0.01, 1.0, 10.0, 1.0, 0.01, 1.0, 0.01]),
        desired_base_velocity=jnp.array([0.1, 0.0, 0.0]),
        desired_base_angular_velocity=jnp.array([0.0, 0.0, 0.0]),
        apex_height=0.05,
        contact_schedule=jnp.array([[1, 1, 1, 1]]),
        at_des_position=jnp.array([[1, 1, 1, 1]])
    )
    
    h = jnp.array(get_flat_heightmap(a=50, b=50, height=0.0))
    terrain = Terrain(
        heightmap=h,
        grid_cell_length=0.04,
        mu=0.6,
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )
    
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
    
    # State-space
    problem_state = TAMOLS(
        gait, terrain, robot,
        num_samples=50, num_iterations=5,
        use_control_space_mppi=False
    )
    x_state, info_state = problem_state.run_single_optimization()
    
    # Control-space
    problem_control = TAMOLS(
        gait, terrain, robot,
        num_samples=50, num_iterations=5,
        use_control_space_mppi=True,
        control_horizon=10
    )
    x_control, info_control = problem_control.run_single_optimization()
    
    print(f"  State-space objective: {info_state['obj_val']:.6f}")
    print(f"  Control-space objective: {info_control['obj_val']:.6f}")
    
    # Both should produce valid solutions
    assert np.isfinite(info_state['obj_val']), "State-space objective is not finite"
    assert np.isfinite(info_control['obj_val']), "Control-space objective is not finite"
    
    print("✓ Comparison test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("TAMOLS Integration Tests")
    print("=" * 60)
    
    test_tamols_state_space()
    test_tamols_control_space()
    test_tamols_comparison()
    
    print("\n" + "=" * 60)
    print("All integration tests passed! ✓")
    print("=" * 60)
