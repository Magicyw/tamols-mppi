import numpy as np

from tamols.tamols_dataclasses import Terrain, Gait, Robot
from tamols.helpers import bilinear_interp, evaluate_spline_position, euler_xyz_to_matrix
import math

def batch_search(
    terrain: Terrain,
    gait: Gait,
    robot: Robot,
    grad_h_s1_x: np.ndarray,
    grad_h_s1_y: np.ndarray,
    h_s2: np.ndarray,
    laplacian_h: np.ndarray,
    sv_sol: dict,
    search_radius: float = 0.4,
) -> dict:
    heightmap = np.asarray(terrain.heightmap, dtype=float)
    cell_length = float(terrain.grid_cell_length)

    p_last = int(gait.n_phases) - 1
    t_end = float(gait.tau_k[p_last])
    # Evaluate once and keep as NumPy
    p_end = np.asarray(evaluate_spline_position(sv_sol["a_pos"][p_last], t_end), dtype=float)
    r_end = np.asarray(evaluate_spline_position(sv_sol["a_rot"][p_last], t_end), dtype=float)
    R_end = np.asarray(euler_xyz_to_matrix(r_end), dtype=float)

    hip_B = np.stack([np.asarray(robot.r_1), np.asarray(robot.r_2),
                      np.asarray(robot.r_3), np.asarray(robot.r_4)], axis=0)
    hip_W = p_end[None, :] + (R_end @ hip_B.T).T  # (4,3)
    l_max = float(robot.l_max)

    refined_sv = dict(sv_sol)

    for i in range(4):
        p_nom = np.asarray(sv_sol[f"p_{i+1}"], dtype=float)
        cx, cy = float(p_nom[0]), float(p_nom[1])
        best_cost = float("inf")
        best = p_nom.copy()

        r_cells = int(math.ceil(search_radius / cell_length))
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                if math.hypot(dx * cell_length, dy * cell_length) > search_radius:
                    continue
                x = cx + dx * cell_length
                y = cy + dy * cell_length
                z = float(bilinear_interp(heightmap, [x, y, 0.0], cell_length))
                if math.isnan(z):
                    continue
                # Reachability in NumPy
                d = np.linalg.norm(np.array([x, y, z]) - hip_W[i])
                if d > l_max:
                    continue

                p = np.array([x, y, z], dtype=float)
                gx = float(bilinear_interp(grad_h_s1_x, p, cell_length))
                gy = float(bilinear_interp(grad_h_s1_y, p, cell_length))
                lap = float(bilinear_interp(laplacian_h, p, cell_length))
                h2 = float(bilinear_interp(h_s2, p, cell_length))

                slope2 = gx*gx + gy*gy
                curv2 = lap*lap
                vert2 = (h2 - z)*(h2 - z)
                prev2 = (p[0] - cx)**2 + (p[1] - cy)**2

                cost = 10.0*slope2 + 5.0*curv2 + 2.0*vert2 + 0.1*prev2
                if cost < best_cost:
                    best_cost = cost
                    best = p

        refined_sv[f"p_{i+1}"] = best.astype(np.float32)

    return refined_sv