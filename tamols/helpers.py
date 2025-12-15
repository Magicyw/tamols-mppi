
import numpy as np

def euler_xyz_to_matrix(phi):
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrix.
    """
    roll, pitch, yaw = phi

    cx, cy, cz = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sx, sy, sz = np.sin(roll), np.sin(pitch), np.sin(yaw)
    
    R = np.array([
        [cy*cz, -cy*sz, sy],
        [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy],
        [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy]
    ])
    return R

def bilinear_interp(heightmap, p, cell_length):
    """
    Bilinear interpolation for heightmaps. The center of the heightmap is at (0,0).
    Args:
        heightmap: 2D Jax numpy array representing the heightmap
        p: position vector (length-3 array-like, [x, y, z]), only x and y are used
        cell_length: length of one grid cell (float)

    Returns:
        Interpolated height at position p
    """
    h, w = heightmap.shape
    # Shift (0,0) to the center of the heightmap, scale by cell_length
    x = p[0] / cell_length + w / 2
    y = p[1] / cell_length + h / 2

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    Q11 = heightmap[y0, x0]
    Q12 = heightmap[y1, x0]
    Q21 = heightmap[y0, x1]
    Q22 = heightmap[y1, x1]

    wx = x - x0
    wy = y - y0

    return (1 - wx) * (1 - wy) * Q11 + \
           (1 - wx) * wy * Q12 + \
           wx * (1 - wy) * Q21 + \
           wx * wy * Q22

def evaluate_spline_position(a, t):
    """
    Evaluate the position of an n-th order spline at time t.
    Args:
        a: Coefficients of the spline (shape [d, n+1] array-like, d: dimension, n: order)
        t: Time variable

    Returns:
        Position at time t (shape [d])
    """
    n = a.shape[1] - 1
    t_powers = np.array([t**i for i in range(n + 1)])
    return np.dot(a, t_powers)

def evaluate_spline_velocity(a, t):
    """
    Evaluate the velocity of an n-th order spline at time t.
    Args:
        a: Coefficients of the spline (shape [d, n+1] array-like, d: dimension, n: order)
        t: Time variable

    Returns:
        Velocity at time t (shape [d])
    """
    n = a.shape[1] - 1
    t_powers = np.array([i * t**(i-1) if i >= 1 else 0 for i in range(n + 1)])
    return np.dot(a, t_powers)

def evaluate_spline_acceleration(a, t):
    """
    Evaluate the acceleration of an n-th order spline at time t.
    Args:
        a: Coefficients of the spline (shape [d, n+1] array-like, d: dimension, n: order)
        t: Time variable

    Returns:
        Acceleration at time t (shape [d])
    """
    n = a.shape[1] - 1
    # Compute second derivative coefficients: i*(i-1)*t**(i-2) for i >= 2, else 0
    t_powers = np.array([i * (i - 1) * t**(i - 2) if i >= 2 else 0 for i in range(n + 1)])
    return np.dot(a, t_powers)

def robust_norm(x, eps=1e-8):
    """
    Compute the robust norm of a vector x, adding a small epsilon to avoid division by zero.
    Args:
        x: Input vector
        eps: Small constant to avoid division by zero

    Returns:
        Robust norm of x
    """
    return np.sqrt(np.sum(x**2) + eps)

def constraint_hess_vec(con_fun):
    def hess_vec(x, d_params, v):
        m = con_fun(x, d_params).shape[0]
        n = x.shape[0]
        return sum(v[i] * jacfwd(jacrev(lambda x: con_fun(x, d_params)[i]))(x) for i in range(m))
    return hess_vec

def scalar_triple_product(a, b, c):
    """
    Compute the scalar triple product of three vectors a, b, c.
    Args:
        a: First vector
        b: Second vector
        c: Third vector

    Returns:
        Scalar triple product of a, b, c
    """
    return np.dot(a, np.cross(b, c))

def quat_to_rot(q):
    # q = [w, x, y, z]
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y)]
    ])
    return R

def transform_inertia(I_A, m, r, q):
    R = quat_to_rot(q)             # maps A->base
    I_rot = R @ I_A @ R.T          # inertia in base axes but about A origin
    r = np.asarray(r).reshape(3,1)
    I_pa = m * (np.dot(r.T, r).item() * np.eye(3) - (r @ r.T))
    I_base = I_rot + I_pa
    return I_base

def euler_xyz_rates_to_body_omega_alpha(phi: np.ndarray,
                                        phi_dot: np.ndarray,
                                        phi_ddot: np.ndarray):
    """
    Map Euler XYZ angle rates to body angular velocity/acceleration.
    Angles: phi = [roll_x, pitch_y, yaw_z] = [r, p, y]
    Sequence: ZYX-intrinsic (equivalent to XYZ-extrinsic used by euler_xyz_to_matrix)
    ω_b = E(r,p) * φ̇
    α_b = E(r,p) * φ̈ + Ė(r,p,φ̇) * φ̇
    """
    r, p, y = phi
    r_d, p_d, y_d = phi_dot
    # Precompute trig
    sr, cr = np.sin(r), np.cos(r)
    sp, cp = np.sin(p), np.cos(p)

    # Mapping matrix E(r,p) for ZYX-intrinsic (roll-pitch-yaw) to body rates
    E = np.array([
        [1.0, 0.0,    -sp],
        [0.0,   cr,  sr*cp],
        [0.0,  -sr,  cr*cp],
    ])

    # Time derivative Ė = ∂E/∂r * ṙ + ∂E/∂p * ṗ
    dE_dr = np.array([
        [0.0,  0.0,          0.0],
        [0.0, -sr,      cr*cp],
        [0.0, -cr,     -sr*cp],
    ])
    dE_dp = np.array([
        [0.0,  0.0,        -cp],
        [0.0,  0.0,   -sr*sp],
        [0.0,  0.0,   -cr*sp],
    ])
    E_dot = dE_dr * r_d + dE_dp * p_d

    omega_b = E @ phi_dot
    alpha_b = E @ phi_ddot + E_dot @ phi_dot
    return omega_b, alpha_b

def angular_momentum_dot_world_from_euler_xyz(phi: np.ndarray,
                                              phi_dot: np.ndarray,
                                              phi_ddot: np.ndarray,
                                              inertia_body: np.ndarray):
    """
    World-frame derivative: L̇_w = R(φ) · ḣ_b
    Uses euler_xyz_to_matrix only to rotate the body-vector to world.
    """
    omega_b, alpha_b = euler_xyz_rates_to_body_omega_alpha(phi, phi_dot, phi_ddot)
    hdot_b = inertia_body @ alpha_b + np.cross(omega_b, inertia_body @ omega_b)
    R = euler_xyz_to_matrix(phi)
    return R @ hdot_b

