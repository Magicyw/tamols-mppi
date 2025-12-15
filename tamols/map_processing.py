from scipy import ndimage
import numpy as np
from pathlib import Path
from PIL import Image  # Pillow is standard for PNG writing
import matplotlib.pyplot as plt

def compute_heightmap_gradients(height_map,grid_cell_length):
    """
    Compute gradients using 1D 5-point central finite difference kernel
    Args:
        height_map: 2D height map array
    Returns:
        gradient_x: Gradient in x direction
        gradient_y: Gradient in y direction
    """
    kernel = np.array([-1, 8, 0, -8, 1]) / (12 * grid_cell_length)
    
    # Compute first derivatives
    gradient_x = ndimage.convolve1d(height_map, kernel, axis=1)
    gradient_y = ndimage.convolve1d(height_map, kernel, axis=0)
    
    return gradient_x, gradient_y

def get_hs1(h_raw,sigma=0.5):
    """
    Compute the Gaussian filtered heightmap.
    Args:
        h_raw: Raw heightmap data
    Returns:
        h_s1: Smoothed heightmap
    """
    h_s1 = ndimage.gaussian_filter(h_raw, sigma)
    return h_s1

def get_hs2(h_raw, median_kernel=25, dilation_kernel=25, sigma=5.0, limit=0.05):
    """
    Compute the virtual floor heightmap.
    Args:
        h_s1: Smoothed heightmap from get_hs1
    Returns:
        h_s2: Virtual floor heightmap
    """
    h_median= ndimage.median_filter(h_raw, median_kernel)

    delta_h=h_raw-h_median

    mask=np.zeros_like(h_raw)
    mask[delta_h>limit] = 1
    mask[delta_h<-limit] = 1

    h_dilated=ndimage.maximum_filter(h_raw,dilation_kernel)
    h_s2 = np.where(mask==1, h_dilated, h_raw)
    h_s2 = ndimage.gaussian_filter(h_s2, sigma)

    return h_s2

def get_laplacian(h: np.ndarray, grid_cell_length: float) -> np.ndarray:
    """
    Compute the Laplacian of the heightmap using finite differences.
    Args:
        h: 2D heightmap array
        grid_cell_length: Length of each grid cell in the heightmap
    Returns:
        laplacian: 2D array of the same shape as h representing the Laplacian
    """
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]]) / (grid_cell_length ** 2)
    
    laplacian = ndimage.convolve(h, kernel, mode='nearest')
    return laplacian

def save_heightmap_to_png(heightmap, filepath, vmin=None, vmax=None, invert=False, ensure_parent=True):
    """Save a 2D heightmap array as an 8-bit grayscale PNG (black=low, white=high).

    Args:
        heightmap: 2D array-like (NumPy array) of shape (H, W).
        filepath: Output file path (str or Path). Extension ".png" will be appended if missing.
        vmin: Optional lower bound for normalization. If None, uses heightmap min.
        vmax: Optional upper bound for normalization. If None, uses heightmap max.
        invert: If True, invert grayscale (so higher values darker instead of lighter).
        ensure_parent: Create parent directories if they don't exist.

    Behavior:
        Scales values linearly: val -> (val - vmin)/(vmax - vmin) * 255, clipped to [0,255].
        If vmin == vmax (constant map), outputs mid-gray (128) everywhere.

    Raises:
        ValueError: If input is not 2D.
        RuntimeError: If Pillow is not installed.
    """
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required to save PNGs. Install with 'pip install Pillow'.")

    arr = np.array(heightmap)  # Ensure heightmap is a NumPy array
    if arr.ndim != 2:
        raise ValueError(f"Heightmap must be 2D, got shape {arr.shape}.")

    # Determine normalization range
    data_min = float(arr.min()) if vmin is None else float(vmin)
    data_max = float(arr.max()) if vmax is None else float(vmax)

    if data_max == data_min:
        norm = np.full_like(arr, 128, dtype=np.uint8)
    else:
        scale = 255.0 / (data_max - data_min)
        norm = ((arr - data_min) * scale).clip(0, 255)
        if invert:
            norm = 255 - norm
        norm = norm.astype(np.uint8)

    fp = Path(filepath)
    if fp.suffix.lower() != '.png':
        fp = fp.with_suffix('.png')
    if ensure_parent:
        fp.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(norm, mode='L')  # 'L' = 8-bit grayscale
    img.save(fp)
    return fp

    return heightmap

def show_map(h):

    h_s1 = get_hs1(h)
    h_s2 = get_hs2(h)

    y = np.arange(h.shape[0])
    x = np.arange(h.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(15, 5))

    zmin,zmax= -0.1, 0.5

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, h, cmap='terrain')
    ax1.set_title("Original h")
    ax1.set_zlim(zmin, zmax)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Height (Z)")

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, h_s1, cmap='terrain')
    ax2.set_title("h_s1")
    ax2.set_zlim(zmin, zmax)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Height (Z)")

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, h_s2, cmap='terrain')
    ax3.set_title("h_s2")
    ax3.set_zlim(zmin, zmax)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Height (Z)")

    # Plot gradients on a separate 2D figure
    grad_x, grad_y = compute_heightmap_gradients(h_s1, grid_cell_length=0.05)
    skip = 1

    fig2, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(
        h_s1, origin='lower', cmap='terrain',
        extent=[0, h.shape[1], 0, h.shape[0]]
    )
    ax.quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        grad_x[::skip, ::skip], grad_y[::skip, ::skip],
        color='red', scale=10, width=0.003
    )

    ax.set_title("Gradient field on h_s1")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()