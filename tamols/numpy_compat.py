"""
NumPy compatibility layer to replace JAX functionality.
Provides minimal implementations of JAX features using pure NumPy.
"""

import numpy as np
from typing import Callable, Any, Tuple
import functools


def jit(func: Callable) -> Callable:
    """
    No-op JIT decorator. In JAX, jit compiles functions for performance.
    With NumPy, we just return the function as-is.
    """
    return func


def vmap(func: Callable, in_axes=0, out_axes=0) -> Callable:
    """
    Vectorized map. Maps a function over array axes.
    This is a simplified implementation that handles the most common case.
    """
    @functools.wraps(func)
    def vectorized(*args):
        # Handle simple case: single argument
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, (np.ndarray, list)):
                return np.array([func(item) for item in arg])
            else:
                return func(arg)
        
        # Handle multiple arguments
        # Assume all arguments are arrays with same length
        results = []
        if isinstance(in_axes, int):
            # All arguments vectorized along same axis
            n = len(args[0])
            for i in range(n):
                inputs = [arg[i] for arg in args]
                results.append(func(*inputs))
        else:
            # Custom axes per argument (not fully implemented)
            raise NotImplementedError("vmap with custom in_axes not fully supported")
        
        result = np.array(results)
        return result
    
    return vectorized


def ravel_pytree(pytree: Any) -> Tuple[np.ndarray, Callable]:
    """
    Flatten a pytree (nested dict/list of arrays) into a 1D array.
    Returns the flattened array and a function to unflatten it.
    
    This is a simplified version that handles dicts of arrays.
    """
    flat_arrays = []
    structure = []
    
    def flatten(tree, prefix=""):
        if isinstance(tree, dict):
            for key, value in sorted(tree.items()):
                flatten(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(tree, np.ndarray):
            flat_arrays.append(tree.ravel())
            structure.append((prefix, tree.shape, tree.dtype))
        elif isinstance(tree, (list, tuple)):
            for i, item in enumerate(tree):
                flatten(item, f"{prefix}[{i}]")
        else:
            # Scalar or other type - convert to array
            arr = np.asarray(tree)
            flat_arrays.append(arr.ravel())
            structure.append((prefix, arr.shape, arr.dtype))
    
    flatten(pytree)
    
    if not flat_arrays:
        flattened = np.array([])
    else:
        flattened = np.concatenate([arr.astype(np.float64) for arr in flat_arrays])
    
    def unflatten(flat):
        """Reconstruct the pytree from flattened array."""
        flat = np.asarray(flat, dtype=np.float64)
        idx = 0
        reconstructed = {}
        
        for path, shape, dtype in structure:
            size = np.prod(shape, dtype=int)
            arr = flat[idx:idx+size].reshape(shape).astype(dtype)
            idx += size
            
            # Rebuild nested structure
            keys = path.split('.')
            current = reconstructed
            for i, key in enumerate(keys[:-1]):
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = arr
        
        return reconstructed
    
    return flattened, unflatten


class RandomState:
    """Simple random state wrapper to mimic JAX random."""
    
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
    
    def split(self, num=2):
        """Split random state. Returns new states."""
        seeds = self.rng.randint(0, 2**31, size=num)
        return tuple(RandomState(s) for s in seeds)
    
    def normal(self, shape, dtype=np.float64):
        """Generate normal random numbers."""
        return self.rng.normal(size=shape).astype(dtype)
    
    def uniform(self, shape, minval=0.0, maxval=1.0, dtype=np.float64):
        """Generate uniform random numbers."""
        return self.rng.uniform(minval, maxval, size=shape).astype(dtype)


class random:
    """Random number generation compatible with JAX API."""
    
    @staticmethod
    def PRNGKey(seed):
        """Create a PRNG key from seed."""
        return RandomState(seed)
    
    @staticmethod
    def split(key, num=2):
        """Split a PRNG key into multiple keys."""
        if isinstance(key, RandomState):
            return key.split(num)
        else:
            # Handle tuple of keys
            return tuple(k.split(num)[0] for k in key[:num])
    
    @staticmethod
    def normal(key, shape, dtype=np.float64):
        """Generate normal random numbers."""
        if isinstance(key, RandomState):
            return key.normal(shape, dtype)
        else:
            # key is a tuple
            return key[0].normal(shape, dtype)


# Stub implementations for unused JAX features
def value_and_grad(func):
    """Stub - not implemented."""
    raise NotImplementedError("value_and_grad not implemented in NumPy compat layer")


def jacfwd(func):
    """Stub - not implemented."""
    raise NotImplementedError("jacfwd not implemented in NumPy compat layer")


def jacrev(func):
    """Stub - not implemented."""
    raise NotImplementedError("jacrev not implemented in NumPy compat layer")
