import numpy as np
import mlx.core as mx
import math
import numpy as np
from numba import njit, prange, types, complex128, int32, float64
import time 

mx.set_default_device(mx.gpu)

@njit("complex128(complex128, complex128, int32)", fastmath=True, cache=True)
def julia_equation(z: np.complex128, c:np.complex128, eqn:np.int32):
    if eqn==0:
        return z*z*z*z*z*z - z*z*z*z + c
    elif eqn==1:
        return np.exp(1j*2*np.pi*np.abs(z)) + z + c
    elif eqn==2:
        return z*z + c
    elif eqn==3:
        return z*z*z + c
    return z*z + c

@njit("int32(complex128, complex128, int32, int32)", fastmath=True, cache=True)
def _julia_escape_single(
    z0: np.complex128,
    c: np.complex128,
    eqn: int = 0,
    max_iter: int = 400,
) -> np.int32:
    z = z0
    for k in range(max_iter):
        z = julia_equation(z,c,eqn)
        if (z.real*z.real + z.imag*z.imag) > 4.0: return k+1
    return max_iter

# vectorized, parallel caller
@njit("int32[:](complex128[:], complex128, int32, int32)",parallel=True, fastmath=True, cache=True)
def julia_escape_vec(z0, c, eqn, max_iter):
    n = z0.size
    out = np.empty(n, np.int32)
    for i in prange(n):
        out[i] = _julia_escape_single(z0[i], c, eqn, max_iter)
    return out


# Example polynomial: f(z, c) = z^2 + c
def f_quadratic(z: mx.array, c: mx.array) -> mx.array:
    return z * z + c

def escape_vec(z0, c, max_iter: int, bailout: float = 2.0,
                    f=f_quadratic) -> mx.array:
    """
    Same semantics as escape_vec, but avoids host sync in the loop.
    """
    # Move to MLX + complex
    z = mx.array(z0, dtype=mx.complex64)
    c_arr = mx.array(c, dtype=mx.complex64)

    # Broadcast c if scalar
    if c_arr.size == 1:
        c_arr = mx.full(z.shape, c_arr.item(), dtype=mx.complex64)

    # Iter counts, init to max_iter (i.e. "never escaped")
    iters = mx.full(z.shape, max_iter, dtype=mx.int32)

    # Points still being iterated
    active = mx.ones(z.shape, dtype=mx.bool_)

    bailout_sq = bailout * bailout

    for k in range(max_iter):
        # Only advance active points; escaped ones keep their old z
        z = mx.where(active, f(z, c_arr), z)

        # Compute abs^2 without an extra sqrt
        mag_sq = mx.real(z) * mx.real(z) + mx.imag(z) * mx.imag(z)

        escaped_now = active & (mag_sq > bailout_sq)

        # Record first-escape iteration (k+1)
        iters = mx.where(escaped_now,
                         mx.array(k + 1, iters.dtype),
                         iters)

        # Once escaped, no longer active
        active = active & (~escaped_now)

    mx.eval(iters)
    return iters



# Build a grid of starting z’s
N = 5000
x = np.linspace(-1.5, 1.5, N, dtype=np.float32)
y = np.linspace(-1.5, 1.5, N, dtype=np.float32)
X, Y = np.meshgrid(x, y)
z0 = X + 1j * Y

c = -0.8 + 0.156j
max_iter = 300

t0 = time.perf_counter()
iters = escape_vec(z0, c, max_iter)
iter1 = np.array(iters)
t1 = time.perf_counter()
print(f"MLX GPU:  {t1 - t0:.4f} s")


zz0 = z0.ravel().astype(np.complex128)
_ = julia_escape_vec(zz0[:10], c, 2, max_iter)
t0 = time.perf_counter()
iter2 = julia_escape_vec(zz0,np.complex128(c),np.int32(2),np.int32(max_iter))
t1 = time.perf_counter()
print(f"Numba CPU:  {t1 - t0:.4f} s")


