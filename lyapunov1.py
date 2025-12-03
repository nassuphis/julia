#!/usr/bin/env python
"""
lyapounov.py

Lyapunov fractal renderer for 1‑D maps with A/B forcing, driven by
specparser/expandspec.

Key idea: every map is defined by a *single* symbolic expression

    f(x, r; alpha, beta, delta, epsilon)

where:
    - x is the state
    - r is the driven parameter (A/B alternating via a sequence)
    - alpha, beta, delta, epsilon are optional extra parameters

For each expression we automatically:
    - build a Numba‑compatible stepping function
    - build its symbolic derivative df/dx using SymPy
    - JIT both with the same (x, r, params) signature

The Lyapunov code then treats all maps generically; adding a new map is
just:
    1) add an entry in MAP_TEMPLATES with an expression string
    2) (optionally) set default (A,B) window and parameter defaults
"""

import sys
sys.path.insert(0, "/Users/nicknassuphis")
import time
import math
import cmath
import argparse
import re as regex
import numpy as np
import sympy as sp
from numba import njit, types, prange

from specparser import specparser, expandspec
from rasterizer import raster
from rasterizer import colors

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_MAP_NAME = "logistic"
DEFAULT_SEQ      = "AB"
DEFAULT_TRANS    = 200
DEFAULT_ITER     = 1000
DEFAULT_X0       = 0.5
DEFAULT_EPS_LYAP = 1e-12
DEFAULT_CLIP     = None     # auto from data
DEFAULT_GAMMA    = 1.0      # colormap gamma


# ---------------------------------------------------------------------------
# Symbolic derivative helper (x derivative of map expression)
# ---------------------------------------------------------------------------

def _sympy_deriv(expr_str: str) -> str:
    """
    Return d/dx of the given expression string in SymPy's sstr format.

    The expression can use:
        x, r,  x, r, a, b, c, d, epsilon, eps, zeta, eta
        sin, cos, tan, sec, cosh, exp, sign, abs/Abs, max
        step (Heaviside), DiracDelta, pow, mod1, Mod, pi
    """
    x, r = sp.symbols("x r")
    a, b, c, d = sp.symbols("a b c d")
    eps, zeta, eta, epsilon = sp.symbols("eps zeta eta epsilon")

    locs = {
        "x": x,
        "r": r,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "epsilon": epsilon,
        "eps": eps,
        "zeta": zeta,
        "eta": eta,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sec": sp.sec,
        "cosh": sp.cosh,
        "exp": sp.exp,
        "sign": sp.sign,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "max": sp.Max,
        "min": sp.Min,
        "step": sp.Heaviside,
        "Heaviside": sp.Heaviside,
        "DiracDelta": sp.DiracDelta,
        "pow": sp.Pow,
        "apow": lambda x, a: sp.sign(x) * sp.Pow(sp.Abs(x),a),
        "mod1": lambda v: sp.Mod(v, 1),
        "Mod": sp.Mod,
        "pi": sp.pi,
        "floor": sp.floor,
        "ceil": sp.ceiling,
    }

    expr = sp.sympify(expr_str, locals=locs)
    expr_der = sp.diff(expr, x)
    return sp.sstr(expr_der)


def _sympy_jacobian_2d(expr_x: str, expr_y: str):
    """
    Return 4 SymPy sstr expressions for the Jacobian of a 2‑D map:
        expr_x = f(x, y, r, s, ...)
        expr_y = g(x, y, r, s, ...)
    """
    x, y, r, s = sp.symbols("x y r s")
    a, b, c, d = sp.symbols("a b c d")
    eps, zeta, eta, epsilon = sp.symbols("eps zeta eta epsilon")

    locs = {
        "x": x,
        "y": y,
        "r": r,
        "s": s,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "epsilon": epsilon,
        "eps": eps,
        "zeta": zeta,
        "eta": eta,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sec": sp.sec,
        "cosh": sp.cosh,
        "exp": sp.exp,
        "sign": sp.sign,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "max": sp.Max,
        "min": sp.Min,
        "step": sp.Heaviside,
        "Heaviside": sp.Heaviside,
        "DiracDelta": sp.DiracDelta,
        "pow": sp.Pow,
        "apow": lambda x, a: sp.sign(x) * sp.Pow(sp.Abs(x), a),
        "mod1": lambda v: sp.Mod(v, 1),
        "Mod": sp.Mod,
        "pi": sp.pi,
        "floor": sp.floor,
        "ceil": sp.ceiling,
    }

    fx = sp.sympify(expr_x, locals=locs)
    fy = sp.sympify(expr_y, locals=locs)

    dfx_dx = sp.diff(fx, x)
    dfx_dy = sp.diff(fx, y)
    dfy_dx = sp.diff(fy, x)
    dfy_dy = sp.diff(fy, y)

    return tuple(sp.sstr(e) for e in (dfx_dx, dfx_dy, dfy_dx, dfy_dy))


# ---------------------------------------------------------------------------
# Tiny Numba helpers used inside map expressions
# ---------------------------------------------------------------------------

@njit(types.float64(types.float64), cache=True, fastmath=False)
def DiracDelta(x):
    # We ignore distributional spikes; enough for Lyapunov purposes.
    return 0.0


@njit(types.float64(types.float64), cache=True, fastmath=False)
def Heaviside(x):
    return 1.0 if x > 0.0 else 0.0


@njit(types.float64(types.float64), cache=True, fastmath=False)
def step(x):
    return 1.0 if x > 0.0 else 0.0


@njit(types.float64(types.float64), cache=True, fastmath=False)
def sign(x):
    return 1.0 if x > 0.0 else -1.0


@njit(types.float64(types.float64), cache=True, fastmath=False)
def Abs(x):
    return np.abs(x)


@njit(types.float64(types.float64), cache=True, fastmath=False)
def re(x):
    return x


@njit(types.float64(types.float64), cache=True, fastmath=False)
def im(x):
    return 0.0


@njit(types.float64(types.float64), cache=True, fastmath=False)
def sec(x):
    return 1.0 / np.cos(x)


@njit(types.float64(types.float64), cache=True, fastmath=False)
def mod1(x):
    return x % 1.0


@njit(types.float64(types.float64, types.float64), cache=True, fastmath=False)
def Mod(x, v):
    return x % v


@njit(types.float64(types.float64, types.float64), cache=True, fastmath=False)
def Derivative(x, v):
    # never actually used; placeholder to keep SymPy happy if it
    # sneaks in.
    return 1.0


@njit(types.float64(types.float64, types.float64), cache=True, fastmath=False)
def apow(x, a):
    return np.sign(x)*np.pow(np.abs(x),a)

@njit(types.float64(types.float64), cache=True, fastmath=False)
def floor(x):
    return math.floor(x)


@njit(types.float64(types.float64), cache=True, fastmath=False)
def ceil(x):
    return math.ceil(x)

@njit(types.float64(types.float64,types.float64,), cache=True, fastmath=False)
def abs_cap(x,cap):
    return min(abs(x),cap)*sign(x)

@njit
def j0s(x):
    ax = abs(x)
    if ax < 8.0:
        y = x*x
        return (1.0 - y*(0.25 - y*(0.046875 - y*(0.003255208333))))
    else:
        z = 8.0/ax
        y = z*z
        xx = ax - 0.7853981633974483096
        return np.sqrt(0.636619772/ax) * (
            np.cos(xx)*(1 - y*(0.001098628627 - y*0.000002073)) -
            np.sin(xx)*(0.01562499997 - y*(0.000143048876 - y*0.000000243))
        )
    
@njit("float64(float64)", fastmath=True, cache=True)
def j0(x):
    ax = x if x >= 0.0 else -x

    # Near zero: J0(x) ≈ 1 - x²/4
    if ax < 1e-8:
        return 1.0 - 0.25 * x * x

    # Power series: J0(x) = Σ (-1)^k (x²/4)^k / (k!)²
    if ax < 20.0:
        y = (x * x) * 0.25
        term = 1.0
        s = 1.0
        # 20 terms is plenty for double precision on this range
        for k in range(1, 20):
            term *= -y / (k * k)
            s += term
        return s

    # Asymptotic for large |x|: J0(x) ~ sqrt(2/(πx)) cos(x - π/4)
    t = ax
    return math.sqrt(2.0 / (math.pi * t)) * math.cos(t - 0.25 * math.pi)


@njit("float64(float64)", fastmath=True, cache=True)
def j1(x):
    ax = x if x >= 0.0 else -x

    # Near zero: J1(x) ≈ x/2
    if ax < 1e-8:
        return 0.5 * x

    # Power series: J1(x) = Σ (-1)^k (x/2)^{2k+1} / (k!(k+1)!)
    if ax < 20.0:
        # k = 0 term
        term = 0.5 * x
        s = term
        y = (x * x) * 0.25
        for k in range(1, 20):
            term *= -y / (k * (k + 1))
            s += term
        return s

    # Asymptotic: J1(x) ~ sqrt(2/(πx)) cos(x - 3π/4)
    t = ax
    val = math.sqrt(2.0 / (math.pi * t)) * math.cos(t - 0.75 * math.pi)
    # J1(-x) = -J1(x)
    return -val if x < 0.0 else val


@njit("float64(float64)", fastmath=True, cache=True)
def i0(x):
    ax = x if x >= 0.0 else -x

    # Near zero: I0(x) ≈ 1 + x²/4
    if ax < 1e-8:
        return 1.0 + 0.25 * x * x

    # Power series: I0(x) = Σ (x²/4)^k / (k!)²
    if ax < 15.0:
        y = 0.25 * x * x
        term = 1.0
        s = 1.0
        for k in range(1, 50):
            term *= y / (k * k)
            s += term
        return s

    # Asymptotic: I0(x) ~ exp(x)/sqrt(2πx)
    t = ax
    val = math.exp(t) / math.sqrt(2.0 * math.pi * t)
    return val


@njit("float64(float64)", fastmath=True, cache=True)
def i1(x):
    ax = x if x >= 0.0 else -x

    # Near zero: I1(x) ≈ x/2
    if ax < 1e-8:
        return 0.5 * x

    # Power series: I1(x) = Σ (x/2)^{2k+1} / (k!(k+1)!)
    if ax < 15.0:
        y = 0.25 * x * x
        term = 0.5 * x  # k=0
        s = term
        for k in range(1, 50):
            term *= y / (k * (k + 1))
            s += term
        return s

    # Asymptotic: I1(x) ~ exp(x)/sqrt(2πx)
    t = ax
    val = math.exp(t) / math.sqrt(2.0 * math.pi * t)
    # I1(-x) = -I1(x)
    return -val if x < 0.0 else val

# Ai(0), Ai'(0), Bi(0), Bi'(0)
AI0  = 0.3550280538878172
AI0P = -0.2588194037928068
BI0  = 0.6149266274460007
BI0P = 0.4482883573538264


@njit(types.float64(types.float64, types.float64, types.float64), fastmath=True, cache=True)
def _airy_series(x, c0, c1):
    """
    Generic power series for a solution of y'' - x y = 0
    with y(0) = c0, y'(0) = c1.

    Uses the recurrence from the ODE:
        c_{n+3} = c_n / ((n+3)(n+2))
    split into two nonzero branches n ≡ 0,1 (mod 3).
    """
    x3 = x * x * x
    y = 0.0
    kmax = 50
    tol = 1e-16

    # branch n0 = 0: n = 0, 3, 6, ...
    n = 0.0
    term = c0
    y += term
    for _ in range(kmax):
        denom = (n + 3.0) * (n + 2.0)
        term *= x3 / denom
        y += term
        n += 3.0
        if math.fabs(term) < tol:
            break

    # branch n0 = 1: n = 1, 4, 7, ...
    n = 1.0
    term = c1 * x
    y += term
    for _ in range(kmax):
        denom = (n + 3.0) * (n + 2.0)
        term *= x3 / denom
        y += term
        n += 3.0
        if math.fabs(term) < tol:
            break

    return y


@njit(types.float64(types.float64), fastmath=True, cache=True)
def airy_ai(x):
    """
    Numba-friendly Airy Ai(x).

    - |x| <= 5: power series around 0
    - x  >  5: decaying asymptotic
    - x  < -5: oscillatory asymptotic
    """
    if x > 5.0:
        # Ai(x) ~ (1 / (2√π)) x^{-1/4} exp(-2/3 x^{3/2})
        t = (2.0 / 3.0) * (x ** 1.5)
        amp = 1.0 / (2.0 * math.sqrt(math.pi) * (x ** 0.25))
        return amp * math.exp(-t)
    elif x < -5.0:
        # Ai(x) ~ (1 / (√π |x|^{1/4})) * sin(2/3 |x|^{3/2} + π/4)
        z = -x
        t = (2.0 / 3.0) * (z ** 1.5)
        amp = 1.0 / (math.sqrt(math.pi) * (z ** 0.25))
        return amp * math.sin(t + 0.25 * math.pi)
    else:
        return _airy_series(x, AI0, AI0P)


@njit(types.float64(types.float64), fastmath=True, cache=True)
def airy_bi(x):
    """
    Numba-friendly Airy Bi(x).

    - |x| <= 5: power series around 0
    - x  >  5: growing asymptotic
    - x  < -5: oscillatory asymptotic
    """
    if x > 5.0:
        # Bi(x) ~ (1 / √π) x^{-1/4} exp(+2/3 x^{3/2})
        t = (2.0 / 3.0) * (x ** 1.5)
        amp = 1.0 / (math.sqrt(math.pi) * (x ** 0.25))
        return amp * math.exp(t)
    elif x < -5.0:
        # Bi(x) ~ (1 / (√π |x|^{1/4})) * cos(2/3 |x|^{3/2} + π/4)
        z = -x
        t = (2.0 / 3.0) * (z ** 1.5)
        amp = 1.0 / (math.sqrt(math.pi) * (z ** 0.25))
        return amp * math.cos(t + 0.25 * math.pi)
    else:
        return _airy_series(x, BI0, BI0P)
    
PI = math.pi
PI_OVER_2 = 0.5 * PI


@njit(types.float64(types.float64), fastmath=True, cache=True)
def fresnel_c(x):
    """
    Numba-friendly Fresnel C(x) = ∫_0^x cos(π t^2 / 2) dt

    - |x| <= 2: power series around 0
    - |x|  > 2: simple asymptotic (good qualitatively)
    """
    # C(x) is odd: C(-x) = -C(x)
    sign = 1.0
    if x < 0.0:
        sign = -1.0
        x = -x

    if x <= 2.0:
        # Power series:
        # C(x) = Σ_{k=0}^∞ (-1)^k ( (π/2)^{2k} x^{4k+1} ) / ( (2k)! (4k+1) )
        max_k = 10
        result = 0.0

        # Precompute some powers iteratively
        x2 = x * x
        x4 = x2 * x2
        p = 1.0                    # (π/2)^(2k)
        xpow = x                   # x^(4k+1), start at k=0 -> x^1
        sign_k = 1.0               # (-1)^k
        fact2k = 1.0               # (2k)! (start with 0! = 1)

        for k in range(max_k):
            term = sign_k * p * xpow / (fact2k * (4.0 * k + 1.0))
            result += term

            # Prepare next k
            # sign
            sign_k = -sign_k

            # (π/2)^{2(k+1)} = (π/2)^{2k} * (π/2)^2
            p *= (PI_OVER_2 * PI_OVER_2)

            # x^{4(k+1)+1} = x^{4k+1} * x^4
            xpow *= x4

            # (2(k+1))! from (2k)!:
            # multiply by (2k+1)*(2k+2)
            n1 = 2 * k + 1
            n2 = 2 * k + 2
            fact2k *= n1 * n2

        return sign * result

    # Asymptotic region: x > 2
    # Use a simple two-term asymptotic:
    # C(x) ≈ 1/2 + f(x)*sin(π x^2/2) - g(x)*cos(π x^2/2)
    # with f(x) ≈ 1/(π x), g(x) ≈ 1/(π^2 x^3)
    t = PI_OVER_2 * x * x
    sin_t = math.sin(t)
    cos_t = math.cos(t)
    f = 1.0 / (PI * x)
    g = 1.0 / (PI * PI * x * x * x)
    result = 0.5 + f * sin_t - g * cos_t

    return sign * result


@njit(types.float64(types.float64), fastmath=True, cache=True)
def fresnel_s(x):
    """
    Numba-friendly Fresnel S(x) = ∫_0^x sin(π t^2 / 2) dt

    - |x| <= 2: power series around 0
    - |x|  > 2: simple asymptotic (good qualitatively)
    """
    # S(x) is odd: S(-x) = -S(x)
    sign = 1.0
    if x < 0.0:
        sign = -1.0
        x = -x

    if x <= 2.0:
        # Power series:
        # S(x) = Σ_{k=0}^∞ (-1)^k ( (π/2)^{2k+1} x^{4k+3} ) / ( (2k+1)! (4k+3) )
        max_k = 10
        result = 0.0

        x2 = x * x
        x4 = x2 * x2
        p = PI_OVER_2              # (π/2)^(2k+1); start k=0 -> (π/2)^1
        xpow = x * x2              # x^(4k+3), start at k=0 -> x^3
        sign_k = 1.0               # (-1)^k
        fact2k1 = 1.0              # (2k+1)!; start at k=0 -> 1! = 1

        for k in range(max_k):
            term = sign_k * p * xpow / (fact2k1 * (4.0 * k + 3.0))
            result += term

            # Prepare next k
            sign_k = -sign_k
            p *= (PI_OVER_2 * PI_OVER_2)
            xpow *= x4

            # (2(k+1)+1)! from (2k+1)!:
            # multiply by (2k+2)*(2k+3)
            n1 = 2 * k + 2
            n2 = 2 * k + 3
            fact2k1 *= n1 * n2

        return sign * result

    # Asymptotic region: x > 2
    # S(x) ≈ 1/2 - f(x)*cos(π x^2/2) - g(x)*sin(π x^2/2)
    t = PI_OVER_2 * x * x
    sin_t = math.sin(t)
    cos_t = math.cos(t)
    f = 1.0 / (PI * x)
    g = 1.0 / (PI * PI * x * x * x)
    result = 0.5 - f * cos_t - g * sin_t

    return sign * result

# Tolerance & max iterations for the alternating series
ZETA_TOL = 1e-15
ZETA_MAX_N = 200


@njit(types.float64(types.float64), fastmath=True, cache=True)
def zeta(s):
    """
    Numba-safe Riemann zeta ζ(s) for real s with s > 0, s ≠ 1.

    Uses the alternating Dirichlet eta series:
        η(s) = Σ_{n>=1} (-1)^{n-1} / n^s
        ζ(s) = η(s) / (1 - 2^{1-s})
    """
    # outside domain: just return NaN
    if s <= 0.0:
        return math.nan

    # zeta has a pole at s=1
    if abs(s - 1.0) < 1e-10:
        # large placeholder instead of +∞
        return 1e30

    # compute eta(s)
    eta = 0.0
    sign = 1.0
    for n in range(1, ZETA_MAX_N + 1):
        term = sign / (n ** s)
        eta += term
        if math.fabs(term) < ZETA_TOL:
            break
        sign = -sign

    # denominator 1 - 2^{1-s}
    denom = 1.0 - (2.0 ** (1.0 - s))
    return eta / denom


@njit(types.float64(types.float64), fastmath=True, cache=True)
def lambertw(x):
    """
    Real principal branch of Lambert W, W0(x), for x >= -1/e.

    Uses Halley's method with a simple initial guess.
    Suitable for maps, not for hardcore special-function work.
    """
    # domain check: real W0 exists for x >= -1/e
    x_min = -1.0 / math.e
    if x < x_min:
        return math.nan

    if x == 0.0:
        return 0.0

    # initial guess
    if x < 1.0:
        w = x        # near zero, W(x) ~ x
    else:
        w = math.log(x) - math.log(math.log(x + 1.0))

    for _ in range(40):
        e = math.exp(w)
        we = w * e
        f = we - x            # f(w) = w*e^w - x

        if math.fabs(f) < 1e-14:
            break

        # avoid division near w = -1
        wp1 = w + 1.0
        if math.fabs(wp1) < 1e-7:
            wp1 = 1e-7 if wp1 >= 0.0 else -1e-7

        # Halley step
        denom = e * wp1 - (wp1 + 1.0) * f / (2.0 * wp1)
        w = w - f / denom

    return w

@njit(types.float64(types.float64, types.float64), fastmath=True, cache=True)
def gammainc(a, x):
    """
    Regularized lower incomplete gamma P(a, x) for a > 0, x >= 0.

        P(a,x) = γ(a,x) / Γ(a)

    Uses:
    - series for x < a + 1
    - continued fraction for x >= a + 1
    """
    if a <= 0.0 or x < 0.0:
        return math.nan

    if x == 0.0:
        return 0.0

    # ln Γ(a)
    gln = math.lgamma(a)

    # Series representation
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if math.fabs(delta) < math.fabs(summ) * 1e-15:
                break
        return summ * math.exp(-x + a * math.log(x) - gln)

    # Continued fraction for Q(a,x) = Γ(a,x)/Γ(a); then P = 1 - Q
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if math.fabs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if math.fabs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if math.fabs(delta - 1.0) < 1e-15:
            break

    Q = math.exp(-x + a * math.log(x) - gln) * h
    P = 1.0 - Q
    return P

SQRT_PI = math.sqrt(math.pi)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def dawson(x):
    """
    Dawson's integral F(x):

        F(x) = e^{-x^2} ∫_0^x e^{t^2} dt

    Piecewise:
    - |x| < 0.5: power series
    - |x| >= 0.5: asymptotic expansion
    """
    ax = x if x >= 0.0 else -x

    # Near zero: F(x) = x - 2/3 x^3 + 4/15 x^5 - 8/105 x^7 + ...
    if ax < 0.5:
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        return (x
                - (2.0 / 3.0) * x * x2
                + (4.0 / 15.0) * x * x4
                - (8.0 / 105.0) * x * x6
                + (16.0 / 945.0) * x * x8)

    # Asymptotic: F(x) ~ 1/(2x) + 1/(4x^3) + 3/(8x^5) + 15/(16x^7) + ...
    ax2 = ax * ax
    ax3 = ax * ax2
    ax5 = ax3 * ax2
    ax7 = ax5 * ax2

    f = (1.0 / (2.0 * ax)
         + 1.0 / (4.0 * ax3)
         + 3.0 / (8.0 * ax5)
         + 15.0 / (16.0 * ax7))

    return f if x >= 0.0 else -f

@njit(types.float64(types.float64), fastmath=True, cache=True)
def erfi(x):
    """
    Imaginary error function erfi(x), via Dawson:

        erfi(x) = 2 / sqrt(pi) * e^{x^2} * F(x)
    """
    return (2.0 / SQRT_PI) * math.exp(x * x) * dawson(x)

@njit(types.float64(types.int64, types.float64), fastmath=True, cache=True)
def legendre(n, x):
    """
    Legendre polynomial P_n(x), n >= 0, |x| <= 1 recommended.

    Recurrence:
        P_0(x) = 1
        P_1(x) = x
        (n+1) P_{n+1}(x) = (2n+1)x P_n(x) - n P_{n-1}(x)
    """
    if n < 0:
        return math.nan
    if n == 0:
        return 1.0
    if n == 1:
        return x

    Pnm1 = 1.0   # P_0
    Pn = x       # P_1

    for k in range(1, n):
        kf = float(k)
        Pnp1 = ((2.0 * kf + 1.0) * x * Pn - kf * Pnm1) / (kf + 1.0)
        Pnm1 = Pn
        Pn = Pnp1

    return Pn

# ---------------------------------------------------------------------------
# build python function text
# ---------------------------------------------------------------------------

def _funtext_1d(name:str ,expr:str, dict) -> str:
    lines = [
        f"def {name}(x, forced):",
    ]
    for i, (key, value) in enumerate(dict.items()):
        if not isinstance(key, str):
            raise TypeError(f"Only str keys supported, got {type(key)!r}")
        if not key.isidentifier():
            raise ValueError(f"Key {key!r} is not a valid Python identifier")
        lines.extend([
        f"    {key} = {value}"
        ])
    lines.extend([
        f"    x_next = {expr}",
        f"    return x_next"
    ])
    source = "\n".join(lines)
    return source

def _funtext_2d_ab_step(name: str, expr_x: str, expr_y: str, dict) -> str:
    lines = [
        f"def {name}(x, y, forced):",
    ]
    for i, (key, value) in enumerate(dict.items()):
        if not isinstance(key, str):
            raise TypeError(f"Only str keys supported, got {type(key)!r}")
        if not key.isidentifier():
            raise ValueError(f"Key {key!r} is not a valid Python identifier")
        lines.extend([
        f"    {key} = {value}"
        ])
    lines.extend([
        f"    x_next = {expr_x}",
        f"    y_next = {expr_y}",
        f"    return x_next, y_next",
    ])
    source = "\n".join(lines)
    return source

def _funtext_2d_ab_jac(
     name: str, dXdx: str, dXdy: str, dYdx: str, dYdy: str, dict
) -> str:
    lines = [
        f"def {name}(x, y, forced):",
    ]
    for i, (key, value) in enumerate(dict.items()):
        if not isinstance(key, str):
            raise TypeError(f"Only str keys supported, got {type(key)!r}")
        if not key.isidentifier():
            raise ValueError(f"Key {key!r} is not a valid Python identifier")
        lines.extend([
        f"    {key} = {value}"
        ])
    lines.extend([
        f"    dxdx = {dXdx}",
        f"    dxdy = {dXdy}",
        f"    dydx = {dYdx}",
        f"    dydy = {dYdy}",
        f"    return dxdx, dxdy, dydx, dydy",
    ])
    source = "\n".join(lines)
    return source

def _funtext_2d_step(name: str, expr_x: str, expr_y: str, dict) -> str:
    lines = [
        f"def {name}(x, y, first, second):",
    ]
    for i, (key, value) in enumerate(dict.items()):
        if not isinstance(key, str):
            raise TypeError(f"Only str keys supported, got {type(key)!r}")
        if not key.isidentifier():
            raise ValueError(f"Key {key!r} is not a valid Python identifier")
        lines.extend([
        f"    {key} = {value}"
        ])
    lines.extend([
        f"    x_next = {expr_x}",
        f"    y_next = {expr_y}",
        f"    return x_next, y_next",
    ])
    source = "\n".join(lines)
    return source

def _funtext_2d_jac(
     name: str, dXdx: str, dXdy: str, dYdx: str, dYdy: str, dict
) -> str:
    lines = [
        f"def {name}(x, y, first, second):",
    ]
    for i, (key, value) in enumerate(dict.items()):
        if not isinstance(key, str):
            raise TypeError(f"Only str keys supported, got {type(key)!r}")
        if not key.isidentifier():
            raise ValueError(f"Key {key!r} is not a valid Python identifier")
        lines.extend([
        f"    {key} = {value}"
        ])
    lines.extend([
        f"    dxdx = {dXdx}",
        f"    dxdy = {dXdy}",
        f"    dydx = {dYdx}",
        f"    dydy = {dYdy}",
        f"    return dxdx, dxdy, dydx, dydy",
    ])
    source = "\n".join(lines)
    return source

# ---------------------------------------------------------------------------
# build python functions from text
# ---------------------------------------------------------------------------

NS = {
    "step": step,
    "Heaviside": Heaviside,
    "DiracDelta": DiracDelta,
    "sign": sign,
    "Abs": Abs,
    "abs": Abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sec": sec,
    "cosh": np.cosh,
    "sinh": np.sinh,
    "exp": np.exp,
    "pow": np.power,
    "apow": apow,
    "log": np.log,
    "mod1": mod1,
    "Mod": Mod,
    "Derivative": Derivative,
    "re": re,
    "im": im,
    "pi": np.pi,
    "max": max,
    "min": min,
    "floor": floor,
    "ceil": ceil,
    "abs_cap": abs_cap,
    "np": np,
    "j0s": j0s,
    "j0": j0,
    "j1": j1,
    "i0": i0,
    "i1": i1,
    "lgamma": math.lgamma,
    "airy_ai": airy_ai,
    "airy_bi": airy_bi,
    "fresnel_c": fresnel_c,
    "fresnel_s": fresnel_s,
    "zeta": zeta,
    "lambertw": lambertw,
    "gammainc": gammainc,
    "dawson": dawson,
    "legendre": legendre,
    "math": math,
    "cmath": cmath,
}

# 1D forced step + deriv
def _funpy_1d(expr: str, dict):
    ns = NS.copy()
    src = _funtext_1d("impl",expr, dict)
    exec(src, ns, ns)
    return ns["impl"]

# 2D forced step 
def _funpy_2d_ab_step(expr_x: str, expr_y: str, dict):
    ns = NS.copy()
    src = _funtext_2d_ab_step("impl2_step", expr_x, expr_y, dict)
    exec(src, ns, ns)
    return ns["impl2_step"]


# 2D forced jacobian
def _funpy_2d_ab_jac(dXdx, dXdy, dYdx, dYdy,dict):
    ns = NS.copy()
    src = _funtext_2d_ab_jac("impl2_jac", dXdx, dXdy, dYdx, dYdy, dict)
    exec(src, ns, ns)
    return ns["impl2_jac"]

# 2D 
def _funpy_2d_step(expr_x: str, expr_y: str, dict):
    ns = NS.copy()
    src = _funtext_2d_step("impl2_step", expr_x, expr_y, dict)
    exec(src, ns, ns)
    return ns["impl2_step"]

# 2D jacobian
def _funpy_2d_jac(dXdx, dXdy, dYdx, dYdy,dict):
    ns = NS.copy()
    src = _funtext_2d_jac("impl2_jac", dXdx, dXdy, dYdx, dYdy, dict)
    exec(src, ns, ns)
    return ns["impl2_jac"]

# ---------------------------------------------------------------------------
# jit function
# ---------------------------------------------------------------------------

# All jitted step/deriv functions will share these signatures
#

STEP_SIG = types.float64(
    types.float64,   # x, the mapped variable
    types.float64,   # forced
)

STEP2_AB_SIG = types.UniTuple(types.float64, 2)(
    types.float64,  # x, mapped variable
    types.float64,  # y, mapped variable
    types.float64,  # forced
)

JAC2_AB_SIG = types.UniTuple(types.float64, 4)(
    types.float64,  # x, mapped variable
    types.float64,  # y, mapped variable
    types.float64,  # forced
)

STEP2_SIG = types.UniTuple(types.float64, 2)(
    types.float64,  # x, mapped variable
    types.float64,  # y, mapped variable
    types.float64,  # first
    types.float64,  # second
)

JAC2_SIG = types.UniTuple(types.float64, 4)(
    types.float64,  # x, mapped variable
    types.float64,  # y, mapped variable
    types.float64,  # first
    types.float64,  # second
)


def _funjit_1d(expr:str, dict):
    fun = _funpy_1d(expr,dict)
    jit = njit(STEP_SIG, cache=False, fastmath=False)(fun)
    return jit

def _funjit_2d_ab_step(xexpr:str, yexpr:str, dict):
    fun = _funpy_2d_ab_step(xexpr, yexpr, dict)
    jit = njit(STEP2_AB_SIG, cache=False, fastmath=False)(fun)
    return jit

def _funjit_2d_ab_jag(dxdx:str, dxdy:str, dydx:str, dydy:str, dict):
    fun = _funpy_2d_ab_jac( dxdx, dxdy, dydx, dydy, dict )
    jit = njit(JAC2_AB_SIG, cache=False, fastmath=False)(fun)
    return jit

def _funjit_2d_step(xexpr:str, yexpr:str, dict):
    fun = _funpy_2d_step(xexpr, yexpr, dict)
    jit = njit(STEP2_SIG, cache=False, fastmath=False)(fun)
    return jit

def _funjit_2d_jag(dxdx:str, dxdy:str, dydx:str, dydy:str, dict):
    fun = _funpy_2d_jac( dxdx, dxdy, dydx, dydy, dict )
    jit = njit(JAC2_SIG, cache=False, fastmath=False)(fun)
    return jit

# ---------------------------------------------------------------------------
# Map templates: add / tweak here to define all maps
# ---------------------------------------------------------------------------

# type "step1d"    : 1 parameter and 1d derivative (no jacobian!)
# type "step2d_ab" : 1 parameter and 2d derivative (with jacobian)
# type "step2d"    : 2 parameters and 2d derivative (with jacobian)

# 1-parameter maps are converted into 2d by forcing
# 2-parameter maps need no forcing

# parameters are static or scanned (scan: means scan over the domain)
# scanned parameters are changed by the field calculators
# scanned parameters always come first
# depending on the field calculator, either the first
# or both the second and the first parameters are
# scanned


MAP_TEMPLATES: dict[str, dict] = {

    "cardiac": dict(
        type   = "step2d",
        domain=[20.0, 0.0, 140.0, 150.0],
        pardict=dict(
            tmin   = "second",
            r_p    = "first",
            A      = 270.0,
            B1     = 2441,
            B2     = 90.02, 
            tau1   = 19.6,
            tau2   = 200.5,
            r_eff  = "max(r_p,1e-12)",
            k      = "math.floor((tmin + x) / r_eff) + 1.0",
            t      = "k * r_eff - x",
            F1     = "abs_cap(-t/tau1,50)",
            e1     = "exp(F1)",
            F2     = "abs_cap(-t/tau2,50)", 
            e2     = "exp(F2)",
        ),
        expr_x = "A - B1 * e1 - B2 * e2",
        expr_y = "y",
        jac_exprs=(
            "-(B1 / tau1) * e1 - (B2 / tau2) * e2",
            "0.0",
            "0.0",
            "0.0"
        ),
        x0=5.0,       # initial x_n (duration)
        y0=0.0,       # dummy y
        trans=100,    # n_prev
        iter=200,     # n_max
        eps_floor=1e-16,
    ),
    
    "predprey": dict(
        type="step2d_ab",
        expr_x="abs_cap(r * x * (1.0 - x - y),1e6)",
        expr_y="abs_cap(b * x * y,1e6)",
        jac_exprs=(
            "r * (1.0 - 2.0 * x - y)", # dXdx
            "-r * x", # dXdy
            "b * y",  # dYdx
            "b * x",  # dYdy
        ),
        domain=[-0.04, -0.6, 4.5, 6.6],
        pardict=dict(
            r="forced",
            b=3.569985,
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=200,
        eps_floor=1e-16,
    ),

    "parasite": dict(
        # parasite, k:3.1, rgb:mh:0.25:red:black:yellow
        type="step2d",
        expr_x="abs_cap(x * exp( abs_cap( r * (1.0 - x/K) - a*y, 50.0 ) ),1e6)",
        expr_y="abs_cap(x * (1.0 - exp( abs_cap( -a*y, 50.0 ) ) ),1e6)",
        jac_exprs=(
        "exp( abs_cap( r * (1.0 - x/K) - a*y, 50.0 ) ) * (1.0 - r * x / K)",
        "-a * x * exp( abs_cap( r * (1.0 - x/K) - a*y, 50.0 ) )",
        "1.0 - exp( abs_cap( -a*y, 50.0 ) )",
        "x * a * exp( abs_cap( -a*y, 50.0 ) )",
        ),
        domain=[-0.1, -0.1, 4.0, 7.0],
        pardict=dict(
            r="first",  
            a="second", 
            K=3.1,
        ),
        x0=0.5,
        y0=0.5,
        trans=200,
        iter=800,
        eps_floor=1e-16,
    ),

    "kicked": dict(
        # kicked:BA, ll:-3.215:-2.45, ul:-6.35:0.9325, lr:1.4325:1.85744, a:0.3, b:3
        type="step2d_ab",
        expr_x=(
            "mod1("
            "x + s*(1 + ((1-exp(-b))/b)*y) "
            "+ a*s*((1-exp(-b))/b)*cos(2*pi*x)"
            ")"
        ),
        expr_y="exp(-b)*(y + a*cos(2*pi*x))",
        jac_exprs=(
            "1 - a*s*((1-exp(-b))/b)*2*pi*sin(2*pi*x)",  # dXdx
            "s*((1-exp(-b))/b)",                         # dXdy
            "-exp(-b)*a*2*pi*sin(2*pi*x)",               # dYdx
            "exp(-b)",                                   # dYdy
        ),
        domain=[-2.45, -6.35, 1.85744, 1.4325],
        pardict=dict(
            s="forced",
            a=0.3,
            b=3.0,
        ),
        x0 = 0.4,
        y0 = 0.4,
        trans = 100,
        iter = 200,
    ),

    "henzi": dict( #henon-lozi map
        # fig 9.11 :  henzi:BA,ll:1.483:2.35,ul:2.15:1.794,lr:-0.35:0.15,gamma:0.25
        type="step2d_ab",
        # FIXME: add a default "seq" key
        # seq="AB" or somthing
        expr_x="1-s*abs(x)+y",
        expr_y="a*x",
        domain=[-10,-10,10,10],
        pardict=dict(
            s="forced",
            a=0.994,
        ),
        x0 = 0.4,
        y0 = 0.4,
        trans = 100,
        iter = 200,
    ),

    "adbash": dict( #adams-bashworth
        type="step2d_ab",
        expr_x="x+(h/2)*(3*r*x*(1-x)-r*y*(1-y))",
        expr_y="x",
        domain=[-5,-5,5,5],
        pardict=dict(
            r  = "forced",
            h  = 1.0,
        ),
        x0 = 0.5,
        y0 = 0.5,
        trans = 200,
        iter = 200,
    ),

    "degn_nn": dict( #NN's degn's map
        type="step2d",
        expr_x="s*(x-0.5)+0.5+a*sin(2*pi*r*y) ",
        expr_y="y+s*(x-0.5)+0.5+a*sin(2*pi*r*y)*mod1(b/s)",
        domain=[-2,-5,2,5],
        pardict=dict(
            r  = "first",
            s  = "second",
            a  = 0.1,
            b  = 1e8,
        ),
        x0    = 0.4,
        y0    = 0.4,
        trans = 100,
        iter  = 300,
    ),

    "degn": dict(  # Degn's map, Eqs. (9.20, 9.21)
        type="step2d",
        expr_common=dict(
            x1="c*(x - 0.5) + 0.5 + rho*sin(2*pi*r*y)"
        ),
        expr_x="{x1}",
        expr_y="Mod(y + ({x1}), k/b)",
        jac_exprs=(
            "c",                                   # dXdx
            "2*pi*rho*r*cos(2*pi*r*y)",            # dXdy
            "c",                                   # dYdx
            "1 + 2*pi*rho*r*cos(2*pi*r*y)",        # dYdy
        ),
        domain=[0.2, -1.15, 2.9, 1.15],
        pardict=dict(
            r="first",    # vertical axis: r
            b="second",   # horizontal axis: b
            rho=0.1,      # fixed R
            k=1.0,
            c="b",     
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=300,
    ),

    "eqn941": dict(  # Eqs. (9.41, 9.42)
        type="step2d",
        expr_x="Mod( x + 2*pi*k + b*sin(x) + r*cos(y) , 2*pi )",
        expr_y="Mod( y + 2*pi*omega, 2*pi )",
        # Jacobian, treating 'mod' as identity for derivatives
        jac_exprs=(
            "+1 + b*cos(x)",  # dXdx
            "-r * sin(y)",    # dXdy
            "0",              # dYdx
            "1",              # dYdy
        ),
        # Default rectangle matching the caption (b, r)
        domain=[0.2, -1.15, 2.9, 1.15],
        pardict=dict(
            r="first",     # vertical axis: r
            b="second",    # horizontal axis: b
            k=0.28,        # fixed R
            omega="(pow(5,0.5)-1)/2",
        ),
        x0=0.1,
        y0=0.5,
        trans=100,
        iter=300,
    ),

    "eqn941_ab": dict(  # Eqs. (9.41, 9.42)
        type="step2d_ab",
        expr_x="Mod( x + 2*pi*k + b*sin(x) + r*cos(y) , 2*pi )",
        expr_y="Mod( y + 2*pi*omega, 2*pi )",
        # Jacobian, treating 'mod' as identity for derivatives
        jac_exprs=(
            "+1 + b*cos(x)",  # dXdx
            "-r * sin(y)",    # dXdy
            "0",              # dYdx
            "1",              # dYdy
        ),
        # Default rectangle matching the caption (b, r)
        domain=[0.2, -1.15, 2.9, 1.15],
        pardict=dict(
            r="forced", # vertical axis: r
            b=1.075,    # horizontal axis: b
            k=0.28,        # fixed R
            omega="(pow(5,0.5)-1)/2",
        ),
        x0=0.1,
        y0=0.5,
        trans=100,
        iter=300,
    ),

    "henon": dict(
        type="step2d",
        # (x, y) -> (x', y'), using r,s as axis parameters
        expr_x="1 + y - r * x * x",  # Henon-like
        expr_y="s * x",
        domain=[1.0, 0.1, 1.4, 0.3],  # r0,s0,r1,s1
        pardict=dict(
            r  = "second",
            s  = "first",
        ),
        x0=0.1,
        y0=0.1,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
        # optionally: manual Jacobian override
        # jac_exprs=(" -2*r*x", "1", "s", "0"),
    ),

    "henon2": dict(
        type="step2d",
        # (x, y) -> (x', y'), using r,s as axis parameters
        expr_x="r - x*x + s*y",  # Henon-like
        expr_y="a*x+b*x*x",
        pardict=dict(
            r  = "second",
            s  = "first",
            a=1.0,
            b= 0.0,
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=2000,
        # optionally: manual Jacobian override
        # jac_exprs=(" -2*r*x", "1", "s", "0"),
    ),

    "kst2d": dict(
        type="step2d",
        # (x, y) -> (x', y'), using r,s as axis parameters
        expr_x="1+s*apow(x-c,r)-a*pow(abs(x),b)",  
        expr_y="0",
        domain=[1,1.33,3.5,2.5],  # r0,s0,r1,s1
        pardict=dict(
            r  = "second",
            s  = "first",
            a  = 3.0,
            b  = 2.0,
            c  = 0.0, 
        ),
        x0=0.5,
        y0=0.0,
        trans=100,
        iter=500,
        # optionally: manual Jacobian override
    ),

    "logistic2d": dict(
        type="step2d",
        # text is "r vs a"
        # s is the LHS of the text 
        # r is the RHS of the text
        expr_x="(1-s*x*x)*step(x)+(r-s*x*x)*(1-step(x))",  # Henon-like
        expr_y="y",
        domain=[0.66,-0.05,3,1.66],  # r0,s0,r1,s1
        pardict=dict(
            s  = "first",
            r  = "second",
        ),
        x0 = 0.5,
        y0 = 1.0,
        trans = 100,
        iter = 300,
        # optionally: manual Jacobian override
    ),

    "logistic": dict( # Classic logistic
        expr="r * x * (1.0 - x)",
        domain=[2.5, 2.5, 4.0, 4.0],  # A0, B0, A1, B1
        pardict=dict(r  = "forced"),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "sine": dict( # Sine map (classical Lyapunov variant: r sin(pi x))
        expr="r * sin(pi * x)",
        domain=[0.0, 2.0, 0.0, 2.0],
        pardict=dict(r  = "forced"),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "tent": dict(  # Tent map
        expr="r*x*(1-step(x-0.5)) + r*(1-x)*step(x-0.5)",
        domain=[0.0, 0.0, 2.0, 2.0],
        pardict=dict(r  = "forced"),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    
    "heart": dict( # Heart-cell map: x_{n+1} = sin(alpha x_n) + r_n
        expr="sin(a * x) + r",
        domain=[0.0, 0.0, 15.0, 15.0], 
        pardict=dict(
            r  = "forced",
            a  = 1.0,
        ),    
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "cardiac1d": dict(
        # Guevara cardiac map, eq. (8.4), AB‑forcing in r.
        #
        # Parameters (via a,b,c,d):
        #   a = A       (default 270)
        #   b = B1      (default 2441)
        #   c = B2      (default 90.02)
        #   d = t_min   (default 53.5)
        #
        # We use tau1 = 19.6, tau2 = 200.5 as in the caption.
        # t_n is encoded as:
        #   t_n = r - Mod(x + d, r) + d
        # so that k_n r - x_n > t_min with minimal integer k_n.
        expr=(
            "a"
            " - b*exp(-(r - Mod(x + d, r) + d)/19.6)"
            " - c*exp(-(r - Mod(x + d, r) + d)/200.5)"
        ),

        # Manual derivative d x_{n+1} / d x_n:
        #   = -(B1/tau1) * exp(-t/tau1) - (B2/tau2) * exp(-t/tau2)
        deriv_expr=(
            "-(b/19.6)*exp(-(r - Mod(x + d, r) + d)/19.6)"
            " - (c/200.5)*exp(-(r - Mod(x + d, r) + d)/200.5)"
        ),

        # Default AB‑plane window in r_A, r_B (you'll probably override).
        domain=[50,50,150,150],

        # [A, B1, B2, t_min] defaults from the text + t_min = 53.5
        pardict=dict(
            r  = "forced",
            a  = 270.0,
            b  = 2441.0,
            c  = 90.02, 
            d  = 53.5,
        ),

        x0=5.0,
        trans=100,   # n_prev
        iter=200,    # n_max
    ),

    "econ882": dict(  # Equations (8.19, 8.20), interdependent economies
        # x_{n+1} = μ x_n (1 - x_n) + γ y_n
        # y_{n+1} = μ y_n (1 - y_n) + γ x_n
        type="step2d_ab",
        expr_x="mu * x * (1.0 - x) + gamma * y",
        expr_y="mu * y * (1.0 - y) + gamma * x",
        jac_exprs=(
            "mu * (1.0 - 2.0 * x)",  # dXdx
            "gamma",                 # dXdy
            "gamma",                 # dYdx
            "mu * (1.0 - 2.0 * y)",  # dYdy
        ),
        # B vs A plane: (A,B) = (mu_A, mu_B)
        # LL:(2, 2.40625), UL:(2.24375, 2.65), LR:(2.40625, 2)
        domain=[2.0, 2.40625, 2.24375, 2.65, 2.40625, 2.0],
        pardict=dict(
            mu="forced",     # A/B-forced parameter μ
            gamma=0.43,      # γ₁ = γ₂ = 0.43
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=300,
        eps_floor=1e-16,
    ),

    "fishery": dict(  # Equations (8.23, 8.24)
        type="step2d",

        # x = D (crab biomass), y = P (pots)
        expr_x="x * exp(ad + bd*x + cd*y)",
        expr_y="y * exp(ap + bp*y + cp*x)",

        # Jacobian:
        # let g1 = aD + bD x + cD y; f1 = x e^{g1}
        # dXdx = e^{g1} + x e^{g1} bD = e^{g1} * (1 + bD x)
        # dXdy = x e^{g1} cD
        # let g2 = aP + bP y + cP x; f2 = y e^{g2}
        # dYdx = y e^{g2} cP
        # dYdy = e^{g2} + y e^{g2} bP = e^{g2} * (1 + bP y)
        jac_exprs=(
            "exp(ad + bd*x + cd*y) * (1 + bd*x)",       # dXdx
            "x * cd * exp(ad + bd*x + cd*y)",           # dXdy
            "y * cp * exp(ap + bp*y + cp*x)",           # dYdx
            "exp(ap + bp*y + cp*x) * (1 + bp*y)",       # dYdy
        ),

        # c_P vs c_D plane: (first, second) = (cD, cP)
        # LL:(-0.033, 0), UL:(-0.033, 0.105), LR:(0, 0)
        # → domain = [A0,B0,A1,B1] = [ -0.033, 0, 0, 0.105 ]
        domain=[-0.033, 0.0, 0.0, 0.105],

        pardict=dict(
            cd="first",   # horizontal axis
            cp="second",  # vertical axis
            ad=1.0,
            bd=-0.005,
            ap=0.5,
            bp=-0.04,
        ),

        x0=0.5,
        y0=0.5,
        trans=5,    # n_prev
        iter=20,    # n_max
        eps_floor=1e-16,
    ),

    "eq827": dict(  # Equation (8.27) – Angelini antiferromagnetic element
        # piecewise:
        # region1: x < -1/3
        # region2: -1/3 <= x <= 1/3
        # region3: x > 1/3
        expr=(
            # region1: x < -1/3  ->  step(-1/3 - x)
            "(-r/3.0) * exp(b*(x + 1.0/3.0)) * step(-1.0/3.0 - x) + "
            # region3: x > 1/3   ->  step(x - 1/3)
            "( r/3.0) * exp(b*(1.0/3.0 - x)) * step(x - 1.0/3.0) + "
            # region2: middle, complement of the two above
            "r*x * (1.0 - step(-1.0/3.0 - x) - step(x - 1.0/3.0))"
        ),

        # derivative df/dx (same partition, ignoring derivative of step discontinuities)
        deriv_expr=(
            # d/dx of region1 part
            "(-r/3.0) * b * exp(b*(x + 1.0/3.0)) * step(-1.0/3.0 - x) + "
            # d/dx of region3 part
            "(-r/3.0) * b * exp(b*(1.0/3.0 - x)) * step(x - 1.0/3.0) + "
            # d/dx of region2: r
            "r * (1.0 - step(-1.0/3.0 - x) - step(x - 1.0/3.0))"
        ),

        # Fig. 8.17 caption:
        # "Equation (8.27): b = 1. B versus A. r:B5AA B5AA..., nprev = 100, nmax = 200, x0 = 1.
        #  D-shading. LL:(−15, −4.3), UL:(−15, 11.), LR:(35, −4.3)"
        domain=[-15.0, -4.3, -15.0, 11.0],  # will be overridden by ll/ul/lr in spec

        pardict=dict(
            r="forced",  # A/B-forced parameter
            b=1.0,       # fixed b = 1 for this figure
        ),

        x0=1.0,
        trans=100,
        iter=200,
    ),

    "nn1": dict(
        expr="pow(r/abs(x),a)*sign(x)+cos(2*pi*r/2)*sin(2*pi*x/5)",
        #deriv_expr="0",
        domain=[0.1, 0.1, 10, 10], 
        pardict=dict(
            r  = "forced",
            a  = 0.25,
        ),
        x0=0.5,
        trans=600,
        iter=600,
    ),

    "nn1a": dict(
        expr="pow(r/abs(x),a)*sign(x)+pow(abs(cos(2*pi*r/2)*sin(2*pi*x/5)),b)",
        #deriv_expr="0",
        domain=[0.1, 0.1, 10, 10],  
        pardict=dict(
            r  = "forced",
            a  = 0.25,
            b  = 1.0,
        ),
        x0=0.5,
        trans=600,
        iter=600,
    ),

    "nn2": dict(
        expr="pow(r/abs(x),a*cos(r))*sign(x)+cos(2*pi*r/2.25)*sin(2*pi*x/3)",
        #deriv_expr="0",
        domain=[0.0, 0.0, 15.0, 15.0], 
        pardict=dict(
            r  = "forced",
            a  = 1.0,
        ),   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn3": dict(
        expr="cos(2*pi*r*x*(1-x)/a)*pow(abs(x),cos(2*pi*(r+x)/10))*sign(x)-cos(2*pi*r)*step(x)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 25.0,
        ),  
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn4": dict(
        expr="pow(abs(x*x*x-r),cos(r))*sign(x)+pow(abs(r*r*r-x*x*x),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
        ),   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn5": dict(
        expr="pow(abs(x*x*x-pow(r,a)),cos(r))*sign(x)+pow(abs(r*r*r-pow(x,b)),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 3.0,
            b  = 5.0,
        ), 
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn6": dict(
        expr="pow(abs(pow(x,b)-pow(r,a)),cos(r))*sign(x)+pow(abs(pow(r,a)-pow(x,b)),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 3.0,
            b  = 5.0,
        ),  
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn7": dict(
        expr="pow(abs(apow(x,b)-apow(r,a)),cos(r))*sign(x)+pow(abs(apow(r,a)-apow(x,b)),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 3.0,
            b  = 5.0,
        ),
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn8": dict(
        expr="apow(apow(x,b)-apow(r,a),cos(r))+apow(apow(r,a)-apow(x,b),sin(x))",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 3.0,
            b  = 5.0,
        ), 
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn9": dict(
        expr="r*cosh(sin(apow(x,x)))-x*sinh(cos(apow(r,r)))",
        expr1="r*apow(sin(pi*(x-b)),a)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 3.0,
            b  = 5.0,
        ),  
        x0=2.0,
        trans=600,
        iter=600,
    ),

     "nn10": dict(
        expr="c*apow(a*cos(2*pi*r*x*x),b*sin(2*pi*r*x))+d",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 1.0,
            b  = 1.0,
            c  = "x*(1-x)",
            d = 0.0
        ),  
        x0=2.0,
        trans=600,
        iter=600,
    ),

# python lyapunov1.py --spec \
# 'map:nn11:AB:-10:-10:10:10,
# c:x*(1-x)*cos(x)*cos(r)*sin(exp(x)),
# b:1*(cos(x-r))**2+0*cos(x-r),
# x0:0.5,trans:200,iter:200,
# rgb:mh_eq:1:seagreen:black:copper,
# hist:5:5:128' \           
# --pix 2000 --out tst7.png

    "nn11": dict(
        expr="a*(b+c)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a  = 1.0,
            b  = 1.0,
            c  = "x*(1-x)",
            d = 0.0
        ),  
        x0=2.0,
        trans=600,
        iter=600,
    ),

    # a:cos(2*pi*{0.1:0.4:4}),b:sin(2*pi*${1}*${1}),c:0.2*cos(2*pi*exp(${1})),d:{0.2:0.8:4},e:-0.5*${2},f:0
     "nn12": dict(
        expr="term1+term2",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            a = 0.0, # polynomial coefficients
            b = 0.0,
            c = 0.0,
            d = 1.0,
            e = 0.0,
            f = 0.0,
            v  = "cos(x-r)",
            term1  = "a*pow(v,5)+b*pow(v,4)+c*pow(v,3)+d*pow(v,2)+e*v+f",
            term2  = "x*(1-x)*cos(x)*cos(r)*sin(exp(x))",
        ),  
        x0=0.5,
        trans=200,
        iter=200,
    ),

     "nn13": dict(
        expr="final",
        deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            c8 = 1.0,
            c7 = 1.0, # polynomial coefficients
            c6 = 1.0, 
            c5 = 1.0, 
            c4 = 1.0,
            c3 = 1.0,
            c2 = 1.0,
            c1 = 1.0,
            c0 = 1.0,
            v  = "cos(x-r)",
            poly  = "c8*v**8+c7*v**7+c6*v**6+c5*v**5+c4*v**4+c3*v**3+c2*v**2+c1*v+c0",
            final = "exp(cos(poly)*sin(poly))",
        ),  
        x0=0.5,
        trans=200,
        iter=200,
    ),


    "nn14": dict(
        expr="np.real(m)",
        deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        pardict=dict(
            r  = "forced",
            l  = "r*x*(1-x)",
            m = "lgamma(l)*j1(l)*j0(l)*sin(l)*cos(l)*np.exp(x+1j*l)",
        ),  
        x0=0.5,
        trans=200,
        iter=200,
    ),

    "eq86": dict(
        expr="x + r*pow(abs(x),b)*sin(x)",
        # A8B8
        domain=[2,2,2.75,2.75],
        pardict=dict(
            r  = "forced",
            b  = 0.3334,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq826": dict(
        expr="x * exp((r/(1+x))-b)",
        # A8B8
        domain=[10,10,40, 40],
        pardict=dict(
            r  = "forced",
            b  = 11.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq95": dict(
        expr=" (1-r*x*x)*step(x)+(a-r*x*x)*(1-step(x))",
        # A8B8
        domain=[-0.5,-0.5,5,5],
        pardict=dict(
            r  = "forced",
            a  = 2.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq96": dict(
        expr=" r*x*(1-x)*step(x-0.5)+(r*x*(1-x)+(a-1)*(r-2)/4)*(1-step(x-0.5))",
        # A8B8
        domain=[2.5,2.5,4, 4],
        pardict=dict(
            r  = "forced",
            a  = 0.4,
        ),  
        x0=0.6,
        trans=100,
        iter=300,
    ),

    "dlog": dict( # same as eq96, but manual derivative to check sympy's derivation
        # Map step = eq. (9.6)
        expr="dlog",
        deriv_expr="r * (1.0 - 2.0 * (dlog))",
        domain=[2.5,2.5,4, 4],
        pardict=dict(
            r  = "forced",
            a  = 0.4,
            dlog = "r*x*(1-x)*step(x-0.5)+(r*x*(1-x)+0.25*(a-1)*(r-2))*(1-step(x-0.5))",
        ),  
        x0=0.6,
        trans=100,
        iter=300,
    ),


    "eq97": dict(
        expr=" a*x*(1-step(x-1))+b*pow(x,1-r)*step(x-1)",
        # A8B8
        domain=[2,0.5,10,1.5],
        pardict=dict(
            r  = "forced",
            a  = 50,
            b  = 50,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq98": dict(
        expr=" 1+r*apow(x,b)-a*apow(x,d)",
        # A8B8
        domain=[-0.25, -0.25, 1.25, 1.25],
        pardict=dict(
            r  = "forced",
            a  = 1.0,
            b  = 1.0,
            d  = 0.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq932": dict(
        expr=" mod1(r*x)",
        deriv_expr="r",      # <--- add this
        # A8B8
        domain=[-0.25, -0.25, 1.25, 1.25],
        pardict=dict(
            r  = "forced",
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq933": dict(
        expr=" 2*x*step(x)*(1-step(x-0.5))+((4*r-2)*x+(2-3*r))*step(x-0.5)*(1-step(x-1.0))",
        # A8B8
        domain=[-0.25, -0.25, 1.25, 1.25],
        pardict=dict(
            r  = "forced",
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq937": dict(
        expr="r * x * (1.0 - x) * step(x-0)*(1-step(x-r))+r*step(x-r)+0*(1-step(x))",
        domain=[0.0, 0.0, 5.0, 5.0],
        pardict=dict(
            r  = "forced",
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq947": dict(
        expr="b*(sin(x+r))**2",
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r  = "forced",
            b  = 1.7,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq948": dict(
        expr=(
            "b * ( sin(x + pow(r,mu) ) )**2"
            " + alpha * pow(r, k) * step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi))"
            " + beta  * pow(r, k) * (1 - step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi)))"
        ),
        deriv_expr = "2*b*sin(x + pow(r,mu))*cos(x + pow(r,mu))",
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r  = "forced",
            b  = 2.0,
            mu = 1.0,
            alpha  = 0.0,
            beta  = 0.0, 
            k  = 2.0,
            n = 1,            
            gamma=1.0,
        ),  
        x0=0.5,
        trans=200,
        iter=200,
    ),

     "eq948a": dict(
        expr=(
            "b * mod1(x + pow(r,mu) )"
            " + alpha * pow(r, k) * step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi))"
            " + beta  * pow(r, k) * (1 - step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi)))"
        ),
        deriv_expr = "2*b*sin(x + pow(r,mu))*cos(x + pow(r,mu))",
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r  = "forced",
            b  = 2.0,
            mu = 1.0,
            alpha  = 0.0,
            beta  = 0.0, 
            k  = 2.0,
            n = 1,            
            gamma=1.0,
        ),  
        x0=0.5,
        trans=200,
        iter=200,
    ),


    "eq948_2d": dict(
        type="step2d",
        expr_x=(
            "b * ( sin(x + pow(r,mu) ) )**2"
            " + alpha * pow(r, k) * step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi))"
            " + beta  * pow(r, k) * (1 - step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi)))"
        ),
        expr_y = "y",
        jac_exprs=(
            "2*b*sin(x + pow(r,mu))*cos(x + pow(r,mu))",  # dXdx
            "pow(r,k)*step(mu*pi/2 - Mod(x + pow(r,n), gamma*pi))", # dXdy
            "0", # dYdx
            "1", # dYdy
        ),
        domain=[0.0, 0.0, 10.0, 10.0],
        pardict=dict(
            r  = "first",
            b  = 2.0,
            mu = 1.0,
            alpha  = "second",
            beta  = 0.0, 
            k  = 2.0,
            n = 1,            
            gamma=1.0,
        ),  
        x0=0.5,
        trans=200,
        iter=200,
    ),

    "eq950": dict(
        # x_{n+1} = [cosh(r x_n)] mod (2/b)
        expr="Mod(cosh(r*x), 2/b)",

        # derivative wrt x, ignoring the outer Mod
        deriv_expr="r * sinh(r*x)",

        # default A/B window for the forced parameter r
        # (adjust from the spec to match the book's figure)
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r  = "forced",  # A/B sequence drives r
            b  = 1.0,       # override with b:... in the spec
        ),

        x0=0.5,
        trans=100,
        iter=300,
    ),

    "eq951": dict(
        expr_common=dict(
            S="sin(1-x)",
            C="cos((x-r)**2)",
        ),
        # x_{n+1} = b r exp(S^3 C) - 1
        expr="b * r * exp({S}**3 * {C}) - 1",

        # derivative f'(x) = b r exp(S^3 C) * ( -3 S^2 cos(1-x) C
        #                                     -2 (x-r) S^3 sin((x-r)^2) )
        deriv_expr=(
            "b * r * exp({S}**3 * {C}) * ("
            " -3*{S}**2 * cos(1-x) * {C}"
            " -2*(x-r) * {S}**3 * sin((x-r)**2)"
            ")"
        ),

        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced", # A/B‑forced parameter
            b  = 1.0,      # default b, override with b:... in the spec if needed
        ),
        x0=0.5,
        trans=100,
        iter=100,
    ),

    "eq952": dict(
        # x_{n+1} = b sin[(x_n - r)^3] e^{-(x_n - r)^2}
        expr=(
            "b * sin(pow(x - r, 3)) * exp(-pow(x - r, 2))"
        ),

        # derivative wrt x, ignoring any forcing / AB structure
        deriv_expr=(
            "b * exp(-pow(x - r, 2)) * ("
            "  3*pow(x - r, 2)*cos(pow(x - r, 3))"
            " - 2*(x - r)*sin(pow(x - r, 3))"
            ")"
        ),

        # choose a default (A,B) window for r_A,r_B;
        # tweak in the spec to match the book's figure
        domain=[-4.0, -4.0, 4.0, 4.0],

        pardict=dict(
            r  = "forced",   # driven parameter (A/B sequence)
            b  = 3.2,        # amplitude; override with b:...
        ),

        x0=0.5,
        trans=25,
        iter=50,
    ),

    "eq953": dict(
        # x_{n+1} = b sin^4(x_n - r)
        expr="b * pow(sin(x - r), 4)",

        # derivative wrt x (ignore any forcing structure)
        deriv_expr="4 * b * pow(sin(x - r), 3) * cos(x - r)",

        # default (A,B) window for r_A,r_B; tweak in spec if needed
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # driven parameter (A/B sequence)
            b=1.0,        # amplitude; override with b:... in spec
        ),

        x0=0.5,
        trans=400,
        iter=400,
    ),

    "eq954": dict(
        # x_{n+1} = cos(x_n + r) cos(1 - x_n)
        expr="cos(x + r) * cos(b - x)",

        # derivative wrt x
        deriv_expr=(
            "-sin(x + r) * cos(b - x)"
            " + cos(x + r) * sin(b - x)"
        ),

        # default (A,B) window for r_A, r_B – tune from spec as needed
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # r is the driven A/B parameter
            b=1.0,
        ),

        x0=0.5,
        trans=500,
        iter=1000,
    ),

    "eq955": dict(
        expr=(
            "b * pow(x - 1, 2) * pow(sin(r - x), 2)"
        ),
       deriv_expr=(
            "2*b*(x - 1)*pow(sin(r - x), 2)"
            " - 2*b*pow(x - 1, 2)*sin(r - x)*cos(r - x)"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",   # A/B sequence drives 'r'
            b=0.8,        # amplitude parameter (override with b:…)
        ),
        x0=0.5,
        trans=100,
        iter=500,
    ),

    "eq959": dict(  # Eq. (9.59)
        # x_{n+1} = (b + r) * exp(sin(1 - x)^3 * cos((x - r)^2)) - 1
        expr=(
            "(b + r) * exp(pow(sin(1 - x), 3) * cos(pow(x - r, 2))) - 1"
        ),

        # derivative wrt x
        deriv_expr=(
            "(b + r) * exp(pow(sin(1 - x), 3) * cos(pow(x - r, 2))) * ("
            " -3*pow(sin(1 - x), 2)*cos(1 - x)*cos(pow(x - r, 2))"
            " -2*(x - r)*pow(sin(1 - x), 3)*sin(pow(x - r, 2))"
            ")"
        ),

        # default A/B window (you’ll override via ll/ul/lr for this figure)
        domain=[-1.0, -1.0, 1.0, 1.0],

        pardict=dict(
            r="forced",   # r is A/B-forced
            b=0.6,        # for Fig. 9.152
        ),

        x0=0.5,
        trans=100,
        iter=200,
    ),

    "eq961": dict(  # Eq. (9.61)
        expr=(
            "b * cos(exp(-pow(x - r, 2)))"
        ),
        deriv_expr=(
            "2 * b * (x - r) * exp(-pow(x - r, 2)) "
            "* sin(exp(-pow(x - r, 2)))"
        ),
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r="forced",   # driven parameter (A/B sequence)
            b=5.0,        # amplitude; override with b:...
        ),
        x0=0.5,
        trans=25,
        iter=50,
    ),

    "eq962": dict(
        expr="b * r*r * exp( sin( pow(1 - x, 3) ) ) - 1",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq963": dict(
        expr="b * exp( pow( sin(1 - x), 3 ) ) + r",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq964": dict(
        expr="r * exp( -pow(x - b, 2) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq965": dict(
        expr="b * exp( sin(r * x) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq966": dict(
        expr="pow( abs(b*b - pow(x - r, 2)), 0.5 ) + 1",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq967": dict(
        expr="pow( b + pow( sin(r * x), 2 ), -1 )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq968": dict(
        expr="b * exp( r * pow( sin(x) + cos(x), -1 ) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.3,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq969": dict(
        expr="b * (x - r) * exp( -pow(x - r, 3) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq970": dict(
        # x_{n+1} = b * exp(cos(1 - x) * sin(pi/2) + sin(r))
        expr="b * exp( cos(1 - x) * sin(pi/2) + sin(r) )",

        # derivative wrt x:
        # f'(x) = b * exp(cos(1-x) + sin(r)) * sin(1-x)
        deriv_expr=(
            "b * exp( cos(1 - x) * sin(pi/2) + sin(r) ) * sin(1 - x)"
        ),

        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # A/B-driven
            b=1.0,        # override to 1.5 for this fig
        ),

        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq971": dict(  # Eq. (9.71)
        # x_{n+1} = b * r * exp( sin(x - r)^4 )
        expr=(
            "b * r * exp(pow(sin(x - r), 4))"
        ),

        # derivative wrt x:
        # let s = sin(x - r);  f(x) = b r e^{s^4}
        # f'(x) = b r e^{s^4} * 4 s^3 cos(x - r)
        deriv_expr=(
            "4 * b * r * exp(pow(sin(x - r), 4))"
            " * pow(sin(x - r), 3) * cos(x - r)"
        ),

        # default window – overridden by ll/ul/lr in specs
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # A/B-driven parameter r
            b=0.5,        # placeholder; will be overridden to 1.5
        ),

        x0=0.5,
        trans=25,
        iter=50,
    ),

    "eq972": dict(  # Eq. (9.72)
        # x_{n+1} = b * r * exp( sin(1 - x)^3 )
        expr=(
            "b * r * exp(pow(sin(1 - x), 3))"
        ),

        # derivative wrt x
        deriv_expr=(
            "-3 * b * r * exp(pow(sin(1 - x), 3))"
            " * pow(sin(1 - x), 2) * cos(1 - x)"
        ),

        # default A/B window for r_A, r_B (tune from spec to match the plate)
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # driven parameter (A/B sequence)
            b=1.0,        # amplitude; override with b:... in spec
        ),

        x0=0.5,
        trans=100,
        iter=300,
    ),

    "eq973": dict(  # Eq. (9.73)
        # x_{n+1} = b * r * sin^2(b x + r^2) * cos^2(b x - r^2)
        expr=(
            "b * r * pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2)"
        ),

        # derivative wrt x
        deriv_expr=(
            "2*pow(b, 2)*r*("
            " sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")"
        ),

        # default (A,B) window for (r_A,r_B); you'll override via ll/ul/lr
        domain=[-2.5, -2.5, 2.5, 2.5],

        pardict=dict(
            r="forced",   # A/B-driven parameter r
            b=1.1,        # for Fig. 9.141; override with b:... if desired
        ),

        x0=0.5,
        trans=125,
        iter=250,
    ),

    "eq974": dict(
        expr="pow( abs(r*r - pow(x - b, 2)), 0.5 )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq975": dict(
        expr="b*cos(x-r)*sin(x+r)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq976": dict(
        expr="(x-r)*sin( pow(x-b,2))",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq977": dict(
        expr="r*sin(pi*r)*sin(pi*x)*step(x-0.5)+b*r*sin(pi*r)*sin(pi*x)*step(0.5-x)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq978": dict(  # Eq. (9.78)
        # x_{n+1} = r * sin(pi*r) * sin(pi*(x - b))
        expr=(
            "r * sin(pi*r) * sin(pi*(x - b))"
        ),

        # derivative wrt x (r treated as parameter)
        deriv_expr=(
            "r * sin(pi*r) * pi * cos(pi*(x - b))"
        ),

        # A/B window: A,B ∈ [0,2] (LL:(0,0), UL:(0,2), LR:(2,0))
        domain=[0.0, 0.0, 2.0, 2.0],

        pardict=dict(
            r="forced",   # A/B-driven parameter
            b=0.5,        # fixed b for Fig. 9.147 (override with b:... if you like)
        ),

        x0=0.5,
        trans=500,
        iter=1000,
    ),

    "eq979_old": dict(
        type="step2d",

        # (x, y) -> (x', y'), parameters (first, second) = (b, r)
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - r"
        ),
        expr_y="y",   # dummy y-dimension

        # Jacobian: only dXdx is non-zero
        jac_exprs=(
            "2*pow(b, 2)*r*("
            "  sin(b*x + r*r)*cos(b*x + r*r)"
            "  *pow(cos(b*x - r*r - r), 2)"
            " - pow(sin(b*x + r*r), 2)"
            "  *sin(b*x - r*r - r)*cos(b*x - r*r - r)"
            ")",  # dXdx
            "0",   # dXdy
            "0",   # dYdx
            "1",   # dYdy
        ),

        # (b,r) window matching the caption: LL:(0,0), UL:(0,7.66), LR:(4.3,0)
        # i.e. b in [0,4.3], r in [0,7.66]
        domain=[0.0, 0.0, 4.3, 7.66],

        pardict=dict(
            b="first",   # horizontal axis: b
            r="second",  # vertical axis: r
        ),

        x0=0.5,
        y0=0.0,
        trans=100,   # n_prev
        iter=200,    # n_max
    ),

    "eq979": dict(
        type="step2d",

        # x_{n+1} = b*r*sin^2(b*x + r^2)*cos^2(b*x - r^2) - r
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - r"
        ),
        expr_y="0",  # dummy y-dimension

        jac_exprs=(
            # dXdx
            "2*pow(b, 2)*r*("
            "  sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")",
            "0",  # dXdy
            "0",  # dYdx
            "0",  # dYdy
        ),

        # (b,r) window as before (from caption)
        # LL:(0,0), UL:(0,7.66), LR:(4.3,0)
        domain=[0.0, 0.0, 4.3, 7.66],

        pardict=dict(
            b="first",   # horizontal axis
            r="second",  # vertical axis
        ),

        x0=0.5,
        y0=0.0,
        trans=100,
        iter=200,
    ),

    "eq979_ab": dict(
        type="step2d_ab",

        # x_{n+1} = b*r*sin^2(b*x + r^2)*cos^2(b*x - r^2) - r
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - r"
        ),
        expr_y="0",  # dummy y-dimension

        jac_exprs=(
            # dXdx
            "2*pow(b, 2)*r*("
            "  sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")",
            "0",  # dXdy
            "0",  # dYdx
            "0",  # dYdy
        ),

        # (b,r) window as before (from caption)
        # LL:(0,0), UL:(0,7.66), LR:(4.3,0)
        domain=[0.0, 0.0, 4.3, 7.66],

        pardict=dict(
            r="forced",  # vertical axis
            b=1,   # horizontal axis
            
        ),

        x0=1.5,
        y0=0.0,
        trans=100,
        iter=200,
    ),

    "eq980": dict(
        type="step2d",

        # (x, y) -> (x', y'), parameters (first, second) = (r, b)
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - 1"
        ),
        expr_y="0",   # dummy y-dimension

        # Jacobian: only dXdx is non-zero; y-dimension is dead
        jac_exprs=(
            "2*pow(b, 2)*r*("  # dXdx
            "  sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")",
            "0",  # dXdy
            "0",  # dYdx
            "0",  # dYdy
        ),

        # default rectangle for (r,b); you’ll override via ll/ul/lr
        domain=[0.26, 1.36, 1.44, 3.85],

        pardict=dict(
            r="first",   # horizontal axis: r
            b="second",  # vertical axis: b
        ),

        x0=0.5,
        y0=0.0,
        trans=100,   # n_prev
        iter=200,    # n_max
    ),

    "eq980_ab": dict(
        type="step2d_ab",

        # (x, y) -> (x', y'), parameters (first, second) = (r, b)
        expr_x=(
            "b*r*pow(sin(b*x + r*r), 2) * pow(cos(b*x - r*r), 2) - 1"
        ),
        expr_y="y",   # dummy y-dimension

        # Jacobian: only dXdx is non-zero; y-dimension is dead
        jac_exprs=(
            "2*pow(b, 2)*r*("  # dXdx
            "  sin(b*x + r*r)*cos(b*x + r*r)*pow(cos(b*x - r*r), 2)"
            " - pow(sin(b*x + r*r), 2)*sin(b*x - r*r)*cos(b*x - r*r)"
            ")",
            "0",  # dXdy
            "0",  # dYdx
            "1",  # dYdy
        ),

        # default rectangle for (r,b); you’ll override via ll/ul/lr
        domain=[0.26, 1.36, 1.44, 3.85],

        pardict=dict(
            r="forced",   # horizontal axis: r
            b=0.9,  # vertical axis: b
        ),

        x0=0.5,
        y0=0.0,
        trans=100,   # n_prev
        iter=200,    # n_max
    ),



    "eq981": dict(
        # x_{n+1} = b / ( 2 + sin( (x mod 1) - r ) )
        expr=(
            "b * pow(2 + sin(mod1(x) - r), -1)"
        ),

        # derivative wrt x, treating mod1 as locally identity:
        # f'(x) = -b * cos((x mod 1) - r) / (2 + sin((x mod 1) - r))^2
        deriv_expr=(
            "-b * cos(mod1(x) - r) * pow(2 + sin(mod1(x) - r), -2)"
        ),

        # default A/B window (you'll override per-figure)
        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # A/B sequence drives r
            b=1.0,        # will override to b=2 for Fig. 9.153
        ),

        x0=0.5,
        trans=100,
        iter=200,
    ),

    "eq982": dict(
        expr="b*r*exp(exp(exp(x*x*x)))",
        domain=[0.0, 2.0, 0.0, 2.0],
        pardict=dict(
            r  = "forced",
            b  = 0.1,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq983": dict(
        expr="b*r* exp(pow(sin(1-x*x),4))",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
  
    "eq984": dict(
        expr="r*(sin(x)+b*sin(9.0*x))",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq985": dict(
        # x_{n+1} = b * exp(tan(r*x) - x)
        expr="b * exp(tan(r*x) - x)",

        # derivative wrt x:
        # f'(x) = b * exp(tan(r*x) - x) * (r * sec^2(r*x) - 1)
        deriv_expr=(
            "b * exp(tan(r*x) - x) * (r * pow(sec(r*x), 2) - 1)"
        ),

        domain=[0.0, 0.0, 4.0, 4.0],

        pardict=dict(
            r="forced",   # A/B-forced parameter
            b=1.0,        # override via b:...
        ),

        x0=0.5,
        trans=100,      # caption uses n_prev = 100
        iter=200,       # caption uses n_max = 200
    ),

    "eq986": dict(
        expr="b*exp(cos(x*x*x*r-b)-r)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

}

# Lazy map cache: compiled maps live here, built on first use.
MAPS: dict[str, dict] = {}

def substitute_common(x,d):
    if d is None:
        return x 
    x=x.format(**d)
    return x


def _build_map(name: str) -> dict:
    """
    Build (compile) a single map configuration from MAP_TEMPLATES[name].
    This does the same work the old _build_maps() did, but for one name.
    """
    if name not in MAP_TEMPLATES:
        raise KeyError(f"Unknown map '{name}'")

    cfg = MAP_TEMPLATES[name]
    new_cfg = dict(cfg)
    type = cfg.get("type", "step1d")
    pardict = cfg.get("pardict",dict())
    new_cfg["pardict"] = pardict
    new_cfg["domain"]  = np.asarray(cfg.get("domain", [0.0, 0.0, 1.0, 1.0]),dtype=np.float64)
    new_cfg["type"] = type

    if type == "step1d":
        expr = substitute_common(cfg["expr"],cfg.get("expr_common"))
        new_cfg["step"]  =  _funjit_1d(expr,pardict)
        if "deriv_expr" in cfg:
            deriv_expr = substitute_common(cfg["deriv_expr"],cfg.get("expr_common"))
        else:
            deriv_expr =  _sympy_deriv(substitute_common(cfg.get("expr"),cfg.get("expr_common"))) 
        new_cfg["deriv"] =  _funjit_1d(deriv_expr,pardict)
        new_cfg["eps_floor"] = cfg.get("eps_floor", 1e-16)
        return new_cfg
    
    if type == "step2d":
        if "step2_func" in cfg and "jac2_func" in cfg:
            new_cfg["step2"] = njit(STEP2_SIG, cache=False, fastmath=False)(cfg["step2_func"])
            new_cfg["jac2"]  = njit(JAC2_SIG, cache=False, fastmath=False)(cfg["jac2_func"])
        else:
            expr_x = substitute_common( cfg["expr_x"], cfg.get("expr_common") )
            expr_y = substitute_common( cfg["expr_y"], cfg.get("expr_common") )
            new_cfg["step2"] = _funjit_2d_step(expr_x,expr_y,pardict)
            if "jac_exprs" in cfg:
                dXdx, dXdy, dYdx, dYdy = cfg["jac_exprs"]
                dXdx = substitute_common( dXdx, cfg.get("expr_common") )
                dXdy = substitute_common( dXdy, cfg.get("expr_common") )
                dYdx = substitute_common( dYdx, cfg.get("expr_common") )
                dYdy = substitute_common( dYdy, cfg.get("expr_common") )
            else:
                dXdx, dXdy, dYdx, dYdy = _sympy_jacobian_2d(expr_x, expr_y)
            new_cfg["jac2"] = _funjit_2d_jag(dXdx,dXdy,dYdx,dYdy,pardict)
        new_cfg["eps_floor"] = cfg.get("eps_floor", 1e-16)
        return new_cfg
    
    if type == "step2d_ab":
        if "step2_func" in cfg and "jac2_func" in cfg:
            new_cfg["step2_ab"] = njit(STEP2_AB_SIG, cache=False, fastmath=False)(cfg["step2_func"])
            new_cfg["jac2_ab"]  = njit(JAC2_AB_SIG, cache=False, fastmath=False)(cfg["jac2_func"])
        else:
            expr_x = substitute_common( cfg["expr_x"], cfg.get("expr_common") )
            expr_y = substitute_common( cfg["expr_y"], cfg.get("expr_common") )
            new_cfg["step2_ab"] = _funjit_2d_ab_step(expr_x,expr_y,pardict)
            if "jac_exprs" in cfg:
                dXdx, dXdy, dYdx, dYdy = cfg["jac_exprs"]
                dXdx = substitute_common( dXdx, cfg.get("expr_common") )
                dXdy = substitute_common( dXdy, cfg.get("expr_common") )
                dYdx = substitute_common( dYdx, cfg.get("expr_common") )
                dYdy = substitute_common( dYdy, cfg.get("expr_common") )
            else:
                dXdx, dXdy, dYdx, dYdy = _sympy_jacobian_2d(expr_x, expr_y)
            new_cfg["jac2_ab"] = _funjit_2d_ab_jag(dXdx,dXdy,dYdx,dYdy,pardict)
        new_cfg["eps_floor"] = cfg.get("eps_floor", 1e-16)
        return new_cfg
    
   
    raise ValueError(f"Unsupported type={type} for map '{name}'")



def _get_map(name: str) -> dict:
    """
    Return compiled map config for 'name', building it on first use.
    """
    cfg = MAPS.get(name)
    #if cfg is not None:
    #    return cfg
    # build + cache
    print(f"Compiling map '{name}'")
    cfg = _build_map(name)
    MAPS[name] = cfg
    return cfg


# ---------------------------------------------------------------------------
# Sequence handling (A/B patterns)
# ---------------------------------------------------------------------------

SEQ_ALLOWED_RE = regex.compile(r"^[AaBb0-9()]+$")


def _looks_like_sequence_token(tok: str) -> bool:
    s = tok.strip()
    if not s:
        return False
    if not SEQ_ALLOWED_RE.match(s):
        return False
    # must contain at least one A/B or '(' so "123" isn't treated as seq
    return any(ch in "AaBb(" for ch in s)


def _decode_sequence_token(tok: str, default_seq: str = DEFAULT_SEQ) -> str:
    """
    Decode a sequence token into a string of 'A' and 'B'.

    Supported syntax:

        ABBA        -> ABBA
        A5B5        -> AAAAA BBBBB
        AB3A2       -> A B B B A A
        (AB)40      -> AB repeated 40 times
        A2(BA)3B    -> AA BABABA B
    """
    s = tok.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    if not s:
        return default_seq

    out_parts = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        # Single letter A/B with optional count
        if ch in "AaBb":
            letter = ch.upper()
            i += 1
            j = i
            while j < n and s[j].isdigit():
                j += 1
            if j == i:
                count = 1
            else:
                try:
                    count = int(s[i:j])
                except Exception:
                    return default_seq
                if count < 0:
                    return default_seq
            out_parts.append(letter * count)
            i = j
            continue

        # Parenthesised group (AB...) with optional count: (AB)40
        if ch == "(":
            j = s.find(")", i + 1)
            if j == -1:
                return default_seq
            group_str = s[i + 1 : j]
            if not group_str:
                return default_seq
            if any(c not in "AaBb" for c in group_str):
                return default_seq
            group = "".join(c.upper() for c in group_str)

            k = j + 1
            while k < n and s[k].isdigit():
                k += 1
            if k == j + 1:
                count = 1
            else:
                try:
                    count = int(s[j + 1 : k])
                except Exception:
                    return default_seq
                if count < 0:
                    return default_seq

            out_parts.append(group * count)
            i = k
            continue

        # Anything else -> fall back to default
        return default_seq

    seq = "".join(out_parts)
    return seq or default_seq


def _seq_to_array(seq_str: str) -> np.ndarray:
    s = (seq_str or "").strip().upper()
    if not s:
        raise ValueError("Sequence must be non-empty (e.g. 'AB')")
    data = []
    for ch in s:
        if ch == "A":
            data.append(0)
        elif ch == "B":
            data.append(1)
        else:
            raise ValueError(f"Invalid symbol '{ch}' in sequence; use only A/B.")
    return np.asarray(data, dtype=np.int32)


# ---------------------------------------------------------------------------
# Lyapunov field 
# ---------------------------------------------------------------------------

@njit(cache=False, fastmath=False)
def map_logical_to_physical(domain, u, v):
    llx, lly, ulx, uly, lrx, lry = domain
    ex = lrx - llx
    ey = lry - lly
    fx = ulx - llx
    fy = uly - lly
    A = llx + u*ex + v*fx
    B = lly + u*ey + v*fy
    return A, B

@njit(cache=False, fastmath=False, parallel=True)
def _lyapunov_field_1d(
    step,
    deriv,
    seq,        
    domain,     # <- 1D float64 array: [llx, lly, ulx, uly, lrx, lry]
    pix,
    x0,
    n_transient,
    n_iter,
    eps,
):
    """
    Generic λ-field for a 1‑D map with A/B forcing, over an arbitrary
    parallelogram in (A,B):

        (u,v) in [0,1]^2   (logical)
        (A,B) = LL + u (LR-LL) + v (UL-LL)

    where A,B are the two parameter values used in the A/B sequence.
    """
    seq_len = seq.size
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        for i in range(pix):

            A,B = map_logical_to_physical(domain, i / denom, j / denom)
            x = x0
            acc = 0.0

            for n in range(n_transient + n_iter):
                force = seq[n % seq_len] % 2
                forced_param = A if force==0 else B
                d = deriv(x, forced_param)
                x = step(x, forced_param)

                if not np.isfinite(x):
                    x = 0.5

                if n >= n_transient:
                    ad = abs(d)
                    if (not np.isfinite(ad)) or ad < eps:
                        ad = eps
                    acc += math.log(ad)

            out[j, i] = acc / float(n_iter)

    return out

# 2d map with single forced parameter
@njit(cache=False, fastmath=False, parallel=True)
def _lyapunov_field_2d_ab(
    step2_ab,
    jac2_ab,
    seq,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (A,B)-plane
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    eps_floor,
):
    seq_len = seq.size
    out = np.empty((pix, pix), dtype=np.float64)

    if eps_floor <= 0.0:
        eps_floor = 1e-16

    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        for i in range(pix):
            A, B = map_logical_to_physical(domain,  i / denom, j / denom)
            x = x0
            y = y0
            vx = 1.0
            vy = 0.0
            acc = 0.0

            for n in range(n_transient + n_iter):
                force = seq[n % seq_len] % 2
                forced_param = A if force==0 else B
 
                dXdx, dXdy, dYdx, dYdy = jac2_ab(x, y, forced_param)

                vx_new = dXdx * vx + dXdy * vy
                vy_new = dYdx * vx + dYdy * vy
                vx, vy = vx_new, vy_new

                x_next, y_next = step2_ab(x, y, forced_param)

                if not np.isfinite(x_next) or not np.isfinite(y_next):
                    x_next = 0.5
                    y_next = 0.5

                norm = math.sqrt(vx * vx + vy * vy)

                if norm < eps_floor:
                    norm = eps_floor

                if n >= n_transient:
                    acc += math.log(norm)

                vx /= norm
                vy /= norm

                x = x_next
                y = y_next

            out[j, i] = acc / float(n_iter)

    return out

@njit(cache=False, fastmath=False, parallel=True)
def _lyapunov_field_2d(
    step2,
    jac2,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (r,s)-plane
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    eps_floor,
):
    """
    Generic λ-field for a 2‑D map over an arbitrary parallelogram in the
    (r,s) parameter plane.

        (u,v) in [0,1]^2
        (r,s) = LL + u (LR-LL) + v (UL-LL)
    """
    out = np.empty((pix, pix), dtype=np.float64)

    if eps_floor <= 0.0:
        eps_floor = 1e-16

    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        for i in range(pix):

            first_param,second_param = map_logical_to_physical(domain, i / denom, j / denom)

            x = x0
            y = y0
            vx = 1.0
            vy = 0.0
            acc = 0.0

            for n in range(n_transient + n_iter):
                x_next, y_next = step2(x, y, first_param,second_param)
                if not np.isfinite(x_next) or not np.isfinite(y_next):
                    x_next = 0.5
                    y_next = 0.0

                dXdx, dXdy, dYdx, dYdy = jac2(x, y, first_param,second_param)

                vx_new = dXdx * vx + dXdy * vy
                vy_new = dYdx * vx + dYdy * vy
                vx, vy = vx_new, vy_new

                norm = math.sqrt(vx * vx + vy * vy)
                if norm < 1e-16:
                    norm = 1e-16

                if n >= n_transient:
                    acc += math.log(norm)

                inv_norm = 1.0 / norm
                vx *= inv_norm
                vy *= inv_norm

                x, y = x_next, y_next

            out[j, i] = acc / float(n_iter)

    return out


# ---------------------------------------------------------------------------
# Entropy field 
# ---------------------------------------------------------------------------

@njit(cache=False, fastmath=False)
def _entropy_from_amplitudes(A):
    """
    Shannon entropy of non-negative amplitudes A[0..K-1],
    normalized to [0,1].
    """
    K = A.size
    S = 0.0
    for k in range(K):
        S += A[k]
    if S <= 0.0:
        return 0.0

    H = 0.0
    for k in range(K):
        p = A[k] / S
        if p > 0.0:
            H -= p * math.log(p)

    # normalize to [0,1]
    return H / math.log(K)


@njit(cache=False, fastmath=False, parallel=True)
def _entropy_field_1d(
    step,
    seq,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    n_transient,
    n_iter,
    omegas,     # 1D float64 array of frequencies
):
    """
    Streaming spectral-entropy field for a 1-D AB-forced map.
    Uses x_n as the observable, but with the mean removed
    (x_n - running_mean) to avoid DC dominance.
    """
    seq_len = seq.size
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)
    K = omegas.size

    # global per-call frequency increments
    cw = np.empty(K)
    sw = np.empty(K)
    for k in range(K):
        w = omegas[k]
        cw[k] = math.cos(w)
        sw[k] = math.sin(w)

    for j in prange(pix):
        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)

            # burn-in
            x = x0
            for n in range(n_transient):
                force = seq[n % seq_len] % 2
                forced_param = A if force == 0 else B
                x = step(x, forced_param)
                if not np.isfinite(x):
                    x = 0.5

            # streaming "DFT" accumulators
            C = np.zeros(K)
            S = np.zeros(K)

            # phase state for each frequency, start at angle = 0
            cos_n = np.ones(K)
            sin_n = np.zeros(K)

            # running mean of x (for DC removal)
            mean = 0.0

            for n in range(n_iter):
                force = seq[n % seq_len] % 2
                forced_param = A if force == 0 else B
                x = step(x, forced_param)
                if not np.isfinite(x):
                    x = 0.5

                # online mean update (n goes 0..n_iter-1)
                mean += (x - mean) / (n + 1.0)
                obs = x - mean  # de-meaned observable

                for k in range(K):
                    c = cos_n[k]
                    s = sin_n[k]

                    # accumulate with current phase
                    C[k] += obs * c
                    S[k] += obs * s

                    # rotate phase: θ -> θ + ω_k
                    cwk = cw[k]
                    swk = sw[k]
                    c_new = c * cwk - s * swk
                    s_new = c * swk + s * cwk

                    cos_n[k] = c_new
                    sin_n[k] = s_new

            Avals = np.empty(K)
            for k in range(K):
                Avals[k] = math.sqrt(C[k] * C[k] + S[k] * S[k])

            out[j, i] = _entropy_from_amplitudes(Avals)

    return out


@njit(cache=False, fastmath=False, parallel=True)
def _entropy_field_2d_ab(
    step2_ab,
    seq,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    omegas,
):
    """
    Streaming spectral-entropy field for a 2-D AB-forced map.
    Observable = x-component, demeaned (x - mean(x)) to avoid DC dominance.
    """
    seq_len = seq.size
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)
    K = omegas.size

    # global per-call frequency increments
    cw = np.empty(K)
    sw = np.empty(K)
    for k in range(K):
        w = omegas[k]
        cw[k] = math.cos(w)
        sw[k] = math.sin(w)

    for j in prange(pix):
        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)

            x = x0
            y = y0

            # burn-in
            for n in range(n_transient):
                force = seq[n % seq_len] % 2
                forced_param = A if force == 0 else B
                x, y = step2_ab(x, y, forced_param)
                if not np.isfinite(x) or not np.isfinite(y):
                    x = 0.5
                    y = 0.5

            # streaming accumulators
            C = np.zeros(K)
            S = np.zeros(K)

            # phase state for each frequency, start at angle = 0
            cos_n = np.ones(K)
            sin_n = np.zeros(K)

            # running mean of x for DC removal
            mean = 0.0

            for n in range(n_iter):
                force = seq[n % seq_len] % 2
                forced_param = A if force == 0 else B
                x, y = step2_ab(x, y, forced_param)
                if not np.isfinite(x) or not np.isfinite(y):
                    x = 0.5
                    y = 0.5

                # update running mean of x
                mean += (x - mean) / (n + 1.0)
                obs = x - mean  # de-meaned observable

                for k in range(K):
                    c = cos_n[k]
                    s = sin_n[k]

                    # accumulate with current phase
                    C[k] += obs * c
                    S[k] += obs * s

                    # rotate phase: θ -> θ + ω_k
                    cwk = cw[k]
                    swk = sw[k]
                    c_new = c * cwk - s * swk
                    s_new = c * swk + s * cwk

                    cos_n[k] = c_new
                    sin_n[k] = s_new

            Avals = np.empty(K)
            for k in range(K):
                Avals[k] = math.sqrt(C[k] * C[k] + S[k] * S[k])

            out[j, i] = _entropy_from_amplitudes(Avals)

    return out


@njit(cache=False, fastmath=False, parallel=True)
def _entropy_field_2d(
    step2,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (first,second)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    omegas,
):
    """
    Streaming spectral-entropy field for a non-forced 2-D map.
    Parameters are (first, second) from the domain; observable = x (demeaned).
    """
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)
    K = omegas.size

    # global per-call frequency increments
    cw = np.empty(K)
    sw = np.empty(K)
    for k in range(K):
        w = omegas[k]
        cw[k] = math.cos(w)
        sw[k] = math.sin(w)

    for j in prange(pix):
        for i in range(pix):
            first_param, second_param = map_logical_to_physical(
                domain, i / denom, j / denom
            )

            x = x0
            y = y0

            # burn-in
            for n in range(n_transient):
                x, y = step2(x, y, first_param, second_param)
                if not np.isfinite(x) or not np.isfinite(y):
                    x = 0.5
                    y = 0.0

            # streaming accumulators
            C = np.zeros(K)
            S = np.zeros(K)

            # phase state for each frequency, start at angle = 0
            cos_n = np.ones(K)
            sin_n = np.zeros(K)

            # running mean of x for DC removal
            mean = 0.0

            for n in range(n_iter):
                x, y = step2(x, y, first_param, second_param)
                if not np.isfinite(x) or not np.isfinite(y):
                    x = 0.5
                    y = 0.0

                # update running mean and de-meaned observable
                mean += (x - mean) / (n + 1.0)
                obs = x - mean   # or math.hypot(x, y) - mean if you prefer radius

                for k in range(K):
                    c = cos_n[k]
                    s = sin_n[k]

                    # accumulate with current phase
                    C[k] += obs * c
                    S[k] += obs * s

                    # rotate phase: θ -> θ + ω_k
                    cwk = cw[k]
                    swk = sw[k]
                    c_new = c * cwk - s * swk
                    s_new = c * swk + s * cwk

                    cos_n[k] = c_new
                    sin_n[k] = s_new

            Avals = np.empty(K)
            for k in range(K):
                Avals[k] = math.sqrt(C[k] * C[k] + S[k] * S[k])

            out[j, i] = _entropy_from_amplitudes(Avals)

    return out

# ---------------------------------------------------------------------------
# stat field 
# ---------------------------------------------------------------------------

@njit
def hist_fixed_bins_inplace(bins, x, xmin, xmax):
    nbins = bins.size
    for k in range(nbins): bins[k] = 0
    if xmax <= xmin: xmax = xmin + 1e-12
    scale = nbins / (xmax - xmin)
    for val in x:
        j = int((val - xmin) * scale)
        if 0 <= j < nbins:
            bins[j] += 1


@njit
def compute_orbit(step, x0, A, B, seq, n_transient, n_iter, xs):
    seq_len = seq.size
    x = x0
    for n in range(n_transient):
        force = seq[n % seq_len] & 1
        forced_param = A if force == 0 else B
        x = step(x, forced_param)
        if not math.isfinite(x): x = 0.5
    for n in range(n_iter):
        force = seq[n % seq_len] & 1
        forced_param = A if force == 0 else B
        x = step(x, forced_param)
        if not math.isfinite(x): x = 0.5
        xs[n] = x
    return

@njit
def copy(xs, vs):
    for n in range(xs.size): vs[n] = xs[n]
    return 
    
@njit
def negabs(xs, vs):
    for n in range(xs.size): vs[n] = -math.fabs(xs[n])
    return 
    
@njit
def modulo1(xs, vs):
    for n in range(xs.size): vs[n] = xs[n] % 1
    return 

@njit
def slope(xs, vs):
    px = xs[0]
    for n in range(1, xs.size):
        x = xs[n]
        vs[n] = x - px
        px = x
    vs[0] = vs[1] # preserve range
    return

@njit
def convexity(xs, vs):
    N = xs.size
    if N < 3:
        for n in range(N): vs[n] = 0.0
        return
    ppx = xs[0]
    px  = xs[1]
    for n in range(2, N):
        x = xs[n]
        vs[n] = x - 2.0 * px + ppx
        ppx = px
        px = x
    vs[0] = vs[2] # preserve range
    vs[1] = vs[2]
    return 

@njit
def curvature(xs, vs):
    N = xs.size
    if N < 3:
        for n in range(N): vs[n] = 0.0
        return
    ppx = xs[0]
    px  = xs[1]
    for n in range(2, N):
        x = xs[n]
        num = math.fabs(x - 2.0 * px + ppx)
        denom = math.pow(1.0 + (x - px)**2, 3/2)
        if denom > 0.0: v = num / denom
        else: v = 0.0
        vs[n] = v
        ppx = px
        px = x
    vs[0] = vs[2] # preserve range
    vs[1] = vs[2]
    return

@njit 
def product(xs, vs):
    px = xs[0]
    for n in range(1, xs.size):
        x = xs[n]
        vs[n] = x * px
        px = x
    vs[0]=vs[1] # preserve range
    return 

@njit 
def ema(xs, vs):
    v = xs[0]
    for n in range(1, xs.size):
        x = xs[n]
        v = v * 0.95 + x * 0.05
        vs[n] = v
    vs[0]=vs[1] # preserve range
    return

@njit 
def rsi(xs, vs):
    px = xs[0]
    u = 0.0
    w = 0.0
    v = 0.0
    for n in range(1, xs.size):
        x = xs[n]
        dx = x - px
        u = 0.9 * u + 0.1 * dx
        w = 0.9 * w + 0.1 * abs(dx)
        if w > 0.0: v = u / w
        vs[n] = v
        px = x
    vs[0]=vs[1] # preserve range
    return

@njit
def transform_values(vcalc, xs, vs):
    N = xs.size
    if N == 0: return
    if vcalc == 0: copy(xs,vs)
    elif vcalc == 1: slope(xs,vs)
    elif vcalc == 2: convexity(xs,vs)
    elif vcalc == 3: curvature(xs,vs)
    elif vcalc == 4: product(xs,vs)
    elif vcalc == 5: ema(xs,vs)
    elif vcalc == 6: rsi(xs,vs)
    elif vcalc == 7: negabs(xs,vs)
    elif vcalc == 8: modulo1(xs,vs)
    else: copy(xs,vs)
    return

@njit
def entropy(hist):
    total = float(np.sum(hist))
    e=0.0
    if total > 0.0:
        for b in hist:
            if b > 0:
                p = b / total
                e += p * math.log(p)
        e = e/math.log(hist.size)
    return e

@njit
def zerocross(hist):
    m=np.mean(hist)
    s=np.sign(hist-m)
    c=s[1:]*s[:-1]
    e = np.sum(c>0)/hist.size
    return e
    
@njit 
def slopehist(hist):
    for k in range(hist.size - 1):
        hist[k] = hist[k + 1] - hist[k]
    e = np.std(hist[:-1])
    return e

@njit
def convhist(hist):
    for k in range(hist.size - 2):
        hist[k] = hist[k + 2] - 2*hist[k+1] + hist[k]
    e = np.std(hist[:-2])
    return e

@njit
def skewhist(hist):
    a = hist-np.mean(hist)
    m2 = np.mean(a*a)
    m3 = np.mean(a*a*a)
    if m2 > 0:
        e  = m3 / (m2 ** 1.5)
    else:
        e  = 0.0
    return e


@njit
def sumabschange(hist):
    e = 0.0
    for k in range(hist.size-1): 
        e += abs(hist[k+1] - hist[k])
    return e

@njit
def maxratio(hist):
    hmax = np.max(hist)
    if hmax==0: return 0.0
    hmean = np.mean(hist)
    return float(hmax/hmean)

@njit
def lrratio(hist):
    leftsum = np.sum(hist[int(hist.size/2):])
    if leftsum==0: return 0.0
    rightsum = np.sum(hist[0:int(hist.size/2)])
    return float(rightsum/leftsum)

@njit
def tailratio(hist):
    tail = int(hist.size/4)
    tailsum = np.sum(hist[:tail])+np.sum(hist[hist.size-tail:])
    midsum = np.sum(hist[tail:hist.size-tail])
    if midsum==0: return 0.0
    return float(tailsum/midsum)


@njit
def transform_hist(hcalc,hist):
    if   hcalc==0: return np.std(hist)
    elif hcalc==1: return entropy(hist)
    elif hcalc==2: return zerocross(hist)
    elif hcalc==3: return slopehist(hist)
    elif hcalc==4: return convhist(hist)
    elif hcalc==5: return skewhist(hist)
    elif hcalc==6: return sumabschange(hist)
    elif hcalc==7: return maxratio(hist)
    elif hcalc==8: return lrratio(hist)
    elif hcalc==9: return tailratio(hist)
    return 0.0



@njit(cache=False, fastmath=False, parallel=True)
def _hist_field_1d(
    step,
    seq,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    n_transient,
    n_iter,
    vcalc=0,
    hcalc=0,
    hbins=32,
):

    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        xs = np.empty(n_iter, dtype=np.float64)
        vs = np.empty(n_iter, dtype=np.float64)
        hist = np.zeros(hbins, dtype=np.int64)
        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)
            compute_orbit(step, x0, A, B, seq, n_transient, n_iter, xs)
            transform_values(vcalc, xs, vs)
            vmin = 1e300
            vmax = -1e300
            for n in range(n_iter):
                v = vs[n]
                if v < vmin: vmin = v
                if v > vmax: vmax = v
            for k in range(hist.size): hist[k] = 0  # reset
            hist_fixed_bins_inplace(hist, vs, vmin, vmax)
            e=transform_hist(hcalc,hist)
            out[j, i] = -e

    return out

@njit(cache=False, fastmath=False, parallel=True)
def _hist_field_2d_ab(
    step2_ab,
    seq,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    vcalc=0,
    hcalc=0
):
    """
    Histogram-based texture field for a 2-D AB-forced map.
    Observable is derived from x via vcalc:
        0: value      -> v = x
        1: slope      -> v = x - px
        2: convexity  -> v = x - 2*px + ppx
        3: curvature  -> v = |x-2px+ppx| / (1 + (x-px)^2)^(3/2)
    hcalc selects the histogram functional (same as 1-D version):
        0: std(bins)
        1: entropy(bins)
        2: "zero-crossings" of bins about their mean
        3: std(diff(bins))
    """
    seq_len = seq.size
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        vs   = np.zeros(n_iter, dtype=np.float64)
        bins = np.empty(n_iter, dtype=np.int64)

        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)

            # burn-in
            x = x0
            y = y0
            px = x0
            ppx = x0
            for n in range(n_transient):
                force = seq[n % seq_len] & 1
                forced_param = A if force == 0 else B
                x, y = step2_ab(x, y, forced_param)
                if (not np.isfinite(x)) or (not np.isfinite(y)):
                    x = 0.5
                    y = 0.5
                ppx = px
                px  = x

            vmin = 1e6
            vmax = -1e6

            for n in range(n_iter):
                force = seq[n % seq_len] & 1
                forced_param = A if force == 0 else B
                x, y = step2_ab(x, y, forced_param)
                if (not np.isfinite(x)) or (not np.isfinite(y)):
                    x = 0.5
                    y = 0.5

                if vcalc == 0:        # value
                    v = x
                elif vcalc == 1:      # slope
                    v = x - px
                elif vcalc == 2:      # convexity
                    v = x - 2.0*px + ppx
                elif vcalc == 3:      # curvature
                    num = math.fabs(x - 2.0*px + ppx)
                    den = math.pow(1.0 + (x - px)*(x - px), 1.5)
                    if den > 0.0:
                        v = num / den
                    else:
                        v = 0.0
                else:
                    v = x

                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v

                vs[n] = v
                ppx = px
                px  = x

            hist_fixed_bins_inplace(bins, vs, vmin, vmax)

            e = 0.0
            if hcalc == 0:      # stdev of bin counts
                e = np.std(bins)
            elif hcalc == 1:    # entropy
                total = float(np.sum(bins))
                if total > 0.0:
                    H = 0.0
                    for b in bins:
                        if b > 0:
                            p = b / total
                            H += p * math.log(p)
                    e = H / math.log(bins.size)
            elif hcalc == 2:    # "zero-crossing" of bins around mean
                m = float(np.mean(bins))
                s = np.sign(bins - m)
                c = s[1:] * s[:-1]
                e = np.sum(c > 0) / bins.size
            elif hcalc == 3:    # std of changes
                for k in range(bins.size - 1):
                    bins[k] = bins[k + 1] - bins[k]
                e = np.std(bins[:-1])
            else:
                e = 0.0

            out[j, i] = -e

    return out

@njit(cache=False, fastmath=False, parallel=True)
def _hist_field_2d(
    step2,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (first,second)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    vcalc=0,
    hcalc=0
):
    """
    Histogram-based texture field for a 2-D non-forced map.
    Parameters are (first, second) from the domain; observable derived from x.
    vcalc and hcalc as in _hist_field_2d_ab.
    """
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        vs   = np.zeros(n_iter, dtype=np.float64)
        bins = np.empty(n_iter, dtype=np.int64)

        for i in range(pix):
            first_param, second_param = map_logical_to_physical(
                domain, i / denom, j / denom
            )

            x = x0
            y = y0
            px = x0
            ppx = x0

            # burn-in
            for n in range(n_transient):
                x, y = step2(x, y, first_param, second_param)
                if (not np.isfinite(x)) or (not np.isfinite(y)):
                    x = 0.5
                    y = 0.0
                ppx = px
                px  = x

            vmin = 1e6
            vmax = -1e6

            for n in range(n_iter):
                x, y = step2(x, y, first_param, second_param)
                if (not np.isfinite(x)) or (not np.isfinite(y)):
                    x = 0.5
                    y = 0.0

                if vcalc == 0:        # value
                    v = x
                elif vcalc == 1:      # slope
                    v = x - px
                elif vcalc == 2:      # convexity
                    v = x - 2.0*px + ppx
                elif vcalc == 3:      # curvature
                    num = math.fabs(x - 2.0*px + ppx)
                    den = math.pow(1.0 + (x - px)*(x - px), 1.5)
                    if den > 0.0:
                        v = num / den
                    else:
                        v = 0.0
                else:
                    v = x

                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v

                vs[n] = v
                ppx = px
                px  = x

            hist_fixed_bins_inplace(bins, vs, vmin, vmax)

            e = 0.0
            if hcalc == 0:
                e = np.std(bins)
            elif hcalc == 1:
                total = float(np.sum(bins))
                if total > 0.0:
                    H = 0.0
                    for b in bins:
                        if b > 0:
                            p = b / total
                            H += p * math.log(p)
                    e = H / math.log(bins.size)
            elif hcalc == 2:
                m = float(np.mean(bins))
                s = np.sign(bins - m)
                c = s[1:] * s[:-1]
                e = np.sum(c > 0) / bins.size
            elif hcalc == 3:
                for k in range(bins.size - 1):
                    bins[k] = bins[k + 1] - bins[k]
                e = np.std(bins[:-1])
            else:
                e = 0.0

            out[j, i] = -e

    return out


# ---------------------------------------------------------------------------
# Color mapping: Lyapunov exponent or Entropy or Custom -> RGB (schemes)
# ---------------------------------------------------------------------------

# Scheme registry: ADD NEW SCHEMES HERE ONLY
RGB_SCHEMES: dict[str, dict] = {
    "mh": dict(
        func=colors.rgb_scheme_mh,
        defaults=dict(
            gamma=0.25,
            pos_color="FF0000",
            zero_color="000000",
            neg_color="FFFF00",
        ),
    ),

    "mh_eq": dict(
        func=colors.rgb_scheme_mh_eq,
        defaults=dict(
            gamma=1,
            pos_color="FF0000",
            zero_color="000000",
            neg_color="FFFF00",
            nbins=2048,
        ),
    ),

    "palette": dict(
        func=colors.rgb_scheme_palette_eq,
        defaults=dict(
            palette="bauhaus_primaries",
            gamma=1,
            nbins=2048,
        ),
    ),

    "multi": dict(
        func=colors.rgb_scheme_multipoint,
        defaults=dict(
            palette="bauhaus_primaries",
            gamma=1,
            nbins=2048,
        ),
    ),
}

DEFAULT_RGB_SCHEME = "mh"


def lyapunov_to_rgb(lyap: np.ndarray, specdict: dict) -> np.ndarray:
    """
    Apply a colorization scheme to the λ-field based on the 'rgb' spec.

    Syntax:
        # Markus–Hess style:
        rgb:mh                          -> use mh defaults
        rgb:mh:0.25                     -> override gamma
        rgb:mh:*:#FF0000:#FFFF00        -> keep gamma, set pos/neg colors

        # Equalized variant (γ, pos_color, neg_color, nbins):
        rgb:mh_eq                       -> defaults
        rgb:mh_eq:0.3                   -> gamma=0.3
        rgb:mh_eq:*:#00FF00:#0000FF:512 -> custom colors, nbins=512

    """
    # --- 1) choose scheme ---
    rgb_vals = specdict.get("rgb")
    if rgb_vals:
        scheme_name = str(rgb_vals[0]).strip().lower()
    else:
        scheme_name = DEFAULT_RGB_SCHEME

    scheme_cfg = RGB_SCHEMES.get(scheme_name, RGB_SCHEMES[DEFAULT_RGB_SCHEME])

    # --- 2) start from scheme defaults ---
    params = dict(scheme_cfg["defaults"])  # shallow copy

    # --- 3) optional global gamma: override if present and scheme uses gamma ---
    gamma_vals = specdict.get("gamma")
    if gamma_vals and "gamma" in params:
        try:
            params["gamma"] = float(_eval_number(gamma_vals[0]).real)
        except Exception:
            pass

     # --- 4) parse positional args from rgb:scheme:arg1:arg2:... ---
    if rgb_vals and len(rgb_vals) > 1:
        arg_tokens = rgb_vals[1:]

        # order is exactly the insertion order of defaults
        defaults = scheme_cfg["defaults"]
        order = list(defaults.keys())

        for idx, tok in enumerate(arg_tokens):
            if idx >= len(order):
                break
            name = order[idx]
            if name not in params:
                continue

            default_val = params[name]
            tok_str = str(tok).strip()
            if tok_str == "*":
                # '*' -> keep default
                continue

            # parse based on type of default
            if isinstance(default_val, (float, int)):
                try:
                    params[name] = float(_eval_number(tok_str).real)
                except Exception:
                    pass
            elif isinstance(default_val, str):
                params[name] = tok_str
            else:
                # unsupported type, leave default
                pass

    func = scheme_cfg["func"]
    return func(lyap, params)


# ---------------------------------------------------------------------------
# Spec helpers using specparser.split_chain
# ---------------------------------------------------------------------------

def _eval_number(tok: str) -> complex:
    return specparser.simple_eval_number(tok)


def _get_float(d: dict, key: str, default: float) -> float:
    vals = d.get(key)
    if not vals:
        return float(default)
    try:
        return float(_eval_number(vals[0]).real)
    except Exception:
        return float(default)


def _get_int(d: dict, key: str, default: int) -> int:
    vals = d.get(key)
    if not vals:
        return int(default)
    try:
        return int(round(float(_eval_number(vals[0]).real)))
    except Exception:
        return int(default)

# ---------------------------------------------------------------------------
# Domain / affine mapping helpers
# ---------------------------------------------------------------------------

def _get_corner(d: dict, key: str, default_x: float, default_y: float):
    """
    Parse a corner operator like:

        ll:x:y
        ul:*:y
        lr:x:*
        ll:x        (x only, y from default)
        ll          (no args, all defaults)

    '*' means "keep default". Missing args also keep defaults.
    """
    vals = d.get(key)
    if not vals:
        return float(default_x), float(default_y)

    x = default_x
    y = default_y

    try:
        # First argument: x
        if len(vals) >= 1:
            v0 = vals[0].strip()
            if v0 != "*":
                x = float(_eval_number(v0).real)

        # Second argument: y
        if len(vals) >= 2:
            v1 = vals[1].strip()
            if v1 != "*":
                y = float(_eval_number(v1).real)

    except Exception:
        # On parse error, fall back to defaults
        x, y = default_x, default_y

    return float(x), float(y)


def _build_affine_domain(
    specdict: dict,
    a0: float,
    b0: float,
    a1: float,
    b1: float,
) -> np.ndarray:
    """
    Build a 2‑D affine domain mapping from logical (u,v) in [0,1]^2
    to physical (A,B) coordinates.

    We use three corners:

        LL = lower-left   (u=0, v=0)
        UL = upper-left   (u=0, v=1)
        LR = lower-right  (u=1, v=0)

    The user can override them via:

        ll:x:y   ul:x:y   lr:x:y

    with '*' as "keep default" and optional 1-arg forms ll:x, etc.

    Additionally, 'ur:x:y' can be used to complete a rectangle when
    ul/lr are not given explicitly:

        ll:x:y, ur:ux:uy

    means "axis-aligned rectangle" from (x,y) to (ux,uy).
    """

    # 0) defaults: axis-aligned rectangle from [a0,a1] x [b0,b1]
    llx, lly = a0, b0
    ulx, uly = a0, b1
    lrx, lry = a1, b0

    # 1) apply ll/ul/lr with '*' semantics
    llx, lly = _get_corner(specdict, "ll", llx, lly)
    ulx, uly = _get_corner(specdict, "ul", ulx, uly)
    lrx, lry = _get_corner(specdict, "lr", lrx, lry)

    # 2) ur, if present and ul/lr not explicitly given, completes rectangle
    if "ur" in specdict:
        urx, ury = _get_corner(specdict, "ur", a1, b1)

        # Only fill UL/LR from UR if user *didn't* specify them directly
        if "ul" not in specdict:
            ulx, uly = llx, ury
        if "lr" not in specdict:
            lrx, lry = urx, lly

    # 3) fine-grained llx/lly/ulx/... overrides (power user layer)
    llx = _get_float(specdict, "llx", llx)
    lly = _get_float(specdict, "lly", lly)
    ulx = _get_float(specdict, "ulx", ulx)
    uly = _get_float(specdict, "uly", uly)
    lrx = _get_float(specdict, "lrx", lrx)
    lry = _get_float(specdict, "lry", lry)

    domain_affine = np.asarray(
        [llx, lly, ulx, uly, lrx, lry],
        dtype=np.float64,
    )

    # Optional sanity check: are the three points colinear?
    vx0 = lrx - llx
    vy0 = lry - lly
    vx1 = ulx - llx
    vy1 = uly - lly
    area = abs(vx0 * vy1 - vx1 * vy0)
    if area == 0.0:
        print("WARNING: affine domain is degenerate (LL, UL, LR colinear)")

    return domain_affine

def debug_affine_for_spec(spec: str) -> None:
    """
    Print the resolved affine domain and a few logical->physical
    sample points for the given spec string.
    """
    specdict = specparser.split_chain(spec)

    map_name = None
    for op in specdict.keys():
        if op in MAP_TEMPLATES:
            map_name = op
            break
    if map_name is None:
        print(f"No map name found in spec {spec}")
        return

    map_cfg = MAP_TEMPLATES[map_name]
    type = map_cfg.get("type", "step1d")
    domain = map_cfg["domain"].copy()

    use_seq = (type=="step1d") or (type=="step2d_ab")
    domain_idx = 0
    for i, v in enumerate(specdict[map_name]):
        if use_seq and i == 0 and _looks_like_sequence_token(v):
            continue
        try:
            domain_component = float(specparser.simple_eval_number(v).real)
        except Exception:
            continue
        if domain_idx < domain.size:
            domain[domain_idx] = domain_component
            domain_idx += 1

    a0 = _get_float(specdict, "a0", domain[0])
    b0 = _get_float(specdict, "b0", domain[1])
    a1 = _get_float(specdict, "a1", domain[2])
    b1 = _get_float(specdict, "b1", domain[3])

    domain_affine = _build_affine_domain(specdict, a0, b0, a1, b1)
    llx, lly, ulx, uly, lrx, lry = domain_affine

    print("Affine domain:")
    print(f"  LL = ({llx}, {lly})")
    print(f"  UL = ({ulx}, {uly})")
    print(f"  LR = ({lrx}, {lry})")

    def map_uv(u, v):
        A = llx + u * (lrx - llx) + v * (ulx - llx)
        B = lly + u * (lry - lly) + v * (uly - lly)
        return A, B

    samples = [
        (0.0, 0.0, "(0,0)"),
        (1.0, 0.0, "(1,0)"),
        (0.0, 1.0, "(0,1)"),
        (1.0, 1.0, "(1,1)"),
        (0.5, 0.5, "(0.5,0.5)"),
    ]
    print("Sample logical -> physical mapping:")
    for u, v, label in samples:
        A, B = map_uv(u, v)
        print(f"  {label}: (u={u}, v={v}) -> ({A}, {B})")


# ---------------------------------------------------------------------------
# spec2lyapunov: parse spec -> RGB tile
# ---------------------------------------------------------------------------

def get_map_name(spec: str)-> str:
    specdict = specparser.split_chain(spec)
    if not "map" in specdict:
        raise SystemExit(f"No 'map' found in spec {spec}")
    map_spec = specdict["map"]
    if len(map_spec)<1:
        raise SystemExit(f"map needs to specify map name")
    map_name = map_spec[0]
    if not map_name in  MAP_TEMPLATES:
        raise SystemExit(f"{map_name} not in MAP_TEMPLATES")
    return map_name

def make_cfg(spec:str):

    map_name = get_map_name(spec)

    specdict = specparser.split_chain(spec)
    map_spec = specdict["map"]
    map_temp = MAP_TEMPLATES[map_name]

    if not "pardict" in map_temp:
        raise SystemExit(f"{map_name} needs a pardict")

    pardict = map_temp["pardict"]
    for i,(key,value) in enumerate(pardict.items()):
        if specdict.get(key) is not None:
            param_value = specdict.get(key)[0]
        else:
            param_value = value
        pardict[key] = param_value

    map_cfg = _build_map(map_name)

    map_cfg["map_name"] = map_name

    map_type = map_cfg.get("type", "step1d")
    map_cfg["type"] = map_type
    domain = map_cfg["domain"]
    
    use_seq = (map_type=="step1d") or (map_type=="step2d_ab")
    seq_arr = _seq_to_array(DEFAULT_SEQ) if use_seq else None

    if len(map_spec)>1:
        domain_idx = 0
        for i, v in enumerate(map_spec[1:]):

            if use_seq and i == 0 and _looks_like_sequence_token(v):
                seq_str = _decode_sequence_token(v, DEFAULT_SEQ)
                seq_arr = _seq_to_array(seq_str)
                continue
            try:
                domain_component = float(specparser.simple_eval_number(v).real)
            except Exception:
                continue

            if domain_idx < domain.size:
                domain[domain_idx] = domain_component
                domain_idx += 1
    
    map_cfg["seq_arr"]=seq_arr

    a0 = _get_float(specdict, "a0", domain[0])
    b0 = _get_float(specdict, "b0", domain[1])
    a1 = _get_float(specdict, "a1", domain[2])
    b1 = _get_float(specdict, "b1", domain[3])

    map_cfg["domain_affine"] = _build_affine_domain(specdict, a0, b0, a1, b1)

    map_cfg["x0"]    = _get_float(specdict, "x0", map_cfg.get("x0", 0.5))
    map_cfg["y0"]    = _get_float(specdict, "y0", map_cfg.get("y0", 0.5))
    map_cfg["n_tr"]  = _get_int(specdict, "trans", map_cfg.get("trans", DEFAULT_TRANS))
    map_cfg["n_it"]  = _get_int(specdict, "iter", map_cfg.get("iter",  DEFAULT_ITER))
    map_cfg["eps"]   = _get_float(specdict, "eps",   DEFAULT_EPS_LYAP)

    if "entropy" in specdict:
        map_cfg["type"]=map_cfg["type"]+"_entropy"
        K = _get_int(specdict, "k", 32)
        w0 = _get_float(specdict, "w0", 0.1)
        w1 = _get_float(specdict, "w1", math.pi)
        K = max(K,2)
        map_cfg["omegas"] = np.linspace(w0, w1, K, dtype=np.float64)
        map_cfg["entropy_sign"] = int(-1)
        if len(specdict["entropy"])>0:
            map_cfg["entropy_sign"] = int(specdict["entropy"][0])


    if "hist" in specdict:
        map_cfg["type"] = map_cfg["type"] + "_hist"
        map_cfg["vcalc"] = int(0)
        map_cfg["hcalc"] = int(0)
        map_cfg["hbins"] = map_cfg["n_it"]
        if len(specdict["hist"])>0:
            map_cfg["vcalc"] = int(specdict["hist"][0])
        if len(specdict["hist"])>1:
            map_cfg["hcalc"] = int(specdict["hist"][1])
        if len(specdict["hist"])>2:
            map_cfg["hbins"] = int(specdict["hist"][2])

    return map_cfg

def spec2lyapunov(spec: str, pix: int = 5000) -> np.ndarray:

    map_cfg = make_cfg(spec)
 
    if map_cfg["type"] == "step1d":

        print("lyapunov_field_generic_1d")

        field = _lyapunov_field_1d(
            map_cfg["step"],
            map_cfg["deriv"],
            map_cfg["seq_arr"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]),
            int(map_cfg["n_tr"]),
            int(map_cfg["n_it"]),
            float(map_cfg["eps"]),
            
        )

    elif map_cfg["type"] == "step2d_ab":

        print("lyapunov_field_generic_2d_ab")

        field = _lyapunov_field_2d_ab(
            map_cfg["step2_ab"],
            map_cfg["jac2_ab"],
            map_cfg["seq_arr"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]), 
            float(map_cfg["y0"]),
            int(map_cfg["n_tr"]), 
            int(map_cfg["n_it"]),
            float(map_cfg.get("eps_floor", 1e-16)),
        )

    elif map_cfg["type"] == "step2d":

        print("lyapunov_field_generic_2d")

        field = _lyapunov_field_2d(
            map_cfg["step2"],
            map_cfg["jac2"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]), 
            float(map_cfg["y0"]),
            int(map_cfg["n_tr"]), 
            int(map_cfg["n_it"]),
            float(map_cfg.get("eps_floor", 1e-16)),
        )

    elif map_cfg["type"] == "step1d_entropy":

        print("entropy_field_generic_1d")

        raw = _entropy_field_1d(
                map_cfg["step"],
                map_cfg["seq_arr"],
                map_cfg["domain_affine"],
                int(pix),
                float(map_cfg["x0"]),
                int(map_cfg["n_tr"]),
                int(map_cfg["n_it"]),
                map_cfg["omegas"],
        )
        # map H∈[0,1] to [-1,1] so your diverging palettes still work:
        field = map_cfg["entropy_sign"] * (2.0 * raw - 1.0)
    
    elif map_cfg["type"] == "step2d_ab_entropy":

        print("entropy_field_generic_2d_ab")

        raw = _entropy_field_2d_ab(
            map_cfg["step2_ab"],
            map_cfg["seq_arr"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]), 
            float(map_cfg["y0"]),
            int(map_cfg["n_tr"]), 
            int(map_cfg["n_it"]),
            map_cfg["omegas"],
        )
        field = map_cfg["entropy_sign"] * (2.0 * raw - 1.0)
    
    elif map_cfg["type"] == "step2d_entropy":

        print("entropy_field_generic_2d")

        raw = _entropy_field_2d(
            map_cfg["step2"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]), 
            float(map_cfg["y0"]),
            int(map_cfg["n_tr"]), 
            int(map_cfg["n_it"]),
            map_cfg["omegas"],
        )
        field = map_cfg["entropy_sign"] * (2.0 * raw - 1.0)
    
    elif map_cfg["type"] == "step1d_hist":

        print("hist_field_1d")

        raw = _hist_field_1d(
            map_cfg["step"],
            map_cfg["seq_arr"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]),
            int(map_cfg["n_tr"]),
            int(map_cfg["n_it"]),
            int(map_cfg["vcalc"]),
            int(map_cfg["hcalc"]),
            int(map_cfg["hbins"]),
        )

        field = raw-np.median(raw)

    elif map_cfg["type"] == "step2d_ab_hist":

        print("hist_field_2d_ab")

        raw = _hist_field_2d_ab(
            map_cfg["step2_ab"],
            map_cfg["seq_arr"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]),
            float(map_cfg["y0"]),
            int(map_cfg["n_tr"]),
            int(map_cfg["n_it"]),
            int(map_cfg["vcalc"]),
            int(map_cfg["hcalc"]),
        )

        field = raw-np.median(raw)

    elif map_cfg["type"] == "step2d_hist":

        print("hist_field_2d")

        raw = _hist_field_2d(
            map_cfg["step2"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]),
            float(map_cfg["y0"]),
            int(map_cfg["n_tr"]),
            int(map_cfg["n_it"]),
            int(map_cfg["vcalc"]),
            int(map_cfg["hcalc"]),
        )

        field = raw-np.median(raw)

    else:
        raise SystemExit(f"Unsupported type={map_cfg['type']} for map '{map_cfg['map_name']}'")

    rgb = lyapunov_to_rgb(field, specparser.split_chain(spec))

    return rgb

# ---------------------------------------------------------------------------
# expansion helpers
# ---------------------------------------------------------------------------

def get_all_palettes(palette_regex, maxp):
    print(f"all palette search:{palette_regex}")
    pat = regex.compile(palette_regex)
    out = []
    for k in colors.COLOR_STRINGS.keys():
        if pat.search(k):
            print(f"found palette: {k}")
            out.append(k)
            if len(out) >= maxp:
                break
    return out

def get_long_palettes(palette_regex, maxp):
    print(f"long palette search:{palette_regex}")
    pat = regex.compile(palette_regex)
    out = []
    for k in colors.COLOR_LONG_STRINGS.keys():
        if pat.search(k):
            print(f"found palette: {k}")
            out.append(k)
            if len(out) >= maxp:
                break
    return out

def get_tri_palettes(palette_regex, maxp):
    print(f"tri palette search:{palette_regex}")
    pat = regex.compile(palette_regex)
    out = []
    for k in colors.COLOR_TRI_STRINGS.keys():
        if pat.search(k):
            print(f"found palette: {k}")
            out.append(k)
            if len(out) >= maxp:
                break
    return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        "lyapounov-cli",
        description=(
            "Lyapunov fractal renderer for 1D maps.\n"
            "Maps are defined symbolically and JIT-compiled; "
            "adding a new map is just editing MAP_TEMPLATES."
        ),
    )

    p.add_argument(
        "--spec",
        required=True,
        help="Lyapunov spec (can include expandspec lists/ranges).",
    )
    p.add_argument(
        "--show-specs",
        action="store_true",
        help="Show expanded specs before rendering.",
    )
    p.add_argument(
        "--pix",
        type=int,
        default=1000,
        help="Tile width/height in pixels (Lyapunov grid resolution).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="lyapunov.png",
        help="Output PNG path (can itself be an expandspec template).",
    )
    p.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Columns if chain expands to multiple tiles.",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Rows if chain expands to multiple tiles.",
    )
    p.add_argument(
        "--invert",
        action="store_true",
        help="Invert final RGB colors (simple negative).",
    )
    p.add_argument(
        "--thumb",
        type=int,
        default=None,
        help="Thumbnail height in pixels (if set, save mosaic as thumbnail).",
    )
    p.add_argument(
        "--const",
        action="append",
        default=[],
        help="Add/override NAME=VALUE (parsed like spec args). Repeatable.",
    )
    p.add_argument(
        "--map",
        type=str,
        default=None,
        help="override map equation",
    )
    p.add_argument(
        "--pal",
        action="append",
        default=[],
        help="add palette",
    )
    p.add_argument(
        "--check-affine",
        action="store_true",
        help="Print affine domain mapping for the first expanded spec and exit.",
    )


    args = p.parse_args()

    if args.map is not None:
        map_name, new_expr = args.map.split("=", 1)
        if map_name in MAP_TEMPLATES:
            new_der_expr = _sympy_deriv(new_expr)
            print(f"map derivative: {new_der_expr}")
            # patch the template; lazy builder will use this
            MAP_TEMPLATES[map_name]["expr"] = new_expr
            MAP_TEMPLATES[map_name]["deriv_expr"] = new_der_expr
            spec_str = f",modify:{map_name}:{new_expr}"
            args.spec = args.spec + spec_str
        else:
            print(f"WARNING: --map refers to unknown map '{map_name}'")

    # Apply constants (like in julia.py)
    for kv in args.const:
        print(f"const {kv}")
        k, v = specparser._parse_const_kv(kv)
        specparser.set_const(k, v)
        expandspec.set_const(k, v)

    for kv in args.pal:
        print(f"adding palette {kv}")
        k, v = kv.split("=", 1)
        colors.COLOR_STRINGS[k]=v

    # Expand output path first
    out_paths = expandspec.expand_cartesian_lists(args.out)
    if not out_paths:
        raise SystemExit("Output expandspec produced no paths")
    pngfile = out_paths[0]
    print(f"will save to {pngfile}")

    # spec expansion helpers
    expandspec.FUNCS["gap"]=get_all_palettes
    expandspec.FUNCS["glp"]=get_long_palettes
    expandspec.FUNCS["gtp"]=get_tri_palettes

    # Expand the main spec chain
    specs = expandspec.expand_cartesian_lists(args.spec)

    if args.check_affine:
        # just inspect the first expanded spec
        debug_affine_for_spec(specs[0])
        return
    
    if args.show_specs:
        for s in specs:
            print(s)

    if not specs:
        raise SystemExit("Spec expansion produced no tiles")

    tiles: list[np.ndarray] = []
    titles: list[str] = []

    for i, spec in enumerate(specs, start=1):
        print(f"{i}/{len(specs)} Rendering {spec}")
        t0 = time.perf_counter()
        rgb = spec2lyapunov(spec, pix=args.pix)
        print(f"field time: {time.perf_counter() - t0:.3f}s")
        # swap A/B axes and flip vertically to match Markus & Hess style
        rgb = np.flipud(rgb)
        tiles.append(rgb)
        titles.append(spec)

    n = len(tiles)
    if args.cols:
        cols = args.cols
    elif args.rows:
        cols = int(round(n / args.rows))
    else:
        cols = max(1, int(round(math.sqrt(n))))

    raster.save_mosaic_png_rgb(
        tiles=tiles,
        titles=titles,
        cols=cols,
        gap=20,
        out_path=pngfile,
        invert=args.invert,
        footer_pad_lr_px=48,
        footer_dpi=300,
        thumbnail=args.thumb,
    )

    print(f"saved: {pngfile}")


if __name__ == "__main__":
    main()

