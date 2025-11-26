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
import argparse
import re as regex
import numpy as np
import sympy as sp
from numba import njit, types, prange

from specparser import specparser, expandspec
from rasterizer import raster

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
    "np": np,
    "math": math,
    "max": max,
    "min": min,
    "floor": floor,
    "ceil": ceil,
    "abs_cap": abs_cap,
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
# Map templates: custom functions
# ---------------------------------------------------------------------------

def parasite_step2_py(x, y, s, r, param):
    EXP_MAX = 50.0
    STATE_MAX = 1e6
    H, P = x, y
    a, rj = r,s
    K = param[0]
    invK = 1.0 / K
    
    F = rj * (1.0 - H * invK) - a * P
    F = min(abs(F),EXP_MAX)*sign(F)
    expF = math.exp(F)

    G = -a * P
    G = min(abs(G),EXP_MAX)*sign(G)
    E = math.exp(G)

    H_next = H * expF
    P_next = H * (1.0 - E)    

    H = min(abs(H_next),STATE_MAX)*sign(H_next)
    P = min(abs(P_next),STATE_MAX)*sign(P_next)

    if not (np.isfinite(H) and np.isfinite(P)): 
        H, P = 1.0, 1.0

    return H, P


def parasite_jac2_py(x, y, s, r, param):

    EXP_MAX = 50.0
    H, P = x, y
    a, rj = r,s
    K = param[0]
    invK = 1.0 / K

    F = rj * (1.0 - H * invK) - a * P
    F = min(abs(F),EXP_MAX)*sign(F)

    expF = math.exp(F)

    G = -a * P
    G = min(abs(G),EXP_MAX)*sign(G)
    E = math.exp(G)

    # Jacobian
    dHdH = expF * (1.0 - rj * H * invK)
    dHdP = -a * H * expF
    dPdH = 1.0 - E
    dPdP = H * a * E
    
    return dHdH, dHdP, dPdH, dPdP

def predprey_step2_py(x, y, forced):
    a_param = forced
    b_param = 3.569985
    x_next = a_param * x * (1.0 - x - y)
    y_next = b_param * x * y
    x_next = min(abs(x_next),1e6)*sign(x_next)
    y_next = min(abs(y_next),1e6)*sign(y_next)
    if not (np.isfinite(x_next) and np.isfinite(y_next)):
        x_next = 0.5
        y_next = 0.5
    return x_next, y_next


def predprey_jac2_py(x, y, forced):
    a_param = forced
    b_param = 3.569985
    dxdx = a_param * (1.0 - 2.0 * x - y)
    dxdy = -a_param * x
    dydx = b_param * y
    dydy = b_param * x
    return dxdx, dxdy, dydx, dydy

def cardiac_step2_py(x, y, s, r, param):
    """
    Guevara cardiac map (Eq. 8.4), embedded in 2D with a dummy y:

        x_{n+1} = A - B1 * exp(-t / tau1) - B2 * exp(-t / tau2)
        t       = k * r_p - x_n

    Here we interpret:
        r = t_min   (vertical axis parameter before transpose/flip)
        s = r_p     (stimulation period, horizontal axis)

    The dummy y is kept at 0 and does not affect the dynamics.
    """
    # parameters
    A    = param[0]
    B1   = param[1]
    B2   = param[2]
    tau1 = param[3]
    tau2 = param[4]

    tmin = r        # r-argument is t_min
    r_p  = s        # s-argument is the period r

    # guard against r_p <= 0 to avoid division by zero
    if r_p <= 0.0:
        r_eff = 1e-12
    else:
        r_eff = r_p

    # k_n = floor((t_min + x_n)/r) + 1  so that k_n r - x_n > t_min
    k = math.floor((tmin + x) / r_eff) + 1.0
    t = k * r_eff - x

    EXP_MAX = 50.0

    F1 = -t / tau1
    if F1 > EXP_MAX:
        F1 = EXP_MAX
    elif F1 < -EXP_MAX:
        F1 = -EXP_MAX
    e1 = math.exp(F1)

    F2 = -t / tau2
    if F2 > EXP_MAX:
        F2 = EXP_MAX
    elif F2 < -EXP_MAX:
        F2 = -EXP_MAX
    e2 = math.exp(F2)

    x_next = A - B1 * e1 - B2 * e2
    if not np.isfinite(x_next):
        x_next = x   # fall back to previous value if something blows up

    # dummy y-dimension, strongly contracting / irrelevant
    y_next = 0.0

    return x_next, y_next


def cardiac_jac2_py(x, y, s, r, param):
    """
    Jacobian for the cardiac map, consistent with cardiac_step2_py.
    We embed the 1D derivative into the (x,x) entry and kill the y-dimension.

    x_{n+1} = A - B1 e^{-t/τ1} - B2 e^{-t/τ2},   t = k r_p - x
    treating k as constant when differentiating (piecewise-smooth approx).
    """
    A    = param[0]
    B1   = param[1]
    B2   = param[2]
    tau1 = param[3]
    tau2 = param[4]

    tmin = r        # r-argument is t_min
    r_p  = s        # s-argument is the period r

    if r_p <= 0.0:
        r_eff = 1e-12
    else:
        r_eff = r_p

    k = math.floor((tmin + x) / r_eff) + 1.0
    t = k * r_eff - x

    EXP_MAX = 50.0

    F1 = -t / tau1
    if F1 > EXP_MAX:
        F1 = EXP_MAX
    elif F1 < -EXP_MAX:
        F1 = -EXP_MAX
    e1 = math.exp(F1)

    F2 = -t / tau2
    if F2 > EXP_MAX:
        F2 = EXP_MAX
    elif F2 < -EXP_MAX:
        F2 = -EXP_MAX
    e2 = math.exp(F2)

    # dt/dx = -1 (k treated as constant), so
    # d x_next / d x = -(B1/τ1) e^{-t/τ1} - (B2/τ2) e^{-t/τ2}
    dXdx = -(B1 / tau1) * e1 - (B2 / tau2) * e2

    dXdy = 0.0
    dYdx = 0.0
    dYdy = 0.0   # y-dimension is dead

    return dXdx, dXdy, dYdx, dYdy


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

# Discontinuous logistic map (eq. 9.6) used by 'dlog'
# x_{n+1} = r_n x_n (1 - x_n)                   if x_n > 0.5
#         = r_n x_n (1 - x_n) + ¼(α - 1)(r_n-2) else
_DLOG_EXPR = (
    "r*x*(1-x)*step(x-0.5) + "
    "(r*x*(1-x) + 0.25*(a-1)*(r-2))*(1-step(x-0.5))"
)

MAP_TEMPLATES: dict[str, dict] = {

    "cardiac": dict(
        # Guevara cardiac map, Eq. (8.4):
        #   x_{n+1} = A - B1 e^{-t_n/τ1} - B2 e^{-t_n/τ2}
        #   t_n = k_n r - x_n, k_n minimum integer with t_n > t_min
        #
        # We treat this as a 2-D map with a dummy y-dimension so that we
        # can scan both r and t_min across the tile while still using
        # the generic 2-D Lyapunov kernel.
        #
        # Axis conventions for this map:
        #   - pre-grid horizontal (i, 'a' axis)  -> t_min in [a0, a1]
        #   - pre-grid vertical   (j, 'b' axis)  -> r     in [b0, b1]
        # After your transpose+flip in the CLI the final image shows
        #   x-axis : r
        #   y-axis : t_min
        type="step2d",
        step2_func=cardiac_step2_py,
        jac2_func=cardiac_jac2_py,
        # domain = [tmin0, r0, tmin1, r1]
        # matching Fig. 8.3: r ∈ [0,150], t_min ∈ [20,140]
        domain=[20.0, 0.0, 140.0, 150.0],
        # params: [A, B1, B2, tau1, tau2]
        # (you can override A,B1,B2,tau1 via a:,b:,c:,d: if you like;
        # tau2 stays at the default unless you edit the map)
        pardict=dict(
            r  = "second",
            s  = "first",
            a=270.0,
            b=2441,
            c=90.02, 
            d=19.6,
            e=200.5,
        ),
        x0=5.0,       # initial x_n (duration)
        y0=0.0,       # dummy y
        trans=100,    # n_prev
        iter=200,     # n_max
        eps_floor=1e-16,
    ),

    "predprey_old": dict(
        type="step2d_ab",
        step2_func=predprey_step2_py,
        jac2_func=predprey_jac2_py,
        # domain: [A0, B0, A1, B1]  (same as old MAP_DEFAULT_DOMAIN)
        domain=[-0.04, -0.6, 4.5, 6.6],
        # params: we store b in param[1]; param[0] is unused
        pardict=dict(
            r="forced",
            a=0.0,
            b= 3.569985,
            c=0.0, 
            d=0.0,
            e=0.0,
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=200,
        eps_floor=1e-16,
    ),
    
    "predprey": dict(
        type="step2d_ab",
        expr_x="abs_cap(r * x * (1.0 - x - y),1e6)",
        expr_y="abs_cap(b * x * y,1e6)",
        jac_exprs=(
            "r * (1.0 - 2.0 * x - y)",  # dXdx
            "-r * x",                   # dXdy
            "b * y",                    # dYdx
            "b * x",                    # dYdy
        ),
        domain=[-0.04, -0.6, 4.5, 6.6],
        pardict=dict(
            r="forced",                 # in impl2_ab_step: r = forced
            b=3.569985,                 # baked in as constant
        ),
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=200,
        eps_floor=1e-16,
    ),

    "parasite": dict(
        type="step2d",
        step2_func=parasite_step2_py,
        jac2_func=parasite_jac2_py,
        #expr_x="x * exp( r*(1.0 - x/a) - s*y )",

        #expr_y="x * (1.0 - exp(-s*y))",
        domain=[-0.1, -0.1, 4, 7],
        pardict=dict(
            r  = "second",
            s  = "first",
            a=2.1,
            b= 0.0,
            c=0.0, 
            d=0.0,
            e=0.0,
        ),
        x0=0.5,
        y0=0.5,
        trans=200,
        iter=800,
        eps_floor=1e-16,  # <- matches original parasite kernel
    ),

    "kicked": dict(
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
        # FIXME: classification needs a single string
        # something like "1D", "2D", "2Dab" 
        # FIXME: add a default "seq" key
        # seq="AB" or somthing
        expr_x="1-s*abs(x)+y",
        expr_y="a*x",
        # FIXME: add a sub_expr dictionary that is 
        # expanded into the formula prior to scypy
        # things inside [...] are looked up 
        # and replaced [xpr1]+ [xpr2]
        # sub_expr={'xpr1':'a+b+c', 'xpr2':'abs(x+b)'}
        domain=[-10,-10,10,10],
        pardict=dict(
            s="forced",
            a=0.994,
        ),
        #FIXME: add pnames=["a","b","c","d"] so formuli can match source
        x0 = 0.4,
        y0 = 0.4,
        # FIXME: make this a list init=[0.4,0.4]
        # so the values can be specced like init:0.4:0.4
        trans = 100,
        iter = 200,
    ),

    "adbash": dict( #adams-bashworth
        type="step2d_ab",
        expr_x="x+(a/2)*(3*s*x*(1-x)-s*y*(1-y))",
        expr_y="y",
        domain=[-5,-5,5,5],
        pardict=dict(
            s  = "forced",
            a=1.0,
        ),
        x0 = 0.5,
        y0 = 0.5,
        trans = 200,
        iter = 200,
    ),

    "degn": dict( #degn's map
        type="step2d",
        # text is "b vs r"
        # s is the LHS of the text 
        # r is the RHS of the text
        expr_x="s*(x-0.5)+0.5+a*sin(2*pi*r*y)",
        expr_y="(y+s*(x-0.5)+0.5+a*sin(2*pi*r*y)*mod1(b/s))",
        domain=[-2,-5,2,5],
        pardict=dict(
            r  = "second",
            s  = "first",
            a=0.1,
            b= 1.0,
        ),
        x0 = 0.4,
        y0 = 0.4,
        trans = 100,
        iter = 300,
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
            a=3.0,
            b= 2.0,
            c=0.0, 
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
            r  = "second",
            s  = "first",
        ),
        x0=0.5,
        y0=1.0,
        trans=100,
        iter=300,
        # optionally: manual Jacobian override
    ),


    "logistic": dict( # Classic logistic
        expr="r * x * (1.0 - x)",
        domain=[2.5, 2.5, 4.0, 4.0],  # A0, B0, A1, B1
        pardict=dict(
            r  = "forced",
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "sine": dict( # Sine map (classical Lyapunov variant: r sin(pi x))
        expr="r * sin(pi * x)",
        domain=[0.0, 2.0, 0.0, 2.0],
        pardict=dict(
            r  = "forced",
        ),
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "tent": dict(  # Tent map
        expr="r*x*(1-step(x-0.5)) + r*(1-x)*step(x-0.5)",
        domain=[0.0, 0.0, 2.0, 2.0],
        pardict=dict(
            r  = "forced",
        ),
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
        expr=_DLOG_EXPR, # FIXME: add a local expansion step, no globals
        deriv_expr="r * (1.0 - 2.0 * (" + _DLOG_EXPR + "))",
        domain=[2.5,2.5,4, 4],
        pardict=dict(
            r  = "forced",
            a  = 0.4,
            b  = 0.0,
            c  = 0.0, 
            d  = 0.0,
            e  = 0.0,
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
        expr="b * exp( cos(1 - x) * sin(pi/2) + sin(r) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq971": dict(
        expr="b * r * exp( pow( sin(x - r), 4 ) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq972": dict(
        expr="b * r * exp( pow( sin(1 - x), 3 ) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq973": dict(
        expr="b * r * pow( sin(b*x + r*r), 2 ) * pow( cos(b*x - r*r), 2 )",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
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

    "eq978": dict(
        expr="r*sin(pi*r)*sin(pi*(x-b))",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq979": dict(
        expr="b*r*pow(sin(b*x+r*r), 2 )*pow( cos(b*x-r*r-r),2)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq980": dict(
        expr="b*r*pow(sin(b*x+r*r),2)*pow(cos(b*x-r*r),2)-1",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 0.5,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq981": dict(
        expr="b/(2+sin(mod1(x))-r)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
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
        expr="b*exp(tan(r*x)-x)",
        domain=[0.0, 0.0, 4.0, 4.0],
        pardict=dict(
            r  = "forced",
            b  = 1.0,
        ),  
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
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
        expr = cfg["expr"]
        new_cfg["step"]  =  _funjit_1d(expr,pardict)
        if "deriv_expr" in cfg:
            deriv_expr = cfg["deriv_expr"]
        else:
            deriv_expr =  _sympy_deriv(expr) 
        new_cfg["deriv"] =  _funjit_1d(deriv_expr,pardict)
        new_cfg["eps_floor"] = cfg.get("eps_floor", 1e-16)
        return new_cfg
    
    if type == "step2d":
        if "step2_func" in cfg and "jac2_func" in cfg:
            new_cfg["step2"] = njit(STEP2_SIG, cache=False, fastmath=False)(cfg["step2_func"])
            new_cfg["jac2"]  = njit(JAC2_SIG, cache=False, fastmath=False)(cfg["jac2_func"])
        else:
            expr_x = cfg["expr_x"]
            expr_y = cfg["expr_y"]
            new_cfg["step2"] = _funjit_2d_step(expr_x,expr_y,pardict)
            if "jac_exprs" in cfg:
                dXdx, dXdy, dYdx, dYdy = cfg["jac_exprs"]
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
            expr_x = cfg["expr_x"]
            expr_y = cfg["expr_y"]
            new_cfg["step2_ab"] = _funjit_2d_ab_step(expr_x,expr_y,pardict)
            if "jac_exprs" in cfg:
                dXdx, dXdy, dYdx, dYdy = cfg["jac_exprs"]
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
    if cfg is not None:
        return cfg
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
# Low-level Lyapunov field (generic, map passed as function)
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
# Color mapping: Lyapunov exponent -> RGB (schemes)
# ---------------------------------------------------------------------------

def _hist_equalize(values: np.ndarray, nbins: int = 256) -> np.ndarray:
    """
    Simple 1D histogram equalization on 'values', returning t in [0,1].

    We use a fixed number of bins and map each value to the CDF bin
    it falls into. This is O(N) and works well for large images.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float64)

    vmin = float(values.min())
    vmax = float(values.max())

    if (not math.isfinite(vmin)) or (not math.isfinite(vmax)) or vmax <= vmin:
        return np.zeros_like(values, dtype=np.float64)

    hist, bin_edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    cdf = hist.cumsum().astype(np.float64)
    if cdf[-1] <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    cdf /= cdf[-1]

    # For each value, find its bin and pick the CDF
    idx = np.searchsorted(bin_edges, values, side="right") - 1
    idx = np.clip(idx, 0, nbins - 1)
    t = cdf[idx]
    return t

# Simple named colors for convenience

_COLOR_NAME_MAP = {
    "black":      "000000",
    "white":      "FFFFFF",
    "red":        "FF0000",
    "green":      "00FF00",
    "blue":       "0000FF",
    "yellow":     "FFFF00",
    "cyan":       "00FFFF",
    "magenta":    "FF00FF",

    # Extra basics / darks
    "gray":       "808080",
    "lightgray":  "D3D3D3",
    "darkgray":   "A9A9A9",
    "navy":       "000080",
    "teal":       "008080",
    "olive":      "808000",
    "maroon":     "800000",

    # Warmer tones
    "orange":     "FFA500",
    "brown":      "A52A2A",
    "gold":       "FFD700",
    "coral":      "FF7F50",

    # Cooler / pastel-ish
    "skyblue":    "87CEEB",
    "darkgreen":  "006400",
    "lightgreen": "90EE90",

    # Purples & pinks
    "purple":     "800080",
    "violet":     "EE82EE",
    "indigo":     "4B0082",
    "pink":       "FFC0CB",
    "orchid":     "DA70D6",

    # Metals
    "silver":     "C0C0C0",

    # Reds / oranges / warm shades
    "crimson":        "DC143C",
    "salmon":         "FA8072",
    "lightsalmon":    "FFA07A",
    "tomato":         "FF6347",
    "orangered":      "FF4500",
    "darkorange":     "FF8C00",
    "goldenrod":      "DAA520",
    "darkgoldenrod":  "B8860B",
    "sienna":         "A0522D",
    "chocolate":      "D2691E",
    "sandybrown":     "F4A460",
    "peru":           "CD853F",
    "tan":            "D2B48C",
    "khaki":          "F0E68C",
    "darkkhaki":      "BDB76B",
    "firebrick":      "B22222",
    "darkred":        "8B0000",

    # Greens
    "lime":           "00FF00",  # alias to green
    "forestgreen":    "228B22",
    "seagreen":       "2E8B57",
    "mediumseagreen": "3CB371",
    "darkseagreen":   "8FBC8F",
    "springgreen":    "00FF7F",
    "lawngreen":      "7CFC00",
    "chartreuse":     "7FFF00",
    "olivedrab":      "6B8E23",

    # Blues / cyans
    "deepskyblue":      "00BFFF",
    "dodgerblue":       "1E90FF",
    "royalblue":        "4169E1",
    "steelblue":        "4682B4",
    "cornflowerblue":   "6495ED",
    "cadetblue":        "5F9EA0",
    "lightsteelblue":   "B0C4DE",
    "powderblue":       "B0E0E6",
    "turquoise":        "40E0D0",
    "mediumturquoise":  "48D1CC",
    "darkturquoise":    "00CED1",
    "lightseagreen":    "20B2AA",
    "midnightblue":     "191970",
    "darkslateblue":    "483D8B",

    # Purples / pinks
    "plum":             "DDA0DD",
    "thistle":          "D8BFD8",
    "mediumorchid":     "BA55D3",
    "darkorchid":       "9932CC",
    "darkmagenta":      "8B008B",
    "mediumvioletred":  "C71585",
    "deeppink":         "FF1493",
    "hotpink":          "FF69B4",
    "palevioletred":    "DB7093",

    # Very light / near-whites
    "mintcream":        "F5FFFA",
    "honeydew":         "F0FFF0",
    "aliceblue":        "F0F8FF",
    "ghostwhite":       "F8F8FF",
    "lavender":         "E6E6FA",
    "ivory":            "FFFFF0",
    "linen":            "FAF0E6",
    "oldlace":          "FDF5E6",
    "seashell":         "FFF5EE",
    "snow":             "FFFAFA",
    "floralwhite":      "FFFAF0",
    "beige":            "F5F5DC",

    # Grays / slates
    "slategray":        "708090",
    "darkslategray":    "2F4F4F",
    "lightslategray":   "778899",
    "gainsboro":        "DCDCDC",
    "dimgray":          "696969",

    # A few fun, slightly custom vibes
    "sunset":           "FFCC66",
    "ocean":            "006994",
    "forest":           "0B6623",
    "rose":             "FF66CC",
    "mint":             "98FF98",
    "peach":            "FFDAB9",
    "sand":             "C2B280",
    "charcoal":         "36454F",
    "coolgray":         "8C92AC",
    "warmwhite":        "FFF8E7",

    # Variants / other metals
    "lightsilver": "D8D8D8",
    "rosegold":    "B76E79",
    "copper":      "B87333",
    "bronze":      "CD7F32",
    "brass":       "B5A642",
    "platinum":    "E5E4E2",
    "steel":       "71797E",
    "iron":        "43464B",

    # What you asked for
    "nickel":      "727472",
    "nikel":       "727472",  # alias, in case you type it this way
    "cobalt":      "0047AB",  # cobalt blue pigment

    # Extra shiny-ish gray
    "chrome":      "B4B4B4",
    "titanium":    "8D8F91",

    # Racing / deep classic greens
    "racinggreen":        "004225",
    "britishracinggreen": "004225",  # alias

    # Gem / luxury greens
    "emerald":            "50C878",
    "jade":               "00A86B",
    "kellygreen":         "4CBB17",
    "shamrock":           "33CC99",

    # Natural / earthy greens
    "mossgreen":          "8A9A5B",
    "fern":               "4F7942",
    "pine":               "01796F",
    "jungle":             "29AB87",
    "avocado":            "568203",
    "sage":               "B2AC88",
    "wasabi":             "A8C545",

    # Soft / pastel greens
    "teagreen":           "D0F0C0",
    "seafoam":            "9FE2BF",

    # Loud / neon greens
    "neongreen":          "39FF14",
    "limepunch":          "C7EA46",

    # Vintage French ink swatches (from the color card)

    "rawsienna": "B87535",        # Rawsienna (Terre de Sienne)
    "terre_de_sienne": "B87535",

    "burntsienna": "97442A",      # Burntsienna (Sienne brûlée)
    "sienne_brulee": "97442A",

    "redbrown": "944544",         # Red-brown (Brun rouge)
    "brun_rouge": "944544",

    "sepia": "55381F",            # Sepia (Sépia)
    "sepia_fr": "55381F",

    "scarlet_ink": "D9534B",      # Scarlet (Écarlate)
    "ecarlate": "D9534B",

    "carmine_ink": "D2475C",      # Carmine (Carmin)
    "carmin": "D2475C",

    "vermillion_ink": "D85734",   # Vermilion (Vermillon)
    "vermillon": "D85734",

    "orange_ink": "DA6F34",       # Orange (Orange)
    "orange_fr": "DA6F34",

    "yellow_ink": "E3BA45",       # Yellow (Jaune)
    "jaune": "E3BA45",

    "lightgreen_ink": "83C0A0",   # Light green (Vert clair)
    "vert_clair": "83C0A0",

    "darkgreen_ink": "8CAD96",    # Dark green (Vert foncé)
    "vert_fonce": "8CAD96",

    "indigo_ink": "191B46",       # Indigo (Indigo)
    "indigo_fr": "191B46",

    "prussianblue_ink": "292791", # Prussian blue (Bleu de Prusse)
    "bleu_de_prusse": "292791",

    "ultramarine_ink": "2234A4",  # Ultramarine (Outremer)
    "outremer": "2234A4",

    "cobaltblue_ink": "4371BC",   # Cobalt blue (Bleu de cobalt)
    "bleu_de_cobalt": "4371BC",

    "purple_ink": "C54B93",       # Purple / magenta-ish (Pourpre)
    "pourpre": "C54B93",

    "lightviolet_ink": "452782",  # Light violet (Violet clair)
    "violet_clair": "452782",

    "darkviolet_ink": "292397",   # Dark violet (Violet foncé)
    "violet_fonce": "292397",

    "gray_ink": "363434",         # Gray (Gris)
    "gris_fr": "363434",

    "neutral_tint": "251E32",     # Neutral tint (Teinte neutre)
    "teinte_neutre": "251E32",

    "black_ink": "111010",        # Black (Noir)
    "noir_fr": "111010",


    # Spectral paper colors (from the wavelength table, approximate RGB)

    "spectral_red": "FF0000",         # Rouge spectral
    "rouge_spectral": "FF0000",

    "vermillion_spectral": "FF5300",  # Vermillon
    "vermillon_spectral": "FF5300",

    "minium_spectral": "FFA900",      # Minium (red lead)
    "minium": "FFA900",

    "spectral_orange": "FFBE00",      # Orangé
    "orange_spectral": "FFBE00",

    "pale_chrome_yellow": "FFF900",   # Jaune (chrome pâle)
    "jaune_chrome_pale": "FFF900",

    "yellow_greenish": "D2FF00",      # Jaune verdâtre
    "jaune_verdatre": "D2FF00",

    "yellow_green": "BFFF00",         # Vert jaune
    "vert_jaune": "BFFF00",

    "spectral_green": "85FF00",       # Vert
    "vert_spectral": "85FF00",

    "emerald_green_spectral": "44FF00",  # Vert émeraude
    "vert_emeraude_spectral": "44FF00",

    "cyan_blue_2": "00FF9D",          # Bleu cyané n° 2
    "bleu_cyane_2": "00FF9D",

    "ultramarine_natural": "00B9FF",  # Outremer naturel
    "outremer_naturel_spectral": "00B9FF",

    "ultramarine_artificial": "0036FF",  # Outremer artificiel
    "outremer_artificiel_spectral": "0036FF",

    "spectral_violet": "5100FF",      # Violet
    "violet_spectral": "5100FF",

}

def _parse_color_spec(spec: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Parse a color spec into (R,G,B) in 0..255.

    Accepts:
        - "RRGGBB" hex
        - "#RRGGBB" hex
        - simple names: red, blue, yellow, ...
    """
    if not isinstance(spec, str):
        return default

    s = spec.strip()
    if not s:
        return default

    if s.startswith("#"):
        s = s[1:]

    # Name → hex mapping
    lower = s.lower()
    if lower in _COLOR_NAME_MAP:
        s = _COLOR_NAME_MAP[lower]

    if len(s) != 6:
        return default

    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError:
        return default

    return float(r), float(g), float(b)


def _rgb_scheme_mh(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Markus & Hess style:

      λ < 0 : black  -> yellow  (periodic / order)
      λ = 0 : black
      λ > 0 : black  -> red or blue (chaos)

    'clip' controls the symmetric |λ| range. If clip is None, auto-range.
    """
    arr = np.asarray(lyap, dtype=np.float64)
    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return rgb

    neg_mask = finite & (arr < 0.0)
    pos_mask = finite & (arr > 0.0)

    gamma = params.get("gamma", DEFAULT_GAMMA)
    gamma = 1 if gamma <= 0.0 else gamma

    pos_spec = params.get("pos_color", "FF0000")   # red
    zero_spec = params.get("zero_color", "000000")   # black
    neg_spec = params.get("neg_color", "FFFF00")   # yellow
    pos_r, pos_g, pos_b = _parse_color_spec(pos_spec, (255.0, 0.0, 0.0))
    zero_r, zero_g, zero_b = _parse_color_spec(zero_spec, (0.0, 0.0, 0.0))
    neg_r, neg_g, neg_b = _parse_color_spec(neg_spec, (255.0, 255.0, 0.0))

    min_neg = float(arr[neg_mask].min()) if np.any(neg_mask) else 0.0
    max_pos = float(arr[pos_mask].max()) if np.any(pos_mask) else 0.0
    scale = max(abs(min_neg), abs(max_pos))
    scale = 1 if (not math.isfinite(scale)) or scale <= 0.0 else scale

    # λ < 0 → black→yellow
    if np.any(neg_mask):
        lam_neg = np.clip(arr[neg_mask], -scale, 0.0)
        t = np.abs(lam_neg) / scale
        t = t ** float(gamma) if gamma != 1.0 else t
        rgb[neg_mask, 0] = np.rint(t * neg_r + (1-t)*zero_r).astype(np.uint8)
        rgb[neg_mask, 1] = np.rint(t * neg_g + (1-t)*zero_g).astype(np.uint8)
        rgb[neg_mask, 2] = np.rint(t * neg_b + (1-t)*zero_b).astype(np.uint8)

    # λ > 0 → black→red/blue
    if np.any(pos_mask):
        lam_pos = np.clip(arr[pos_mask], 0.0, scale)
        t = lam_pos / scale
        t = t ** float(gamma) if gamma != 1.0 else t
        rgb[pos_mask, 0] = np.rint(t * pos_r + (1-t)*zero_r).astype(np.uint8)
        rgb[pos_mask, 1] = np.rint(t * pos_g + (1-t)*zero_g).astype(np.uint8)
        rgb[pos_mask, 2] = np.rint(t * pos_b + (1-t)*zero_b).astype(np.uint8)

    return rgb


def _rgb_scheme_mh_eq(lyap: np.ndarray, params: dict) -> np.ndarray:

    arr = np.asarray(lyap, dtype=np.float64)
    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return rgb

    neg_mask = finite & (arr < 0.0)
    pos_mask = finite & (arr > 0.0)

    gamma = params.get("gamma", DEFAULT_GAMMA)
    gamma = 1 if gamma <= 0.0 else gamma
    nbins = int(params.get("nbins", 256))

    pos_spec = params.get("pos_color", "FF0000")   # red
    zero_spec = params.get("zero_color", "000000")   # black
    neg_spec = params.get("neg_color", "FFFF00")   # yellow
    pos_r, pos_g, pos_b = _parse_color_spec(pos_spec, (255.0, 0.0, 0.0))
    zero_r, zero_g, zero_b = _parse_color_spec(zero_spec, (0.0, 0.0, 0.0))
    neg_r, neg_g, neg_b = _parse_color_spec(neg_spec, (255.0, 255.0, 0.0))


    # λ < 0: equalize |λ|
    if np.any(neg_mask):
        vals = np.abs(arr[neg_mask])
        t = _hist_equalize(vals, nbins=nbins)  # in [0,1]
        t = t ** float(gamma) if gamma != 1.0 else t
        rgb[neg_mask, 0] = np.rint(t * neg_r + (1-t)*zero_r).astype(np.uint8)
        rgb[neg_mask, 1] = np.rint(t * neg_g + (1-t)*zero_g).astype(np.uint8)
        rgb[neg_mask, 2] = np.rint(t * neg_b + (1-t)*zero_b).astype(np.uint8)

    # λ > 0: equalize λ
    if np.any(pos_mask):
        vals = arr[pos_mask]
        t = _hist_equalize(vals, nbins=nbins)  # in [0,1]
        t = t ** float(gamma) if gamma != 1.0 else t
        rgb[pos_mask, 0] = np.rint(t * pos_r + (1-t)*zero_r).astype(np.uint8)
        rgb[pos_mask, 1] = np.rint(t * pos_g + (1-t)*zero_g).astype(np.uint8)
        rgb[pos_mask, 2] = np.rint(t * pos_b + (1-t)*zero_b).astype(np.uint8)

    return rgb

COLOR_TRI_STRINGS = {
    "bauhaus_primaries":  "0038A8:F5F0E6:D00000",  # (-1 deep blue, 0 warm paper, +1 poster red)
    "itten_blue_orange":  "264653:F1FAEE:E76F51",  # (-1 blue-green, 0 light neutral, +1 red-orange)
    "scientific_red_blue":"313695:F7F7F7:A50026",  # (ColorBrewer-style diverging)
    "swiss_modern":       "444444:FAFAFA:E2001A",  # (-1 dark gray, 0 white, +1 Swiss red)
    "pastel_soft":        "A8DADC:F1FAEE:FFB4A2",  # (-1 aqua, 0 off-white, +1 pastel coral)
    "midcentury_earth":   "386641:F2E8CF:BC6C25",  # (-1 forest, 0 cream, +1 ochre/burnt orange)
    "cyberpunk_neon":     "00F5FF:351B3F:FF0A81",  # (-1 cyan, 0 dark purple, +1 hot magenta)
    "thermal_blackbody":  "00004B:3F3F3F:FFE45E",  # (-1 deep blue, 0 dark neutral, +1 hot yellow)
    "vintage_print":      "007F7F:FFF1D0:D7263D",  # (-1 cyan ink, 0 old paper, +1 magenta-red ink)
    "munsell_balanced":   "009080:808080:E05000",  # (-1 teal, 0 mid gray, +1 orange; similar lightness)
    #
    "bauhaus_blue_yellow":  "002C7F:F6F1E1:FFC400",  # -1 deep ultramarine, 0 warm paper, +1 Bauhaus yellow
    "fauvism_wild":         "005F73:FFF3B0:FF006E",  # -1 teal, 0 acid pale yellow, +1 hot magenta
    "impressionist_pastel": "8E9FE6:FFF7E6:F6B189",  # -1 periwinkle, 0 light cream, +1 peach
    "expressionist_moody":  "1B3B6F:4A4A4A:D1495B",  # -1 dark blue, 0 charcoal, +1 moody red‑pink
    "brutalist_concrete":   "1E1E1E:C4C4C4:FFD100",  # -1 near‑black, 0 concrete gray, +1 hazard yellow
    "minimal_blue":         "101820:F4F4F4:2D7FF9",  # -1 deep ink, 0 clean white, +1 accent blue
    "material_teal_orange": "00695C:FAFAFA:FF8F00",  # -1 teal, 0 off‑white, +1 material amber
    "terrain_elevation":    "005B96:A4DE02:C38E70",  # -1 sea blue, 0 bright grass, +1 highland brown
    "oceanographic_deep":   "03045E:CAF0F8:FF6F61",  # -1 abyss blue, 0 pale water, +1 living coral
    "psychedelic_70s":      "008B8B:FFF7D6:B5179E",  # -1 dark cyan, 0 warm cream, +1 electric purple
    "grayscale_magnitude":  "1A1A1A:7F7F7F:F5F5F5",  # -1 dark, 0 mid gray, +1 light; no hue, just magnitude
    "artdeco_jewel":        "004E64:F5F0E8:D4A017",  # -1 teal jewel, 0 ivory, +1 antique gold

     # --- Natural materials & metal (-1 shadow, 0 material, +1 metal/highlight) ---

    "stone_bronze":   "545E63:C2B8A3:CD7F32",  # slate shadow, warm stone, bronze
    "wood_copper":    "3B2F2F:8B5A2B:B87333",  # bark, mid wood, copper
    "concrete_steel": "2F3133:A8A8A8:D0D7DF",  # dark steel, concrete, polished steel
    "sand_iron":      "43464B:E2C290:FFF1C1",  # iron shadow, desert sand, sunlit sand
    "marble_gold":    "6E7B8B:F5F5F0:FFD700",  # cool marble veins, white marble, gold

    # --- Marvel comics (bold, punchy primaries) ---

    "marvel_spiderman": "003366:F0F0F0:C00018",  # navy, city paper, Spidey red
    "marvel_ironman":   "1A1A1A:851313:FFD700",  # dark armor, deep red, gold
    "marvel_captain":   "002868:FFFFFF:BF0A30",  # flag blue, star white, flag red
    "marvel_hulk":      "1A1433:3A5F0B:B2FF59",  # gamma purple, Hulk green, glowing green
    "marvel_cosmic":    "240046:0B1724:FF6FF2",  # deep space purple, night, cosmic magenta

    # --- Pop art (bright, poster-y, halftone vibes) ---

    "pop_primary_dots": "0057A8:FFF4C7:F0142F",  # comic blue, pale yellow paper, pop red
    "pop_cyan_magenta": "00B8FF:FFEEDB:FF1E8A",  # cyan, warm off-white, hot pink
    "pop_lipstick":     "001B44:FCE4EC:E0004D",  # navy outline, light skin/pink, lipstick red
    "pop_banana":       "004D40:FFF59D:FFB300",  # deep teal, banana yellow, golden peel
    "pop_tv":           "3F51B5:F5F5F5:FFEB3B",  # TV blue, static white, bright yellow

    # --- Andy Warhol (acid color combos, Marilyn / Campbell vibes) ---

    "warhol_marilyn_cyan":  "008FD3:FFE0B2:FF00A0",  # cyan, peach skin, hot pink
    "warhol_marilyn_lime":  "4141FF:FFE0B2:C6FF00",  # cobalt, peach, neon lime
    "warhol_campbell":      "2E7D32:FFF3E0:C62828",  # can green, label cream, soup red
    "warhol_fluoro_blocks": "00E5FF:FFEB3B:FF3D00",  # aqua, acid yellow, fluoro orange
    "warhol_duotone_purple":"311B92:EDE7F6:F50057",  # deep violet, lilac paper, pink

    # --- Rocket launch (night sky, flame, smoke, glow) ---

    "rocket_night_launch":   "020819:4B4B55:FF6A00",  # deep night, smoke, flame orange
    "rocket_sunrise_launch": "1B2A49:FFECB3:FF7043",  # pre-dawn blue, sunrise, plume orange
    "rocket_plume_blue":     "001B44:CFD8DC:64FFDA",  # night blue, exhaust haze, turquoise plume
    "rocket_control_room":   "0D1B2A:1B263B:FFB703",  # dark consoles, panels, warning amber
    "rocket_heatshield":     "40241A:9E5E34:FFD166",  # char, hot tile, glowing heat

    # --- Tropical fish (reef, neon stripes, coral) ---

    "tropical_parrotfish":   "004E7A:00B8A9:F8E16C",  # deep teal, bright teal, yellow fin
    "tropical_clownfish":    "003049:FFFFFF:FF7B00",  # deep sea, white stripe, clown orange
    "tropical_reef":         "14213D:25CED1:FCA311",  # deep reef, turquoise, coral gold
    "tropical_angel":        "011627:FDFCDC:FF3366",  # dark water, pale body, pink accents
    "tropical_neon_tetra":   "001B44:00B4D8:FF006E",  # night blue, neon blue stripe, magenta
}

COLOR_LONG_STRINGS = {

    "test_long": (
        "red:green:blue:red:green:blue:red:green:blue"
    ),

    # Fantastic Four: deep space, suit white, FF blue, various blues & highlights
    "marvel_fantasticfour_long": (
        "001533:F5F8FF:0059D6:001A4D:1FA4FF:003A8C:"
        "F5F8FF:00214D:4DA3FF:000814:CCE0FF:003366:FFFFFF"
    ),
    #   0 deep navy (neg)
    #   1 suit white (zero)
    #   2 FF blue (pos)
    #   6 suit white again (middle, zero)

    # The Thing: rocky orange vs blue trunks
    "marvel_thing_long": (
        "3A1C0A:D77A21:004FAD:4D2A10:FF9B30:D77A21:"
        "0B1A3D:FFB866:05122C:7FB0FF:000814"
    ),
    #   0 deep rock shadow (neg)
    #   1 warm rock orange (zero)
    #   2 blue trunks (pos)
    #   5 rock orange again (middle, zero)

    # Doctor Doom: Latverian green cloak and cold armor
    "marvel_drdoom_long": (
        "10120F:1F5D3A:C8CACC:0B2618:3E8C54:1F5D3A:"
        "555B60:8FA29F:1D2227:C8CACC:050608"
    ),
    #   0 dark iron (neg)
    #   1 Doom green (zero)
    #   2 bright steel (pos)
    #   5 Doom green again (middle, zero)

    # Ultron: gunmetal body with red energy core
    "marvel_ultron_long": (
        "12141A:70747C:FF1133:2B2F36:9EA3AC:70747C:"
        "3A3F49:C3CAD2:1C1F26:FF5A6B:0A0B0F"
    ),
    #   0 deep gunmetal (neg)
    #   1 mid steel gray (zero)
    #   2 red core glow (pos)
    #   5 mid steel gray again (middle, zero)

    # Spidey: navy -> paper -> red, then wavy between dark/light blues & warms
    "marvel_spiderman_long": (
        "003366:F0F0F0:C00018:001D3D:FF4B2B:F0F0F0:"
        "8C001A:FF8C00:001B44:FFD700:000814"
    ),
    #   0  navy (neg)
    #   1  paper white (zero)
    #   2  Spidey red (pos)
    #   5  paper again (middle, zero)
    #   dark/light interleaved across blues, reds, oranges, yellow

    # Iron Man: armor shadow -> red core -> gold, with bouncing brightness
    "marvel_ironman_long": (
        "1A1A1A:851313:FFD700:3D0A0A:FF8F00:851313:"
        "FFE082:3B2F2F:FFF176:2B2B2B:FFFDE7"
    ),
    #   0  dark armor (neg)
    #   1  deep red (zero)
    #   2  gold (pos)
    #   5  deep red again (middle, zero)
    #   wavy between dark maroons, hot ambers, and pale golds

    # Hulk: purple shadows + gamma greens + neon pops
    "marvel_hulk_long": (
        "1A1433:3A5F0B:B2FF59:143D12:8BC34A:3A5F0B:"
        "4CAF50:10300A:81C784:1B5E20:E6FFB3"
    ),
    #   0  purple shadow (neg)
    #   1  dark Hulk green (zero)
    #   2  neon green (pos)
    #   5  dark Hulk green again (middle, zero)
    #   oscillates between deep greens and bright/lime/pale greens

    # Cosmic: deep space, magenta, cyan, yellow, back into dark & starlight
    "marvel_cosmic_long": (
        "240046:0B1724:FF6FF2:111827:4CC9F0:3C096C:"
        "0B1724:7B2CBF:FEE440:4361EE:FF9E00:03071E:F0F0F5"
    ),
    #   0  deep purple (neg)
    #   1  dark navy (zero)
    #   2  cosmic magenta (pos)
    #   6  dark navy again (middle, zero)
    #   lots of dark/light bouncing via cyan, yellow, blue, orange, starlight

    # Team palette: Cap blue / white / red + gold, teal, purple, etc.
    "marvel_team_long": (
        "002868:F5F5F5:BF0A30:1A1A1A:FFE082:0D1B2A:"
        "F5F5F5:00B4D8:FF6F00:7B2CBF:FFE082:003366:FFFDE7"
    ),
    #   0  flag blue (neg)
    #   1  paper white (zero)
    #   2  flag red (pos)
    #   6  paper white again (middle, zero)
    #   weaves through darks and brights: gold, deep blue, cyan, orange, purple

    # 1) Hard black/white striping + midgray center
    #    3-color version: black / white / black (pretty ugly)
    #    multipoint: strong banding / stripe-like transitions
    "debug_stripes_bw_11": (
        "000000:FFFFFF:000000:FFFFFF:000000:808080:"
        "FFFFFF:000000:FFFFFF:000000:FFFFFF"
    ),
    # idx 0..10, N=11 -> center at idx 5 (808080)

    # 2) Chaotic rainbow zigzag; center is neutral gray
    #    3-color: dark purple / neon yellow / electric blue (harsh)
    #    multipoint: wild oscillation through the spectrum
    "debug_rainbow_zigzag_13": (
        "2E004F:FFF700:0011FF:00FF5F:B00000:00F0FF:"
        "808080:FF00B4:004F4F:FF7F00:001A72:7FFF00:000000"
    ),
    # center (idx 6) = 808080

    # 3) "Thermal sawtooth": hot/cold repeating, zero is a dark gray
    #    3-color: deep navy / pale warm / bright orange (not balanced)
    #    multipoint: repeated dark->bright jumps like a saw
    "debug_thermal_saw_11": (
        "00003C:FFEEAA:FF8000:1C1C1C:FFD000:404040:"
        "FFA000:606060:FFE080:808080:FFF8C0"
    ),
    # center (idx 5) = 404040

    # 4) Bit-ish RGB ladder: marching through dark primaries to brights
    #    3-color: black / dark red / dark green (bad diverging)
    #    multipoint: interesting as a weird categorical-ish gradient
    "debug_rgb_bits_11": (
        "000000:200000:002000:000020:FF0000:00FF00:"
        "0000FF:00FFFF:FF00FF:FFFF00:FFFFFF"
    ),
    # center (idx 5) = 00FF00

    # 5) Sawtooth luminance: dark/light/dark/light with tiny hue shifts
    #    3-color: black / dark gray / light gray (very low interest)
    #    multipoint: a high-frequency luminance pattern along the range
    "debug_saw_luma_13": (
        "000000:333333:AAAAAA:222222:DDDDDD:111111:"
        "F0F0F0:2A2A2A:C8C8C8:1A1A1A:B0B0B0:080808:FFFFFF"
    ),
    # center (idx 6) = F0F0F0 (nearly white)
    "metal_checkerboard_13": (
        "101215:E2E3E5:2A2D32:F5F5F7:3B3F45:E0DFD8:7F848C:"
        "E0DFD8:3B3F45:F5F5F7:2A2D32:E2E3E5:101215"
    ),

     # 1) Cool chrome / mirror steel
    "metal_chrome_13": (
        "090B10:BFC3C9:F5F7FA:2C3139:D3D7DE:1B2027:"
        "BFC3C9:1B2027:D3D7DE:2C3139:F5F7FA:BFC3C9:090B10"
    ),

    # 2) Brushed aluminum (subtle, cool gray waves)
    "metal_brushed_aluminum_13": (
        "0E1013:AEB2B8:E4E7EB:3A3E44:C8CCD2:272B30:"
        "AEB2B8:272B30:C8CCD2:3A3E44:E4E7EB:AEB2B8:0E1013"
    ),

    # 3) Bronze (warm, statuary metal)
    "metal_bronze_13": (
        "23140C:9A6434:E0A35C:5B3419:C4833D:3A2111:"
        "9A6434:3A2111:C4833D:5B3419:E0A35C:9A6434:23140C"
    ),

    # 4) Copper (pipes, cookware, patina-ready)
    "metal_copper_13": (
        "1F120E:A15C34:E39A6A:5A2815:C87546:3A1B10:"
        "A15C34:3A1B10:C87546:5A2815:E39A6A:A15C34:1F120E"
    ),

    # 5) Rusted steel (oxidized orange + cool steel)
    "metal_rusted_steel_13": (
        "15171B:6E716F:C26A3A:243137:9A4F2C:3A474C:"
        "6E716F:3A474C:9A4F2C:243137:C26A3A:6E716F:15171B"
    ),

    # 6) Titanium (cool gray with blue/violet sheens)
    "metal_titanium_13": (
        "0C1016:7E8187:C4C7D0:303848:969AAF:222733:"
        "7E8187:222733:969AAF:303848:C4C7D0:7E8187:0C1016"
    ),

    # 7) Bi-metal: copper below zero, iron above
    "metal_bimetal_copper_iron_13": (
        "2A1308:7A746B:B7BCC3:4A2712:B66A36:D8945A:"
        "7A746B:9B9FA6:4A4F55:D7DADF:353A40:B7BCC3:101317"
    ),
    #   < 0 → mostly copper browns/oranges into the center
    #   > 0 → iron/steel grays and highlights

    # 8) Bi-metal: gold below zero, steel above
    "metal_bimetal_gold_steel_13": (
        "3A290A:8C7A3A:C0CBD5:A67C1F:F2C649:FFF0B0:"
        "8C7A3A:AEB7C1:4B5460:D9E1EA:333A44:C0CBD5:111318"
    ),

    # 9) Bi-metal: bronze below zero, silver above
    "metal_bimetal_bronze_silver_13": (
        "24130B:8F6033:D0D4DB:5A3118:C5843D:E8AF6A:"
        "8F6033:C0C5CC:4A4E54:E4E8EE:32353A:D0D4DB:0E1013"
    ),

    # 10) Bi-metal: cobalt-ish blue metal below, nickel above
    "metal_bimetal_cobalt_nickel_13": (
        "020819:3C5D9A:B5BABE:16356A:4F7AC4:A0B5E6:"
        "3C5D9A:9B9FA2:4E5356:D1D5D8:303335:B5BABE:0B0D0F"
    ),

    # Strong primaries, poster-y
    "bauhaus_primary_13": (
        "0038A8:F5F0E6:D00000:000000:F9E547:0038A8:F5F0E6:D00000:111111:F9E547:1E5AA8:F5F0E6:D00000"
    ),
    #   -1 deep cobalt blue, 0 warm paper, +1 poster red

    # Blue–yellow emphasis, very “schoolbook Bauhaus”
    "bauhaus_yellow_blue_13": (
        "002B7F:FAF3DD:FFC600:111111:005BBB:FFF1B2:FAF3DD:FFC600:003566:FFDD00:1A1A1A:FAF3DD:FFC600"
    ),

    # Red / black / white / yellow – classic graphic posters
    "bauhaus_red_black_13": (
        "111111:F5F5F5:D10000:2B2B2B:FFCC00:111111:F5F5F5:D10000:4A4A4A:FFE066:000000:F5F5F5:D10000"
    ),

    # Softer Bauhaus-influenced pastels
    "bauhaus_pastel_13": (
        "005F73:F8F3E6:FFB703:0B3C49:EAE2B7:0081A7:F8F3E6:FFB703:FF7B00:F4D58D:001427:F8F3E6:FFB703"
    ),

    # Minimal, UI-ish but still Bauhaus primaries
    "bauhaus_minimal_13": (
        "202124:FAFAFA:E53935:121212:FBC02D:1E88E5:FAFAFA:E53935:424242:FFF59D:1565C0:F5F5F5:E53935"
    ),

    # Grid / composition with black lines and primary blocks
    "bauhaus_grid_13": (
        "000000:F5F0E6:FF0000:0038A8:F9E547:000000:F5F0E6:FF0000:0038A8:F9E547:111111:F5F0E6:FF0000"
    ),

    # Shapes: blue (square), yellow (circle), red (triangle), black outlines
    "bauhaus_shapes_13": (
        "0038A8:FFF7E0:FFD000:000000:D10000:F4E4B0:FFF7E0:FFD000:0038A8:FFE066:111111:FFF7E0:FFD000"
    ),

    # Muted Bauhaus – deep blues, warm creams, red accent
    "bauhaus_muted_13": (
        "1C2A3A:F4F1E8:E53935:3F4A5A:F2D16B:26415A:F4F1E8:E53935:5A6B7A:F7E4A8:121212:F4F1E8:E53935"
    ),

    # Blue / yellow with modern aqua variants
    "bauhaus_blue_yellow_13": (
        "003F88:FAF3DD:FFB703:001219:8ECAE6:FFCB77:FAF3DD:FFB703:219EBC:FFDD00:000000:FAF3DD:FFB703"
    ),

    # Mostly monochrome with sharp color accents
    "bauhaus_monochrome_accent_13": (
        "202124:FAFAFA:FFC107:111111:FF5252:1E88E5:FAFAFA:FFC107:424242:FFCA28:1565C0:F5F5F5:FFC107"
    ),

     # 1) Deep blue night, spiritual yellow, red accents
    "blaue_reiter_night_11": (
        "08152A:F4F0E8:F6E45C:102A54:C0392B:F4F0E8:"
        "26499B:F2994A:050814:C6D4F5:0B1B32"
    ),
    # -1 deep blue, 0 warm paper, +1 luminous yellow
    # Wavy through dark/light blues, red, orange, pale sky

    # 2) Franz Marc-ish horses: blue, green, yellow, orange
    "blaue_reiter_marc_horses_9": (
        "0B2340:F5EBDD:3FA34D:F2C94C:F5EBDD:"
        "2F6DB5:F2994A:6C4AA5:050814"
    ),
    # -1 deep blue, 0 light parchment, +1 vivid green
    # Center is parchment again; fields of blue, yellow, orange, violet around it

    # 3) Kandinsky-like multipatch: red, yellow, blue, green, purple
    "blaue_reiter_kandinsky_11": (
        "151640:F6F3EB:D6453D:F6E45C:1E5AA8:F6F3EB:"
        "2F9E44:5C3C8C:F2994A:2CA5A5:08152A"
    ),
    # -1 deep indigo, 0 light paper, +1 warm red
    # Middle is paper again; ring of yellow, blue, green, purple, orange, turquoise

    # 4) Spiritual blue/white: quiet and luminous
    "blaue_reiter_spiritual_7": (
        "0A1638:F7F4ED:1E5AA8:F7F4ED:7EC3E6:5C3C8C:020616"
    ),
    # -1 deep ultramarine, 0 ivory, +1 strong blue
    # Center ivory again, then soft cyan, violet, midnight

    # 5) Fields & hills: greens, yellows, blues
    "blaue_reiter_fields_9": (
        "062218:F6F2E6:8CBF26:F2C94C:F6F2E6:"
        "2C7C78:2F6DB5:4C3C72:050814"
    ),
    # -1 deep forest, 0 light cream, +1 yellow‑green
    # Middle cream again; waves of teal, blue, violet, night

    # 6) Storm over the city: dramatic sky, lightning, red
    "blaue_reiter_storm_11": (
        "101319:F5F0E7:233B8B:050709:D9A441:F5F0E7:"
        "134A63:B52A2A:2B1640:C8D3EB:000000"
    ),
    # -1 dark slate, 0 warm paper, +1 intense blue
    # Center paper again; mustard lightning, teal, blood red, violet, pale sky, black

    # 1) Ember / cinders: deep char, bone paper, blood red, flare yellows
    "blaue_reiter_apocalypse_ember_11": (
        "140C0C:F5EEDF:D32F2F:2A1A14:FFB300:F5EEDF:"
        "5C2A1A:FFD54F:3B0F0F:FFE082:000000"
    ),
    # -1 deep char red-brown, 0 pale bone paper, +1 blood red

    # 2) Yellow–black judgement: hard yellow & soot
    "blaue_reiter_apocalypse_yellowblack_9": (
        "050507:F6F1DE:FFD000:1A1308:F6F1DE:"
        "3B2907:FFEA7A:000000:FFF8D0"
    ),
    # -1 nearly black, 0 parchment, +1 harsh yellow

    # 3) Blood sky: dark blue horizon, pale light, blood red
    "blaue_reiter_apocalypse_bloodsky_11": (
        "050B1A:F4F0E5:8B0000:0F1C38:F2C94C:F4F0E5:"
        "3C1C1C:FFA726:1B1033:FFE082:000000"
    ),
    # -1 deep navy, 0 warm off-white, +1 dark blood red

    # 4) Four horsemen: gold, plague green, blood, shadow
    "blaue_reiter_apocalypse_four_horsemen_11": (
        "090909:F5EBD8:FFB703:4B1A21:9BC53D:F5EBD8:"
        "5E3C99:F9844A:1A1A1A:FFE066:000000"
    ),
    # -1 black, 0 parchment, +1 gold

    # 5) Scorched earth: burnt browns & embers
    "blaue_reiter_apocalypse_scorched_9": (
        "120A06:F6EDDD:C4511F:3B1C0D:F6EDDD:"
        "6E391A:FFB561:1A0C06:FFE0B2"
    ),
    # -1 very dark brown, 0 light ash paper, +1 scorch orange

    # 6) Smoke and ash: almost monochrome with one molten accent
    "blaue_reiter_apocalypse_smoke_ash_7": (
        "070708:F3F3F3:BB7A00:F3F3F3:514C48:E0D4C4:000000"
    ),
    # -1 black, 0 very light gray, +1 molten amber
    # Gene Davis–ish vertical stripes: black / paper / magenta, then stripey
    "washington_gene_davis_stripes_11": (
        "111111:F5F5F5:FF006E:004E92:FFD500:F5F5F5:"
        "00A8E8:FF5A00:2E2E2E:FFC0CB:000000"
    ),

    # Noland targets / chevrons: blue / paper / red, then teal / gold / blue / salmon
    "washington_noland_target_9": (
        "001F3F:F4F2E8:D62828:008C7E:F4F2E8:"
        "FFB703:2F5DA8:FEC5BB:111111"
    ),

    # Morris Louis veils: deep violet, paper, cool blue, then aqua and warm gold
    "washington_louis_veil_7": (
        "120F2E:F5F1E8:4F7CAC:F5F1E8:9BD5C0:F2C57C:2F1A24"
    ),

    # Downing dots / grids: very poppy primaries on neutral ground
    "washington_downing_dots_9": (
        "000000:F5F5F5:0077B6:F72585:F5F5F5:"
        "3A0CA3:FFBA08:03071E:FFF3B0"
    ),

    # Mehring-ish soft fields: muted but clear color blocks
    "washington_mehring_soft_11": (
        "1C1C25:F7F2EA:FF6F61:88C0D0:F2C94C:F7F2EA:"
        "A3BE8C:5E81AC:F4A261:2E3440:F9E6E6"
    ),

    # Leon Berkowitz luminous atmosphere: blue–violet and warm haze
    "washington_berkowitz_luminous_9": (
        "050716:F6F3ED:527EA3:BFD7EA:F6F3ED:"
        "F5D0A9:C46F70:1E2740:000000"
    ),

    # Sam Gilliam–ish stained / draped color: wine, indigo, orange, green, gold
    "washington_gilliam_drape_11": (
        "050208:F5EFE8:9D174D:2F195F:E87D1E:F5EFE8:"
        "167E5D:FFB703:6A040F:240046:FDF0D5"
    ),

     # High-contrast BW + primaries, classic stripe poster
    "washington_stripe_bw_primary_11": (
        "111111:F5F5F5:D32F2F:000000:FFFFFF:F5F5F5:"
        "000000:0033CC:000000:F5F5F5:FFFF00"
    ),
    # -1 dark gray, 0 light paper, +1 strong red

    # CMYK-ish: black/white, cyan/magenta/yellow between
    "washington_stripe_cmyk_11": (
        "000000:F5F5F5:00B4FF:000000:FF00AA:F5F5F5:"
        "000000:FFFF00:000000:F5F5F5:000000"
    ),
    # -1 black, 0 white, +1 cyan

    # Neon nightclub stripes: black/white + magenta, green, cyan, yellow
    "washington_stripe_neon_13": (
        "050505:F7F7F7:FF00E6:000000:39FF14:000000:"
        "F7F7F7:00F0FF:050505:FFE600:000000:F7F7F7:FF00E6"
    ),
    # -1 near black, 0 cool white, +1 neon magenta

    # Subtle grayscale stripes (good for more restrained stripe textures)
    "washington_stripe_subtle_9": (
        "202124:F5F5F5:8E8E8E:111111:F5F5F5:3A3A3A:"
        "DADADA:111111:E0E0E0"
    ),
    # -1 dark gray, 0 soft white, +1 mid gray

    # Teal–orange stripe pack (cinematic but stripey)
    "washington_stripe_teal_orange_11": (
        "081217:F5F1E8:FF6F00:031017:00A8A8:F5F1E8:"
        "1A1A1A:FFB703:031017:F5F1E8:FF6F00"
    ),
    # -1 deep blue/black, 0 warm off-white, +1 orange

    # Violent rainbow stripes: B/W with RGBY slices
    "washington_stripe_rainbow_13": (
        "000000:F5F5F5:FF0000:000000:FFA500:F5F5F5:"
        "F5F5F5:FFFF00:000000:00FF00:000000:0000FF:000000"
    ),
    # -1 black, 0 white, +1 red

    # Pure “ink on paper” stripes, very print-like
    "washington_stripe_ink_paper_9": (
        "050505:F5F0E6:1A1A1A:000000:F5F0E6:111111:"
        "F5F0E6:000000:F5F0E6"
    ),
    # -1 almost black, 0 warm paper, +1 dark gray

    # Cool–warm alternation: blue/gray vs coral
    "washington_stripe_coolwarm_11": (
        "060B12:F5F5F5:FF6F61:000000:4C82C3:F5F5F5:"
        "111111:F4A261:000000:F5F5F5:264653"
    ),

    "washington_stripe_sorbet_11": (
        "101010:FDF7F2:FF8FA3:000000:FFCFA0:FDF7F2:"
        "000000:7FD1FF:000000:FDF7F2:9C88FF"
    ),
    # -1 deep charcoal, 0 warm off-white, +1 soft pink; pastel stripes with black breaks
    
    # -1 deep navy, 0 white, +1 warm coral
    # Deep sea → sand → pines / rocks
    "beach_coastline_11": (
        "355A7C:EDE7CF:9EAF8A:6BAAD7:D4CFB2:EDE7CF:"
        "B7A88E:999B76:675D45:3E3C2F:191B1F"
    ),

    # Water / reef vibes: deep → turquoise → pale sand
    "beach_reef_11": (
        "355A7C:EDE7CF:6BAAD7:678499:5897D9:EDE7CF:"
        "739AAB:95B7C1:D4CFB2:C4BEA0:B7A88E"
    ),

    # Cliffs & pines: shadows, rock, scrub, sky
    "beach_cliff_pines_11": (
        "191B1F:EDE7CF:89805F:3E3C2F:675D45:EDE7CF:"
        "B7A88E:999B76:9EAF8A:D4CFB2:3B7BF9"
    ),

    # Soft pastel foam & shallows
    "beach_pastel_foam_9": (
        "355A7C:EDE7CF:95B7C1:6BAAD7:EDE7CF:"
        "D4CFB2:C4BEA0:B7A88E:999B76"
    ),

    # Darker / twilight reinterpretation
    "beach_twilight_9": (
        "191B1F:EDE7CF:2574EA:355A7C:EDE7CF:"
        "3E3C2F:675D45:89805F:D4CFB2"
    ),

    # Olive scrub + warm sand
    "beach_olive_sand_7": (
        "3E3C2F:EDE7CF:999B76:EDE7CF:"
        "B7A88E:D4CFB2:9EAF8A"
    ),

    # Clear sky + bright water + warm sand
    "beach_clear_sky_11": (
        "355A7C:EDE7CF:3B7BF9:6BAAD7:D4CFB2:EDE7CF:"
        "5897D9:95B7C1:C4BEA0:2574EA:B7A88E"
    ),
    # -1 deep water blue, 0 warm sand, +1 vivid sky blue

    # Turquoise cove: dark shadow water into light sand + turquoise
    "beach_turquoise_cove_9": (
        "191B1F:EDE7CF:6BAAD7:3E3C2F:EDE7CF:"
        "D4CFB2:739AAB:C4BEA0:B7A88E"
    ),
    # -1 deep shadow, 0 sand, +1 turquoise shallows

    # Pebble shore: rocks, sand, scrub
    "beach_pebble_shore_7": (
        "3E3C2F:EDE7CF:999B76:EDE7CF:"
        "B7A88E:D4CFB2:675D45"
    ),
    # -1 dark rock, 0 pale sand, +1 olive-sand mix

    # Sky gradient over Bugliaca: deep horizon into high, pale sky
    "bugliaca_sky_11": (
        "3F5160:B2D2DF:DEE6E9:5D727F:88BFD7:B2D2DF:"
        "67ABCD:5AA3C8:519CC1:DFE7EA:85AABC"
    ),

    # Mountain faces & snow fields
    "bugliaca_mountains_11": (
        "1A1F23:DFE7EA:85AABC:1D292F:8BAEBE:DFE7EA:"
        "3D647E:324957:BDD9E3:224459:26343B"
    ),

    # Dark pine forest with glimpses of sky
    "bugliaca_forest_11": (
        "1D2B2C:416052:88BFD7:30463C:3F5160:416052:"
        "2A302E:67ABCD:30463C:1D2B2C:253B3D"
    ),

    # Deep valley at dusk, mostly shadow blues and greens
    "bugliaca_valley_dusk_9": (
        "1A1F23:30463C:3D647E:1B3343:30463C:"
        "253B3D:1D292F:202426:1D2B2C"
    ),

    # Fence and meadow in front of the view
    "bugliaca_fence_meadow_9": (
        "202426:506467:416052:253B3D:506467:"
        "30463C:203035:2A302E:B2D2DF"
    ),

    # High glacier light: snow and high-altitude blues
    "bugliaca_glacier_9": (
        "1A1F23:DFE7EA:88BFD7:5D727F:DFE7EA:"
        "BDD9E3:85AABC:519CC1:DEE6E9"
    ),

    # Panorama mix: sky, forest, and rock together
    "bugliaca_panorama_11": (
        "1A1F23:B2D2DF:416052:3D647E:88BFD7:B2D2DF:"
        "30463C:5AA3C8:253B3D:DFE7EA:1D2B2C"
    ),

    # Misty/overcast interpretation of the same view
    "bugliaca_mist_9": (
        "1D292F:8BAEBE:DEE6E9:5D727F:8BAEBE:"
        "30463C:DFE7EA:26343B:BDD9E3"
    ),

    # Evergreen focus: dense dark pines
    "bugliaca_evergreen_7": (
        "1D2B2C:30463C:416052:30463C:2A302E:253B3D:1D2B2C"
    ),

    # Soft gradient, trunks → mid‑snow → highlights
    "winter_forest_soft_11": (
        "141318:6C7788:EFEEEB:262931:545C6B:6C7788:"
        "818B9B:97A0AE:ABB4C0:C0C9D3:EFEEEB"
    ),

    # Darker emphasis, more about trunks and shadow
    "winter_forest_deep_9": (
        "141318:545C6B:ABB4C0:262931:545C6B:"
        "3B414E:6C7788:818B9B:C0C9D3"
    ),

    # High‑contrast, compact palette
    "winter_forest_contrast_7": (
        "141318:818B9B:EFEEEB:818B9B:"
        "3B414E:C0C9D3:6C7788"
    ),

    # Wavy, alternating dark/light like tree vs snow bands
    "winter_forest_wavy_11": (
        "141318:6C7788:EFEEEB:262931:97A0AE:6C7788:"
        "C0C9D3:3B414E:ABB4C0:262931:EFEEEB"
    ),

    # Misty / atmospheric version, skewed to lighter values
    "winter_forest_mist_11": (
        "3B414E:ABB4C0:EFEEEB:545C6B:C0C9D3:ABB4C0:"
        "818B9B:97A0AE:EFEEEB:6C7788:141318"
    ),
    # 6) Deep night snow – darker overall
    "winter_forest_night_11": (
        "141318:6C7788:EFEEEB:262931:818B9B:6C7788:"
        "97A0AE:3B414E:ABB4C0:262931:EFEEEB"
    ),

    # 7) Sky glow – lighter mids, darker ends
    "winter_forest_skyglow_9": (
        "141318:ABB4C0:EFEEEB:3B414E:ABB4C0:"
        "C0C9D3:6C7788:97A0AE:141318"
    ),

    # 8) Blue veils – cool blue‑gray center, soft whites
    "winter_forest_blue_veils_9": (
        "141318:97A0AE:EFEEEB:3B414E:97A0AE:"
        "C0C9D3:545C6B:ABB4C0:141318"
    ),

    # 9) Ink drawing – stark trunks with limited snow tones
    "winter_forest_ink_7": (
        "141318:6C7788:C0C9D3:6C7788:262931:ABB4C0:141318"
    ),

}

COLOR_STRINGS = {}
COLOR_STRINGS.update(COLOR_TRI_STRINGS)
COLOR_STRINGS.update(COLOR_LONG_STRINGS)




def _rgb_scheme_palette_eq(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Colorize using a named palette from COLOR_TRI_STRINGS.
    """
    palette_name = params.get("palette")
    if palette_name is None:
        raise ValueError(
            "params['palette'] must be set to a key in COLOR_TRI_STRINGS"
        )

    if palette_name not in COLOR_TRI_STRINGS:
        raise KeyError(
            f"Unknown palette {palette_name!r}. "
            f"Available: {', '.join(sorted(COLOR_TRI_STRINGS.keys()))}"
        )

    palette_spec = COLOR_TRI_STRINGS[palette_name]
    try:      
        parts = palette_spec.split(":")
        neg_spec, zero_spec, pos_spec = parts[:3]
    except ValueError as exc:
        raise ValueError(
            f"Invalid COLOR_TRI_STRINGS entry for {palette_name!r}: {palette_spec!r} "
            "(expected 'NEG:ZERO:POS')"
        ) from exc

    # Clone params so we don't mutate the caller's dict
    sub_params = dict(params)
    sub_params["neg_color"] = neg_spec
    sub_params["zero_color"] = zero_spec
    sub_params["pos_color"] = pos_spec

    return _rgb_scheme_mh_eq(lyap, sub_params)


def _rgb_scheme_multipoint(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Colorize using N color stops between -1 and +1.

    params:
        palette : str (preferred)
            Name of a palette in COLOR_STRINGS. Its value is a colon-
            separated list "HEX:HEX:HEX:...". All colors are used as
            equidistant stops in [-1, +1].

        color_string : str (optional override)
            If provided, overrides 'palette'. Same format as above.

        gamma : float (optional)
            Gamma applied to normalized coordinate in [0, 1].
            gamma <= 0 is treated as 1 (no gamma).

    Values outside [-1, +1] are clamped before mapping.
    Non-finite entries are left black.
    """
    arr = np.asarray(lyap, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return rgb

    # Choose source of color string
    color_string = params.get("color_string")
    if not color_string:
        palette_name = params.get("palette")
        if not palette_name:
            raise ValueError(
                "scheme_multipoint requires either params['palette'] "
                "or params['color_string']"
            )
        try:
            color_string = COLOR_STRINGS[palette_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown palette {palette_name!r} for scheme_multipoint"
            ) from exc

    # Parse into list of RGB triples
    specs = [s.strip() for s in color_string.split(":") if s.strip()]
    if len(specs) < 2:
        raise ValueError(
            "scheme_multipoint needs at least 2 colors "
            "in color_string / palette"
        )

    colors = []
    for spec in specs:
        r, g, b = _parse_color_spec(spec, (0.0, 0.0, 0.0))
        colors.append((r, g, b))

    colors = np.asarray(colors, dtype=np.float64)
    N = colors.shape[0]

    # Map [-1, +1] -> [0, 1], clamp
    vals = arr[finite]
    t = (np.clip(vals, -1.0, 1.0) + 1.0) * 0.5  # in [0, 1]

    gamma = params.get("gamma", DEFAULT_GAMMA)
    gamma = 1.0 if gamma <= 0.0 else float(gamma)
    if gamma != 1.0:
        t = t ** gamma

    # N colors => N-1 segments in [0,1]
    segment_float = t * (N - 1)
    idx_low = np.floor(segment_float).astype(np.int64)
    idx_low = np.clip(idx_low, 0, N - 2)
    frac = segment_float - idx_low

    c0 = colors[idx_low]           # (M, 3)
    c1 = colors[idx_low + 1]       # (M, 3)
    frac = frac[:, np.newaxis]     # (M, 1) for broadcasting

    rgb_vals = np.rint((1.0 - frac) * c0 + frac * c1).astype(np.uint8)
    rgb[finite] = rgb_vals

    return rgb


# Scheme registry: ADD NEW SCHEMES HERE ONLY
RGB_SCHEMES: dict[str, dict] = {
    "mh": dict(
        func=_rgb_scheme_mh,
        defaults=dict(
            gamma=0.25,
            pos_color="FF0000",
            zero_color="000000",
            neg_color="FFFF00",
        ),
    ),

    "mh_eq": dict(
        func=_rgb_scheme_mh_eq,
        defaults=dict(
            gamma=1,
            pos_color="FF0000",
            zero_color="000000",
            neg_color="FFFF00",
            nbins=2048,
        ),
    ),

    "palette": dict(
        func=_rgb_scheme_palette_eq,
        defaults=dict(
            palette="bauhaus_primaries",
            gamma=1,
            nbins=2048,
        ),
    ),

    "multi": dict(
        func=_rgb_scheme_multipoint,
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

def spec2lyapunov(spec: str, pix: int = 5000) -> np.ndarray:
    specdict = specparser.split_chain(spec)

    map_name = None
    for op in specdict.keys():
        if op in MAP_TEMPLATES:
            map_name = op
            break
    if map_name is None:
        raise SystemExit(f"No map name found in spec {spec}")

    pardict= MAP_TEMPLATES[map_name]["pardict"]
    for i,(key,value) in enumerate(pardict.items()):
        if specdict.get(key) is not None:
            param_value = specdict.get(key)[0]
        else:
            param_value = value
        pardict[key] = param_value
    _build_map(map_name)
    map_cfg =_get_map(map_name)
    type = map_cfg.get("type", "step1d")

     # --- domain + sequence parsing ---
    domain = map_cfg["domain"].copy()

    # do we need an A/B sequence for this map?
    use_seq = (type=="step1d") or (type=="step2d_ab")
    seq_arr = _seq_to_array(DEFAULT_SEQ) if use_seq else None

    domain_idx = 0
    for i, v in enumerate(specdict[map_name]):
        # first positional token can be the sequence (e.g. 'BA')
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

    a0 = _get_float(specdict, "a0", domain[0])
    b0 = _get_float(specdict, "b0", domain[1])
    a1 = _get_float(specdict, "a1", domain[2])
    b1 = _get_float(specdict, "b1", domain[3])

    domain_affine = _build_affine_domain(specdict, a0, b0, a1, b1)

    x0    = _get_float(specdict, "x0", map_cfg.get("x0", 0.5))
    y0    = _get_float(specdict, "y0", map_cfg.get("y0", 0.5))
    n_tr  = _get_int(specdict, "trans", map_cfg.get("trans", DEFAULT_TRANS))
    n_it  = _get_int(specdict, "iter", map_cfg.get("iter",  DEFAULT_ITER))
    eps   = _get_float(specdict, "eps",   DEFAULT_EPS_LYAP)
 
    if type == "step1d":
        lyap = _lyapunov_field_1d(
            map_cfg["step"],
            map_cfg["deriv"],
            seq_arr,
            domain_affine,
            int(pix),
            float(x0),
            int(n_tr),
            int(n_it),
            float(eps),
            
        )
    elif type == "step2d_ab":
        if seq_arr is None:
            raise RuntimeError("internal bug: seq_arr is None for AB-forced map")
        print("lyapunov_field_generic_2d_ab")
        eps_floor = map_cfg.get("eps_floor", 1e-16)
        lyap = _lyapunov_field_2d_ab(
            map_cfg["step2_ab"],
            map_cfg["jac2_ab"],
            seq_arr,
            domain_affine,
            pix,
            x0, 
            y0,
            n_tr, 
            n_it,
            eps_floor,
        )
    elif type == "step2d":
        print("lyapunov_field_generic_2d")
        eps_floor = map_cfg.get("eps_floor", 1e-16)
        lyap = _lyapunov_field_2d(
            map_cfg["step2"],
            map_cfg["jac2"],
            domain_affine,
            int(pix),
            float(x0),
            float(y0),
            int(n_tr),
            int(n_it),
            float(eps_floor),
        )
    else:
        raise SystemExit(f"Unsupported type={type} for map '{map_name}'")

    rgb = lyapunov_to_rgb(lyap, specdict)
    return rgb

# ---------------------------------------------------------------------------
# expansion helpers
# ---------------------------------------------------------------------------

def get_all_palettes(palette_regex, maxp):
    print(f"all palette search:{palette_regex}")
    pat = regex.compile(palette_regex)
    out = []
    for k in COLOR_STRINGS.keys():
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
    for k in COLOR_LONG_STRINGS.keys():
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
    for k in COLOR_TRI_STRINGS.keys():
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
        COLOR_STRINGS[k]=v

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

