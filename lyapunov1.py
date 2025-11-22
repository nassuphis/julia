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

# ---------------------------------------------------------------------------
# Build python functions from expression strings, then JIT them
# ---------------------------------------------------------------------------

def _funtext(expr: str, name: str) -> str:
    """
    Emit a tiny Python function for jit-ing
    """
    lines = [
        f"def {name}(x, r, param):",
        "    a   = param[0]",
        "    b   = param[1]",
        "    c   = param[2]",
        "    d   = param[3]",
       f"    return {expr}",
    ]
    return "\n".join(lines)

def _funtext_2d_step(expr_x: str, expr_y: str, name: str) -> str:
    lines = [
        f"def {name}(x, y, r, s, param):",
        "    a = param[0]",
        "    b = param[1]",
        "    c = param[2]",
        "    d = param[3]",
        f"    x_next = {expr_x}",
        f"    y_next = {expr_y}",
        "    return x_next, y_next",
    ]
    return '\n'.join(lines)


def _funtext_2d_jac(
    dXdx: str, dXdy: str, dYdx: str, dYdy: str, name: str
) -> str:
    lines = [
        f"def {name}(x, y, r, s, param):",
        "    a = param[0]",
        "    b = param[1]",
        "    c = param[2]",
        "    d = param[3]",
        f"    return {dXdx}, {dXdy}, {dYdx}, {dYdy}",
    ]
    return '\n'.join(lines)



def _make_py_func(expr: str):
    """
    Create a plain Python function f(x, r, a) that can be jitted.
    """
    ns = {
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
    }
    src = _funtext(expr, "impl")
    exec(src, ns, ns)
    return ns["impl"]

def _make_py_func_2d_step(expr_x: str, expr_y: str):
    ns = {  # same namespace as _make_py_func, plus what you need
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
    }
    src = _funtext_2d_step(expr_x, expr_y, "impl2_step")
    exec(src, ns, ns)
    return ns["impl2_step"]


def _make_py_func_2d_jac(dXdx, dXdy, dYdx, dYdy):
    ns = {  # same namespace
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
    }
    src = _funtext_2d_jac(dXdx, dXdy, dYdx, dYdy, "impl2_jac")
    exec(src, ns, ns)
    return ns["impl2_jac"]


# All jitted step/deriv functions will share this signature.
STEP_SIG = types.float64(types.float64, types.float64, types.float64[:])

STEP2_SIG = types.UniTuple(types.float64, 2)(
    types.float64,  # x
    types.float64,  # y
    types.float64,  # r
    types.float64,  # s (unused in kicked, but kept for generality)
    types.float64[:],  # param[4]
)

JAC2_SIG = types.UniTuple(types.float64, 4)(
    types.float64,  # x
    types.float64,  # y
    types.float64,  # r
    types.float64,  # s
    types.float64[:],  # param
)


# ---------------------------------------------------------------------------
# Map templates: custom functions
# ---------------------------------------------------------------------------

def parasite_step2_py(x, y, r, s, param):
    EXP_MAX = 50.0
    STATE_MAX = 1e6
    H, P = x, y
    a, rj = r,s
    K = param[0]
    invK = 1.0 / K
    
    F = rj * (1.0 - H * invK) - a * P
    if F > EXP_MAX: F = EXP_MAX
    elif F < -EXP_MAX: F = -EXP_MAX
    expF = math.exp(F)

    G = -a * P
    if G > EXP_MAX: G = EXP_MAX
    elif G < -EXP_MAX: G = -EXP_MAX
    E = math.exp(G)

    H_next = H * expF
    P_next = H * (1.0 - E)    

    H = H_next
    P = P_next

    if H > STATE_MAX: 
        H = STATE_MAX
    elif H < -STATE_MAX: 
        H = -STATE_MAX

    if P > STATE_MAX: 
        P = STATE_MAX
    elif P < -STATE_MAX: 
        P = -STATE_MAX

    if not (np.isfinite(H) and np.isfinite(P)): 
        H, P = 1.0, 1.0

    return H, P


def parasite_jac2_py(x, y, r, s, param):

    EXP_MAX = 50.0
    H, P = x, y
    a, rj = r,s
    K = param[0]
    invK = 1.0 / K

    F = rj * (1.0 - H * invK) - a * P

    if F > EXP_MAX: 
        F = EXP_MAX
    elif F < -EXP_MAX: 
        F = -EXP_MAX

    expF = math.exp(F)

    G = -a * P
    if G > EXP_MAX: 
        G = EXP_MAX
    elif G < -EXP_MAX: 
        G = -EXP_MAX
    E = math.exp(G)

    # Jacobian
    dHdH = expF * (1.0 - rj * H * invK)
    dHdP = -a * H * expF
    dPdH = 1.0 - E
    dPdP = H * a * E
    
    return dHdH, dHdP, dPdH, dPdP

def predprey_step2_py(x, y, r, s, param):
    """
    One step of the predator–prey map with periodically forced 'a':

        x_{n+1} = a_n x_n (1 - x_n - y_n)
        y_{n+1} = b x_n y_n

    Here:
        r      = a_n  (either A or B, chosen by the A/B sequence)
        param[1] = b  (constant over the whole tile)

    We also replicate the STATE_MAX bounding + NaN reset from the
    original _lyapunov_field_predprey.
    """
    STATE_MAX = 1e6

    a_param = r
    b_param = param[1]   # param[1] will be DEFAULT_PRED_B, overridden by b:...

    x_next = a_param * x * (1.0 - x - y)
    y_next = b_param * x * y

    # crude bounding to avoid numeric blowup (same thresholds as original)
    if x_next > STATE_MAX:
        x_next = STATE_MAX
    elif x_next < -STATE_MAX:
        x_next = -STATE_MAX
    if y_next > STATE_MAX:
        y_next = STATE_MAX
    elif y_next < -STATE_MAX:
        y_next = -STATE_MAX

    if not (np.isfinite(x_next) and np.isfinite(y_next)):
        x_next = 0.5
        y_next = 0.5

    return x_next, y_next


def predprey_jac2_py(x, y, r, s, param):
    """
    Jacobian of the predator–prey map at (x, y) for a_n = r, b = param[1].
    """
    a_param = r
    b_param = param[1]

    dxdx = a_param * (1.0 - 2.0 * x - y)
    dxdy = -a_param * x
    dydx = b_param * y
    dydy = b_param * x

    return dxdx, dxdy, dydx, dydy

def cardiac_step2_py(x, y, r, s, param):
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


def cardiac_jac2_py(x, y, r, s, param):
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
        dim=2,
        step2_func=cardiac_step2_py,
        jac2_func=cardiac_jac2_py,
        # domain = [tmin0, r0, tmin1, r1]
        # matching Fig. 8.3: r ∈ [0,150], t_min ∈ [20,140]
        domain=[20.0, 0.0, 140.0, 150.0],
        # params: [A, B1, B2, tau1, tau2]
        # (you can override A,B1,B2,tau1 via a:,b:,c:,d: if you like;
        # tau2 stays at the default unless you edit the map)
        params=[270.0, 2441.0, 90.02, 19.6, 200.5],
        x0=5.0,       # initial x_n (duration)
        y0=0.0,       # dummy y
        trans=100,    # n_prev
        iter=200,     # n_max
        eps_floor=1e-16,
    ),

    "predprey": dict(
        dim=2,
        forcing="ab",                # use the A/B-forced 2D kernel
        step2_func=predprey_step2_py,
        jac2_func=predprey_jac2_py,
        # domain: [A0, B0, A1, B1]  (same as old MAP_DEFAULT_DOMAIN)
        domain=[-0.04, -0.6, 4.5, 6.6],
        # params: we store b in param[1]; param[0] is unused
        params=[0.0, 3.569985, 0.0, 0.0],
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=200,
        eps_floor=1e-16,
    ),
    
    "parasite": dict(
        dim=2,
        step2_func=parasite_step2_py,
        jac2_func=parasite_jac2_py,
        #expr_x="x * exp( s*(1.0 - x/a) - r*y )",
        #expr_y="x * (1.0 - exp(-r*y))",
        domain=[-2.4, -0.1, 8.1, 3.57],
        params=[2.1, 0.0, 0.0, 0.0],
        x0=0.5,
        y0=0.5,
        trans=200,
        iter=800,
        eps_floor=1e-16,  # <- matches original parasite kernel
    ),

    "kicked": dict(
        dim=2,
        forcing="ab",   # <--- new flag so spec2lyapunov knows to use the AB kernel
        expr_x=(
            "mod1("
            "x + r*(1 + ((1-exp(-b))/b)*y) "
            "+ a*r*((1-exp(-b))/b)*cos(2*pi*x)"
            ")"
        ),
        expr_y="exp(-b)*(y + a*cos(2*pi*x))",
        jac_exprs=(
            "1 - a*r*((1-exp(-b))/b)*2*pi*sin(2*pi*x)",  # dXdx
            "r*((1-exp(-b))/b)",                         # dXdy
            "-exp(-b)*a*2*pi*sin(2*pi*x)",               # dYdx
            "exp(-b)",                                   # dYdy
        ),
        domain=[-2.45, -6.35, 1.85744, 1.4325],
        params=[0.3, 3.0, 0.0, 0.0],
        x0 = 0.4,
        y0 = 0.4,
        trans = 100,
        iter = 200,
    ),

    "henzi": dict( #henon-lozi map
        # fig 9.11 :  henzi:BA,ll:1.483:2.35,ul:2.15:1.794,lr:-0.35:0.15,gamma:0.25
        dim=2,
        forcing="ab",   
        # FIXME: classification needs a single string
        # something like "1D", "2D", "2Dab" 
        # FIXME: add a default "seq" key
        # seq="AB" or somthing
        expr_x="1-r*abs(x)+y",
        expr_y="a*x",
        # FIXME: add a sub_expr dictionary that is 
        # expanded into the formula prior to scypy
        # things inside [...] are looked up 
        # and replaced [xpr1]+ [xpr2]
        # sub_expr={'xpr1':'a+b+c', 'xpr2':'abs(x+b)'}
        domain=[-10,-10,10,10],
        params=[0.994,0.0, 0.0, 0.0],
        #FIXME: add pnames=["a","b","c","d"] so formuli can match source
        x0 = 0.4,
        y0 = 0.4,
        # FIXME: make this a list init=[0.4,0.4]
        # so the values can be specced like init:0.4:0.4
        trans = 100,
        iter = 200,
    ),

    "adbash": dict( #adams-bashworth
        dim=2,
        forcing="ab",  
        expr_x="x+(a/2)*(3*r*x*(1-x)-r*y*(1-y))",
        expr_y="x",
        domain=[-5,-5,5,5],
        params=[1,0.0, 0.0, 0.0],
        x0 = 0.5,
        y0 = 0.5,
        trans = 200,
        iter = 200,
    ),

    "degn": dict( #degn's map
        dim=2,  
        # text is "b vs r"
        # s is the LHS of the text 
        # r is the RHS of the text
        expr_x="s*(x-0.5)+0.5+a*sin(2*pi*r*y)",
        expr_y="(y+s*(x-0.5)+0.5+a*sin(2*pi*r*y)*mod1(b/s))",
        domain=[-2,-5,2,5],
        params=[0.1,1.0, 0.0, 0.0],
        x0 = 0.4,
        y0 = 0.4,
        trans = 100,
        iter = 300,
    ),

    "henon": dict(
        dim=2,
        # (x, y) -> (x', y'), using r,s as axis parameters
        expr_x="1 + y - r * x * x",  # Henon-like
        expr_y="s * x",
        domain=[1.0, 0.1, 1.4, 0.3],  # r0,s0,r1,s1
        params=[0.0, 0.0, 0.0, 0.0],
        x0=0.1,
        y0=0.1,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
        # optionally: manual Jacobian override
        # jac_exprs=(" -2*r*x", "1", "s", "0"),
    ),

    "henon2": dict(
        dim=2,
        # (x, y) -> (x', y'), using r,s as axis parameters
        expr_x="r - x*x + s*y",  # Henon-like
        expr_y="a*x+b*x*x",
        domain=[0,-0.25,2,1],  # r0,s0,r1,s1
        params=[1, 0.0, 0.0, 0.0],
        x0=0.4,
        y0=0.4,
        trans=100,
        iter=2000,
        # optionally: manual Jacobian override
        # jac_exprs=(" -2*r*x", "1", "s", "0"),
    ),

    "kst2d": dict(
        dim=2,
        # (x, y) -> (x', y'), using r,s as axis parameters
        expr_x="1+s*apow(x-c,r)-a*pow(abs(x),b)",  
        expr_y="0",
        domain=[1,1.33,3.5,2.5],  # r0,s0,r1,s1
        params=[3,2, 0.0, 0.0],
        x0=0.5,
        y0=0.0,
        trans=100,
        iter=500,
        # optionally: manual Jacobian override
    ),


    "logistic2d": dict(
        dim=2,
        # text is "r vs a"
        # s is the LHS of the text 
        # r is the RHS of the text
        expr_x="(1-s*x*x)*step(x)+(r-s*x*x)*(1-step(x))",  # Henon-like
        expr_y="0",
        domain=[0.66,-0.05,3,1.66],  # r0,s0,r1,s1
        params=[2, 0.0, 0.0, 0.0],
        x0=0.5,
        y0=1.0,
        trans=100,
        iter=300,
        # optionally: manual Jacobian override
    ),


    "logistic": dict( # Classic logistic
        expr="r * x * (1.0 - x)",
        domain=[2.5, 2.5, 4.0, 4.0],  # A0, B0, A1, B1
        params=[0.0, 0.0, 0.0, 0.0],  # a, b, c, d,
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "sine": dict( # Sine map (classical Lyapunov variant: r sin(pi x))
        expr="r * sin(pi * x)",
        domain=[0.0, 2.0, 0.0, 2.0],
        params=[0.0, 0.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "tent": dict(  # Tent map
        expr="r*x*(1-step(x-0.5)) + r*(1-x)*step(x-0.5)",
        domain=[0.0, 0.0, 2.0, 2.0],
        params=[0.0, 0.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    
    "heart": dict( # Heart-cell map: x_{n+1} = sin(alpha x_n) + r_n
        expr="sin(a * x) + r",
        domain=[0.0, 0.0, 15.0, 15.0], 
        params=[1.0, 0.0, 0.0, 0.0],    
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
        params=[270.0, 2441.0, 90.02, 53.5],

        x0=5.0,
        trans=100,   # n_prev
        iter=200,    # n_max
    ),

    "nn1": dict(
        expr="pow(r/abs(x),a)*sign(x)+cos(2*pi*r/2)*sin(2*pi*x/5)",
        #deriv_expr="0",
        domain=[0.1, 0.1, 10, 10], 
        params=[0.25, 0.0, 0.0, 0.0], 
        x0=0.5,
        trans=600,
        iter=600,
    ),

    "nn1a": dict(
        expr="pow(r/abs(x),a)*sign(x)+pow(abs(cos(2*pi*r/2)*sin(2*pi*x/5)),b)",
        #deriv_expr="0",
        domain=[0.1, 0.1, 10, 10],  
        params=[0.25, 1.0, 0.0, 0.0], 
        x0=0.5,
        trans=600,
        iter=600,
    ),

    "nn2": dict(
        expr="pow(r/abs(x),a*cos(r))*sign(x)+cos(2*pi*r/2.25)*sin(2*pi*x/3)",
        #deriv_expr="0",
        domain=[0.0, 0.0, 15.0, 15.0], 
        params=[1.0, 0.0, 0.0, 0.0],   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn3": dict(
        expr="cos(2*pi*r*x*(1-x)/a)*pow(abs(x),cos(2*pi*(r+x)/10))*sign(x)-cos(2*pi*r)*step(x)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        params=[25, 0.0, 0.0, 0.0],   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn4": dict(
        expr="pow(abs(x*x*x-r),cos(r))*sign(x)+pow(abs(r*r*r-x*x*x),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        params=[25, 0.0, 0.0, 0.0],   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn5": dict(
        expr="pow(abs(x*x*x-pow(r,a)),cos(r))*sign(x)+pow(abs(r*r*r-pow(x,b)),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        params=[3, 5, 0.0, 0.0],   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn6": dict(
        expr="pow(abs(pow(x,b)-pow(r,a)),cos(r))*sign(x)+pow(abs(pow(r,a)-pow(x,b)),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        params=[3, 5, 0.0, 0.0],   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn7": dict(
        expr="pow(abs(apow(x,b)-apow(r,a)),cos(r))*sign(x)+pow(abs(apow(r,a)-apow(x,b)),sin(x))*sign(r)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        params=[3, 5, 0.0, 0.0],   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn8": dict(
        expr="apow(apow(x,b)-apow(r,a),cos(r))+apow(apow(r,a)-apow(x,b),sin(x))",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        params=[3, 5, 0.0, 0.0],   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn9": dict(
        expr1="r*cosh(sin(apow(x,x)))-x*sinh(cos(apow(r,r)))",
        expr="r*apow(sin(pi*(x-b)),a)",
        #deriv_expr="0",
        domain=[-10, -10, 10.0, 10.0], 
        params=[3, 5, 0.0, 0.0],   
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "eq86": dict(
        expr="x + r*pow(abs(x),b)*sin(x)",
        # A8B8
        domain=[2,2,2.75,2.75],
        params=[0,0.3334,0,0], 
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq826": dict(
        expr="x * exp((r/(1+x))-b)",
        # A8B8
        domain=[10,10,40, 40],
        params=[0,11,0,0], 
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq95": dict(
        expr=" (1-r*x*x)*step(x)+(a-r*x*x)*(1-step(x))",
        # A8B8
        domain=[-0.5,-0.5,5,5],
        params=[2,2,2,0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq96": dict(
        expr=" r*x*(1-x)*step(x-0.5)+(r*x*(1-x)+(a-1)*(r-2)/4)*(1-step(x-0.5))",
        # A8B8
        domain=[2.5,2.5,4, 4],
        params=[0.4,0,0,0], 
        x0=0.6,
        trans=100,
        iter=300,
    ),

    "dlog": dict( # same as eq96, but manual derivative to check sympy's derivation
        # Map step = eq. (9.6)
        expr=_DLOG_EXPR,
        deriv_expr="r * (1.0 - 2.0 * (" + _DLOG_EXPR + "))",
        domain=[2.5,2.5,4, 4],
        params=[0.4, 0.0, 0.0, 0.0],
        x0=0.6,
        trans=100,
        iter=300,
    ),


    "eq97": dict(
        expr=" a*x*(1-step(x-1))+b*pow(x,1-r)*step(x-1)",
        # A8B8
        domain=[2,0.5,10,1.5],
        params=[50,50, 2.0, 0.0], 
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq98": dict(
        expr=" 1+r*apow(x,b)-a*apow(x,d)",
        # A8B8
        domain=[-0.25, -0.25, 1.25, 1.25],
        params=[1.0, 1.0, 2.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq932": dict(
        expr=" mod1(r*x)",
        deriv_expr="r",      # <--- add this
        # A8B8
        domain=[-0.25, -0.25, 1.25, 1.25],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq933": dict(
        expr=" 2*x*step(x)*(1-step(x-0.5))+((4*r-2)*x+(2-3*r))*step(x-0.5)*(1-step(x-1.0))",
        # A8B8
        domain=[-0.25, -0.25, 1.25, 1.25],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq937": dict(
        expr="r * x * (1.0 - x) * step(x-0)*(1-step(x-r))+r*step(x-r)+0*(1-step(x))",
        domain=[0.0, 0.0, 5.0, 5.0],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq947": dict(
        expr="b*pow(sin(x+r),2)",
        domain=[0.0, 0.0, 10.0, 10.0],
        params=[0.0, 1.7, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq962": dict(
        expr="b * r*r * exp( sin( pow(1 - x, 3) ) ) - 1",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq963": dict(
        expr="b * exp( pow( sin(1 - x), 3 ) ) + r",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq964": dict(
        expr="r * exp( -pow(x - b, 2) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq965": dict(
        expr="b * exp( sin(r * x) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq966": dict(
        expr="pow( abs(b*b - pow(x - r, 2)), 0.5 ) + 1",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq967": dict(
        expr="pow( b + pow( sin(r * x), 2 ), -1 )",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq968": dict(
        expr="b * exp( r * pow( sin(x) + cos(x), -1 ) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.3, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq969": dict(
        expr="b * (x - r) * exp( -pow(x - r, 3) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq970": dict(
        expr="b * exp( cos(1 - x) * sin(pi/2) + sin(r) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq971": dict(
        expr="b * r * exp( pow( sin(x - r), 4 ) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq972": dict(
        expr="b * r * exp( pow( sin(1 - x), 3 ) )",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq973": dict(
        expr="b * r * pow( sin(b*x + r*r), 2 ) * pow( cos(b*x - r*r), 2 )",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq974": dict(
        expr="pow( abs(r*r - pow(x - b, 2)), 0.5 )",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq975": dict(
        expr="b*cos(x-r)*sin(x+r)",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq976": dict(
        expr="(x-r)*sin( pow(x-b,2))",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq977": dict(
        expr="r*sin(pi*r)*sin(pi*x)*step(x-0.5)+b*r*sin(pi*r)*sin(pi*x)*step(0.5-x)",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq978": dict(
        expr="r*sin(pi*r)*sin(pi*(x-b))",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq979": dict(
        expr="b*r*pow(sin(b*x+r*r), 2 )*pow( cos(b*x-r*r-r),2)",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq980": dict(
        expr="b*r*pow(sin(b*x+r*r),2)*pow(cos(b*x-r*r),2)-1",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq981": dict(
        expr="b/(2+sin(mod1(x))-r)",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq982": dict(
        expr="b*r*exp(exp(exp(x*x*x)))",
        domain=[0.0, 2.0, 0.0, 2.0],
        params=[0.0, 0.1, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq983": dict(
        expr="b*r* exp(pow(sin(1-x*x),4))",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
  
    "eq984": dict(
        expr="r*(sin(x)+b*sin(9.0*x))",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq985": dict(
        expr="b*exp(tan(r*x)-x)",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq986": dict(
        expr="b*exp(cos(x*x*x*r-b)-r)",
        domain=[0.0, 0.0, 4.0, 4.0],
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

}


def _build_maps() -> dict:
    out = {}

    for name, cfg in MAP_TEMPLATES.items():
        dim = cfg.get("dim", 1)

        if dim == 1:
            expr = cfg["expr"]
            step_py = _make_py_func(expr)
            der_expr = cfg.get("deriv_expr")
            if der_expr is None:
                der_expr = _sympy_deriv(expr)
            der_py = _make_py_func(der_expr)

            step_jit = njit(STEP_SIG, cache=False, fastmath=False)(step_py)
            der_jit  = njit(STEP_SIG, cache=False, fastmath=False)(der_py)

            params_default = np.asarray(
                cfg.get("params", [0.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            domain_default = np.asarray(
                cfg.get("domain", [0.0, 0.0, 1.0, 1.0]),
                dtype=np.float64,
            )

            new_cfg = dict(cfg)
            new_cfg["step"] = step_jit
            new_cfg["deriv"] = der_jit
            new_cfg["params"] = params_default
            new_cfg["domain"] = domain_default
            new_cfg["dim"] = 1
            new_cfg["eps_floor"] = cfg.get("eps_floor", 1e-16)
            out[name] = new_cfg

        elif dim == 2:
            if "step2_func" in cfg and "jac2_func" in cfg:
                step2_py = cfg["step2_func"]
                jac2_py  = cfg["jac2_func"]
                print(f"Compiling manual functions for {name}")
            else:
                expr_x = cfg["expr_x"]
                expr_y = cfg["expr_y"]
                step2_py = _make_py_func_2d_step(expr_x, expr_y)
                if "jac_exprs" in cfg:
                    dXdx, dXdy, dYdx, dYdy = cfg["jac_exprs"]
                else:
                    dXdx, dXdy, dYdx, dYdy = _sympy_jacobian_2d(expr_x, expr_y)
                jac2_py = _make_py_func_2d_jac(dXdx, dXdy, dYdx, dYdy)
            step2_jit = njit(STEP2_SIG, cache=False, fastmath=False)(step2_py)
            jac2_jit  = njit(JAC2_SIG,  cache=False, fastmath=False)(jac2_py)
            params_default = np.asarray(
                cfg.get("params", [0.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            domain_default = np.asarray(
                cfg.get("domain", [0.0, 0.0, 1.0, 1.0]),
                dtype=np.float64,
            )
            new_cfg = dict(cfg)
            new_cfg["step2"] = step2_jit
            new_cfg["jac2"] = jac2_jit
            new_cfg["params"] = params_default
            new_cfg["domain"] = domain_default
            new_cfg["dim"] = 2
            new_cfg["forcing"] = cfg.get("forcing", None)
            out[name] = new_cfg

        else:
            raise ValueError(f"Unsupported dim={dim} for map '{name}'")

    return out


# Master map configuration: everything lives here.
MAPS = _build_maps()


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
def _lyapunov_field_generic(
    step,
    deriv,
    seq,
    pix,
    domain,         # <- 1D float64 array: [llx, lly, ulx, uly, lrx, lry]
    x0,
    n_transient,
    n_iter,
    eps,
    params,
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
        v = j / denom
        for i in range(pix):
            u = i / denom

            # Logical -> physical mapping
            Ai, Bj = map_logical_to_physical(domain, u, v)

            x = x0
            acc = 0.0

            for n in range(n_transient + n_iter):
                s_idx = seq[n % seq_len]
                r = Ai if s_idx == 0 else Bj

                d = deriv(x, r, params)
                x = step(x, r, params)
                if not np.isfinite(x):
                    x = 0.5

                if n >= n_transient:
                    ad = abs(d)
                    if (not np.isfinite(ad)) or ad < eps:
                        ad = eps
                    acc += math.log(ad)

            out[j, i] = acc / float(n_iter)

    return out

@njit(cache=False, fastmath=False, parallel=True)
def _lyapunov_field_generic_2d(
    step2,
    jac2,
    pix,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (r,s)-plane
    x0,
    y0,
    n_transient,
    n_iter,
    eps_floor,
    params,
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
        v = j / denom
        for i in range(pix):
            u = i / denom

            r,s = map_logical_to_physical(domain, u, v)

            x = x0
            y = y0
            vx = 1.0
            vy = 0.0
            acc = 0.0

            for n in range(n_transient + n_iter):
                x_next, y_next = step2(x, y, r, s, params)

                dXdx, dXdy, dYdx, dYdy = jac2(x, y, r, s, params)

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

@njit(cache=False, fastmath=False, parallel=True)
def _lyapunov_field_generic_2d_ab(
    step2,
    jac2,
    seq,
    pix,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (A,B)-plane
    x0,
    y0,
    n_transient,
    n_iter,
    eps_floor,
    params,
):
    """
    Largest Lyapunov exponent field for a 2‑D map with single-parameter
    A/B forcing over an arbitrary parallelogram in the (A,B) plane.

        (u,v) in [0,1]^2
        (A,B) = LL + u (LR-LL) + v (UL-LL)

    For each pixel, A and B are the two values used in the sequence.
    """
    seq_len = seq.size
    out = np.empty((pix, pix), dtype=np.float64)

    if eps_floor <= 0.0:
        eps_floor = 1e-16

    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        v = j / denom
        for i in range(pix):
            u = i / denom

            A, B = map_logical_to_physical(domain, u, v)

            x = x0
            y = y0
            vx = 1.0
            vy = 0.0
            acc = 0.0

            for n in range(n_transient + n_iter):
                s_idx = seq[n % seq_len]
                r = A if s_idx == 0 else B

                # Jacobian at (x,y; r)
                dXdx, dXdy, dYdx, dYdy = jac2(x, y, r, 0.0, params)

                vx_new = dXdx * vx + dXdy * vy
                vy_new = dYdx * vx + dYdy * vy
                vx, vy = vx_new, vy_new

                x_next, y_next = step2(x, y, r, 0.0, params)
                if not np.isfinite(x_next) or not np.isfinite(y_next):
                    x_next = 0.5
                    y_next = 0.0

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

    clip = params.get("clip", DEFAULT_CLIP)
    gamma = params.get("gamma", DEFAULT_GAMMA)
    pos_color = params.get("pos_color", "red")

    # symmetric |λ| scale
    if clip is not None and clip > 0 and math.isfinite(clip):
        scale = float(clip)
    else:
        min_neg = float(arr[neg_mask].min()) if np.any(neg_mask) else 0.0
        max_pos = float(arr[pos_mask].max()) if np.any(pos_mask) else 0.0
        scale = max(abs(min_neg), abs(max_pos))
        if (not math.isfinite(scale)) or scale <= 0.0:
            scale = 1.0

    if gamma <= 0.0:
        gamma = 1.0

    # λ < 0 → black→yellow
    if np.any(neg_mask):
        lam_neg = np.clip(arr[neg_mask], -scale, 0.0)
        t = np.abs(lam_neg) / scale
        if gamma != 1.0:
            t = t ** float(gamma)
        r = t
        g = t
        rgb[neg_mask, 0] = np.rint(r * 255.0).astype(np.uint8)
        rgb[neg_mask, 1] = np.rint(g * 255.0).astype(np.uint8)
        rgb[neg_mask, 2] = 0

    # λ > 0 → black→red/blue
    if np.any(pos_mask):
        lam_pos = np.clip(arr[pos_mask], 0.0, scale)
        t = lam_pos / scale
        if gamma != 1.0:
            t = t ** float(gamma)

        if pos_color.lower().startswith("b"):
            r = np.zeros_like(t)
            g = np.zeros_like(t)
            b = t
        else:
            r = t
            g = np.zeros_like(t)
            b = np.zeros_like(t)

        rgb[pos_mask, 0] = np.rint(r * 255.0).astype(np.uint8)
        rgb[pos_mask, 1] = np.rint(g * 255.0).astype(np.uint8)
        rgb[pos_mask, 2] = np.rint(b * 255.0).astype(np.uint8)

    return rgb


def _rgb_scheme_mh_eq(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Markus & Hess style with histogram equalization.

    - λ < 0 : equalize |λ| over negative values
    - λ > 0 : equalize λ over positive values

    So each 1/255-ish step tries to hold ~equal mass in λ, on each side.
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
    pos_color = params.get("pos_color", "red")
    nbins = int(params.get("nbins", 256))

    if gamma <= 0.0:
        gamma = 1.0

    # λ < 0: equalize |λ|
    if np.any(neg_mask):
        vals = np.abs(arr[neg_mask])
        t = _hist_equalize(vals, nbins=nbins)  # in [0,1]
        if gamma != 1.0:
            t = t ** float(gamma)
        r = t
        g = t
        rgb[neg_mask, 0] = np.rint(r * 255.0).astype(np.uint8)
        rgb[neg_mask, 1] = np.rint(g * 255.0).astype(np.uint8)
        rgb[neg_mask, 2] = 0

    # λ > 0: equalize λ
    if np.any(pos_mask):
        vals = arr[pos_mask]
        t = _hist_equalize(vals, nbins=nbins)  # in [0,1]
        if gamma != 1.0:
            t = t ** float(gamma)

        if pos_color.lower().startswith("b"):
            r = np.zeros_like(t)
            g = np.zeros_like(t)
            b = t
        else:
            r = t
            g = np.zeros_like(t)
            b = np.zeros_like(t)

        rgb[pos_mask, 0] = np.rint(r * 255.0).astype(np.uint8)
        rgb[pos_mask, 1] = np.rint(g * 255.0).astype(np.uint8)
        rgb[pos_mask, 2] = np.rint(b * 255.0).astype(np.uint8)

    return rgb


# Scheme registry: ADD NEW SCHEMES HERE ONLY
RGB_SCHEMES: dict[str, dict] = {
    "mh": dict(
        func=_rgb_scheme_mh,
        defaults=dict(
            clip=DEFAULT_CLIP,       # None -> auto symmetric |λ|
            gamma=DEFAULT_GAMMA,
            pos_color="red",
        ),
    ),

    "mh_eq": dict(
        func=_rgb_scheme_mh_eq,
        defaults=dict(
            gamma=DEFAULT_GAMMA,
            pos_color="red",
            nbins=256,
        ),
    ),
}

DEFAULT_RGB_SCHEME = "mh"


def lyapunov_to_rgb(lyap: np.ndarray, specdict: dict) -> np.ndarray:
    """
    Apply a colorization scheme to the λ-field based on the 'rgb' spec.

    Syntax:
        rgb:mh                -> use 'mh' defaults
        rgb:mh:clip           -> set clip
        rgb:mh:clip:gamma     -> set clip & gamma
        rgb:mh:*:*:blue       -> keep defaults for clip/gamma, pos_color='blue'

        rgb:mh_eq             -> equalized scheme with defaults
        rgb:mh_eq:0.25        -> mh_eq with gamma=0.25
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
        if op in MAPS:
            map_name = op
            break
    if map_name is None:
        print(f"No map name found in spec {spec}")
        return

    map_cfg = MAPS[map_name]
    dim = map_cfg.get("dim", 1)
    forcing = map_cfg.get("forcing", None)

    params = map_cfg["params"].copy()
    params[0] = _get_float(specdict, "a", params[0])
    params[1] = _get_float(specdict, "b", params[1])
    params[2] = _get_float(specdict, "c", params[2])
    params[3] = _get_float(specdict, "d", params[3])

    domain = map_cfg["domain"].copy()

    use_seq = (dim == 1) or (dim == 2 and forcing == "ab")
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
        if op in MAPS:
            map_name = op
            break
    if map_name is None:
        raise SystemExit(f"No map name found in spec {spec}")

    map_cfg = MAPS[map_name]
    dim = map_cfg.get("dim", 1)
    forcing = map_cfg.get("forcing", None)

    params = map_cfg["params"].copy()
    params[0] = _get_float(specdict, "a", params[0])
    params[1] = _get_float(specdict, "b", params[1])
    params[2] = _get_float(specdict, "c", params[2])
    params[3] = _get_float(specdict, "d", params[3])

     # --- domain + sequence parsing ---
    domain = map_cfg["domain"].copy()

    # do we need an A/B sequence for this map?
    use_seq = (dim == 1) or (dim == 2 and forcing == "ab")
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

    x0 = _get_float(specdict, "x0", map_cfg.get("x0", 0.5))
    y0 = _get_float(specdict, "y0", map_cfg.get("y0", 0.5))
    n_tr  = _get_int(specdict, "trans", map_cfg.get("trans", DEFAULT_TRANS))
    n_it  = _get_int(specdict, "iter", map_cfg.get("iter",  DEFAULT_ITER))
    eps   = _get_float(specdict, "eps",   DEFAULT_EPS_LYAP)
 

    if dim == 1:
        lyap = _lyapunov_field_generic(
            map_cfg["step"],
            map_cfg["deriv"],
            seq_arr,
            int(pix),
            domain_affine,
            float(x0),
            int(n_tr),
            int(n_it),
            float(eps),
            params,
        )
    elif dim == 2 and forcing == "ab":
        if seq_arr is None:
            raise RuntimeError("internal bug: seq_arr is None for AB-forced map")
        print("lyapunov_field_generic_2d_ab")
        eps_floor = map_cfg.get("eps_floor", 1e-16)
        lyap = _lyapunov_field_generic_2d_ab(
            map_cfg["step2"],
            map_cfg["jac2"],
            seq_arr,
            pix,
            domain_affine,
            x0, 
            y0,
            n_tr, 
            n_it,
            eps_floor,
            params,
        )
    elif dim == 2:
        print("lyapunov_field_generic_2d")
        eps_floor = map_cfg.get("eps_floor", 1e-16)
        lyap = _lyapunov_field_generic_2d(
            map_cfg["step2"],
            map_cfg["jac2"],
            int(pix),
            domain_affine,
            float(x0),
            float(y0),
            int(n_tr),
            int(n_it),
            float(eps_floor),
            params,
        )
    else:
        raise SystemExit(f"Unsupported dim={dim} for map '{map_name}'")

    rgb = lyapunov_to_rgb(lyap, specdict)
    return rgb


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
        "--check-affine",
        action="store_true",
        help="Print affine domain mapping for the first expanded spec and exit.",
    )


    args = p.parse_args()

    if args.map is not None:
        map_name, new_expr = args.map.split("=", 1)
        if map_name in MAPS:
            new_der_expr = _sympy_deriv(new_expr)
            print(f"map derivative:{new_der_expr}")
            new_step_py = _make_py_func(new_expr)
            new_der_py = _make_py_func(new_der_expr)
            new_step_jit = njit(STEP_SIG, cache=False, fastmath=False)(new_step_py)
            new_der_jit  = njit(STEP_SIG, cache=False, fastmath=False)(new_der_py)
            MAPS[map_name]["step"] = new_step_jit
            MAPS[map_name]["deriv"] = new_der_jit
            spec_str=f",modify:{map_name}:{new_expr}"
            args.spec=args.spec+spec_str

    # Apply constants (like in julia.py)
    for kv in args.const:
        print(f"const {kv}")
        k, v = specparser._parse_const_kv(kv)
        specparser.set_const(k, v)
        expandspec.set_const(k, v)

    # Expand output path first
    out_paths = expandspec.expand_cartesian_lists(args.out)
    if not out_paths:
        raise SystemExit("Output expandspec produced no paths")
    pngfile = out_paths[0]
    print(f"will save to {pngfile}")

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

