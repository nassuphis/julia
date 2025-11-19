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

import math
import argparse
import re as regex

import numpy as np
import sympy as sp
from numba import njit, types, prange

from specparser import specparser, expandspec
from rasterizer import raster
import time

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
        x, r, alpha, beta, delta, epsilon, eps, zeta, eta
        sin, cos, tan, sec, cosh, exp, sign, abs/Abs, max
        step (Heaviside), DiracDelta, pow, mod1, Mod, pi
    """
    x, r = sp.symbols("x r")
    alpha, beta, delta, epsilon = sp.symbols("alpha beta delta epsilon")
    eps_var, zeta, eta = sp.symbols("eps zeta eta")

    locs = {
        "x": x,
        "r": r,
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "epsilon": epsilon,
        "eps": eps_var,
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
        "step": sp.Heaviside,
        "Heaviside": sp.Heaviside,
        "DiracDelta": sp.DiracDelta,
        "pow": sp.Pow,
        "mod1": lambda v: sp.Mod(v, 1),
        "Mod": sp.Mod,
        "pi": sp.pi,
    }

    expr = sp.sympify(expr_str, locals=locs)
    d = sp.diff(expr, x)
    return sp.sstr(d)


# ---------------------------------------------------------------------------
# Tiny Numba helpers used inside map expressions
# ---------------------------------------------------------------------------

@njit(types.float64(types.float64), cache=True, fastmath=True)
def DiracDelta(x):
    # We ignore distributional spikes; enough for Lyapunov purposes.
    return 0.0


@njit(types.float64(types.float64), cache=True, fastmath=True)
def Heaviside(x):
    return 1.0 if x > 0.0 else 0.0


@njit(types.float64(types.float64), cache=True, fastmath=True)
def step(x):
    return 1.0 if x > 0.0 else 0.0


@njit(types.float64(types.float64), cache=True, fastmath=True)
def sign(x):
    return 1.0 if x > 0.0 else -1.0


@njit(types.float64(types.float64), cache=True, fastmath=True)
def Abs(x):
    return np.abs(x)


@njit(types.float64(types.float64), cache=True, fastmath=True)
def re(x):
    return x


@njit(types.float64(types.float64), cache=True, fastmath=True)
def im(x):
    return 0.0


@njit(types.float64(types.float64), cache=True, fastmath=True)
def sec(x):
    return 1.0 / np.cos(x)


@njit(types.float64(types.float64), cache=True, fastmath=True)
def mod1(x):
    return x % 1.0


@njit(types.float64(types.float64, types.float64), cache=True, fastmath=True)
def Mod(x, v):
    return x % v


@njit(types.float64(types.float64, types.float64), cache=True, fastmath=True)
def Derivative(x, v):
    # never actually used; placeholder to keep SymPy happy if it
    # sneaks in.
    return 1.0



# ---------------------------------------------------------------------------
# Build python functions from expression strings, then JIT them
# ---------------------------------------------------------------------------

def _funtext(expr: str, name: str) -> str:
    """
    Emit a tiny Python function:

        def name(x, r, a):
            alpha   = a[0]
            beta    = a[1]
            delta   = a[2]
            epsilon = a[3]
            return <expr>
    """
    lines = [
        f"def {name}(x, r, a):",
        "    alpha   = a[0]",
        "    beta    = a[1]",
        "    delta   = a[2]",
        "    epsilon = a[3]",
        f"    return {expr}",
    ]
    return "\n".join(lines)


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
        "exp": np.exp,
        "pow": np.power,
        "mod1": mod1,
        "Mod": Mod,
        "Derivative": Derivative,
        "re": re,
        "im": im,
        "pi": np.pi,
        "np": np,
        "math": math,
    }
    src = _funtext(expr, "impl")
    exec(src, ns, ns)
    return ns["impl"]


# All jitted step/deriv functions will share this signature.
STEP_SIG = types.float64(types.float64, types.float64, types.float64[:])


# ---------------------------------------------------------------------------
# Map templates: add / tweak here to define all 1‑D maps
# ---------------------------------------------------------------------------

MAP_TEMPLATES: dict[str, dict] = {
    # Classic logistic
    "logistic": dict(
        expr="r * x * (1.0 - x)",
        domain=(2.5, 4.0, 2.5, 4.0),  # A0, A1, B0, B1
        params=[0.0, 0.0, 0.0, 0.0],  # alpha, beta, delta, epsilon
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    # Sine map (classical Lyapunov variant: r sin(pi x))
    "sine": dict(
        expr="r * sin(pi * x)",
        domain=(0.0, 2.0, 0.0, 2.0),
        params=[0.0, 0.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    # Tent map
    "tent": dict(
        expr="r*x*step(0.5-x) + r*(1-x)*step(x-0.5)",
        domain=(0.0, 2.0, 0.0, 2.0),
        params=[0.0, 0.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    # Heart-cell map: x_{n+1} = sin(alpha x_n) + r_n
    "heart": dict(
        expr="sin(alpha * x) + r",
        domain=(0.0, 15.0, 0.0, 15.0),  # B versus A as in the book
        params=[1.0, 0.0, 0.0, 0.0],    # alpha default = 1
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn2": dict(
        expr="pow(r/abs(x),alpha*cos(r))*sign(x)+cos(2*pi*r/2.25)*sin(2*pi*x/3)",
        #deriv_expr="0",
        domain=(0.0, 15.0, 0.0, 15.0),  # B versus A as in the book
        params=[1.0, 0.0, 0.0, 0.0],    # alpha default = 1
        x0=2.0,
        trans=600,
        iter=600,
    ),

    "nn1": dict(
        expr="pow(r/abs(x),alpha)*sign(x)+cos(2*pi*r/2)*sin(2*pi*x/5)",
        #deriv_expr="0",
        domain=(0.1, 0.1, 10, 10),  # B versus A as in the book
        params=[0.25, 0.0, 0.0, 0.0],    # alpha default = 0.25
        x0=0.5,
        trans=600,
        iter=600,
    ),

    "nn1a": dict(
        expr="pow(r/abs(x),alpha)*sign(x)+pow(cos(2*pi*r/2)*sin(2*pi*x/5),beta)",
        #deriv_expr="0",
        domain=(0.1, 0.1, 10, 10),  # B versus A as in the book
        params=[0.25, 1.0, 0.0, 0.0],    # alpha default = 0.25
        x0=0.5,
        trans=600,
        iter=600,
    ),

    "eq86": dict(
        expr=" x + r*pow(abs(x),beta)*sin(x)",
        # A8B8
        domain=(2,2,2.75,2.75),
        params=[0,0.3334,0,0], # default [alpha, beta, delta, epsilon]
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq826": dict(
        expr=" x * exp((r/(1+x))-beta)",
        # A8B8
        domain=(10,10,40, 40),
        params=[0,11,0,0], # default [alpha, beta, delta, epsilon]
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq95": dict(
        expr=" (1-r*x*x)*step(x)+(alpha-r*x*x)*(1-step(x))",
        # A8B8
        domain=(2.5,2.5,4, 4),
        params=[0.4,0,0,0], # default [alpha, beta, delta, epsilon]
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq96": dict(
        expr=" r*x*(1-x)*step(x-0.5)+(r*x*(1-x)+(alpha-1)*(r-2)/4)*(1-step(x-0.5))",
        # A8B8
        domain=(2.5,2.5,4, 4),
        params=[0.4,0,0,0], # default [alpha, beta, delta, epsilon]
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq97": dict(
        expr=" beta*x*(1-step(x-1))+beta*pow(x,1-r)*step(x-1)",
        # A8B8
        domain=(-0.25, -0.25, 1.25, 1.25),
        params=[2.0,0.5, 2.0, 0.0], # default [alpha, beta, delta, epsilon]
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq98": dict(
        expr=" 1+r*pow(abs(x),beta)*sign(x)-alpha*pow(x,delta)",
        # A8B8
        domain=(-0.25, -0.25, 1.25, 1.25),
        params=[1.0, 1.0, 2.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    "eq932": dict(
        expr=" mod1(r*x)",
        deriv_expr="r",      # <--- add this
        # A8B8
        domain=(-0.25, -0.25, 1.25, 1.25),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq933": dict(
        expr=" 2*x*step(x)*(1-step(x-0.5))+((4*r-2)*x+(2-3*r))*step(x-0.5)*(1-step(x-1.0))",
        # A8B8
        domain=(-0.25, -0.25, 1.25, 1.25),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),


    "eq937": dict(
        expr="r * x * (1.0 - x) * step(x-0)*(1-step(x-r))+r*step(x-r)+0*(1-step(x))",
        domain=(0.0, 0.0, 5.0, 5.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),


    "eq947": dict(
        expr="beta * pow(sin(x+r),2)",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),

    # Assorted one‑dimensional maps from the book (9.6x..9.8x)
    "eq962": dict(
        expr="beta * r*r * exp( sin( pow(1 - x, 3) ) ) - 1",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq963": dict(
        expr="beta * exp( pow( sin(1 - x), 3 ) ) + r",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq964": dict(
        expr="r * exp( -pow(x - beta, 2) )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq965": dict(
        expr="beta * exp( sin(r * x) )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq966": dict(
        expr="pow( abs(beta*beta - pow(x - r, 2)), 0.5 ) + 1",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq967": dict(
        expr="pow( beta + pow( sin(r * x), 2 ), -1 )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq968": dict(
        expr="beta * exp( r * pow( sin(x) + cos(x), -1 ) )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.3, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq969": dict(
        expr="beta * (x - r) * exp( -pow(x - r, 3) )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq970": dict(
        expr="beta * exp( cos(1 - x) * sin(pi/2) + sin(r) )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq971": dict(
        expr="beta * r * exp( pow( sin(x - r), 4 ) )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq972": dict(
        expr="beta * r * exp( pow( sin(1 - x), 3 ) )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq973": dict(
        expr="beta * r * pow( sin(beta*x + r*r), 2 ) * pow( cos(beta*x - r*r), 2 )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq974": dict(
        expr="pow( abs(r*r - pow(x - beta, 2)), 0.5 )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq975": dict(
        expr="beta * cos(x - r) * sin(x + r)",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq976": dict(
        expr="(x - r) * sin( pow(x - beta, 2) )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq977": dict(
        expr="r*sin(pi*r)*sin(pi*x)*step(x-0.5) + beta*r*sin(pi*r)*sin(pi*x)*step(0.5-x)",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq978": dict(
        expr="r * sin(pi*r) * sin(pi*(x - beta))",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq979": dict(
        expr="beta * r * pow( sin(beta*x + r*r), 2 ) * pow( cos(beta*x - r*r - r), 2 )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq980": dict(
        expr="beta*r*pow(sin(beta*x+r*r),2)*pow(cos(beta*x-r*r),2)-1",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq981": dict(
        expr="beta/(2+sin(mod1(x))-r)",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq982": dict(
        expr="beta * r * exp(exp(exp(x*x*x)))",
        domain=(0.0, 2.0, 0.0, 2.0),
        params=[0.0, 0.1, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq983": dict(
        expr="beta * r * exp( pow( sin(1-x*x), 4 ) )",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    # The two you explicitly asked for
    "eq984": dict(
        expr="r * (sin(x) + beta * sin(9.0 * x))",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq985": dict(
        expr="beta * exp(tan(r * x) - x)",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 1.0, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
    "eq986": dict(
        expr="beta * exp( cos(x*x*x*r - beta) - r)",
        domain=(0.0, 4.0, 0.0, 4.0),
        params=[0.0, 0.5, 0.0, 0.0],
        x0=0.5,
        trans=DEFAULT_TRANS,
        iter=DEFAULT_ITER,
    ),
}


def _build_maps() -> dict:
    """
    From MAP_TEMPLATES, build:
        - jitted stepping function val(x,r,a)
        - jitted derivative der(x,r,a)
        - default parameter vector (float64[4])
    """
    out = {}

    for name, cfg in MAP_TEMPLATES.items():
        expr = cfg["expr"]
        step_py = _make_py_func(expr)
        der_expr = cfg.get("deriv_expr")
        if der_expr is None:
            der_expr = _sympy_deriv(expr)
        der_py = _make_py_func(der_expr)

        # cache=False because these come from dynamically generated
        # source strings (Numba's cache wants a real file).
        step_jit = njit(STEP_SIG, cache=False, fastmath=True)(step_py)
        der_jit  = njit(STEP_SIG, cache=False, fastmath=True)(der_py)

        params_default = np.asarray(
            cfg.get("params", [0.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )

        new_cfg = dict(cfg)
        new_cfg["step"] = step_jit
        new_cfg["deriv"] = der_jit
        new_cfg["params_default"] = params_default
        out[name] = new_cfg

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

@njit(cache=False, fastmath=True, parallel=True)
def _lyapunov_field_generic(
    step,
    deriv,
    seq,
    width,
    height,
    a0,
    a1,
    b0,
    b1,
    x0,
    n_transient,
    n_iter,
    eps,
    params,
):
    """
    Generic λ-field for a 1‑D map:

        x_{n+1} = step(x_n, r_n, params)
        λ  ~  <log |∂x_{n+1}/∂x_n|>

    where r_n alternates between A and B according to `seq`.
    """
    seq_len = seq.size
    out = np.empty((height, width), dtype=np.float64)

    for j in prange(height):
        if height > 1:
            bj = b0 + (b1 - b0) * (j / (height - 1.0))
        else:
            bj = 0.5 * (b0 + b1)

        for i in range(width):
            if width > 1:
                ai = a0 + (a1 - a0) * (i / (width - 1.0))
            else:
                ai = 0.5 * (a0 + a1)

            x = x0
            acc = 0.0

            for n in range(n_transient + n_iter):
                s = seq[n % seq_len]
                r = ai if s == 0 else bj

                # derivative at x_n
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


# ---------------------------------------------------------------------------
# Color mapping: Lyapunov exponent -> RGB
# ---------------------------------------------------------------------------

def lyapunov_to_rgb(
    lyap: np.ndarray,
    clip: float | None = DEFAULT_CLIP,
    gamma: float = DEFAULT_GAMMA,
    pos_color: str = "red",  # "red" or "blue"
) -> np.ndarray:
    """
    Markus & Hess style color map:

      λ < 0 : black  -> yellow  (periodic / order)
      λ = 0 : black
      λ > 0 : black  -> red or blue (chaos)
    """
    arr = np.asarray(lyap, dtype=np.float64)
    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return rgb

    neg_mask = finite & (arr < 0.0)
    pos_mask = finite & (arr > 0.0)

    # symmetric |λ| scale
    if clip is not None and clip > 0 and math.isfinite(clip):
        scale = float(clip)
    else:
        min_neg = float(arr[neg_mask].min()) if np.any(neg_mask) else 0.0
        max_pos = float(arr[pos_mask].max()) if np.any(pos_mask) else 0.0
        scale = max(abs(min_neg), abs(max_pos))
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0

    if gamma <= 0:
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
# dict2lyapunov: parse spec dict -> RGB tile
# ---------------------------------------------------------------------------

def dict2lyapunov(d: dict, *, pix: int) -> tuple[np.ndarray, dict]:
    """
    Interpret one split spec dict as Lyapunov parameters.

    Map token:

        MAP[:SEQ][:a0][:b0][:a1][:b1]

    Examples:

        logistic:AB:2.5:4:2.5:4

    Other keys:

        trans : transient iterations
        iter  : iterations used to accumulate λ
        x0    : initial x
        eps   : epsilon for log(|f'| + eps) (Lyapunov floor)
        clip  : |λ| clipping scale (per-tile)
        gamma : gamma correction for color map

        alpha, beta, delta, epsilon : map parameters stored in a[0..3]
    """
    map_name = DEFAULT_MAP_NAME
    seq_str  = DEFAULT_SEQ
    seq_raw  = DEFAULT_SEQ

    a0_dom = a1_dom = b0_dom = b1_dom = None

    # --- 1) Find which map this spec uses ---
    for cand in MAPS.keys():
        if cand in d:
            vals = d[cand]
            map_name = cand
            idx = 0

            if len(vals) > 0:
                first = vals[0].strip()
                if _looks_like_sequence_token(first):
                    seq_raw = first
                    seq_str = _decode_sequence_token(first, DEFAULT_SEQ)
                    idx = 1
                else:
                    numeric = True
                    try:
                        _ = _eval_number(first)
                    except Exception:
                        numeric = False
                    if numeric:
                        idx = 0
                    else:
                        seq_raw = first
                        seq_str = _decode_sequence_token(first, DEFAULT_SEQ)
                        idx = 1

            def parse_float_tok(tok: str) -> float:
                return float(_eval_number(tok).real)

            if idx < len(vals):
                try:
                    a0_dom = parse_float_tok(vals[idx])
                except Exception:
                    pass
                idx += 1
            if idx < len(vals):
                try:
                    b0_dom = parse_float_tok(vals[idx])
                except Exception:
                    pass
                idx += 1
            if idx < len(vals):
                try:
                    a1_dom = parse_float_tok(vals[idx])
                except Exception:
                    pass
                idx += 1
            if idx < len(vals):
                try:
                    b1_dom = parse_float_tok(vals[idx])
                except Exception:
                    pass
                idx += 1

            break

    cfg = MAPS.get(map_name, MAPS[DEFAULT_MAP_NAME])
    A0_def, B0_def, A1_def, B1_def = cfg.get("domain", (2.5, 2.5, 4.0, 4.0))

    a0 = a0_dom if a0_dom is not None else A0_def
    b0 = b0_dom if b0_dom is not None else B0_def
    a1 = a1_dom if a1_dom is not None else A1_def
    b1 = b1_dom if b1_dom is not None else B1_def

    # Optional explicit overrides
    a0 = _get_float(d, "a0", a0)
    a1 = _get_float(d, "a1", a1)
    b0 = _get_float(d, "b0", b0)
    b1 = _get_float(d, "b1", b1)

    # Dynamics defaults for this map
    x0    = _get_float(d, "x0",    cfg.get("x0",    DEFAULT_X0))
    n_tr  = _get_int  (d, "trans", cfg.get("trans", DEFAULT_TRANS))
    n_it  = _get_int  (d, "iter",  cfg.get("iter",  DEFAULT_ITER))
    eps   = _get_float(d, "eps",   DEFAULT_EPS_LYAP)
    gamma = _get_float(d, "gamma", DEFAULT_GAMMA)

    # Map parameter vector (alpha,beta,delta,epsilon)
    params = cfg["params_default"].copy()
    params[0] = _get_float(d, "alpha",   params[0])
    params[1] = _get_float(d, "beta",    params[1])
    params[2] = _get_float(d, "delta",   params[2])
    params[3] = _get_float(d, "epsilon", params[3])

    # Sequence
    seq_arr = _seq_to_array(seq_str)

    # --- 2) Compute Lyapunov field ---
    lyap = _lyapunov_field_generic(
        cfg["step"],
        cfg["deriv"],
        seq_arr,
        int(pix),
        int(pix),
        float(a0),
        float(a1),
        float(b0),
        float(b1),
        float(x0),
        int(n_tr),
        int(n_it),
        float(eps),
        params,
    )

    # --- 3) Color mapping ---
    clip_raw = _get_float(d, "clip", DEFAULT_CLIP if DEFAULT_CLIP is not None else -1.0)
    if DEFAULT_CLIP is None and (clip_raw <= 0 or not math.isfinite(clip_raw)):
        clip_used = None
    else:
        clip_used = clip_raw

    rgb = lyapunov_to_rgb(lyap, clip=clip_used, gamma=gamma)

    meta = dict(
        map_name=map_name,
        seq=seq_str,
        seq_raw=seq_raw,
        a0=a0,
        a1=a1,
        b0=b0,
        b1=b1,
        x0=x0,
        params=list(params),
        trans=n_tr,
        iter=n_it,
        eps=eps,
        clip=clip_used,
        gamma=gamma,
        spec=d.get("__spec__", ""),
    )
    return rgb, meta


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

    args = p.parse_args()

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
    if args.show_specs:
        for s in specs:
            print(s)

    if not specs:
        raise SystemExit("Spec expansion produced no tiles")

    dicts: list[dict] = []
    for s in specs:
        d = specparser.split_chain(s)
        d["__spec__"] = s
        dicts.append(d)

    tiles: list[np.ndarray] = []
    titles: list[str] = []

    for i, d in enumerate(dicts, start=1):
        spec_str = d.get("__spec__", "")
        print(f"{i}/{len(dicts)} Rendering {spec_str}")
        t0 = time.perf_counter()
        rgb, meta = dict2lyapunov(d, pix=args.pix)
        print(f"field time: {time.perf_counter() - t0:.3f}s")
        # swap A/B axes and flip vertically to match Markus & Hess style
        rgb = np.flipud(np.transpose(rgb, (1, 0, 2)))
        tiles.append(rgb)
        titles.append(spec_str)

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

