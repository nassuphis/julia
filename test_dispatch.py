#
# test dispatch of jited lyapunov maps
# maps can be referenced by names, no
# stupid elif chains, no id values all 
# over the place. 
# readable and simple to update
# map derivatives are computed symolically
# at initialization time and jited 
# for speed
#

#%load_ext autoreload
#%autoreload 2

import sympy as sp
from numba import njit
from numba import types
import numpy as np
import math


def ddx(expr_str: str) -> str:
    x=sp.symbols('x')
    expr = sp.sympify(
        expr_str,
        locals={
            'x': sp.symbols('x'),
            'r': sp.symbols('r'),
            'alpha': sp.symbols('alpha'),
            'beta': sp.symbols('beta'),
            'delta': sp.symbols('delta'),
            'eps': sp.symbols('eps'),
            'epsilon': sp.symbols('epsilon'),
            'zeta': sp.symbols('zeta'),
            'eta': sp.symbols('eta'),
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'sec': sp.sec,
            'cosh': sp.cosh,
            'exp': sp.exp,
            'sign': sp.sign,
            'abs': sp.Abs,
            'max': sp.Max,
            'step': sp.Heaviside,
            'dirac': sp.DiracDelta,
            'pow': sp.Pow,
            'mod1': lambda x: sp.Mod(x, 1),
            'pi': np.pi,
        }
    )
    print(sp.sstr(expr))
    d = sp.diff(expr, x)
    return sp.sstr(d)

#########################################
# numba implementations for missing
# functions
#########################################

@njit(types.float64(types.float64),cache=True, fastmath=True)
def DiracDelta(x):
    return 0.0

@njit(types.float64(types.float64),cache=True, fastmath=True)
def Heaviside(x):
    if x>0: return 1.0
    return 0.0

@njit(types.float64(types.float64),cache=True, fastmath=True)
def sign(x):
    if x>0: return 1.0
    return -1.0

@njit(types.float64(types.float64),cache=True, fastmath=True)
def Abs(x):
    return np.abs(x)

@njit(types.float64(types.float64),cache=True, fastmath=True)
def re(x):
    return x

@njit(types.float64(types.float64),cache=True, fastmath=True)
def im(x):
    return 0.0

@njit(types.float64(types.float64),cache=True, fastmath=True)
def step(x):
    return 1.0 if x > 0.0 else 0.0

@njit(types.float64(types.float64),cache=True, fastmath=True)
def sec(x):
    return 1.0 / np.cos(x)

@njit(types.float64(types.float64),cache=True, fastmath=True)
def mod1(x):
    return x % 1

@njit(types.float64(types.float64,types.float64),cache=True, fastmath=True)
def Mod(x,v):
    return x % v

@njit(types.float64(types.float64,types.float64),cache=True, fastmath=True)
def Derivative(x,v):
    return 1.0

#########################################
# the lyapunov map function has 
# x, r and some parameter inputs
#########################################
def funtext(calc,name):
    lines = [f"def {name}(x,r,a):"]
    lines.append(f"    alpha=a[0]")
    lines.append(f"    beta=a[1]")
    lines.append(f"    delta=a[2]")
    lines.append(f"    epsilon=a[3]")
    lines.append(f"    return {calc}")
    src = "\n".join(lines)
    return src

#############################################
# make a njit-compatible function from text
#############################################
def func(calc):
    maps={
        "step": step,
        "sign": sign,
        "DiracDelta": DiracDelta,
        "Heaviside": Heaviside,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sec": sec,
        "cosh": np.cosh,
        "exp": np.exp,
        "pow": np.power,
        "sign": sign,
        "im": im,
        "re": re,
        "Abs": Abs,
        "mod1": mod1,
        "Mod": Mod,
        "Derivative": Derivative,
        "pi": np.pi
    }
    #print(f"Making: {calc}")
    txt=funtext(calc,"something")
    maps["_source"]=txt
    #print(f"{txt}")
    exec(txt,maps,maps)
    return maps["something"]

#########################################
# make a function from text
#########################################
mapdict = {
    "logistic": {"map": "r * x * (1.0 - x)" },
    "sine":     {"map": "r * sin(x)" },
    "tent":     {"map": "r*x*step(0.5-x)+r*(1-x)*step(x-0.5)" },
    "heart":    {"map": "sin(alpha * x) + r" },
    "eq962":    {"map": "beta * r*r * exp( sin( pow(1 - x, 3) ) ) - 1" },
    "eq963":    {"map": "beta * exp( pow( sin(1 - x), 3 ) ) + r" },
    "eq964":    {"map": "r * exp( -pow(x - beta, 2) )" },
    "eq965":    {"map": "beta * exp( sin(r * x) )" },
    "eq966":    {"map": "pow( abs(beta*beta - pow(x - r, 2)), 0.5 ) + 1" },
    "eq967":    {"map": "pow( beta + pow( sin(r * x), 2 ), -1 )" },
    "eq968":    {"map": "beta * exp( r * pow( sin(x) + cos(x), -1 ) )" },
    "eq969":    {"map": "beta * (x - r) * exp( -pow(x - r, 3) )" },
    "eq970":    {"map": "beta * exp( cos(1 - x) * sin(pi/2) + sin(r) )" },
    "eq971":    {"map": "beta * r * exp( pow( sin(x - r), 4 ) )" },
    "eq972":    {"map": "beta * r * exp( pow( sin(1 - x), 3 ) )" },
    "eq973":    {"map": "beta * r * pow( sin(beta*x + r*r), 2 ) * pow( cos(beta*x - r*r), 2 )" },
    "eq974":    {"map": "pow( abs(r*r - pow(x - beta, 2)), 0.5 )" },
    "eq975":    {"map": "beta * cos(x - r) * sin(x + r)" },
    "eq976":    {"map": "(x - r) * sin( pow(x - beta, 2) )" },
    "eq977":    {"map": "r*sin(pi*r)*sin(pi*x)*step(x-0.5) + beta*r*sin(pi*r)*sin(pi*x)*step(0.5-x)" },
    "eq978":    {"map": "r * sin(pi*r) * sin(pi*(x - beta))" },
    "eq979":    {"map": "beta * r * pow( sin(beta*x + r*r), 2 ) * pow( cos(beta*x - r*r - r), 2 )" },
    "eq980":    {"map": "beta*r*pow(sin(beta*x+r*r),2)*pow(cos(beta*x-r*r),2)-1" },
    "eq981":    {"map": "beta/(2+sin(mod1(x))-r)" },
    "eq982":    {"map": "beta * r * exp(exp(exp(x*x*x)))" },
    "eq983":    {"map": "beta * r * exp( pow( sin(1-x*x), 4 ) )" },
    "eq984":    {"map": "r * (sin(x) + beta * sin(9.0 * x))"},
    "eq985":    {"map": "beta * exp(tan(r * x) - x)" },
    "eq986":    {"map": "beta * exp( cos(x*x*x*r - beta) - r)" }
}

#########################################
# populate funcdict with jited 
# map function and its derivative
#########################################
sig = types.float64(types.float64,types.float64,types.float64[:])
if True:
    for key, entry in mapdict.items():
        print(f"Compiling: {key}:")
        map = entry["map"]
        f_map  = func(map)
        f_mapder = func(ddx(map))
        entry["val"]  = njit(sig, cache=False, fastmath=True)(f_map)
        entry["der"] = njit(sig, cache=False, fastmath=True)(f_mapder)

#########################################
# heavy use of map happens here
#########################################
@njit(cache=True, fastmath=True)
def _lyapunov_lambda(step, deriv, x0, r, a):
    n_transient = 100
    n_iter = 200
    eps = 1e-12
    x = x0
    for _ in range(n_transient): x = step(x, r, a)
    acc = 0.0
    for _ in range(n_iter):
        d = deriv(x, r, a)       # derivative at current x
        ad = abs(d)
        if (not np.isfinite(ad)) or ad < eps:
            ad = eps
        acc += math.log(ad)
        x = step(x, r, a)        # move to next iterate
    return acc / n_iter

#########################################
# dispatch function to jited heavy user
#########################################
def dispatch(s,x,r,a):
    fs = mapdict[s] # get the map
    field = _lyapunov_lambda(fs["val"],fs["der"],x,r,a) # heavy calculation
    return field


