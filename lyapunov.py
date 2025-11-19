#!/usr/bin/env python
"""
lyapounov.py  (spelling per user request)

Lyapunov fractal renderer for 1D maps (logistic, sine, tent),
driven entirely by specparser/expandspec.

Key idea: each map is specified *self-contained* in a single token, e.g.:

    logistic:AAAABBBB:-4:-4:4:4

which is interpreted as:

    map_name = "logistic"
    sequence = "AAAABBBB"    # A/B pattern
    A-range  = [-4, 4]
    B-range  = [-4, 4]

More generally, a map token looks like:

    MAP[:SEQ][:a0][:a1][:b0][:b1]

MAP is one of: logistic, sine, tent.

If SEQ is omitted, defaults to "AB".
If a0/a1/b0/b1 are omitted, map-specific defaults are used.

Additional parameters are passed as normal key:value pairs in the spec:

    trans:N   -> transient iterations
    iter:N    -> iterations used to accumulate λ
    x0:X      -> initial x
    eps:E     -> epsilon for log(|f'| + eps)
    clip:C    -> |λ| clipping scale for color mapping
    gamma:G   -> gamma correction in color mapping

Examples:
---------

    # Classic logistic window, simple sequence
    "logistic:AB:2.5:4:2.5:4"

    # Different sequences in one mosaic:
    "[logistic:AB:2.5:4:2.5:4,logistic:AAAABBBB:2.5:4:2.5:4,logistic:ABBABA:2.5:4:2.5:4]"

    # Sine map with custom dynamics:
    "sine:AB:0:2:0:2,trans:400,iter:1200,x0:0.3,clip:2,gamma:0.8"

CLI is intentionally minimal: only "spec", mosaic layout, output path, etc.
"""

import sys
sys.path.insert(0, "/Users/nicknassuphis")

import math
import numpy as np
from numba import njit, prange, float64, int32
import argparse

from specparser import specparser, expandspec
from rasterizer import raster
import re



# logistic 

@njit(float64(float64, float64), fastmath=True, cache=True)
def logistic_step(x, r):
    return r * x * (1.0 - x)


@njit(float64(float64, float64), fastmath=True, cache=True)
def logistic_deriv(x, r):
    return r * (1.0 - 2.0 * x)


@njit(float64(float64, float64), fastmath=True, cache=True)
def sine_step(x, r):
    return r * math.sin(math.pi * x)


@njit(float64(float64, float64), fastmath=True, cache=True)
def sine_deriv(x, r):
    return r * math.pi * math.cos(math.pi * x)


@njit(float64(float64, float64), fastmath=True, cache=True)
def tent_step(x, r):
    return r * x if x < 0.5 else r * (1.0 - x)


@njit(float64(float64, float64), fastmath=True, cache=True)
def tent_deriv(x, r):
    return r if x < 0.5 else -r

@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def heart_step(x, r, alpha):
    # x_{n+1} = sin(α x_n) + r
    return math.sin(alpha * x) + r


@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def heart_deriv(x, r, alpha):
    # derivative wrt x_n: d/dx [sin(α x) + r] = α cos(α x)
    return alpha * math.cos(alpha * x)

@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def eq984_step(x, r, beta):
    # x_{n+1} = r ( sin x + beta sin 9x )
    return r * (math.sin(x) + beta * math.sin(9.0 * x))


@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def eq984_deriv(x, r, beta):
    # d/dx [ r (sin x + beta sin 9x) ]
    return r * (math.cos(x) + beta * 9.0 * math.cos(9.0 * x))


@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def eq985_step(x, r, beta):
    # x_{n+1} = beta * exp(tan(r x) - x)
    return beta * math.exp(math.tan(r * x) - x)


@njit(float64(float64, float64, float64, float64), fastmath=True, cache=True)
def eq985_deriv(x, r, beta, x_next):
    """
    d/dx [ beta * exp(tan(r x) - x) ]
      = beta * exp(tan(r x) - x) * (r sec^2(r x) - 1)
      = x_next * (r sec^2(r x) - 1)
    """
    rx = r * x
    c = math.cos(rx)
    # sec^2 = 1 / cos^2; guard tiny cos to avoid insane blowups
    if abs(c) < 1e-12:
        c = 1e-12 if c >= 0 else -1e-12
    sec2 = 1.0 / (c * c)
    return x_next * (r * sec2 - 1.0)


# ---------------------------------------------------------------------------
# Map definitions and defaults
# ---------------------------------------------------------------------------



MAP_NAME_TO_ID = {
    "logistic": 0,
    "sine": 1,
    "tent": 2,
    "kickedrotator": 3,     # NEW
    "parasite": 4,          # NEW
    "predprey": 6,       # NEW
    "dlog": 7,          # NEW
    "heart": 8,          # NEW
    "eq984": 9,      # NEW
    "eq985": 10,     # NEW
}
MAP_ID_TO_NAME = {v: k for (k, v) in MAP_NAME_TO_ID.items()}
# handy integer aliases for numba
MAP_ID_LOGISTIC = 0
MAP_ID_SINE     = 1
MAP_ID_TENT     = 2
MAP_ID_HEART    = 8
MAP_ID_EQ984    = 9
MAP_ID_EQ985    = 10

# Default (A,B) parameter windows per map: (a0, a1, b0, b1)
MAP_DEFAULT_DOMAIN = {
    MAP_NAME_TO_ID["logistic"]: (2.5, 2.5, 4.0, 4.0),
    MAP_NAME_TO_ID["sine"]:     (0.0, 0.0, 2.0, 2.0),
    MAP_NAME_TO_ID["tent"]:     (0.0, 0.0, 2.0, 2.0),
    MAP_NAME_TO_ID["kickedrotator"]: (-2.45, -6.35, 1.85744, 1.4325),
    MAP_NAME_TO_ID["parasite"]:      (-2.4, -0.1, 8.1, 3.57),
    MAP_NAME_TO_ID["predprey"]:      (-0.04, -0.6,  4.5,    6.6),
    MAP_NAME_TO_ID["dlog"]:          (0.6,   0.6,   2.15,   1.375),
    MAP_NAME_TO_ID["heart"]:         (0.0,  0.0, 15.0, 15.0),
    MAP_NAME_TO_ID["eq984"]:         (0.0,  0.0,  4.0,  4.0),   # r in [0,4]×[0,4]
    MAP_NAME_TO_ID["eq985"]:         (0.0,  0.0,  4.0,  4.0),
}

DEFAULT_MAP_NAME = "logistic"
DEFAULT_SEQ      = "AB"
DEFAULT_TRANS    = 200
DEFAULT_ITER     = 1000
DEFAULT_X0       = 0.5
DEFAULT_EPS      = 1e-12
DEFAULT_CLIP     = None   # auto from data
DEFAULT_GAMMA    = 1.0

# ---- Kicked rotator–specific defaults (from Fig. 8.24 caption) ----
DEFAULT_Y0        = 0.4     # y0 = 0.4
DEFAULT_KR_EPS    = 0.3     # ε = 0.3
DEFAULT_KR_GAMMA  = 3.0     # γ = 3
DEFAULT_KR_TRANS  = 100     # n_prev = 100
DEFAULT_KR_ITER   = 200     # n_max  = 200

DEFAULT_PARASITE_X0    = 0.5   # x0 = H0
DEFAULT_PARASITE_Y0    = 0.5   # y0 = P0
DEFAULT_PARASITE_K     = 2.1   # K = 2.1
DEFAULT_PARASITE_TRANS = 200   # n_prev
DEFAULT_PARASITE_ITER  = 800   # n_max

DEFAULT_PRED_X0   = 0.4     # x0 = y0 = 0.4 in Fig. 8.21
DEFAULT_PRED_Y0   = 0.4
DEFAULT_PRED_B    = 3.569985  # b = 3.569985
DEFAULT_PRED_TRANS = 100    # n_prev
DEFAULT_PRED_ITER  = 200    # n_max

DEFAULT_DLOG_ALPHA = 0.4   # α used in Fig. 9.6
DEFAULT_DLOG_X0    = 0.6   # x0 = 0.6
DEFAULT_DLOG_TRANS = 100   # n_prev
DEFAULT_DLOG_ITER  = 300   # n_max

DEFAULT_HEART_ALPHA = 1.0   # α = 1 in Fig. 8.4
DEFAULT_HEART_X0    = 2.0   # x0 = 2
DEFAULT_HEART_TRANS = 600   # n_prev
DEFAULT_HEART_ITER  = 600   # n_max

DEFAULT_EQ984_BETA = 0.5   # arbitrary but nontrivial; override with beta:...
DEFAULT_EQ985_BETA = 1.0


# ---------------------------------------------------------------------------
# Core Lyapunov field computation (Numba)
# ---------------------------------------------------------------------------

@njit(
    float64[:, :](
        int32[:],   # seq: 0/1 for A/B
        int32,      # width
        int32,      # height
        float64,    # a0
        float64,    # a1
        float64,    # b0
        float64,    # b1
        int32,      # map_id
        float64,    # x0
        int32,      # n_transient
        int32,      # n_iter
        float64,    # eps
        float64,    # alpha (heart, etc.)
        float64,    # beta  (eq984, eq985)
    ),
    parallel=True,
    fastmath=True,
    cache=True,
)
def _lyapunov_field(
    seq,
    width,
    height,
    a0,
    a1,
    b0,
    b1,
    map_id,
    x0,
    n_transient,
    n_iter,
    eps,
    alpha,
    beta,
):
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

                # --- map selection ---
                if map_id == MAP_ID_LOGISTIC:
                    d = logistic_deriv(x, r)
                    x = logistic_step(x, r)

                elif map_id == MAP_ID_SINE:
                    d = sine_deriv(x, r)
                    x = sine_step(x, r)

                elif map_id == MAP_ID_TENT:
                    d = tent_deriv(x, r)
                    x = tent_step(x, r)

                elif map_id == MAP_ID_HEART:
                    d = heart_deriv(x, r, alpha)
                    x = heart_step(x, r, alpha)

                elif map_id == MAP_ID_EQ984:
                    d = eq984_deriv(x, r, beta)
                    x = eq984_step(x, r, beta)

                elif map_id == MAP_ID_EQ985:
                    x_next = eq985_step(x, r, beta)
                    d = eq985_deriv(x, r, beta, x_next)
                    x = x_next

                else:
                    # fallback: logistic
                    d = logistic_deriv(x, r)
                    x = logistic_step(x, r)

                if not np.isfinite(x):
                    x = 0.5

                if n >= n_transient:
                    ad = abs(d)
                    if (not np.isfinite(ad)) or ad < eps:
                        ad = eps
                    acc += math.log(ad)

            out[j, i] = acc / float(n_iter)
    return out


@njit(
    float64[:, :](
        int32[:],   # seq: 0/1 for A/B
        int32,      # width
        int32,      # height
        float64,    # a0
        float64,    # a1
        float64,    # b0
        float64,    # b1
        float64,    # x0
        float64,    # y0
        float64,    # eps_kick (ε in the text)
        float64,    # gamma   (γ in the text)
        int32,      # n_transient
        int32,      # n_iter
    ),
    parallel=True,
    fastmath=True,
    cache=True,
)
def _lyapunov_field_kicked(
    seq,
    width,
    height,
    a0,
    a1,
    b0,
    b1,
    x0,
    y0,
    eps_kick,
    gamma,
    n_transient,
    n_iter,
):
    """
    Largest Lyapunov exponent field for the kicked rotator:

        x_{n+1} = [ x_n + r(1 + μ y_n) + ε r μ cos(2π x_n) ] mod 1
        y_{n+1} = e^{-γ} [ y_n + ε cos(2π x_n) ]

    where μ = (1 - e^{-γ}) / γ, r is modulated as A/B by `seq`,
    and each pixel corresponds to a particular (A,B).
    """
    seq_len = seq.size
    out = np.empty((height, width), dtype=np.float64)

    E = math.exp(-gamma)
    if gamma != 0.0:
        mu = (1.0 - E) / gamma
    else:
        # limit γ -> 0
        mu = 1.0

    two_pi = 2.0 * math.pi

    for j in prange(height):
        # vertical axis = B parameter
        if height > 1:
            bj = b0 + (b1 - b0) * (j / (height - 1.0))
        else:
            bj = 0.5 * (b0 + b1)

        for i in range(width):
            # horizontal axis = A parameter
            if width > 1:
                ai = a0 + (a1 - a0) * (i / (width - 1.0))
            else:
                ai = 0.5 * (a0 + a1)

            # initial state
            x = x0
            y = y0

            # tangent vector (vx, vy)
            vx = 1.0
            vy = 0.0

            acc = 0.0

            for n in range(n_transient + n_iter):
                s = seq[n % seq_len]
                r = ai if s == 0 else bj

                sin2pi = math.sin(two_pi * x)
                cos2pi = math.cos(two_pi * x)

                # map
                x_pre  = x + r * (1.0 + mu * y) + eps_kick * r * mu * cos2pi
                x_next = x_pre - math.floor(x_pre)  # mod 1
                y_next = E * (y + eps_kick * cos2pi)

                # Jacobian J_n = d(x_{n+1},y_{n+1})/d(x_n,y_n)
                dXdx = 1.0 - eps_kick * r * mu * two_pi * sin2pi
                dXdy = r * mu
                dYdx = -E * eps_kick * two_pi * sin2pi
                dYdy = E

                # propagate tangent vector
                vx_new = dXdx * vx + dXdy * vy
                vy_new = dYdx * vx + dYdy * vy
                vx = vx_new
                vy = vy_new

                # renormalize and accumulate after transients
                norm = math.sqrt(vx * vx + vy * vy)
                if norm < 1e-16:
                    norm = 1e-16
                if n >= n_transient:
                    acc += math.log(norm)
                vx /= norm
                vy /= norm

                x = x_next
                y = y_next

            out[j, i] = acc / float(n_iter)

    return out

@njit(
    float64[:, :](
        int32,   # width
        int32,   # height
        float64, # r0 (min r)
        float64, # r1 (max r)
        float64, # alpha0 (min α)
        float64, # alpha1 (max α)
        float64, # x0 (H0)
        float64, # y0 (P0)
        float64, # K
        int32,   # n_transient
        int32,   # n_iter
    ),
    parallel=True,
    fastmath=True,
    cache=True,
)
def _lyapunov_field_parasite(
    width,
    height,
    r0,
    r1,
    alpha0,
    alpha1,
    x0,
    y0,
    K,
    n_transient,
    n_iter,
):
    """
    Host–parasitoid map (Beddington et al.):

        H_{n+1} = H_n * exp( r * (1 - H_n / K) - α P_n )
        P_{n+1} = H_n * (1 - exp(-α P_n))

    We treat:
        pre-vertical (j)  -> r   in [r0, r1]
        pre-horizontal (i)-> α   in [alpha0, alpha1]

    After the transpose+flip you already do in the CLI,
    the final image has:
        x-axis: r   (abscissa)
        y-axis: α   (ordinate)
    matching Fig. 8.19.
    """
    out = np.empty((height, width), dtype=np.float64)
    invK = 1.0 / K
    EXP_MAX = 50.0     # clamp exponents to avoid overflow
    STATE_MAX = 1e6    # clamp state to avoid runaway blowup

    for j in prange(height):
        # vertical index j → r
        if height > 1:
            rj = r0 + (r1 - r0) * (j / (height - 1.0))
        else:
            rj = 0.5 * (r0 + r1)

        for i in range(width):
            # horizontal index i → α
            if width > 1:
                a = alpha0 + (alpha1 - alpha0) * (i / (width - 1.0))
            else:
                a = 0.5 * (alpha0 + alpha1)

            H = x0
            P = y0

            # tangent vector for largest Lyapunov exponent
            vH = 1.0
            vP = 0.0
            acc = 0.0

            for n in range(n_transient + n_iter):
                # Map step with exponent clamping
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

                H_next = H * expF
                P_next = H * (1.0 - E)

                # Jacobian
                dHdH = expF * (1.0 - rj * H * invK)
                dHdP = -a * H * expF
                dPdH = 1.0 - E
                dPdP = H * a * E

                # tangent update
                vH_new = dHdH * vH + dHdP * vP
                vP_new = dPdH * vH + dPdP * vP
                vH, vP = vH_new, vP_new

                norm = math.sqrt(vH * vH + vP * vP)
                if norm < 1e-16:
                    norm = 1e-16

                if n >= n_transient:
                    acc += math.log(norm)

                inv_norm = 1.0 / norm
                vH *= inv_norm
                vP *= inv_norm

                H = H_next
                P = P_next

                # keep state in a sane numeric range
                if H > STATE_MAX:
                    H = STATE_MAX
                elif H < -STATE_MAX:
                    H = -STATE_MAX
                if P > STATE_MAX:
                    P = STATE_MAX
                elif P < -STATE_MAX:
                    P = -STATE_MAX

                if not (np.isfinite(H) and np.isfinite(P)):
                    H = 1.0
                    P = 1.0

            out[j, i] = acc / float(n_iter)

    return out


@njit(
    float64[:, :](
        int32[:],   # seq: 0/1 for A/B (controls a_n)
        int32,      # width
        int32,      # height
        float64,    # a0 (A_min)
        float64,    # a1 (A_max)
        float64,    # b0 (B_min)
        float64,    # b1 (B_max)
        float64,    # x0
        float64,    # y0
        float64,    # b_param (constant b)
        int32,      # n_transient
        int32,      # n_iter
    ),
    parallel=True,
    fastmath=True,
    cache=True,
)
def _lyapunov_field_predprey(
    seq,
    width,
    height,
    a0,
    a1,
    b0,
    b1,
    x0,
    y0,
    b_param,
    n_transient,
    n_iter,
):
    """
    Predator–prey map with periodically forced 'a':

        x_{n+1} = a_n x_n (1 - x_n - y_n)
        y_{n+1} = b x_n y_n

    where a_n alternates between A and B according to 'seq'.
    Pre-grid:
      horizontal i -> A in [a0, a1]
      vertical   j -> B in [b0, b1]

    After your transpose+flip in the CLI, the final image is
    B (abscissa) versus A (ordinate), as in Fig. 8.21.
    """
    seq_len = seq.size
    out = np.empty((height, width), dtype=np.float64)

    STATE_MAX = 1e6

    for j in prange(height):
        # vertical: B
        if height > 1:
            B = b0 + (b1 - b0) * (j / (height - 1.0))
        else:
            B = 0.5 * (b0 + b1)

        for i in range(width):
            # horizontal: A
            if width > 1:
                A = a0 + (a1 - a0) * (i / (width - 1.0))
            else:
                A = 0.5 * (a0 + a1)

            x = x0
            y = y0

            vx = 1.0
            vy = 0.0
            acc = 0.0

            for n in range(n_transient + n_iter):
                s = seq[n % seq_len]
                a_param = A if s == 0 else B   # A ↔ 'A', B ↔ 'B'

                # map
                x_next = a_param * x * (1.0 - x - y)
                y_next = b_param * x * y

                # Jacobian at (x,y)
                dxdx = a_param * (1.0 - 2.0 * x - y)
                dxdy = -a_param * x
                dydx = b_param * y
                dydy = b_param * x

                vx_new = dxdx * vx + dxdy * vy
                vy_new = dydx * vx + dydy * vy
                vx, vy = vx_new, vy_new

                norm = math.sqrt(vx * vx + vy * vy)
                if norm < 1e-16:
                    norm = 1e-16

                if n >= n_transient:
                    acc += math.log(norm)

                inv_norm = 1.0 / norm
                vx *= inv_norm
                vy *= inv_norm

                x = x_next
                y = y_next

                # crude bounding to avoid numeric blowup
                if x > STATE_MAX:
                    x = STATE_MAX
                elif x < -STATE_MAX:
                    x = -STATE_MAX
                if y > STATE_MAX:
                    y = STATE_MAX
                elif y < -STATE_MAX:
                    y = -STATE_MAX

                if not (np.isfinite(x) and np.isfinite(y)):
                    x = 0.5
                    y = 0.5

            out[j, i] = acc / float(n_iter)

    return out

@njit(
    float64[:, :](
        int32[:],   # seq: 0/1 for A/B
        int32,      # width
        int32,      # height
        float64,    # a0
        float64,    # a1
        float64,    # b0
        float64,    # b1
        float64,    # x0
        int32,      # n_transient
        int32,      # n_iter
        float64,    # eps
        float64,    # alpha
    ),
    parallel=True,
    fastmath=True,
    cache=True,
)
def _lyapunov_field_dlog(
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
    alpha,
):
    """
    Discontinuous logistic map (eq. 9.6):

      x_{n+1} = r_n x_n (1 - x_n)                   if x_n > 0.5
              = r_n x_n (1 - x_n) + ¼(α - 1)(r_n-2) else

    r_n alternates between A and B according to 'seq'.

    Pre-grid:
      horizontal i -> A ∈ [a0, a1]
      vertical   j -> B ∈ [b0, b1]

    After the transpose+flip you already do in the CLI,
    the final image is B (abscissa) versus A (ordinate).
    """
    seq_len = seq.size
    out = np.empty((height, width), dtype=np.float64)

    for j in prange(height):
        # vertical: B
        if height > 1:
            bj = b0 + (b1 - b0) * (j / (height - 1.0))
        else:
            bj = 0.5 * (b0 + b1)

        for i in range(width):
            # horizontal: A
            if width > 1:
                ai = a0 + (a1 - a0) * (i / (width - 1.0))
            else:
                ai = 0.5 * (a0 + a1)

            x = x0
            acc = 0.0

            for n in range(n_transient + n_iter):
                s = seq[n % seq_len]
                r = ai if s == 0 else bj

                # piecewise map
                core = r * x * (1.0 - x)
                if x <= 0.5:
                    core += 0.25 * (alpha - 1.0) * (r - 2.0)
                x = core

                if not np.isfinite(x):
                    x = 0.5

                if n >= n_transient:
                    # derivative w.r.t x (constant term drops out)
                    d = logistic_deriv(x, r)   # same formula as plain logistic
                    ad = abs(d)
                    if ad < eps:
                        ad = eps
                    acc += math.log(ad)

            out[j, i] = acc / float(n_iter)

    return out

# ---------------------------------------------------------------------------
# Sequence & generator helpers
# ---------------------------------------------------------------------------
# Allowed characters in a sequence token: A/B, digits, parentheses
SEQ_ALLOWED_RE = re.compile(r'^[AaBb0-9()]+$')

def _looks_like_sequence_token(tok: str) -> bool:
    """
    Heuristic: does this look like a sequence (A/B with optional counts
    and/or parenthesized groups), as opposed to a numeric value?
    """
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

    Rules:
        - Token is stripped and unquoted if wrapped in '...' or "...".
        - Grammar (no nesting of parentheses):
              seq := item+
              item := 'A' [digits] | 'B' [digits] | '(' [AB]+ ')' [digits]
        - Digits give repetition count; default count is 1.
        - On any parse error, we fall back to default_seq.
    """
    s = tok.strip()
    # Strip surrounding quotes
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1]
    if not s:
        return default_seq

    out_parts: list[str] = []
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

        # Parenthesized group (AB...) with optional count: (AB)40
        if ch == "(":
            j = s.find(")", i + 1)
            if j == -1:
                return default_seq
            group_str = s[i + 1 : j]
            if not group_str:
                return default_seq
            # group must be only A/B letters
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

        # Unknown character -> fail back to default
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


def generate_lyapunov(
    pix: int,
    sequence: str,
    map_name: str,
    a0: float | None,
    a1: float | None,
    b0: float | None,
    b1: float | None,
    x0: float,
    n_transient: int,
    n_iter: int,
    eps: float,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """
    Generic 1-D Lyapunov field for map_name in
    {logistic, sine, tent, heart, eq984, eq985}.
    """
    if map_name not in MAP_NAME_TO_ID:
        raise ValueError(f"Unknown 1D map '{map_name}'")

    map_id = MAP_NAME_TO_ID[map_name]
    d0_a0, d0_b0, d0_a1, d0_b1 = MAP_DEFAULT_DOMAIN[map_id]

    if a0 is None: a0 = d0_a0
    if a1 is None: a1 = d0_a1
    if b0 is None: b0 = d0_b0
    if b1 is None: b1 = d0_b1

    seq_arr = _seq_to_array(sequence)

    return _lyapunov_field(
        seq_arr,
        int(pix),
        int(pix),
        float(a0),
        float(a1),
        float(b0),
        float(b1),
        int(map_id),
        float(x0),
        int(n_transient),
        int(n_iter),
        float(eps),
        float(alpha),
        float(beta),
    )


def generate_lyapunov_kicked(
    pix: int,
    sequence: str,
    a0: float | None,
    a1: float | None,
    b0: float | None,
    b1: float | None,
    x0: float,
    y0: float,
    eps_kick: float,
    gamma: float,
    n_transient: int,
    n_iter: int,
) -> np.ndarray:
    """
    Kicked‑rotator Lyapunov field (2‑D map), B versus A.

    Parameters
    ----------
    sequence : A/B pattern, e.g. "(BA)2" → BABA...
    a0,a1,b0,b1 : parameter window for (A,B).
    eps_kick : ε in Eqs. (8.48),(8.49)
    gamma    : γ in Eqs. (8.48),(8.49)
    """
    map_id = MAP_NAME_TO_ID["kickedrotator"]
    d0_a0, d0_b0, d0_a1, d0_b1 = MAP_DEFAULT_DOMAIN[map_id]

    if a0 is None:
        a0 = d0_a0
    if a1 is None:
        a1 = d0_a1
    if b0 is None:
        b0 = d0_b0
    if b1 is None:
        b1 = d0_b1

    seq_arr = _seq_to_array(sequence)

    lyap = _lyapunov_field_kicked(
        seq_arr,
        int(pix),
        int(pix),
        float(a0),
        float(a1),
        float(b0),
        float(b1),
        float(x0),
        float(y0),
        float(eps_kick),
        float(gamma),
        int(n_transient),
        int(n_iter),
    )
    return lyap


def generate_lyapunov_parasite(
    pix: int,
    a0: float | None,
    a1: float | None,
    b0: float | None,
    b1: float | None,
    x0: float,
    y0: float,
    K: float,
    n_transient: int,
    n_iter: int,
) -> np.ndarray:
    """
    a0,a1 = α-range (vertical) ; b0,b1 = r-range (horizontal)

    If any are None, fall back to MAP_DEFAULT_DOMAIN for 'parasite'.
    """
    map_id = MAP_NAME_TO_ID["parasite"]
    d0_a0, d0_b0, d0_a1, d0_b1 = MAP_DEFAULT_DOMAIN[map_id]

    alpha0 = d0_a0 if a0 is None else a0
    alpha1 = d0_a1 if a1 is None else a1
    r0     = d0_b0 if b0 is None else b0
    r1     = d0_b1 if b1 is None else b1

    return _lyapunov_field_parasite(
        int32(pix),
        int32(pix),
        float64(r0),
        float64(r1),
        float64(alpha0),
        float64(alpha1),
        float64(x0),
        float64(y0),
        float64(K),
        int32(n_transient),
        int32(n_iter),
    )

def generate_lyapunov_predprey(
    pix: int,
    sequence: str,
    a0: float | None,
    a1: float | None,
    b0: float | None,
    b1: float | None,
    x0: float,
    y0: float,
    b_param: float,
    n_transient: int,
    n_iter: int,
) -> np.ndarray:
    map_id = MAP_NAME_TO_ID["predprey"]
    d0_a0, d0_b0, d0_a1, d0_b1 = MAP_DEFAULT_DOMAIN[map_id]

    A0 = d0_a0 if a0 is None else a0
    A1 = d0_a1 if a1 is None else a1
    B0 = d0_b0 if b0 is None else b0
    B1 = d0_b1 if b1 is None else b1

    seq_arr = _seq_to_array(sequence)

    return _lyapunov_field_predprey(
        seq_arr,
        int32(pix),
        int32(pix),
        float64(A0),
        float64(A1),
        float64(B0),
        float64(B1),
        float64(x0),
        float64(y0),
        float64(b_param),
        int32(n_transient),
        int32(n_iter),
    )

def generate_lyapunov_dlog(
    pix: int,
    sequence: str,
    a0: float | None,
    a1: float | None,
    b0: float | None,
    b1: float | None,
    x0: float,
    alpha: float,
    n_transient: int,
    n_iter: int,
    eps: float,
) -> np.ndarray:
    """
    Discontinuous logistic, AB-plane in (A,B), α fixed.
    """
    map_id = MAP_NAME_TO_ID["dlog"]
    d0_a0, d0_b0, d0_a1, d0_b1 = MAP_DEFAULT_DOMAIN[map_id]

    if a0 is None:
        a0 = d0_a0
    if a1 is None:
        a1 = d0_a1
    if b0 is None:
        b0 = d0_b0
    if b1 is None:
        b1 = d0_b1

    seq_arr = _seq_to_array(sequence)

    lyap = _lyapunov_field_dlog(
        seq_arr,
        int(pix),
        int(pix),
        float(a0),
        float(a1),
        float(b0),
        float(b1),
        float(x0),
        int(n_transient),
        int(n_iter),
        float(eps),
        float(alpha),
    )
    return lyap


# ---------------------------------------------------------------------------
# Color mapping: Lyapunov exponent -> RGB (H x W x 3, uint8)
# ---------------------------------------------------------------------------

def lyapunov_to_rgb(
    lyap: np.ndarray,
    clip: float | None = DEFAULT_CLIP,
    gamma: float = DEFAULT_GAMMA,
    pos_color: str = "red",  # "red" (Fig. 1 style) or "blue" (Fig. 2 style)
) -> np.ndarray:
    """
    Markus & Hess style color map:

      λ < 0 : black  -> yellow  (periodic / order)
      λ = 0 : black  (discontinuity boundary)
      λ > 0 : black  -> red or blue (chaos)

    'clip' sets a symmetric max |λ|; if None/<=0, we infer it from data.
    'gamma' applies to both branches.
    """
    arr = np.asarray(lyap, dtype=np.float64)
    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return rgb

    neg_mask = finite & (arr < 0.0)
    pos_mask = finite & (arr > 0.0)

    # ---- symmetric scaling for |λ| ----
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

    # ---- λ < 0  →  black -> yellow ----
    if np.any(neg_mask):
        lam_neg = np.clip(arr[neg_mask], -scale, 0.0)
        t = np.abs(lam_neg) / scale  # 0..1, 0 at λ=0, 1 at λ=-scale
        if gamma != 1.0:
            t = t ** float(gamma)
        r = t
        g = t
        b = np.zeros_like(t)

        rgb[neg_mask, 0] = np.rint(r * 255.0).astype(np.uint8)
        rgb[neg_mask, 1] = np.rint(g * 255.0).astype(np.uint8)
        rgb[neg_mask, 2] = 0

    # ---- λ > 0  →  black -> red or blue ----
    if np.any(pos_mask):
        lam_pos = np.clip(arr[pos_mask], 0.0, scale)
        t = lam_pos / scale  # 0..1, 0 at λ=0, 1 at λ=+scale
        if gamma != 1.0:
            t = t ** float(gamma)

        if pos_color.lower().startswith("b"):
            # black -> blue (Fig. 2 style)
            r = np.zeros_like(t)
            g = np.zeros_like(t)
            b = t
        else:
            # black -> red (Fig. 1 style)
            r = t
            g = np.zeros_like(t)
            b = np.zeros_like(t)

        rgb[pos_mask, 0] = np.rint(r * 255.0).astype(np.uint8)
        rgb[pos_mask, 1] = np.rint(g * 255.0).astype(np.uint8)
        rgb[pos_mask, 2] = np.rint(b * 255.0).astype(np.uint8)

    # λ == 0 stays black by default: rgb[...] already zero there
    return rgb


# ---------------------------------------------------------------------------
# Spec helpers using specparser.split_chain
# ---------------------------------------------------------------------------

def _eval_number(tok: str) -> complex:
    """Use specparser's scalar evaluator."""
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


def dict2lyapunov(d: dict, *, pix: int) -> tuple[np.ndarray, dict]:
    """
    Interpret one split spec dict as Lyapunov parameters.

    Expected usage:

      logistic[:SEQ][:a0][:a1][:b0][:b1][,x0:...,trans:...,iter:...,eps:...,clip:...,gamma:...]

    where the map token is something like:

      logistic:AAAABBBB:-4:-4:4:4

    Parsing logic for the map token:

      vals = d["logistic"]  # for example

      - If vals[0] exists and "looks like" a sequence (A/B or non-numeric),
        it's SEQ, and remaining values (if any) are a0,a1,b0,b1.

      - Otherwise, vals[0] is treated as a0 (numeric), and SEQ falls back to DEFAULT_SEQ.

    Other keys:

      trans : transient iterations
      iter  : iterations used to accumulate λ
      x0    : initial x
      eps   : epsilon added inside log(|f'| + eps)
      clip  : |λ| clipping scale (per-tile)
      gamma : gamma correction (per-tile)
    """
    map_name = DEFAULT_MAP_NAME
    seq_str  = DEFAULT_SEQ
    seq_raw  = DEFAULT_SEQ

    a0_dom = a1_dom = b0_dom = b1_dom = None

    alpha = None   # parameter for maps
    beta   = None   # NEW

    # --- 1) Self-contained map token (logistic/sine/tent) ---
    for cand in MAP_NAME_TO_ID.keys():
        if cand in d:
            vals = d[cand]
            map_name = cand
            idx = 0

            if len(vals) > 0:
                first = vals[0].strip()

                # heuristic: sequence vs numeric
                if _looks_like_sequence_token(first):
                    # explicit sequence token like A5B5 or (AB)40
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
                        # first value is numeric (A0/B0/etc); keep seq_raw as default
                        idx = 0
                    else:
                        # treat as sequence anyway (odd but allowed)
                        seq_raw = first
                        seq_str = _decode_sequence_token(first, DEFAULT_SEQ)
                        idx = 1

            def parse_float_tok(tok: str) -> float:
                return float(_eval_number(tok).real)

            if idx < len(vals):
                try: a0_dom = parse_float_tok(vals[idx])
                except Exception: pass
                idx += 1
            if idx < len(vals):
                try: b0_dom = parse_float_tok(vals[idx])
                except Exception: pass
                idx += 1
            if idx < len(vals):
                try: a1_dom = parse_float_tok(vals[idx])
                except Exception: pass
                idx += 1
            if idx < len(vals):
                try: b1_dom = parse_float_tok(vals[idx])
                except Exception: pass
                idx += 1

            break  # only one map token is expected

    # --- 2) Map defaults for domain ---
    map_id = MAP_NAME_TO_ID.get(map_name, MAP_NAME_TO_ID[DEFAULT_MAP_NAME])
    d0_a0, d0_b0, d0_a1, d0_b1 = MAP_DEFAULT_DOMAIN[map_id]

    a0 = a0_dom if a0_dom is not None else d0_a0
    b0 = b0_dom if b0_dom is not None else d0_b0
    a1 = a1_dom if a1_dom is not None else d0_a1
    b1 = b1_dom if b1_dom is not None else d0_b1

    # overrides via named a0/a1/b0/b1 if present
    a0 = _get_float(d, "a0", a0)
    a1 = _get_float(d, "a1", a1)
    b0 = _get_float(d, "b0", b0)
    b1 = _get_float(d, "b1", b1)  

    # --- 3) Dynamics & color parameters (map-specific defaults) ---
    x0 = _get_float(d, "x0", DEFAULT_X0)

    if map_name == "kickedrotator":
        # you already had these defaults
        y0    = _get_float(d, "y0", DEFAULT_Y0)
        n_tr  = _get_int  (d, "trans", DEFAULT_KR_TRANS)
        n_it  = _get_int  (d, "iter",  DEFAULT_KR_ITER)
        eps   = _get_float(d, "eps",   DEFAULT_KR_EPS)
        gamma = _get_float(d, "gamma", DEFAULT_KR_GAMMA)

        lyap = generate_lyapunov_kicked(
            pix=pix,
            sequence=seq_str,
            a0=a0,
            a1=a1,
            b0=b0,
            b1=b1,
            x0=x0,
            y0=y0,
            eps_kick=eps,
            gamma=gamma,
            n_transient=n_tr,
            n_iter=n_it,
        )

    elif map_name == "parasite":
        # Host–parasitoid map defaults from Fig. 8.19
        y0    = _get_float(d, "y0",    DEFAULT_PARASITE_Y0)
        K     = _get_float(d, "k",     DEFAULT_PARASITE_K)   # spec key 'k:2.1'
        n_tr  = _get_int  (d, "trans", DEFAULT_PARASITE_TRANS)
        n_it  = _get_int  (d, "iter",  DEFAULT_PARASITE_ITER)
        gamma = _get_float(d, "gamma", DEFAULT_GAMMA)
        eps   = 0.0  # NEW: parasite map has no epsilon; keep meta happy

        lyap = generate_lyapunov_parasite(
            pix=pix,
            a0=a0,
            a1=a1,
            b0=b0,
            b1=b1,
            x0=x0,
            y0=y0,
            K=K,
            n_transient=n_tr,
            n_iter=n_it,
        )
    
    elif map_name == "predprey":
        # Predator–prey with forced 'a' (Eqs. 8.44, 8.45)
        x0    = _get_float(d, "x0",  DEFAULT_PRED_X0)
        y0    = _get_float(d, "y0",  DEFAULT_PRED_Y0)
        bpar  = _get_float(d, "b",   DEFAULT_PRED_B)      # constant b
        n_tr  = _get_int  (d, "trans", DEFAULT_PRED_TRANS)
        n_it  = _get_int  (d, "iter",  DEFAULT_PRED_ITER)
        gamma = _get_float(d, "gamma", DEFAULT_GAMMA)

        eps   = 0.0  # not used in this map; keep meta happy

        lyap = generate_lyapunov_predprey(
            pix=pix,
            sequence=seq_str,
            a0=a0,
            a1=a1,
            b0=b0,
            b1=b1,
            x0=x0,
            y0=y0,
            b_param=bpar,
            n_transient=n_tr,
            n_iter=n_it,
        )

    elif map_name == "dlog":
        # Discontinuous logistic (eq. 9.6)
        x0    = _get_float(d, "x0",    DEFAULT_DLOG_X0)
        alpha = _get_float(d, "alpha", DEFAULT_DLOG_ALPHA)
        n_tr  = _get_int  (d, "trans", DEFAULT_DLOG_TRANS)
        n_it  = _get_int  (d, "iter",  DEFAULT_DLOG_ITER)
        eps   = _get_float(d, "eps",   DEFAULT_EPS)
        gamma = _get_float(d, "gamma", DEFAULT_GAMMA)

        lyap = generate_lyapunov_dlog(
            pix=pix,
            sequence=seq_str,
            a0=a0,
            a1=a1,
            b0=b0,
            b1=b1,
            x0=x0,
            alpha=alpha,
            n_transient=n_tr,
            n_iter=n_it,
            eps=eps,
        )

    elif map_name == "heart":
        # Discrete heart cell map: x_{n+1} = sin(α x_n) + r_n
        x0    = _get_float(d, "x0",    DEFAULT_HEART_X0)
        alpha = _get_float(d, "alpha", DEFAULT_HEART_ALPHA)
        n_tr  = _get_int  (d, "trans", DEFAULT_HEART_TRANS)
        n_it  = _get_int  (d, "iter",  DEFAULT_HEART_ITER)
        eps   = _get_float(d, "eps",   DEFAULT_EPS)
        gamma = _get_float(d, "gamma", DEFAULT_GAMMA)
        # clip handled later, same as other maps

        lyap = generate_lyapunov(
            pix=pix,
            sequence=seq_str,
            map_name=map_name,
            a0=a0,
            a1=a1,
            b0=b0,
            b1=b1,
            x0=x0,
            n_transient=n_tr,
            n_iter=n_it,
            eps=eps,
            alpha=alpha,
        )

    elif map_name == "eq984":
        # x_{n+1} = r ( sin x_n + beta sin(9 x_n) )
        x0    = _get_float(d, "x0",    DEFAULT_X0)
        beta  = _get_float(d, "beta",  DEFAULT_EQ984_BETA)
        n_tr  = _get_int  (d, "trans", DEFAULT_TRANS)
        n_it  = _get_int  (d, "iter",  DEFAULT_ITER)
        eps   = _get_float(d, "eps",   DEFAULT_EPS)
        gamma = _get_float(d, "gamma", DEFAULT_GAMMA)

        lyap = generate_lyapunov(
            pix=pix,
            sequence=seq_str,
            map_name=map_name,
            a0=a0,
            a1=a1,
            b0=b0,
            b1=b1,
            x0=x0,
            n_transient=n_tr,
            n_iter=n_it,
            eps=eps,
            alpha=1.0,      # not used
            beta=beta,
        )

    elif map_name == "eq985":
        # x_{n+1} = beta * exp(tan(r x_n) - x_n)
        x0    = _get_float(d, "x0",    DEFAULT_X0)
        beta  = _get_float(d, "beta",  DEFAULT_EQ985_BETA)
        n_tr  = _get_int  (d, "trans", DEFAULT_TRANS)
        n_it  = _get_int  (d, "iter",  DEFAULT_ITER)
        eps   = _get_float(d, "eps",   DEFAULT_EPS)
        gamma = _get_float(d, "gamma", DEFAULT_GAMMA)

        lyap = generate_lyapunov(
            pix=pix,
            sequence=seq_str,
            map_name=map_name,
            a0=a0,
            a1=a1,
            b0=b0,
            b1=b1,
            x0=x0,
            n_transient=n_tr,
            n_iter=n_it,
            eps=eps,
            alpha=1.0,      # not used
            beta=beta,
        )


    else:
        # 1D maps: logistic, sine, tent
        n_tr  = _get_int  (d, "trans", DEFAULT_TRANS)
        n_it  = _get_int  (d, "iter",  DEFAULT_ITER)
        eps   = _get_float(d, "eps",   DEFAULT_EPS)
        gamma = _get_float(d, "gamma", DEFAULT_GAMMA)

        lyap = generate_lyapunov(
            pix=pix,
            sequence=seq_str,
            map_name=map_name,
            a0=a0,
            a1=a1,
            b0=b0,
            b1=b1,
            x0=x0,
            n_transient=n_tr,
            n_iter=n_it,
            eps=eps,
            alpha=alpha,
        )

    # --- 4) Color mapping (same for all maps) ---
    clip_val = _get_float(d, "clip", DEFAULT_CLIP if DEFAULT_CLIP is not None else -1.0)
    if DEFAULT_CLIP is None and (clip_val <= 0 or not math.isfinite(clip_val)):
        clip_used = None
    else:
        clip_used = clip_val

    rgb = lyapunov_to_rgb(
        lyap,
        clip=clip_used,
        gamma=gamma,
    )

    meta = dict(
        map_name=map_name,
        seq=seq_str,
        seq_raw=seq_raw,
        a0=a0,
        a1=a1,
        b0=b0,
        b1=b1,
        x0=x0,
        y0=y0 if map_name in ("kickedrotator", "parasite") else None,
        K=K if map_name == "parasite" else None,
        alpha=alpha,
        beta=beta,
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

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        "lyapounov-cli",
        description=(
            "Lyapunov fractal renderer (logistic / sine / tent maps).\n"
            "Specs are fully self-contained (map + sequence + window)."
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
        rgb, meta = dict2lyapunov(d, pix=args.pix)
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
