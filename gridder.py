#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import julia


def counts_to_monitor(counts, alpha=1.0, m_min=0.3, m_max=3.0):
    """
    Turn 2D bin counts into a monitor field m[i,j].

    counts : 2D array (ny, nx)
      Larger counts -> larger m -> smaller cells in that region.

    alpha  : strength of adaptation.
    m_min, m_max : clamp for stability.
    """
    counts = counts.astype(float)
    mean = counts.mean() + 1e-9
    rho = counts / mean           # normalized around 1
    m = 1.0 + alpha * (rho - 1.0) # m ~ 1 when rho ~ 1
    m = np.clip(m, m_min, m_max)
    return m

def weighted_laplace_grid_from_monitor(m, iters=4000, omega=1.7):
    """
    Elliptic grid generation on [0,1]x[0,1] with monitor m[j,i].

    m : (ny, nx) monitor field, m>0 (large m -> small cells).
    Returns:
        x, y : (ny, nx) physical coordinates in [0,1]x[0,1].
    """
    ny, nx = m.shape

    x = np.zeros((ny, nx))
    y = np.zeros((ny, nx))

    xi  = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(0.0, 1.0, ny)

    # Dirichlet boundary: unit square
    x[0,:]  = xi;   y[0,:]  = 0.0
    x[-1,:] = xi;   y[-1,:] = 1.0
    x[:,0]  = 0.0;  y[:,0]  = eta
    x[:,-1] = 1.0;  y[:,-1] = eta

    # initial interior guess: straight grid
    for j in range(1, ny-1):
        t = eta[j]
        x[j,:] = (1-t)*x[0,:] + t*x[-1,:]
        y[j,:] = (1-t)*y[0,:] + t*y[-1,:]

    # diffusion weights w = 1/m (positive)
    w = 1.0 / m
    wE = 0.5 * (w[:, 1:] + w[:, :-1])
    wW = wE.copy()
    wN = 0.5 * (w[1:, :] + w[:-1, :])
    wS = wN.copy()

    # SOR iteration
    for _ in range(iters):
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                we = wE[j,   i]
                ww = wW[j,   i-1]
                wn = wN[j,   i]
                ws = wS[j-1, i]
                denom = we + ww + wn + ws

                x_new = (we * x[j, i+1] + ww * x[j, i-1] +
                         wn * x[j+1, i] + ws * x[j-1, i]) / denom
                y_new = (we * y[j, i+1] + ww * y[j, i-1] +
                         wn * y[j+1, i] + ws * y[j-1, i]) / denom

                x[j,i] = (1-omega)*x[j,i] + omega*x_new
                y[j,i] = (1-omega)*y[j,i] + omega*y_new

    return x, y

if __name__ == "__main__":

    # --- 1. Sample c vlaues ---
    c=julia.c_sampler(1e5, 0.1, 0.6, 1.5 , 17, 100)

    # --- 2. 2D histogram of z ---
    nx = 51
    ny = 51

    xr = c.real
    yr = c.imag

    # choose bounding box (could also fix to [-1.5, 1.5] etc.)
    xmin, xmax = xr.min(), xr.max()
    ymin, ymax = yr.min(), yr.max()

    counts, xedges, yedges = np.histogram2d(
        yr, xr,                 # note: y first, x second to match (ny, nx)
        bins=(ny, nx),
        range=[[ymin, ymax],[xmin, xmax]]
    )
    # counts has shape (ny, nx)

    # --- 3. Turn counts into monitor, then grid ---
    m = counts_to_monitor(counts, alpha=0.8)
    x, y = weighted_laplace_grid_from_monitor(m, iters=3000, omega=1.7)

    # x,y are in [0,1]x[0,1] logical space; if you want them in the
    # complex-plane box of your histogram:
    X_phys = xmin + x * (xmax - xmin)
    Y_phys = ymin + y * (ymax - ymin)

    # --- 4. Plot for sanity ---
    fig, axes = plt.subplots(1, 2, figsize=(10,5))

    # left: density (histogram)
    axes[0].imshow(
        counts,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="equal"
    )
    axes[0].set_title("2D histogram of Julia samples")

    # right: deformed grid in same physical box
    for j in range(ny):
        axes[1].plot(X_phys[j,:], Y_phys[j,:], color="white", linewidth=0.7)
    for i in range(nx):
        axes[1].plot(X_phys[:,i], Y_phys[:,i], color="white", linewidth=0.7)

    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymin, ymax)
    axes[1].set_aspect("equal", "box")
    axes[1].set_title("Weighted Laplace grid from histogram")
    axes[1].set_facecolor("black")

    plt.tight_layout()
    plt.show()


