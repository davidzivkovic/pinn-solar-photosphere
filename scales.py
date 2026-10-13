import numpy as np

from config import *
from operators import f8, ddx, ddy, ddz, div, grad, curl, interior, interior_xy


def momentum_terms(S):
    """The three groups of terms of the stationary momentum equation.

    A = div(rho v v)              advection
    G = grad p + rho g e_z        pressure and gravity
    L = (1/4pi) (curl B) x B      Lorentz force
    and the equation is A + G - L = 0.
    """
    rho = f8(S['rho'])
    vx, vy, vz = f8(S['vx']), f8(S['vy']), f8(S['vz'])
    Bx, By, Bz = f8(S['Bx']), f8(S['By']), f8(S['Bz'])

    A = [div(rho * vx * vx, rho * vx * vy, rho * vx * vz),
         div(rho * vy * vx, rho * vy * vy, rho * vy * vz),
         div(rho * vz * vx, rho * vz * vy, rho * vz * vz)]

    gx, gy, gz = grad(f8(S['p']))
    G = [gx, gy, gz + rho * G_SUN]

    cx, cy, cz = curl(Bx, By, Bz)
    k = 1.0 / (4 * np.pi)
    L = [k * (cy * Bz - cz * By),
         k * (cz * Bx - cx * Bz),
         k * (cx * By - cy * Bx)]

    return A, G, L


def measure_scales(S, verbose=True):
    """Measure every scale, and the residual the data itself leaves behind."""
    Nz = S['T'].shape[2]
    TR = interior(Nz)       # interior in all three axes
    XY = interior_xy()      # interior horizontally, every z layer
    core = slice(PAD, Nz - PAD)
    left = {}

    # --- div B = 0 -------------------------------------------------------
    # The scale is the mean size of the individual terms that have to cancel.
    Bx, By, Bz = f8(S['Bx']), f8(S['By']), f8(S['Bz'])
    divb = float((np.abs(ddx(Bx)) + np.abs(ddy(By)) + np.abs(ddz(Bz)))[TR].mean())
    left['divB'] = float(np.abs(div(Bx, By, Bz)[TR]).mean() / divb)

    # --- div (rho v) = 0, per layer --------------------------------------
    rho = f8(S['rho'])
    fx, fy, fz = rho * f8(S['vx']), rho * f8(S['vy']), rho * f8(S['vz'])
    cont_z = (np.abs(ddx(fx)) + np.abs(ddy(fy)) + np.abs(ddz(fz)))[XY].mean(axis=(0, 1))
    cont_z = np.maximum(cont_z, 1e-6 * cont_z.max()).astype(np.float32)
    rel = np.abs(div(fx, fy, fz))[XY].mean(axis=(0, 1)) / cont_z
    left['stat_cont'] = float(rel[core].mean())

    # --- momentum, per layer and per component ---------------------------
    # The reference is the largest of the three competing terms, because that
    # is the size the residual has to be compared against.
    A, G, L = momentum_terms(S)
    mom_z = np.stack([
        np.maximum.reduce([np.abs(A[j])[XY].mean(axis=(0, 1)),
                           np.abs(G[j])[XY].mean(axis=(0, 1)),
                           np.abs(L[j])[XY].mean(axis=(0, 1))])
        for j in range(3)], axis=1)
    mom_z = np.maximum(mom_z, 1e-6 * mom_z.max(axis=0, keepdims=True)).astype(np.float32)
    mom_rel = np.stack([
        np.abs(A[j] + G[j] - L[j])[XY].mean(axis=(0, 1)) / mom_z[:, j]
        for j in range(3)], axis=1)
    left['momentum'] = [float(mom_rel[core, j].mean()) for j in range(3)]

    sc = dict(divb=divb, cont_z=cont_z, mom_z=mom_z, left=left)
    if verbose:
        report(sc, S)
    return sc


def report(sc, S):
    """Print the scales and how far the data is from each equation."""
    Nz = S['T'].shape[2]
    lo, hi = [int(round(v * (Nz - 1))) for v in Z_BAND['stat_cont']]
    left = sc['left']
    print("\nresidual the simulation itself leaves (relative, interior only):")
    print(f"  divB       {left['divB'] * 100:6.2f}%")
    print(f"  stat_cont  {left['stat_cont'] * 100:6.2f}%   "
          f"(scale varies {sc['cont_z'].max() / sc['cont_z'].min():.0f}x over height)")
    print(f"  momentum   " + "  ".join(f"{k}:{v * 100:.0f}%"
                                       for k, v in zip('xyz', left['momentum'])))
    print(f"\nstat_cont is imposed on z={lo}..{hi} only, where it holds best.")
    print("A term that the data leaves far from zero is one the network cannot")
    print("satisfy either, so it belongs in the loss with a small weight or not at all.")
