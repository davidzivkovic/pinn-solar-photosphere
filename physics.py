import numpy as np
import torch

from config import *
from data import FIELD_NAMES, CH


def to_torch(a, device):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)


def denorm(pred, X, D):
    """Network output (z-scores) -> physical values, differentiable in z.

    mu and sd are known on the grid layers only. Around the nearest layer they
    are expanded linearly,

        mu(z) = mu[i] + (dmu/dz)[i] * (z - z_i)

    and the same for sd, so autograd gives the full chain rule

        d/dz [sd(z) u + mu(z)] = sd du/dz + u dsd/dz + dmu/dz.

    On a grid point (z - z_i) is zero, so the value is exactly the one used for
    the data, and the slope of the profile is the same central difference the
    numpy diagnostics use.
    """
    Nz = D['Nz']
    zc = X[:, 2].clamp(0.0, 1.0) * (Nz - 1)
    i = zc.detach().round().long().clamp(0, Nz - 1)        # nearest layer
    off = ((zc - i.to(zc.dtype)) * DZ).unsqueeze(1)        # distance from it, cm
    mu = D['mu_t'][i] + D['dmu_t'][i] * off
    sd = D['sd_t'][i] + D['dsd_t'][i] * off
    return pred * sd + mu


def grad3(expr, X, D):
    """(d/dx, d/dy, d/dz) of `expr` in cgs.

    The network sees coordinates on [0,1]^3, so each derivative is divided by
    the physical length of the box along that axis.
    """
    g = torch.autograd.grad(expr, X, torch.ones_like(expr), create_graph=True)[0]
    bx, by, bz = D['box']
    return g[:, 0] / bx, g[:, 1] / by, g[:, 2] / bz


def phys_fields(pred, X, D, with_rho=True):
    """The physical fields at the points X, as a dict. rho comes from the EOS."""
    val = denorm(pred, X, D)
    F = {n: val[:, CH[n]] for n in FIELD_NAMES}
    if with_rho:
        F['rho'] = D['eos'](F['T'], F['p'])
    return F


def layer_index(X, D):
    """Nearest z layer of each point, for the per-layer scales."""
    return (X[:, 2].detach().clamp(0, 1) * (D['Nz'] - 1)).round().long().clamp(0, D['Nz'] - 1)


def res_divB(F, X, D, SC):
    """div B = 0. One component, divided by a single scale."""
    d = (grad3(F['Bx'], X, D)[0]
         + grad3(F['By'], X, D)[1]
         + grad3(F['Bz'], X, D)[2])
    return (d / SC['divb_t']).unsqueeze(1)


def res_stat_cont(F, X, D, SC):
    """div (rho v) = 0, divided by the scale of its own layer."""
    r = F['rho']
    d = (grad3(r * F['vx'], X, D)[0]
         + grad3(r * F['vy'], X, D)[1]
         + grad3(r * F['vz'], X, D)[2])
    return (d / SC['cont_t'][layer_index(X, D)]).unsqueeze(1)


def lorentz(F, X, D):
    """(1/4pi) (curl B) x B."""
    Bx, By, Bz = F['Bx'], F['By'], F['Bz']
    dBx, dBy, dBz = grad3(Bx, X, D), grad3(By, X, D), grad3(Bz, X, D)
    cx = dBz[1] - dBy[2]
    cy = dBx[2] - dBz[0]
    cz = dBy[0] - dBx[1]
    k = 1.0 / (4 * np.pi)
    return k * (cy * Bz - cz * By), k * (cz * Bx - cx * Bz), k * (cx * By - cy * Bx)


def res_momentum(F, X, D, SC):
    """A + G - L = 0, three components, each divided by its layer scale."""
    rho = F['rho']
    vx, vy, vz = F['vx'], F['vy'], F['vz']

    Ax = (grad3(rho * vx * vx, X, D)[0]
          + grad3(rho * vx * vy, X, D)[1]
          + grad3(rho * vx * vz, X, D)[2])
    Ay = (grad3(rho * vy * vx, X, D)[0]
          + grad3(rho * vy * vy, X, D)[1]
          + grad3(rho * vy * vz, X, D)[2])
    Az = (grad3(rho * vz * vx, X, D)[0]
          + grad3(rho * vz * vy, X, D)[1]
          + grad3(rho * vz * vz, X, D)[2])

    Gx, Gy, Gz = grad3(F['p'], X, D)
    Gz = Gz + rho * G_SUN

    Lx, Ly, Lz = lorentz(F, X, D)

    R = torch.stack([Ax + Gx - Lx, Ay + Gy - Ly, Az + Gz - Lz], dim=1)
    return R / SC['mom_t'][layer_index(X, D)]


# Which function belongs to which name, and whether it needs rho (that is,
# whether the EOS has to be evaluated for it).
RESIDUALS = {
    'divB':      res_divB,
    'stat_cont': res_stat_cont,
    'momentum':  res_momentum,
}
NEEDS_RHO = {
    'divB':      False,
    'stat_cont': True,
    'momentum':  True,
}


def torch_scales(sc, device):
    """The measured scales as tensors on the device."""
    return dict(divb_t=to_torch(sc['divb'], device),
                cont_t=to_torch(sc['cont_z'], device),
                mom_t=to_torch(sc['mom_z'], device))


def band_pools(D, verbose=True):
    """Indices of the grid points inside each term's height band.

    Collocation points are drawn from these pools, so none of the budget is
    spent on heights where the equation is not imposed.
    """
    z = D['X_train'][:, 2].detach()
    pools = {}
    for name in PHYS_TERMS:
        if name not in RESIDUALS:
            raise ValueError(f"unknown physics term '{name}'; "
                             f"known terms are {sorted(RESIDUALS)}")
        lo, hi = Z_BAND.get(name, (0.0, 1.0))
        if (lo, hi) not in pools:
            pools[(lo, hi)] = torch.nonzero((z >= lo) & (z <= hi), as_tuple=True)[0]
    if verbose and PHYS_TERMS:
        print(f"\n{'term':12s} {'weight':>7s} {'rho':>5s} {'z band':>12s} "
              f"{'points in band':>15s}")
        for name in PHYS_TERMS:
            lo, hi = Z_BAND.get(name, (0.0, 1.0))
            print(f"{name:12s} {PHYS_TERMS[name]:7.3f} {str(NEEDS_RHO[name]):>5s} "
                  f"{f'{lo:.2f}-{hi:.2f}':>12s} {len(pools[(lo, hi)]):15,d}")
    return pools


def physics_loss(model, D, SC, pools, n_pts=None):
    """Total physics loss and the loss of each term.

    Terms that share a height band share one forward pass and one evaluation
    of the EOS, which is where most of the cost is.
    """
    if not PHYS_TERMS:
        return torch.zeros((), device=D['device']), {}

    n_pts = PHYS_BATCH if n_pts is None else n_pts
    total = torch.zeros((), device=D['device'])
    parts = {}

    groups = {}
    for name in PHYS_TERMS:
        groups.setdefault(Z_BAND.get(name, (0.0, 1.0)), []).append(name)

    for band, names in groups.items():
        pool = pools[band]
        j = pool[torch.randint(0, len(pool), (n_pts,), device=D['device'])]
        X = D['X_train'][j].clone().requires_grad_(True)
        F = phys_fields(model(X), X, D,
                        with_rho=any(NEEDS_RHO[n] for n in names))
        for name in names:
            loss = (RESIDUALS[name](F, X, D, SC) ** 2).mean()
            parts[name] = loss
            total = total + PHYS_TERMS[name] * loss

    return total, parts
