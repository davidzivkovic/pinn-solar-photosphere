"""Reading the cube, choosing the supervised points, and normalization.

The functions here build one dictionary, D, that the rest of the code reads.
It is a plain dict so that you can print any part of it, the keys are listed
in prepare() at the bottom of this file.
"""

import os
import numpy as np
import torch
from config import *

# Channels the network predicts. rho is missing on purpose: it is not an
# independent variable, it comes out of the EOS surrogate from T and p.

FIELD_NAMES = [n for n in NAMES if n != 'rho']
NCH = len(FIELD_NAMES)
CH = {name: i for i, name in enumerate(FIELD_NAMES)}


def load_cube(path=None):
    """Read the configured sub-domain of the cube into a dict of fields."""
    path = CUBE_PATH if path is None else path
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. The paths in config.py are "
                                f"relative, so run the scripts from the "
                                f"repository root.")
    a = np.load(path, mmap_mode='r')
    if a.shape[0] != len(NAMES):
        raise ValueError(f"{path} has {a.shape[0]} channels, expected "
                         f"{len(NAMES)} in the order {NAMES}")
    z1 = a.shape[3] if Z1 is None else Z1
    return {n: np.array(a[i, OD:OD + SUB, OD:OD + SUB, Z0:z1], dtype=np.float32)
            for i, n in enumerate(NAMES)}


def build_data_mask(shape):
    """Boolean mask (Nx, Ny, Nz) of the points that enter the data loss.
    It has to be built before normalization, because mu(z) and sd(z) may use
    the seen layers only.
    """
    Nx, Ny, Nz = shape
    bad = [k for k in Z_KEEP if not 0 <= k < Nz]
    if bad:
        raise ValueError(f"Z_KEEP contains layers outside 0..{Nz - 1}: {bad}. "
                         f"This cube has {Nz} layers.")
    mz = np.zeros(Nz, bool)
    mz[np.asarray(Z_KEEP, dtype=int)] = True
    mx = np.zeros(Nx, bool)
    mx[::STRIDE_X] = True
    my = np.zeros(Ny, bool)
    my[::STRIDE_Y] = True
    m = mx[:, None, None] & my[None, :, None] & mz[None, None, :]
    if FRAC < 1.0:
        rg = np.random.default_rng(SEED_MASK)
        m &= rg.random(m.shape) < FRAC
    return m


def normalized_coords(shape):
    """(Nx*Ny*Nz, 3) coordinates on [0,1]^3, flattened over (x, y, z)."""
    Nx, Ny, Nz = shape
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    z = np.linspace(0, 1, Nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)


def per_z_stats(true_raw, z_seen):
    """mu(z) and sd(z) of every channel, shaped (1, 1, Nz, NCH).
     A channel in PERZ_NAMES gets its own mu and sd in every layer, any other
    channel gets one number for the whole box. Both are read off the seen
    layers only (when PERZ_FROM_SEEN is on) and interpolated in between.
    """
    Nz = true_raw.shape[2]
    perz_ch = [CH[n] for n in PERZ_NAMES]
    zs = np.asarray(z_seen, dtype=int)       # seen layers
    za = np.arange(Nz)                       # all layers

    mu = np.zeros((1, 1, Nz, NCH), dtype=np.float32)
    sd = np.zeros((1, 1, Nz, NCH), dtype=np.float32)

    for c in range(NCH):
        if c in perz_ch:
            m = true_raw[..., c].mean(axis=(0, 1))
            s = true_raw[..., c].std(axis=(0, 1))
            if PERZ_FROM_SEEN:
                uselog = FIELD_NAMES[c] in LOG_PERZ and bool((m[zs] > 0).all())
                if uselog:
                    m = np.exp(np.interp(za, zs, np.log(m[zs])))
                    s = np.exp(np.interp(za, zs, np.log(np.maximum(s[zs], 1e-300))))
                else:
                    m = np.interp(za, zs, m[zs])
                    s = np.interp(za, zs, s[zs])
            mu[0, 0, :, c] = m
            sd[0, 0, :, c] = np.maximum(s, SD_FLOOR * s.max())
        else:
            src = true_raw[:, :, zs, c] if PERZ_FROM_SEEN else true_raw[..., c]
            mu[0, 0, :, c] = src.mean()
            sd[0, 0, :, c] = src.std()

    sd += 1e-8
    return mu, sd


def prepare(device, verbose=True):
    """Everything the model and the physics need, in one dict."""
    S = load_cube()
    Nx, Ny, Nz = S['T'].shape

    mask = build_data_mask((Nx, Ny, Nz))
    z_seen = np.flatnonzero(mask.any(axis=(0, 1)))
    seen = set(z_seen.tolist())
    z_hidden = np.array([k for k in range(Nz) if k not in seen], dtype=int)

    true_raw = np.stack([S[n] for n in FIELD_NAMES], axis=-1).astype(np.float32)
    mu, sd = per_z_stats(true_raw, z_seen)
    targets_norm = ((true_raw - mu) / sd).reshape(-1, NCH).astype(np.float32)

    coords = normalized_coords((Nx, Ny, Nz))
    X_train = torch.from_numpy(coords).to(device)
    Y_train = torch.from_numpy(targets_norm).to(device)

    data_idx = torch.from_numpy(np.flatnonzero(mask.ravel())).to(device)
    X_data = X_train[data_idx]
    Y_data = Y_train[data_idx]

    # mu and sd as torch profiles over z, plus their z derivatives. The
    # derivatives are needed because denorm() in physics.py has to be
    # differentiable in z (chain rule through the profile).
    mu_t = torch.from_numpy(mu[0, 0]).float().to(device)
    sd_t = torch.from_numpy(sd[0, 0]).float().to(device)
    dmu_t = torch.from_numpy(np.gradient(mu[0, 0], DZ, axis=0)).float().to(device)
    dsd_t = torch.from_numpy(np.gradient(sd[0, 0], DZ, axis=0)).float().to(device)

    D = dict(S=S, Nx=Nx, Ny=Ny, Nz=Nz,
             true_raw=true_raw, mu=mu, sd=sd,
             mask=mask, z_seen=z_seen, z_hidden=z_hidden,
             coords=coords, X_train=X_train, Y_train=Y_train,
             X_data=X_data, Y_data=Y_data,
             n_data=int(X_data.shape[0]), n_grid=int(X_train.shape[0]),
             mu_t=mu_t, sd_t=sd_t, dmu_t=dmu_t, dsd_t=dsd_t,
             box=((Nx - 1) * DX, (Ny - 1) * DX, (Nz - 1) * DZ),
             device=device, eos=None)

    if verbose:
        report(D)
    return D


def report(D):
    """Print what was loaded and check that the normalization is sane."""
    Nx, Ny, Nz = D['Nx'], D['Ny'], D['Nz']
    print(f"cube {CUBE_PATH}")
    print(f"sub-domain: {Nx} x {Ny} x {Nz}   (z {Z0}..{Z0 + Nz - 1})")
    for n in NAMES:
        f = D['S'][n]
        print(f"  {n:4s} min={f.min():+.3e}  max={f.max():+.3e}")

    print(f"\nlayers with data    ({len(D['z_seen'])}): {D['z_seen'].tolist()}")
    print(f"layers without data ({len(D['z_hidden'])}): {D['z_hidden'].tolist()}")
    print(f"data points: {D['n_data']:,} / {D['n_grid']:,} "
          f"({D['n_data'] / D['n_grid'] * 100:.1f}%)")
    print(f"physics     : the whole grid, {D['n_grid']:,} points")

    tn = ((D['true_raw'] - D['mu']) / D['sd'])
    zs = D['z_seen']
    perz_ch = [CH[n] for n in PERZ_NAMES]
    print("\nnormalization on the seen layers (should be ~0 mean, ~1 std):")
    for c, n in enumerate(FIELD_NAMES):
        col = tn[:, :, zs, c]
        tag = 'per-z' if c in perz_ch else 'global'
        print(f"  {n:3s} [{tag:6s}] mean={col.mean():+.3f}  std={col.std():.3f}")

    if PERZ_FROM_SEEN and len(D['z_hidden']):
        print("\nleak check: how far the interpolated mu(z) misses on hidden layers")
        print("(0% would mean the true mean of a hidden layer was handed over)")
        for c in perz_ch:
            true_mu = D['true_raw'][..., c].mean(axis=(0, 1))[D['z_hidden']]
            e = np.abs(D['mu'][0, 0, D['z_hidden'], c] - true_mu) / (np.abs(true_mu) + 1e-30)
            print(f"  {FIELD_NAMES[c]:3s} mu off by {e.mean() * 100:5.2f}%")
