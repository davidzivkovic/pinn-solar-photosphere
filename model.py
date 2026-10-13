import numpy as np
import torch
import torch.nn as nn

from config import *
from data import FIELD_NAMES, NCH, CH


def power_spectrum_1d(field, axis):
    """Power spectrum along one axis, averaged over the other two."""
    P = np.abs(np.fft.fft(field, axis=axis)) ** 2
    P = np.moveaxis(P, axis, 0)
    P = P.reshape(P.shape[0], -1).mean(axis=1)
    return P[:len(P) // 2]


def char_freq(field, axis, thresh=0.1):
    """Highest frequency that still carries power, in cycles over the domain.

    The spectrum is divided by its first non-zero mode and the frequency is
    where it drops below `thresh` of that. Above it the field is flat, so the
    network does not need features there.
    """
    n = field.shape[axis]
    if n < 4:
        return 1
    P = power_spectrum_1d(field, axis)
    P = P / (P[1] + 1e-300)
    below = np.where(P < thresh)[0]
    return int(below[0]) if len(below) else len(P)


def pick_sigma(true_raw, verbose=True):
    """Widths of the two frequency bands, from the spectrum of the seen data.

    The spectrum is measured on exactly the points that enter the data loss
    (the Z_KEEP layers, after the strides), because those are the only points
    whose structure the network can actually learn.
    """
    zk = np.sort(np.asarray(Z_KEEP, dtype=int))
    sub = true_raw[::STRIDE_X, ::STRIDE_Y][:, :, zk]
    nx_s, ny_s, nz_s = sub.shape[:3]

    # Nyquist of this sampling, in cycles over the domain. Anything above it
    # is indistinguishable from a lower frequency (aliasing).
    nyq = np.array([nx_s // 2, ny_s // 2, nz_s // 2], dtype=float)

    steps = np.unique(np.diff(zk)) if len(zk) > 1 else np.array([1])
    if verbose and len(steps) > 1:
        print(f"note: Z_KEEP is not equidistant (steps {steps.tolist()}), so the "
              f"spectrum in z is approximate; the Nyquist cap still holds")

    K = np.zeros((NCH, 3), dtype=int)
    K_full = np.zeros((NCH, 3), dtype=int)
    for i in range(NCH):
        for ax in range(3):
            K[i, ax] = char_freq(sub[..., i], ax)
            K_full[i, ax] = char_freq(true_raw[..., i], ax)
    K = np.minimum(K, np.floor(NYQ_SAFETY * nyq)[None, :]).astype(int)

    # x and y are statistically the same direction, so they share one width.
    K_xy = K.copy()
    both = np.maximum(K[:, 0], K[:, 1])
    K_xy[:, 0] = both
    K_xy[:, 1] = both

    idx_high = [CH[n] for n in HIGH_BAND_NAMES]
    idx_low = [i for i in range(NCH) if i not in idx_high]
    agg = {'max': lambda a: a.max(axis=0),
           'median': lambda a: np.median(a, axis=0),
           'mean': lambda a: a.mean(axis=0)}[SIGMA_AGG]

    # A frequency of k cycles over a domain of length 1 is k = 2 pi sigma in
    # the Fourier-feature convention used below, hence the division.
    sigma_high = tuple(float(v) for v in np.round(agg(K_xy[idx_high]) / (2 * np.pi), 3))
    sigma_low = tuple(float(v) for v in np.round(agg(K_xy[idx_low]) / (2 * np.pi), 3))
    sigma_low = (round(sigma_low[0] * SIGMA_XY_LOW, 3),
                 round(sigma_low[1] * SIGMA_XY_LOW, 3),
                 round(sigma_low[2] * SIGMA_Z_FACTOR, 3))
    sigma_high = (round(sigma_high[0] * SIGMA_XY_HIGH, 3),
                  round(sigma_high[1] * SIGMA_XY_HIGH, 3),
                  round(sigma_high[2] * SIGMA_Z_FACTOR, 3))

    if verbose:
        print(f"\nspectrum measured on {nx_s} x {ny_s} x {nz_s} points "
              f"(of {true_raw.shape[0]} x {true_raw.shape[1]} x {true_raw.shape[2]})")
        print(f"Nyquist [cycles/domain]: x={nyq[0]:.0f} y={nyq[1]:.0f} z={nyq[2]:.0f}"
              f"   cap = {NYQ_SAFETY} x Nyquist\n")
        print(f"{'field':6s} {'kx':>4s} {'ky':>4s} {'kz':>4s}   {'(full grid)':>16s}")
        for i, n in enumerate(FIELD_NAMES):
            print(f"{n:6s} {K[i,0]:4d} {K[i,1]:4d} {K[i,2]:4d}   "
                  f"({K_full[i,0]:3d} {K_full[i,1]:3d} {K_full[i,2]:3d})")
        print(f"\naggregation over a band: {SIGMA_AGG}")
        print(f"narrow band (T,p,v) sigma = {sigma_low}"
              f"   -> {sigma_low[0] * 2 * np.pi:.1f} cycles/domain")
        print(f"wide band   (B)     sigma = {sigma_high}"
              f"   -> {sigma_high[0] * 2 * np.pi:.1f} cycles/domain")

    return sigma_low, sigma_high


class GaussianFourierFeatures(nn.Module):
    """Two bands of random Fourier features for coordinates in [0,1]^3."""

    def __init__(self, num_freqs, sigma_low, sigma_high, frac_low, seed=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        n_low = int(num_freqs * frac_low)
        n_high = num_freqs - n_low
        s_lo = torch.tensor(sigma_low, dtype=torch.float32)
        s_hi = torch.tensor(sigma_high, dtype=torch.float32)
        B_lo = torch.randn(n_low, 3, generator=g) * s_lo[None, :]
        B_hi = torch.randn(n_high, 3, generator=g) * s_hi[None, :]
        # Kept as a buffer, not a parameter: the frequencies are fixed, only
        # the MLP after them is trained.
        self.register_buffer("B", torch.cat([B_lo, B_hi], 0) * 2 * np.pi)
        self.out_dim = 3 + 2 * num_freqs

    def forward(self, x):
        proj = x @ self.B.T
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class NeuralField(nn.Module):
    """MLP on the Fourier features, with one skip connection in the middle."""

    def __init__(self, encoder, width, depth, out_ch):
        super().__init__()
        self.enc = encoder
        d = self.enc.out_dim
        self.skip_at = depth // 2
        self.layers = nn.ModuleList([nn.Linear(d, width)])
        for i in range(1, depth):
            self.layers.append(nn.Linear(width + d if i == self.skip_at else width, width))
        self.out = nn.Linear(width, out_ch)
        self.act = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h0 = self.enc(x)
        h = h0
        for i, lin in enumerate(self.layers):
            if i == self.skip_at:
                h = torch.cat([h, h0], dim=-1)
            h = self.act(lin(h))
        return self.out(h)


def build_model(D, verbose=True):
    """Build the network for this cube. Returns (model, arch).

    `arch` holds everything needed to rebuild the same network later, so
    evaluate.py does not have to measure the spectrum again.
    """
    sigma_low, sigma_high = pick_sigma(D['true_raw'], verbose=verbose)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    arch = dict(width=WIDTH, depth=DEPTH, num_freqs=NUM_FREQS,
                sigma_low=sigma_low, sigma_high=sigma_high,
                frac_low=FRAC_LOW, seed=SEED, out_ch=NCH)
    model = model_from_arch(arch).to(D['device'])

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        n_values = D['true_raw'].size
        print(f"\nwidth={arch['width']}  depth={arch['depth']}  "
              f"parameters={n_params:,}")
        print(f"values in the cube: {n_values:,}  -> compression "
              f"{n_values / n_params:.2f}x")
    return model, arch


def model_from_arch(arch):
    """Rebuild a network from the dict stored in a checkpoint."""
    enc = GaussianFourierFeatures(num_freqs=arch['num_freqs'],
                                  sigma_low=arch['sigma_low'],
                                  sigma_high=arch['sigma_high'],
                                  frac_low=arch['frac_low'],
                                  seed=arch['seed'])
    return NeuralField(enc, width=arch['width'], depth=arch['depth'],
                       out_ch=arch['out_ch'])
