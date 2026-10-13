import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from config import *
import data
from data import FIELD_NAMES, NCH, CH
import eos
import model as model_mod
import physics
import scales

LOG_FIELDS = ['p', 'rho']      # fields drawn on a logarithmic colour scale


def predict_cube(net, D, chunk=200_000):
    """The network on every grid point, back in physical units."""
    net.eval()
    out = []
    X = D['X_train']
    with torch.no_grad():
        for s in range(0, X.shape[0], chunk):
            out.append(net(X[s:s + chunk]).cpu().numpy())
    pred_norm = np.concatenate(out, axis=0).reshape(D['Nx'], D['Ny'], D['Nz'], NCH)
    return pred_norm * D['sd'] + D['mu']


def nrmse(pred, true):
    """rmse / std(true), in percent."""
    return np.sqrt(np.mean((pred - true) ** 2)) / (true.std() + 1e-30) * 100


def compare_slice(true_raw, pred_raw, take, title, path, rel):
    """Truth, prediction and a scatter plot, one row per field."""
    fig, axes = plt.subplots(NCH, 3, figsize=(15, 5 * NCH))
    for i, name in enumerate(FIELD_NAMES):
        ft, fp = take(true_raw, i), take(pred_raw, i)
        vmin, vmax = ft.min(), ft.max()
        aspect = 'auto' if ft.shape[0] != ft.shape[1] else None

        uselog = name in LOG_FIELDS and bool((ft > 0).all())
        if uselog:
            vmin = max(vmin, np.percentile(ft, 0.1), 1e-300)
            kw = dict(norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            kw = dict(vmin=vmin, vmax=vmax)

        for j, (field, tag) in enumerate([(ft, 'truth'), (fp, 'PINN')]):
            shown = np.clip(field, vmin * 1e-3, None) if uselog else field
            im = axes[i, j].imshow(shown.T, origin='lower', cmap='plasma',
                                   aspect=aspect, **kw)
            axes[i, j].set_title(f"{name} - {tag}" + (' [log]' if uselog else ''))
            plt.colorbar(im, ax=axes[i, j])

        k = np.random.default_rng(0).choice(ft.size, size=min(5000, ft.size),
                                            replace=False)
        a = axes[i, 2]
        a.scatter(ft.ravel()[k], fp.ravel()[k], alpha=0.3, s=5, color='steelblue')
        if uselog and bool((fp > 0).all()):
            a.set_xscale('log')
            a.set_yscale('log')
        a.plot([vmin, vmax], [vmin, vmax], 'r-', lw=1.5)
        a.set_title(f"{name} - scatter (NRMSE {rel[i]:.2f}%)")
        a.grid(alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.001)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = os.path.join(OUT_DIR, RUN_NAME)
    ckpt_path = os.path.join(run_dir, 'model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found. Run train.py first.")

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if list(ck.get('z_keep', Z_KEEP)) != list(Z_KEEP) or ck.get('cube') != CUBE_PATH:
        print("WARNING: config.py no longer matches this run.")
        print(f"  run was trained with cube={ck.get('cube')} z_keep={ck.get('z_keep')}")
        print(f"  config.py now says    cube={CUBE_PATH} z_keep={Z_KEEP}")
        print("  the normalization will not be the one the network was trained with")

    D = data.prepare(device, verbose=False)
    net = model_mod.model_from_arch(ck['arch']).to(device)
    net.load_state_dict(ck['model_state'])
    print(f"run {RUN_NAME}: best epoch {ck['epoch']}, "
          f"physics {ck.get('phys_terms')}, lambda {ck.get('lambda')}")

    true_raw = D['true_raw']
    pred_raw = predict_cube(net, D)

    # --- error per field ------------------------------------------------
    rel = np.array([nrmse(pred_raw[..., i], true_raw[..., i]) for i in range(NCH)])
    print("\nNRMSE over the whole cube:")
    for i, name in enumerate(FIELD_NAMES):
        print(f"  {name:3s} {rel[i]:6.2f}%")
    vb = [CH[n] for n in ['vx', 'vy', 'vz', 'Bx', 'By', 'Bz']]
    print(f"  mean over v and B: {rel[vb].mean():.2f}%     mean over all: {rel.mean():.2f}%")

    # --- error per layer ------------------------------------------------
    Nz = D['Nz']
    rel_z = np.zeros((Nz, NCH))
    for iz in range(Nz):
        for i in range(NCH):
            rel_z[iz, i] = nrmse(pred_raw[:, :, iz, i], true_raw[:, :, iz, i])

    print(f"\n{'z':>4s} {'data':>7s} " + " ".join(f"{n:>7s}" for n in FIELD_NAMES)
          + f" {'mean':>7s}")
    for iz in range(Nz):
        frac = D['mask'][:, :, iz].mean()
        tag = f"{frac * 100:6.1f}%" if frac > 0 else "      -"
        print(f"{iz:4d} {tag:>7s} " + " ".join(f"{rel_z[iz, i]:7.2f}" for i in range(NCH))
              + f" {rel_z[iz].mean():7.2f}")

    z_seen, z_hidden = D['z_seen'], D['z_hidden']
    seen_mean = float(rel_z[z_seen].mean())
    summary = dict(run=RUN_NAME, cube=CUBE_PATH, best_epoch=int(ck['epoch']),
                   phys_terms=ck.get('phys_terms'), lam=ck.get('lambda'),
                   nrmse_per_field={n: float(rel[i]) for i, n in enumerate(FIELD_NAMES)},
                   nrmse_all=float(rel.mean()), nrmse_vB=float(rel[vb].mean()),
                   nrmse_seen=seen_mean)
    if len(z_hidden):
        hidden_mean = float(rel_z[z_hidden].mean())
        summary['nrmse_hidden'] = hidden_mean
        summary['hidden_over_seen'] = hidden_mean / seen_mean
        print(f"\nmean NRMSE on seen layers   : {seen_mean:.2f}%")
        print(f"mean NRMSE on hidden layers : {hidden_mean:.2f}%")
        print(f"ratio                       : {hidden_mean / seen_mean:.2f}x")
    else:
        print(f"\nevery layer was supervised, mean NRMSE {rel_z.mean():.2f}%")

    if STRIDE_X > 1 or STRIDE_Y > 1 or FRAC < 1.0:
        err = np.sqrt(((pred_raw - true_raw) ** 2).mean(axis=-1))
        sd_all = true_raw.std(axis=(0, 1, 2)).mean()
        mask = D['mask']
        print(f"\ninside the seen layers: on supervised points "
              f"{err[mask].mean() / sd_all * 100:.2f}%, "
              f"on the rest {err[~mask].mean() / sd_all * 100:.2f}%")

    # --- residual of the trained network --------------------------------
    if PHYS_TERMS:
        sc = scales.measure_scales(D['S'], verbose=False)
        SC = physics.torch_scales(sc, device)
        if any(physics.NEEDS_RHO[n] for n in PHYS_TERMS):
            D['eos'] = eos.load_eos(device)
        pools = physics.band_pools(D, verbose=False)
        _, parts = physics.physics_loss(net, D, SC, pools, n_pts=20_000)
        print("\nequation residual of the trained network (rms, relative):")
        summary['residual_after_training'] = {}
        for k, v in parts.items():
            value = float(np.sqrt(v.item()))
            summary['residual_after_training'][k] = value
            print(f"  {k:12s} {value * 100:6.2f}%   "
                  f"(the data itself: {np.atleast_1d(sc['left'][k]).mean() * 100:.2f}%)")

    # --- figures ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(FIELD_NAMES):
        ax.plot(range(Nz), rel_z[:, i], marker='.', ms=3, label=name)
    for iz in z_hidden:
        ax.axvspan(iz - 0.5, iz + 0.5, color='crimson', alpha=0.10, lw=0)
    ax.set_xlabel('z layer')
    ax.set_ylabel('NRMSE [%]')
    ax.set_title(f'{RUN_NAME}: error per layer (red = layer without data)')
    ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'error_per_layer.png'), dpi=110)
    plt.close(fig)

    iz = min(SL_K, Nz - 1)
    compare_slice(true_raw, pred_raw, lambda a, i: a[:, :, iz, i],
                  f'{RUN_NAME}: z layer {iz} of {Nz}',
                  os.path.join(run_dir, f'compare_z{iz}.png'), rel)
    iy = D['Ny'] // 2
    compare_slice(true_raw, pred_raw, lambda a, i: a[:, iy, :, i],
                  f'{RUN_NAME}: vertical cut at y = {iy}',
                  os.path.join(run_dir, f'compare_y{iy}.png'), rel)

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nwrote", os.path.join(run_dir, 'metrics.json'), "and the figures")


if __name__ == '__main__':
    main()
