import json
import os
import shutil
import time

import matplotlib
matplotlib.use('Agg')          # write figures to files, do not open a window
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import *
import data
import eos
import model as model_mod
import physics
import scales


def lam_at(epoch):
    """Weight of the physics loss at this epoch."""
    if not RAMP_EPOCHS:
        return LAMBDA
    return LAMBDA * min(1.0, epoch / RAMP_EPOCHS)


def tail_drop(history, frac=0.2):
    """How much the loss still fell over the last `frac` of the run.

    A fit that has converged shows a small number here. A large one means the
    run was stopped too early, not that the model is at its limit.
    """
    h = np.asarray(history, dtype=float)
    n = min(max(int(len(h) * frac), 5), len(h))
    if n < 3:
        return None, None
    y = np.log(np.maximum(h[-n:], 1e-30))
    slope = np.polyfit(np.arange(n), y, 1)[0]
    return (1 - np.exp(slope * n)) * 100, n


def save_checkpoint(path, model, arch, epoch, best_state, history, final=False):
    torch.save({'epoch': epoch,
                'arch': arch,
                'model_state': model.state_dict(),
                'best_state': best_state,
                'history': history,
                'z_keep': list(Z_KEEP),
                'cube': CUBE_PATH,
                'phys_terms': dict(PHYS_TERMS),
                'lambda': LAMBDA,
                'final': final}, path)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    run_dir = os.path.join(OUT_DIR, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy('config.py', os.path.join(run_dir, 'config_used.py'))
    print("writing to", run_dir)

    D = data.prepare(device)
    sc = scales.measure_scales(D['S'])
    SC = physics.torch_scales(sc, device)

    if any(physics.NEEDS_RHO[n] for n in PHYS_TERMS):
        print(f"\nloading the EOS surrogate from {EOS_PATH}")
        D['eos'] = eos.load_eos(device)

    net, arch = model_mod.build_model(D)
    pools = physics.band_pools(D)

    # One probe before the run: every configured term has to produce a finite
    # gradient. A term that does not would otherwise poison the whole run.
    if PHYS_TERMS:
        total, parts = physics.physics_loss(net, D, SC, pools, n_pts=2000)
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e30).item()
        net.zero_grad(set_to_none=True)
        print("\nprobe of the physics terms:")
        for k, v in parts.items():
            print(f"  {k:12s} loss = {v.item():.3e}")
        print(f"  gradient norm = {gnorm:.3e}  (finite: {bool(np.isfinite(gnorm))})")
        if not np.isfinite(gnorm):
            raise RuntimeError("the physics loss gave a non-finite gradient")

    torch.manual_seed(SEED_RUN)
    torch.cuda.manual_seed_all(SEED_RUN)
    np.random.seed(SEED_RUN)

    steps_per_epoch = max(1, TOTAL_STEPS_TARGET // EPOCHS)
    total_steps = EPOCHS * steps_per_epoch
    bs = min(BATCH_SIZE, D['n_data'])

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=LR * 0.01)
    print(f"\nAdam(lr={LR}) with cosine decay to {LR * 0.01:.1e}")
    print(f"{EPOCHS} epochs x {steps_per_epoch} steps = {total_steps:,} steps, "
          f"batch {bs:,} of {D['n_data']:,} supervised points")

    history = dict(data=[], phys=[], total=[],
                   terms={k: [] for k in PHYS_TERMS})
    best_score = np.inf
    best_data = np.inf
    best_epoch = -1
    best_state = None

    t0 = time.time()
    for epoch in range(EPOCHS):
        lam = lam_at(epoch)
        acc_d = 0.0
        acc_p = 0.0
        acc_terms = {k: 0.0 for k in PHYS_TERMS}

        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            j = torch.randint(0, D['n_data'], (bs,), device=device)
            data_loss = torch.mean((net(D['X_data'][j]) - D['Y_data'][j]) ** 2)
            phys_loss, parts = physics.physics_loss(net, D, SC, pools)
            (data_loss + lam * phys_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP)
            opt.step()
            sched.step()

            acc_d += data_loss.item()
            acc_p += phys_loss.detach().item()
            for k, v in parts.items():
                acc_terms[k] += v.item()

        e_data = acc_d / steps_per_epoch
        e_phys = acc_p / steps_per_epoch
        e_terms = {k: v / steps_per_epoch for k, v in acc_terms.items()}
        e_total = e_data + lam * e_phys

        history['data'].append(e_data)
        history['phys'].append(e_phys)
        history['total'].append(e_total)
        for k in PHYS_TERMS:
            history['terms'][k].append(e_terms[k])

        # The best model is chosen on data + LAMBDA * physics, and only after
        # the ramp is over -- before that the score is not comparable.
        if not (PHYS_TERMS and epoch < (RAMP_EPOCHS or 0)):
            score = e_data + LAMBDA * e_phys
            if score < best_score:
                best_score, best_data, best_epoch = score, e_data, epoch
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}

        if epoch % LOG_EVERY == 0:
            extra = "  ".join(f"{k} {e_terms[k]:.2e}" for k in PHYS_TERMS)
            print(f"epoch {epoch:4d} | data {e_data:.4e} | phys {e_phys:.4e} | "
                  f"total {e_total:.4e} | lambda {lam:.3f}  {extra}")

        if CKPT_EVERY and epoch > 0 and epoch % CKPT_EVERY == 0:
            save_checkpoint(os.path.join(run_dir, 'last.pt'), net, arch, epoch,
                            best_state, history)

    minutes = (time.time() - t0) / 60

    if best_state is None:        # run shorter than the ramp
        best_score = history['total'][-1]
        best_data, best_epoch = history['data'][-1], EPOCHS - 1
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    print(f"\n{minutes:.1f} min | best epoch {best_epoch} | "
          f"best data loss {best_data:.4e} | best score {best_score:.4e}")

    drop, n = tail_drop(history['data'])
    if drop is not None:
        print(f"the data loss fell {drop:.1f}% over the last {n} epochs")

    save_checkpoint(os.path.join(run_dir, 'last.pt'), net, arch, EPOCHS - 1,
                    best_state, history, final=True)
    net.load_state_dict(best_state)
    save_checkpoint(os.path.join(run_dir, 'model.pt'), net, arch, best_epoch,
                    best_state, history, final=True)

    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(dict(history, minutes=minutes, best_epoch=best_epoch,
                       best_data=best_data, best_score=best_score,
                       data_residual=sc['left']), f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history['data'], label='data')
    if PHYS_TERMS:
        ax.semilogy(history['phys'], label='physics')
        for k in PHYS_TERMS:
            ax.semilogy(history['terms'][k], ls=':', label=k)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title(RUN_NAME)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'loss.png'), dpi=110)
    print("\nwrote", os.path.join(run_dir, 'model.pt'))
    print("next:  python evaluate.py")


if __name__ == '__main__':
    main()
