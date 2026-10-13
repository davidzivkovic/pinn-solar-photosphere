import os

import numpy as np
import torch

from config import *
from data import load_cube, build_data_mask
from eos import EOSNet, check_eos


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    S = load_cube()
    Nx, Ny, Nz = S['T'].shape

    if EOS_FROM_SEEN_ONLY:
        mask = build_data_mask((Nx, Ny, Nz))
        T = S['T'][mask]
        p = S['p'][mask]
        rho = S['rho'][mask]
        print(f"training on the seen layers only: {T.size:,} points")
    else:
        T = S['T'].ravel()
        p = S['p'].ravel()
        rho = S['rho'].ravel()
        print(f"training on the whole cube: {T.size:,} points")

    # Work in logarithms, and normalize both sides.
    x = np.stack([np.log(T), np.log(p)], axis=-1).astype(np.float32)
    y = np.log(rho).astype(np.float32)

    in_mu, in_sd = x.mean(axis=0), x.std(axis=0)
    out_mu, out_sd = y.mean(), y.std()

    torch.manual_seed(EOS_SEED)
    net = EOSNet(hidden=EOS_HIDDEN, depth=EOS_DEPTH).to(device)
    net.in_mu.copy_(torch.from_numpy(in_mu))
    net.in_sd.copy_(torch.from_numpy(in_sd))
    net.out_mu.copy_(torch.tensor([out_mu]))
    net.out_sd.copy_(torch.tensor([out_sd]))

    T_t = torch.from_numpy(T.astype(np.float32)).to(device)
    p_t = torch.from_numpy(p.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y).to(device)
    n = T_t.shape[0]

    opt = torch.optim.Adam(net.parameters(), lr=EOS_LR)
    steps_per_epoch = max(1, n // EOS_BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EOS_EPOCHS * steps_per_epoch, eta_min=EOS_LR * 0.01)

    print(f"{EOS_HIDDEN} x {EOS_DEPTH} network, {EOS_EPOCHS} epochs, "
          f"{steps_per_epoch} steps per epoch")
    for epoch in range(EOS_EPOCHS):
        acc = 0.0
        for _ in range(steps_per_epoch):
            j = torch.randint(0, n, (min(EOS_BATCH, n),), device=device)
            # The loss is on log rho, which is what the network outputs.
            pred = torch.log(net(T_t[j], p_t[j]).clamp_min(1e-300))
            loss = torch.mean((pred - y_t[j]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acc += loss.item()
        if epoch % 25 == 0 or epoch == EOS_EPOCHS - 1:
            print(f"epoch {epoch:4d}  mse(log rho) = {acc / steps_per_epoch:.3e}")

    os.makedirs(os.path.dirname(EOS_PATH) or '.', exist_ok=True)
    torch.save({'hidden': EOS_HIDDEN, 'depth': EOS_DEPTH,
                'model_state': net.state_dict(),
                'cube': CUBE_PATH,
                'from_seen_only': EOS_FROM_SEEN_ONLY}, EOS_PATH)
    print("\nsaved", EOS_PATH)

    net.eval()
    print("\ncheck on the cube:")
    check_eos(net, S, device)


if __name__ == '__main__':
    main()
