import numpy as np
import torch
import torch.nn as nn

from config import *


class EOSNet(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("in_mu", torch.zeros(2))
        self.register_buffer("in_sd", torch.ones(2))
        self.register_buffer("out_mu", torch.zeros(1))
        self.register_buffer("out_sd", torch.ones(1))

    def forward(self, T_phys, p_phys):
        lt = torch.log(T_phys.clamp_min(1e-30))
        lp = torch.log(p_phys.clamp_min(1e-30))
        x = torch.stack([lt, lp], dim=-1)
        x = (x - self.in_mu) / self.in_sd
        y = self.net(x).squeeze(-1)
        y = y * self.out_sd + self.out_mu
        return torch.exp(y)


def load_eos(device, path=None):
    """Load a trained surrogate, frozen and in eval mode."""
    path = EOS_PATH if path is None else path
    ck = torch.load(path, map_location=device, weights_only=False)
    net = EOSNet(hidden=ck['hidden'], depth=ck['depth']).to(device)
    net.load_state_dict(ck['model_state'])
    net.eval()
    for prm in net.parameters():
        prm.requires_grad = False
    return net


def predict_rho(net, T, p, device, chunk=500_000):
    """rho on a whole field, as numpy, in chunks so memory stays flat."""
    Tt = torch.from_numpy(np.asarray(T).ravel().astype(np.float32)).to(device)
    pt = torch.from_numpy(np.asarray(p).ravel().astype(np.float32)).to(device)
    out = []
    with torch.no_grad():
        for s in range(0, Tt.shape[0], chunk):
            out.append(net(Tt[s:s + chunk], pt[s:s + chunk]).cpu().numpy())
    return np.concatenate(out).reshape(np.asarray(T).shape)


def check_eos(net, S, device):
    """How well the surrogate reproduces rho, and its gradient, on this cube.

    The gradient matters as much as the value: the physics uses grad rho, so a
    surrogate that gets rho right but its slope wrong would feed a wrong force
    into the loss.
    """
    from operators import f8, grad, interior

    rho_eos = predict_rho(net, S['T'], S['p'], device)
    rho_true = S['rho']
    Nz = rho_true.shape[2]
    TR = interior(Nz)

    rel_std = np.sqrt(np.mean((rho_eos - rho_true) ** 2)) / rho_true.std() * 100
    val_err = np.median(np.abs((rho_eos - rho_true) / (np.abs(rho_true) + 1e-30)))

    gt = grad(rho_true)
    ge = grad(rho_eos)
    gmag_t = np.sqrt(sum(g ** 2 for g in gt))
    gmag_e = np.sqrt(sum(g ** 2 for g in ge))
    grad_err = np.median(np.abs((gmag_e - gmag_t) / (gmag_t + 1e-30)))

    print(f"rho: rel_std = {rel_std:.2f}%   median relative error = {val_err:.2%}")
    print(f"|grad rho|: median relative error = {grad_err:.2%}")
    for name, t, e in zip('xyz', gt, ge):
        r = np.corrcoef(t[TR].ravel(), e[TR].ravel())[0, 1]
        print(f"  correlation of d(rho)/d{name} with the truth: {r:.4f}")
    return rel_std
