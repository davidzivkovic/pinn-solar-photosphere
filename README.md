# pinn-solar-photosphere

This is the code from my Master's thesis, *Modelling the thermodynamic
structure of the solar photosphere with machine-learning methods*
(University of Belgrade, Faculty of Mathematics, 2026; supervisor Dr. Ivan
Milić). It's the thesis pipeline, cleaned up and rewritten as plain scripts
that run on a committed data cutout, if you want to change what it does, 
you edit `config.py` and read the script, there isn't any API.

## What it's doing

A simulation of the solar photosphere gives the full state of the plasma
everywhere in a 3D box: temperature, pressure, density, velocity, magnetic
field. An observation only gives a couple of horizontal layers, the rest are
unknown. So the task is: train a network on a few layers of a simulated
cube, and see whether it can reconstruct the layers it never saw. The layers
it's trained on are `Z_KEEP` in `config.py` everything else is hidden from
training, and the error on those hidden layers is the actual result.

The network takes a position `(x, y, z)` and outputs the fields there:
T, p, vx, vy, vz, Bx, By, Bz. Density isn't one of the outputs because it comes from
a small separate network fitted once to stand in for the equation of state,
`rho = EOS(T, p)` (`eos.py`, `train_eos.py`).

![Network architecture: x, y, z go through a sin/cos Fourier encoder, then
8 hidden layers with a skip connection back to the encoder output at the
middle layer, ending in T, p, v, B](network.png)

The encoder is `GaussianFourierFeatures` in `model.py`, the skip connection
is `NeuralField.skip_at`. 

On top of the data, the network can be also pushed to satisfy three physics
equations everywhere in the box, including the hidden layers. That is the most important part, in hidden
layers that is the only thing that will give some kind of a constrain.

```
divB        div B = 0
stat_cont   div(rho v) = 0                                        (mass continuity)
momentum    div(rho v v) + grad p + rho g z_hat - (curl B) x B / 4pi = 0
```

All three are stationary, so no time derivative, since a single snapshot
doesn't have one. The derivatives in the loss are exact derivatives of the
network (autograd), `operators.py`'s finite differences are only used to measure the data itself, in `scales.py`.

Two things that matter for the result to mean anything, both in `data.py`:
the per-height normalization mu(z)/sigma(z) is computed from the seen layers
only and interpolated across the hidden ones (`PERZ_FROM_SEEN`). Otherwise
the network would get the true mean and amplitude of the hidden layers for
free, and the reconstruction error wouldn't mean much. And each residual is
divided by a scale measured on the data (`scales.py`).

## Files

```
config.py        every parameter, in one place 
operators.py     plain derivatives (numpy)
data.py          load the cube, build the supervision mask, normalize
scales.py        how well the data satisfy each equation, and the scale
                 each residual is divided by
eos.py           the (T, p) -> rho network
train_eos.py     fits it to the cube
model.py         Fourier features + MLP
physics.py       the three residuals, in torch
train.py         the training loop
evaluate.py      errors per field/layer, comparison figures
data/            the two cutouts + the trained EOS network
```

## Running it

```bash
pip install -r requirements.txt
python train_eos.py     # once, fits data/eos_net.pt
python train.py         # trains the main network
python evaluate.py      # NRMSE per field/layer + figures, for that run
```

The four physics setups from the thesis are just `PHYS_TERMS` in
`config.py`: `{}` (no physics), `{'divB': 1.0}`, `{'stat_cont': 0.3}`,
`{'divB': 0.7, 'stat_cont': 0.3}`, `{'momentum': 0.1}`.

A full run (400 epochs, 150,000 steps total) needs a GPU — on a CPU each step
is a couple of seconds, so the whole thing is days, not minutes. Lower
`TOTAL_STEPS_TARGET` for a quick local check that it runs.

## Data

`data/cutout_patch48.npy` is 100 x 100 x 48, channels T, p,
rho, vx, vy, vz, Bx, By, Bz in cgs units, 32 km grid spacing. There's also
`cutout_patch15.npy`, with layers 10-25 of the same patch, for a quicker
test. Both are from a MURaM simulation. The thesis itself used a bigger
300x300 sub-domain of it, which isn't included here.

## Does the data actually satisfy these equations?

Measured on `cutout_patch48.npy`, before any training.

- divB: 4.9% residual, a real constraint
- stat_cont: 38% overall, but 13% on the bottom 15 layers
- momentum: ~110% per component, i.e. not satisfied at all by a single
  snapshot — included anyway (the thesis tested it)

The EOS surrogate reaches 0.44% median error on rho 

## Limitations

One snapshot, so no induction or energy equation, and the three equations
used are stationary approximations the data only satisfy to varying degrees
(see above). These cutouts are smaller than the thesis's own
sub-domain `WIDTH = 128` here gives about the same compression ratio
(~18x) that the thesis had at `WIDTH = 384` (~25x) on the bigger domain, but it's
still a smaller problem.

## Citation

D. Živković, "Modelling the thermodynamic structure of the solar photosphere
with machine-learning methods", Master's thesis, University of Belgrade,
Faculty of Mathematics, 2026.

## License

MIT, see [LICENSE](LICENSE).
