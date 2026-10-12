"""
All parameters are here.

This is the only file you edit between runs. Every other file starts with

from config import *

so a name written here is visible everywhere. Everything in this file is
UPPERCASE, anything lowercase in the other files is a local variable.
"""

# ---------------- paths ----------------
# The cube is a single.npy file of shape (9, nx, ny, nz) in cgs units,
# with the channels in the order of NAMES below.

CUBE_PATH = "data/cutout_patch48.npy"   # 100 x 100 x 48 cutout (the default)
# CUBE_PATH = "data/cutout_patch15.npy" # 15 layers from 10-25 layers of the 100 x 100 x 48 cutout

EOS_PATH = "data/eos_net.pt"            # written by train_eos.py
OUT_DIR  = "runs"                       # one sub-folder per run
RUN_NAME = "run1_divb"                  # name of this run's sub-folder

# ---------------- data and grid ----------------
NAMES = ['T', 'p', 'rho', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz']

OD  = 0          # first x and y index that is read
SUB = 100        # how many points in x and y are read, counting from OD
Z0  = 0          # first z layer that is read (included)
Z1  = None       # last z layer (excluded); None = to the top of the file

DX = 32e5        # cm, horizontal grid step of the simulation
DZ = 32e5        # cm, vertical grid step

G_SUN = 2.74e4   # cm/s^2, surface gravity

PAD  = 4         # cells cut off each horizontal edge before any statistic
SL_K = 8         # z layer shown in the evaluation figures

# ---------------- which points enter the data loss ----------------
# The physics is imposed on the whole grid, this only limits the data loss.
# Z_KEEP are the supervised ("seen") layers. Every other layer is hidden:
# the network never gets data there

Z_KEEP = [0, 6, 8, 10, 12, 18, 25, 32, 39, 47]   # for the 48-layer cutout
# Z_KEEP = [0, 2, 4, 6, 8, 10, 12, 14]           # for the 15-layer cutout

STRIDE_X  = 1     # keep every STRIDE_X-th point in x inside a seen layer
STRIDE_Y  = 1     # the same in y
FRAC      = 1.0   # keep this fraction of the remaining points, at random
SEED_MASK = 0     # seed of that random thinning

# ---------------- normalization ----------------
# Every channel in PERZ_NAMES is normalized per z layer: (f - mu(z)) / sd(z).
# rho is not here because it is not an output of the network, it comes from
# the EOS surrogate.
PERZ_NAMES = ['T', 'p', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz']

# mu(z) and sd(z) are computed from the SEEN layers only and interpolated
# across the hidden ones. With PERZ_FROM_SEEN = False the network would be
# handed the exact mean and amplitude of the hidden layers for free.
PERZ_FROM_SEEN = True

# Channels whose mu(z) and sd(z) are interpolated in the logarithm. Pressure
# falls off exponentially with height, so a straight line between two seen
# layers cuts a chord underneath the real profile.
LOG_PERZ = ['p']

SD_FLOOR = 1e-6   # lower bound on sd(z), as a fraction of max(sd)

# ---------------- model ----------------
NUM_FREQS = 192      # Fourier features per band
FRAC_LOW  = 0.15     # fraction of them in the low-frequency band
DEPTH     = 8        # hidden layers
WIDTH     = 128      # neurons per hidden layer. On the committed 48-layer
                     # cutout this is 216k parameters against 3.84M values in
                     # the cube, so the network stores the cube about 18x
                     # compressed -- the same range as in the thesis, which
                     # used WIDTH = 384 on a 300 x 300 sub-domain.

SEED      = 42       # seed of the weights and of the Fourier matrix

# The two bands are fitted to the measured spectrum of the training data.
# Fields in HIGH_BAND_NAMES set the wide band, the rest set the narrow one.
HIGH_BAND_NAMES = ['Bx', 'By', 'Bz']
SIGMA_AGG       = 'median'   # 'max' | 'median' | 'mean', over the fields of a band
NYQ_SAFETY      = 0.8        # cap on the frequencies: NYQ_SAFETY * Nyquist
SIGMA_XY_LOW    = 1.0        # manual factor on sigma_x,y of the narrow band
SIGMA_XY_HIGH   = 1.0        # the same for the wide band
SIGMA_Z_FACTOR  = 1.0        # manual factor on sigma_z of both bands

# ---------------- physics ----------------
# All three equations are stationary: they hold for a single snapshot and
# need no time derivative.
#   divB       div B = 0
#   stat_cont  div (rho v) = 0
#   momentum   div (rho v v) + grad p + rho g e_z - (1/4pi)(curl B) x B = 0

# The weight of each term is its value in this dict; LAMBDA scales all of them.
PHYS_TERMS = {'divB': 1.0}

LAMBDA      = 0.3      # global weight of the physics loss
RAMP_EPOCHS = 150      # the weight grows linearly from 0 over this many epochs
PHYS_BATCH  = 8000     # collocation points per step, per height band

# Height band in which each term is imposed, in normalized z (0 = bottom,
# 1 = top). Collocation points are drawn from the band only, so a narrow band
# does not need a larger PHYS_BATCH.
Z_BAND = {
    'divB':      (0.0, 1.0),
    'stat_cont': (0.0, 1.0),
    'momentum':  (0.0, 1.0),
}

# ---------------- training ----------------
EPOCHS             = 400
BATCH_SIZE         = 50_000    # supervised points per step
LR                 = 1e-3
TOTAL_STEPS_TARGET = 150_000   # steps in the whole run, sets steps per epoch
SEED_RUN           = 1234      # seed of the batch sampling
GRAD_CLIP          = 1.0       # cap on the gradient norm
LOG_EVERY          = 20        # print a line every this many epochs
CKPT_EVERY         = 25        # overwrite last.pt every this many epochs

# ---------------- EOS surrogate (train_eos.py) ----------------
EOS_HIDDEN = 64
EOS_DEPTH  = 4
EOS_EPOCHS = 1500
EOS_BATCH  = 100_000
EOS_LR     = 1e-3
EOS_SEED   = 0

# Set this to True to train it on the seen layers only, 
# if you want the hidden layers untouched by
# anything at all.

EOS_FROM_SEEN_ONLY = False
