import numpy as np

from config import *


def f8(a):
    """Same array in float64."""
    return np.asarray(a, dtype=np.float64)


def ddx(f):
    return np.gradient(f8(f), DX, axis=0)


def ddy(f):
    return np.gradient(f8(f), DX, axis=1)


def ddz(f):
    return np.gradient(f8(f), DZ, axis=2)


def grad(f):
    return ddx(f), ddy(f), ddz(f)


def div(Fx, Fy, Fz):
    return ddx(Fx) + ddy(Fy) + ddz(Fz)


def curl(Fx, Fy, Fz):
    return (ddy(Fz) - ddz(Fy),
            ddz(Fx) - ddx(Fz),
            ddx(Fy) - ddy(Fx))


def interior(Nz):
    """Slice that drops PAD cells on each horizontal edge and in z."""
    return (slice(PAD, -PAD), slice(PAD, -PAD), slice(PAD, Nz - PAD))


def interior_xy():
    """Slice that drops PAD cells horizontally and keeps every z layer."""
    return (slice(PAD, -PAD), slice(PAD, -PAD))
