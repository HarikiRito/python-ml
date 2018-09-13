import numpy as np


def flip(m):
    if type(m) is np.ndarray:
        r = flip_ndarray(m)
    else:
        r = flip_ndarray(np.asarray(m))
    return r


def flip_ndarray(m):
    return np.rot90(m)[::-1]
