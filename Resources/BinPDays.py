import numpy as np


def bin_pdays(x):
    return np.where(x < 30, 0, 1)
