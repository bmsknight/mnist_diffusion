import numpy as np


def remove_random_points(datapoint, missing_frac):
    datapoint = datapoint.squeeze()
    mask = np.ones_like(datapoint)
    length = datapoint.shape[0]
    indices = np.arange(length)
    missing_frac = round(length * missing_frac)
    nullified_indices = np.random.choice(indices, replace=False, size=missing_frac)

    datapoint[nullified_indices] = 0
    mask[nullified_indices] = 0
    datapoint = np.expand_dims(datapoint, 0)
    mask = np.expand_dims(mask, 0)
    return datapoint, mask
