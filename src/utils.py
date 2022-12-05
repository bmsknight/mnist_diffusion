import numpy as np
import torch


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


def remove_random_points_tensor(batch_data, missing_frac):
    missing_batch = batch_data.squeeze()
    mask = torch.ones_like(missing_batch)
    length = missing_batch.shape[1]
    missing_frac = round(length * missing_frac)
    indices = torch.multinomial(mask, missing_frac)
    missing_batch = missing_batch.scatter(dim=1, index=indices, value=0)
    mask = mask.scatter(dim=1, index=indices, value=0)
    missing_batch = missing_batch.unsqueeze(dim=1)
    mask = mask.unsqueeze(dim=1)
    return missing_batch, mask
