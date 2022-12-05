import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score


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


class Evaluation:
    def __init__(self, predictions, test_set, mask):
        self.missing_values = (mask-1)
        self.rmse_full = mean_squared_error(test_set,predictions,squared=False)
        self.mape_full = mean_absolute_percentage_error(test_set, predictions)
        self.mae_full = mean_absolute_error(test_set, predictions)

        self.rmse_missing_only = mean_squared_error(test_set,predictions,squared=False,sample_weight=(mask - 1))
        self.mape_missing_only = mean_absolute_percentage_error(test_set, predictions, sample_weight=(mask - 1))
        self.mae_missing_only = mean_absolute_error(test_set, predictions, sample_weight=(mask - 1))

    def print(self):
        print("")
        print("Full Dataset Evaluation results")
        print("RMSE \tMAPE \tMAE")
        print("%.4f\t%.4f\t%.4f" % (self.rmse_full, self.mape_full, self.mae_full))
        print("")

        print("Missing Values only Evaluation results")
        print(f"Number of missing values : {self.missing_values.sum()}")
        print("RMSE \tMAPE \tMAE")
        print("%.4f\t%.4f\t%.4f" % (self.rmse_missing_only, self.mape_missing_only, self.mae_missing_only))
