import pandas as pd
from torch.utils.data import Dataset
import numpy as np

import src.constants as const


class UtiliSmartDailyDataset(Dataset):

    def __init__(self, path, user_id, train, transform, *args, **kwargs):
        dataset = pd.read_csv(path)
        meters = dataset[const.METER_ID].unique()
        try:
            meter = meters[user_id]
        except IndexError:
            raise ValueError("Invalid user_id. user_id should be less than ", len(meters))
        dataset = dataset[
                      (dataset[const.METER_ID] == meter) & (dataset[const.READING_TYPE] == const.INTERVAL_READING)].loc[
                  :, [const.READING_TIMESTAMP, const.READING_VALUE, const.READING_STATE]]
        dataset = dataset[(dataset[const.READING_STATE] == const.ACTUAL_READING)]
        dataset[const.READING_TIMESTAMP] = pd.to_datetime(dataset[const.READING_TIMESTAMP])
        dataset[const.K_DAY] = pd.to_datetime(dataset[const.READING_TIMESTAMP].dt.date)
        dataset[const.K_HOUR] = dataset[const.READING_TIMESTAMP].dt.hour
        dataset = dataset.pivot(index=const.K_DAY, columns=const.K_HOUR, values=const.READING_VALUE)

        dataset.sort_index(inplace=True)
        dataset.dropna(inplace=True)
        self.max_val = max(dataset.max())

        train_split_point = int(dataset.shape[0] * (1 - const.TEST_SPLIT_FRAC))
        if train:
            self.dataset = dataset.iloc[:train_split_point, :]
        else:
            self.dataset = dataset.iloc[train_split_point:, :]

        self.length = self.dataset.shape[0]
        self.target_dataset = self.dataset.copy(deep=True)
        self.window_size = 24
        self.transform = transform

        date = pd.to_datetime(self.dataset.index)
        self.week = date.isocalendar().week
        self.day = date.isocalendar().day

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        input_window = self.dataset.iloc[item, :]
        input_window = input_window.to_numpy(dtype='float32') / self.max_val
        input_window = input_window.reshape((1, self.window_size))
        week = int(self.week.iloc[item])
        day = int(self.day.iloc[item])
        return input_window, (week, day)

