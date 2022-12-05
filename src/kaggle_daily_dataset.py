import pandas as pd
from torch.utils.data import Dataset

K_HOUR_LIST = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11", "h12",
               "h13", "h14", "h15", "h16", "h17", "h18", "h19", "h20", "h21", "h22", "h23", "h24"]


class SimpleKaggleDataset(Dataset):
    def __init__(self, path, user_id, train, transform):
        dataset = pd.read_csv(path)
        meters = dataset["zone_id"].unique()
        try:
            meter = meters[user_id]
        except IndexError:
            raise ValueError("Invalid user_id. user_id should be less than ", len(meters))

        dataset = dataset[dataset["zone_id"] == meter]
        dataset["READTS"] = pd.to_datetime(dataset[["year","month","day"]])
        dataset = dataset.drop(columns=["id", "IMPUTED", "zone_id", "year","month","day"])
        dataset.replace(',', '', regex=True, inplace=True)
        dataset[K_HOUR_LIST] = dataset[K_HOUR_LIST].astype('float')
        dataset.sort_values(["READTS"], inplace=True)
        dataset.dropna(inplace=True)
        train_split_point = int(dataset.shape[0] * (1 - 0.2))
        if train:
            self.dataset = dataset.iloc[:train_split_point, :]
        else:
            self.dataset = dataset.iloc[train_split_point:, :]
        self.dataset.set_index("READTS", inplace=True)

        date = self.dataset.index
        self.week = date.isocalendar().week
        self.day = date.isocalendar().day

        self.length = self.dataset.shape[0]
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        input_window = self.dataset.iloc[item,:]
        input_window = input_window.to_numpy(dtype='float32')
        input_window = input_window.reshape((1, 24))
        week = self.week.iloc[item]
        day = self.day.iloc[item]
        return input_window, (week, day)
