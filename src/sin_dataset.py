import numpy as np
from torch.utils.data import Dataset


class SineDataset(Dataset):
    def __init__(self, window_size, transform, length=100000, omega=2 * np.pi / 360, amplitude=1, n_classes=7):
        self.omega = omega
        self.window_size = window_size
        self.transform = transform
        index_array = np.array(range(length), dtype='float32')
        sin_array = np.zeros(shape=(n_classes, length), dtype='float32')
        for i in range(n_classes):
            sin_array[i, :] = amplitude * np.sin(omega * (i + 1) * index_array)
        self.dataset = sin_array
        self.length = length
        self.n_classes = n_classes

    def __len__(self):
        return (self.length // self.window_size) * self.n_classes

    def __getitem__(self, item):
        class_label = item % self.n_classes
        idx = item // self.n_classes * self.window_size
        input_window = self.dataset[class_label, idx:idx + self.window_size]
        input_window = input_window.reshape((1,-1))
        if self.transform:
            input_window = self.transform(input_window)
        return input_window, class_label
