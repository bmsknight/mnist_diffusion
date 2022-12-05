import numpy as np

from src.sin_dataset import SineDataset
from src.kaggle_daily_dataset import SimpleKaggleDataset
from src.visualization_utils import plot_datapoint, plot_multiple_samples

N_CLASSES = 5
N_SAMPLES = 5
dataset = SineDataset(window_size=24, transform=None, omega=np.pi / 24, n_classes=N_CLASSES)
# dataset = SimpleKaggleDataset(path="data/load_history.csv",user_id=0,train=True,transform=None)


def test_plot_datapoints():
    while True:
        ip = input("Enter index : ")
        i = int(ip)
        item, cls = dataset[i]
        fig = plot_datapoint(item, cls)
        fig.savefig("output/{}.png".format(i))


def test_plot_multiple_samples():
    data = np.zeros(shape=(N_SAMPLES, N_CLASSES, 1, 24))
    for idx in range(N_SAMPLES):
        for cls in range(N_CLASSES):
            item = idx * N_CLASSES + cls
            sample, _ = dataset[item]
            data[idx, cls, :, :] = sample
    fig = plot_multiple_samples(data, rows=N_SAMPLES, columns=N_CLASSES)
    fig.savefig("output/multiple.png")


if __name__ == "__main__":
    test_plot_datapoints()
    # test_plot_multiple_samples()
