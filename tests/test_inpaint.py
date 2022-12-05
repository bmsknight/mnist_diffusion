import matplotlib.pyplot as plt
import torch
import yaml

from src.ddpm import DDPM
from src.sin_dataset import SineDataset
from src.unet_1d import ContextualUnet
from src.utils import remove_random_points
from src.visualization_utils import plot_multiple_samples

config = yaml.safe_load(open("config.yml"))

ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
cls = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SineDataset(window_size=24, transform=None, length=100, omega=0.15, amplitude=0.96, n_classes=5)

datapoint,cls_label = dataset[8]
assert cls== cls_label
datapoint_n, mask = remove_random_points(datapoint, missing_frac=0.5)
datapoint_n = torch.from_numpy(datapoint_n)
mask = torch.from_numpy(mask)

model = ContextualUnet(in_channels=1, n_feat=config["N_FEAT"], n_classes=config["N_CLASSES"])
ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=config["n_T"], device=device, drop_prob=0.1)
ddpm.to(device=device)

ddpm.load_state_dict(torch.load("model/model_19.pth"))
ddpm.eval()

with torch.no_grad():
    for w_i, w in enumerate(ws_test):
        x_gen, x_gen_store = ddpm.single_sample_inpaint(datapoint_n, mask, cls, device, guide_w=w)
        x_gen = x_gen.reshape(1, 1, 1, 24).cpu()
        x_real = datapoint_n.reshape(1, 1, 1, 24).cpu()
        x_all = torch.cat([x_gen, x_real])
        x_all = x_all.repeat([1, 2, 1, 1])
        image = plot_multiple_samples(x_all, rows=2, columns=2)

        path = "output/" + f"inpaint_class{cls}_w{w}.png"
        image.savefig(path)
        plt.close(image)
        print('saved image at ' + path)
