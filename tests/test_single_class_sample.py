import matplotlib.pyplot as plt
import torch
import yaml

from src.ddpm import DDPM
from src.unet_1d import ContextualUnet
from src.visualization_utils import plot_multiple_samples

config = yaml.safe_load(open("config.yml"))

ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
n_sample = 4
cls = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ContextualUnet(in_channels=1, n_feat=config["N_FEAT"], n_classes=config["N_CLASSES"])
ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=config["n_T"], device=device, drop_prob=0.1)
ddpm.to(device=device)

ddpm.load_state_dict(torch.load("model/model_19.pth"))
ddpm.eval()

with torch.no_grad():
    for w_i, w in enumerate(ws_test):
        x_gen, x_gen_store = ddpm.single_class_sample(n_sample, cls, (1, 24), device, guide_w=w)
        x_gen = x_gen.reshape(2, 2, 1, 24).cpu()
        image = plot_multiple_samples(x_gen, rows=2, columns=2)

        path = "output/" + f"image_class{cls}_w{w}.png"
        image.savefig(path)
        plt.close(image)
        print('saved image at ' + path)
