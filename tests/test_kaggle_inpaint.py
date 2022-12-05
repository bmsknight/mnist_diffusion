import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score

from src.ddpm import DDPM
from src.kaggle_daily_dataset import SimpleKaggleDataset
from src.unet_1d import ContextualUnet
from src.utils import remove_random_points
from src.visualization_utils import plot_multiple_samples

config = yaml.safe_load(open("config.yml"))

ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SimpleKaggleDataset(path="data/load_history.csv",user_id=0,train=False,transform=None)

datapoint,(week,day) = dataset[72]
day = day-1
datapoint_orig = datapoint.copy()

datapoint_n, mask = remove_random_points(datapoint, missing_frac=0.3)
datapoint_n = torch.from_numpy(datapoint_n)
mask = torch.from_numpy(mask)

model = ContextualUnet(in_channels=1, n_feat=config["N_FEAT"], n_classes=config["N_CLASSES"])
ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=config["n_T"], device=device, drop_prob=0.1)
ddpm.to(device=device)

ddpm.load_state_dict(torch.load("model/kaggle_model_19.pth"))
ddpm.eval()

with torch.no_grad():
    for w_i, w in enumerate(ws_test):
        x_gen, x_gen_store = ddpm.single_sample_inpaint(datapoint_n, mask, day, device, guide_w=w)
        x_gen = x_gen.reshape(1, 1, 1, 24).cpu()
        x_real = datapoint_n.reshape(1, 1, 1, 24).cpu()
        x_all = torch.cat([x_gen, x_real])
        x_all = x_all.repeat([1, 2, 1, 1])
        image = plot_multiple_samples(x_all, rows=2, columns=2)

        path = "output/" + f"inpaint_class{day}_w{w}.png"
        image.savefig(path)
        plt.close(image)
        print('saved image at ' + path)
        temp_dp = datapoint_orig*dataset.max_val
        temp_gen = x_gen*dataset.max_val
        rmse_full = mean_squared_error(temp_dp.reshape((24,)),
                                            temp_gen.reshape((24,)),
                                            squared=False)
        rmse_missing_only = mean_squared_error(temp_dp.reshape((24,)),
                                            temp_gen.reshape((24,)),
                                            squared=False, sample_weight=(mask-1).reshape((24,)))

        mape_full = mean_absolute_percentage_error(temp_dp.reshape((24,)),
                                       temp_gen.reshape((24,)))
        mape_missing_only = mean_absolute_percentage_error(temp_dp.reshape((24,)),
                                               temp_gen.reshape((24,)),
                                               sample_weight=(mask - 1).reshape((24,)))
        print(rmse_full, " ", mape_full)
        print(rmse_missing_only, " ", mape_missing_only)
