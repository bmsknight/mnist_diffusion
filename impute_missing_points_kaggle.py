import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score

from src.ddpm import DDPM
from src.kaggle_daily_dataset import SimpleKaggleDataset
from src.unet_1d import ContextualUnet
from src.utils import remove_random_points, Evaluation
from src.visualization_utils import plot_multiple_samples

config = yaml.safe_load(open("config.yml"))

ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
FRAC = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SimpleKaggleDataset(path="data/load_history.csv",user_id=0,train=False,transform=None)


model = ContextualUnet(in_channels=1, n_feat=config["N_FEAT"], n_classes=config["N_CLASSES"])
ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=config["n_T"], device=device, drop_prob=0.1)
ddpm.to(device=device)

ddpm.load_state_dict(torch.load("model/kaggle_model_19.pth"))
ddpm.eval()

with torch.no_grad():
    for w_i, w in enumerate(ws_test):
        test_set = []
        preds = []
        masks = []
        for datapoint, context in dataset:
            week,day = context
            day = day-1
            datapoint_orig = datapoint.copy()
            datapoint_orig = torch.from_numpy(datapoint_orig)

            datapoint_n, mask = remove_random_points(datapoint, missing_frac=FRAC)
            datapoint_n = torch.from_numpy(datapoint_n)
            mask = torch.from_numpy(mask)

            x_gen, x_gen_store = ddpm.single_sample_inpaint(datapoint_n, mask, day, device, guide_w=w)
            temp_dp = datapoint_orig.reshape((24,))
            temp_dp = temp_dp*dataset.max_val
            temp_gen = x_gen.cpu().reshape((24,))
            temp_gen = temp_gen*dataset.max_val
            test_set.append(temp_dp)
            preds.append(temp_gen)
            masks.append(mask.reshape((24,)))

        preds = torch.concat(preds,dim=0)
        test_set = torch.concat(test_set,dim=0)
        masks = torch.concat(masks,dim=0)

        results = Evaluation(test_set=test_set, predictions=preds,mask=masks)
        print("Results for guidance {}".format(w))
        results.print()
