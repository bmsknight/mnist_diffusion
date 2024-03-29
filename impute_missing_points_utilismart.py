import torch
import yaml
from torch.utils.data import DataLoader
from src.ddpm import DDPM
from src.utilismart_daily_dataset import UtiliSmartDailyDataset
from src.unet_1d import ContextualUnet
from src.utils import Evaluation, remove_random_points_tensor, remove_continuous_points_tensor

config = yaml.safe_load(open("config.yml"))

ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
FRAC = config["MISSING_FRAC"]
HOURS = config["MISSING_HOURS"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = UtiliSmartDailyDataset(path="data/utilismart_dataset2.csv", user_id=config["USER_ID"], train=False,
                                     transform=None)
train_loader = DataLoader(dataset, batch_size=config["BATCH_SIZE"])

n_classes = torch.Tensor(config["N_CLASSES"]).to(device)
model = ContextualUnet(in_channels=1, n_feat=config["N_FEAT"], n_classes=n_classes)
ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=config["n_T"], device=device, drop_prob=0.1)
ddpm.to(device=device)

ddpm.load_state_dict(torch.load(f"model/util_model_19_user{config['USER_ID']}.pth"))
ddpm.eval()

with torch.no_grad():
    for w_i, w in enumerate(ws_test):
        test_set = []
        preds = []
        masks = []
        for datapoint, context in train_loader:
            context = torch.stack(context, dim=-1)
            context = (context - 1).to(device)
            datapoint_orig = datapoint.clone().detach()

            if config["MISSING_TYPE"] == "RANDOM":
                datapoint_n, mask = remove_random_points_tensor(datapoint, missing_frac=FRAC)
            elif config["MISSING_TYPE"] == "CONTINUOUS":
                datapoint_n, mask = remove_continuous_points_tensor(datapoint, hours_to_remove=HOURS)
            else:
                raise ValueError("Unknown MISSING_TYPE")

            x_gen, x_gen_store = ddpm.multiple_sample_inpaint(datapoint_n,mask,context,device,guide_w=w)
            temp_dp = datapoint_orig.squeeze()
            temp_dp = temp_dp * dataset.max_val
            temp_gen = x_gen.cpu().squeeze()
            temp_gen = temp_gen * dataset.max_val
            test_set.append(temp_dp)
            preds.append(temp_gen)
            masks.append(mask.squeeze())
            if len(preds)==2:
                break
        preds = torch.concat(preds, dim=0).reshape(-1)
        test_set = torch.concat(test_set, dim=0).reshape(-1)
        masks = torch.concat(masks, dim=0).reshape(-1)

        results = Evaluation(test_set=test_set, predictions=preds, mask=masks)
        print("Results for guidance {}".format(w))
        results.print()
