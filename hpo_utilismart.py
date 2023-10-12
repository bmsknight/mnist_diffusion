import argparse
import time

import numpy as np
import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from src.ddpm import DDPM
from src.unet_1d import ContextualUnet
from src.utilismart_daily_dataset import UtiliSmartDailyDataset
from src.utils import Evaluation, remove_random_points_tensor, remove_continuous_points_tensor

# Fix for optuna accessing MySQLdb module which is not present
# alternative packages are not present in compute canada
# Hence need to use pymysql as below
pymysql.install_as_MySQLdb()

def main(config, run_id):
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
    best_rmse = float("inf")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training will happen on : ", device)
    n_classes = torch.Tensor(config["N_CLASSES"]).to(device)
    model = ContextualUnet(in_channels=1, n_feat=config["N_FEAT"], n_classes=n_classes)
    ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=config["n_T"], device=device, drop_prob=0.1)
    ddpm.to(device=device)

    # transform = transforms.Compose([transforms.ToTensor()])

    dataset = UtiliSmartDailyDataset(path="data/utilismart_dataset2.csv", user_id=config["USER_ID"], train=True,
                                     transform=None)
    train_loader = DataLoader(dataset, batch_size=config["BATCH_SIZE"])

    test_dataset = UtiliSmartDailyDataset(path="data/utilismart_dataset2.csv", user_id=config["USER_ID"], train=False,
                                          transform=None)
    test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"])

    optim = torch.optim.Adam(ddpm.parameters(), lr=config["LEARNING_RATE"])
    print("Start")
    for epoch in range(config["EPOCHS"]):
        ddpm.train()

        optim.param_groups[0]['lr'] = config["LEARNING_RATE"] * (1 - epoch / config["EPOCHS"])

        loss_ema = None

        for x, context in train_loader:
            optim.zero_grad()
            x = x.to(device)
            # week, day = context
            context = torch.stack(context, dim=-1)
            context = (context - 1).to(device)

            loss = ddpm(x, context)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            optim.step()
            print(f"Epoch: {epoch}, \t Training loss: {loss_ema}")

        ddpm.eval()
        with torch.no_grad():
            FRAC = config["MISSING_FRAC"]
            HOURS = config["MISSING_HOURS"]
            for w_i, w in enumerate(ws_test):
                test_set = []
                preds = []
                masks = []
                for datapoint, context in test_loader:
                    context = torch.stack(context, dim=-1)
                    context = (context - 1).to(device)
                    datapoint_orig = datapoint.clone().detach()

                    if config["MISSING_TYPE"] == "RANDOM":
                        datapoint_n, mask = remove_random_points_tensor(datapoint, missing_frac=FRAC)
                    elif config["MISSING_TYPE"] == "CONTINUOUS":
                        datapoint_n, mask = remove_continuous_points_tensor(datapoint, hours_to_remove=HOURS)
                    else:
                        raise ValueError("Unknown MISSING_TYPE")

                    x_gen, x_gen_store = ddpm.multiple_sample_inpaint(datapoint_n, mask, context, device, guide_w=w)
                    temp_dp = datapoint_orig.squeeze()
                    temp_dp = temp_dp * dataset.max_val
                    temp_gen = x_gen.cpu().squeeze()
                    temp_gen = temp_gen * dataset.max_val
                    test_set.append(temp_dp)
                    preds.append(temp_gen)
                    masks.append(mask.squeeze())

                preds = torch.concat(preds, dim=0).reshape(-1)
                test_set = torch.concat(test_set, dim=0).reshape(-1)
                masks = torch.concat(masks, dim=0).reshape(-1)

                results = Evaluation(test_set=test_set, predictions=preds, mask=masks)
                print("Results for guidance {}".format(w))
                results.print()

                if results.rmse_missing_only < best_rmse:
                    torch.save(ddpm.state_dict(),
                               "model/" + f"util_model_user{config['USER_ID']}_run_{run_id}_epoch_{epoch}.pth")
                    print(f'saved model of Epoch {epoch}, weight {w}')
                    best_rmse = results.rmse_missing_only
                    best_epoch = epoch
                    best_weight = w
                    best_results = results

    print("Best Results for guidance {} in epoch{}".format(best_weight, best_epoch))
    best_results.print()
    return best_rmse


def objective(trial):
    params = yaml.safe_load(open("config.yml"))
    params["LEARNING_RATE"] = trial.suggest_loguniform("LEARNING_RATE", 1e-6, 1e-1)
    params["BATCH_SIZE"] = trial.suggest_int("BATCH_SIZE", 32, 256)
    # params["on_epoch_callbacks"] = [OptunaPruningCallback(trial, metric_name="val_loss")]

    print(f"Initiating Run {trial.number} with params : {trial.params}")

    rmse = main(params, str(trial.number))
    return rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optuna-db", type=str, help="Path to the Optuna Database file",
                        )
    parser.add_argument("-n", "--optuna-study-name", type=str, help="Name of the optuna study",
                        )
    args = parser.parse_args()
    wait_time = np.random.randint(0, 10) * 3
    print(f"Waiting for {wait_time} seconds before starting")
    time.sleep(wait_time)
    study = optuna.create_study(direction="minimize",
                                study_name=args.optuna_study_name,
                                storage=args.optuna_db,
                                load_if_exists=True,
                                )
    study.optimize(objective, n_trials=1)
