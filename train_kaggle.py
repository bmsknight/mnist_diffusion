import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.ddpm import DDPM
from src.kaggle_daily_dataset import SimpleKaggleDataset
from src.unet_1d import ContextualUnet
from src.visualization_utils import plot_multiple_samples


def main(config):
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training will happen on : ", device)
    n_classes = torch.Tensor(config["N_CLASSES"]).to(device)
    model = ContextualUnet(in_channels=1, n_feat=config["N_FEAT"], n_classes=n_classes)
    ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=config["n_T"], device=device, drop_prob=0.1)
    ddpm.to(device=device)

    # transform = transforms.Compose([transforms.ToTensor()])

    dataset = SimpleKaggleDataset(path="data/load_history.csv",user_id=config["USER_ID"],train=True,transform=None)
    train_loader = DataLoader(dataset, batch_size=config["BATCH_SIZE"])
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
            context = torch.stack(context,dim=-1)
            context = (context-1).to(device)

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
            n_sample = 4 * 7
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 24), device, guide_w=w)

                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(7):
                    for j in range(int(n_sample / 7)):
                        try:
                            idx = torch.squeeze((context[1,:] == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k + (j * 7)] = x[idx]

                x_real = x_real.reshape(4, 7, 1, 24)
                x_gen = x_gen.reshape(4, 7, 1, 24)
                print(x_gen.shape)
                print(x_real.shape)
                x_all = torch.cat([x_gen, x_real]).cpu()
                image = plot_multiple_samples(x_all, rows=8, columns=7)

                path = "output/" + f"image_ep{epoch}_w{w}_user{config['USER_ID']}.png"
                image.savefig(path)
                plt.close(image)
                print('saved image at ' + path)

        if epoch == int(config["EPOCHS"] - 1):
            torch.save(ddpm.state_dict(), "model/" + f"kaggle_model_{epoch}_user{config['USER_ID']}.pth")
            print('saved model')


if __name__ == "__main__":
    params = yaml.safe_load(open("config.yml"))
    main(params)
