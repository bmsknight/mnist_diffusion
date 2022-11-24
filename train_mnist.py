import matplotlib.pyplot as plt
import torch
import yaml
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

from src.ddpm import DDPM
from src.unet import ContextualUnet


def main(config):
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
    device = torch.device('cpu')
    print("Training will happen on : ", device)

    model = ContextualUnet(in_channels=1, n_feat=config["N_FEAT"], n_classes=config["N_CLASSES"])
    ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=config["n_T"], device=device, drop_prob=0.1)
    ddpm.to(device=device)

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = MNIST("data", train=True, download=True, transform=transform)
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
            context = context.to(device)

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
            n_sample = 4 * config["N_CLASSES"]
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(config["N_CLASSES"]):
                    for j in range(int(n_sample / config["N_CLASSES"])):
                        try:
                            idx = torch.squeeze((context == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k + (j * config["N_CLASSES"])] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all * -1 + 1, nrow=10)
                path = "output/" + f"image_ep{epoch}_w{w}.png"
                save_image(grid, path)
                print('saved image at ' + path)

                if epoch % 5 == 0 or epoch == int(config["EPOCHS"] - 1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample / config["N_CLASSES"]), ncols=config["N_CLASSES"],
                                            sharex=True, sharey=True, figsize=(8, 3))

                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample / config["N_CLASSES"])):
                            for col in range(config["N_CLASSES"]):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                plots.append(
                                    axs[row, col].imshow(-x_gen_store[i, (row * config["N_CLASSES"]) + col, 0],
                                                         cmap='gray',
                                                         vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots

                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store], interval=200, blit=False, repeat=True,
                                        frames=x_gen_store.shape[0])
                    ani.save("output/" + f"gif_ep{epoch}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print("saved image")

        if epoch == int(config["EPOCHS"] - 1):
            torch.save(ddpm.state_dict(), "model/" + f"model_{epoch}.pth")
            print('saved model')


if __name__ == "__main__":
    params = yaml.safe_load(open("config.yml"))
    main(params)
