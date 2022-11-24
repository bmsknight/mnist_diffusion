import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super(ResidualConvBlock, self).__init__()
        self.same_channels = (in_channels == out_channels)
        self.is_res = is_res

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat([x, skip], 1)
        return self.model(x)


class EmbeddingFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbeddingFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextualUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super(ContextualUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, n_feat * 2)

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(kernel_size=7),
            nn.GELU()
        )

        self.time_embed1 = EmbeddingFC(1, 2 * n_feat)
        self.time_embed2 = EmbeddingFC(1, 1 * n_feat)
        self.context_embed1 = EmbeddingFC(n_classes, 2 * n_feat)
        self.context_embed2 = EmbeddingFC(n_classes, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, kernel_size=7, stride=7),
            nn.GroupNorm(num_groups=8, num_channels=2 * n_feat),
            nn.ReLU()
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, context, time, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hidden_vec = self.to_vec(down2)

        context = nn.functional.one_hot(context, num_classes=self.n_classes).type(torch.float)

        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = (-1 * (1 - context_mask))
        context = context * context_mask

        cemb1 = self.context_embed1(context).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.context_embed2(context).view(-1, self.n_feat, 1, 1)
        temb1 = self.time_embed1(time).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.time_embed2(time).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hidden_vec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat([up3, x], 1))
        return out
