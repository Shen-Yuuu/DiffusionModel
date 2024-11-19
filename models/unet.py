import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from config.config import ModelConfig

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        layers = [
            ResidualConvBlock(in_channels, out_channels), 
            nn.MaxPool2d(2)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.in_channels = config.in_channels
        self.n_feat = config.n_feat
        self.n_classes = config.n_classes
        self.init_conv = ResidualConvBlock(self.in_channels, self.n_feat, is_res=True)
        self.down1 = UnetDown(self.n_feat, self.n_feat)
        self.down2 = UnetDown(self.n_feat, 2 * self.n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(32), nn.GELU())
        self.timeembed1 = EmbedFC(1, 2*self.n_feat)
        self.timeembed2 = EmbedFC(1, 1*self.n_feat)
        self.contextembed1 = EmbedFC(self.n_classes, 2*self.n_feat)
        self.contextembed2 = EmbedFC(self.n_classes, 1*self.n_feat)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.n_feat, 2 * self.n_feat, 32, 32),
            nn.GroupNorm(8, 2 * self.n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * self.n_feat, self.n_feat)
        self.up2 = UnetUp(2 * self.n_feat, self.n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * self.n_feat, self.n_feat, 3, 1, 1),
            nn.GroupNorm(8, self.n_feat),
            nn.ReLU(),
            nn.Conv2d(self.n_feat, self.in_channels, 3, 1, 1),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, 
                context_mask: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)
        context_mask = context_mask.view(-1, 1)
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = (-1*(1-context_mask))
        c = c * context_mask
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out