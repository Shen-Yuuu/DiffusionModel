from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
from PIL import Image
import xml.etree.ElementTree as ET

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
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
            # this adds on correct residual in case channels have increased
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
        '''
        process and downscale the image feature maps
        '''
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),  # 增加一个残差块
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class LocalEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(LocalEnhancementModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x, attention_mask):
        return x + self.conv(x) * (attention_mask > 1.2).float().unsqueeze(1)


class ContextUnet(nn.Module):
    def __init__(self, in_channels=3, n_feat = 128, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)
        self.down4 = UnetDown(4 * n_feat, 8 * n_feat)

        self.cbam1 = CBAM(n_feat)
        self.cbam2 = CBAM(2 * n_feat)
        self.cbam3 = CBAM(4 * n_feat)
        self.cbam4 = CBAM(8 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(8), nn.GELU())

        self.timeembed1 = EmbedFC(1, 8*n_feat)
        self.timeembed2 = EmbedFC(1, 4*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 8*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 4*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(8 * n_feat, 8 * n_feat, 8, 8),
            nn.GroupNorm(8, 8 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(16 * n_feat, 4 * n_feat)
        self.up2 = UnetUp(8 * n_feat, 2 * n_feat)
        self.up3 = UnetUp(4 * n_feat, n_feat)
        self.up4 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

        self.local_enhance = LocalEnhancementModule(n_feat)

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        
        down1 = self.down1(x)
        down1 = self.cbam1(down1)
        
        down2 = self.down2(down1)
        down2 = self.cbam2(down2)
        
        down3 = self.down3(down2)
        down3 = self.cbam3(down3)
        
        down4 = self.down4(down3)
        down4 = self.cbam4(down4)
        
        hiddenvec = self.to_vec(down4)

        c = c.long()
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        
        c = c * context_mask
        
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 8, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 8, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 4, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 4, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down4)
        up3 = self.up2(cemb2*up2 + temb2, down3)
        up4 = self.up3(up3, down2)
        up5 = self.up4(up4, down1)
        up5 = self.local_enhance(up5, context_mask)
        
        out = self.out(torch.cat((up5, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        self.scaler = torch.cuda.amp.GradScaler()

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.n_classes = self.nn_model.n_classes

    def forward(self, x, c, attention_mask):
        """
        this method is used in training, so samples t and noise randomly
        """
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.ones_like(c,dtype=torch.float) * (1 - self.drop_prob)).to(self.device)
        
        # 预测噪声
        predicted_noise = self.nn_model(x_t, c, _ts / self.n_T, context_mask)
        
        # 使用attention_mask加权损失
        attention_mask = attention_mask.to(self.device)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, 3, 1, 1)  # 扩展到3通道
        
        # 计算加权MSE损失
        weighted_mask = torch.where(attention_mask > 1.2, 
                      torch.tensor(3.0).to(self.device),  # 从1.7提高到3.0
                      torch.where(attention_mask > 0.8, 
                          torch.tensor(1.0).to(self.device), 
                          torch.tensor(0.5).to(self.device)))  # 背景从0.3提高到0.5
        
        
        loss = (noise - predicted_noise)**2
        weighted_loss = loss * weighted_mask
        
        high_attention_regions = (attention_mask > 1.2).float().unsqueeze(1)
        feature_consistency_loss = torch.mean(
            torch.abs(
                predicted_noise * high_attention_regions - 
                noise * high_attention_regions
            )
        ) * 2.0  # 额外权重

        total_loss = weighted_loss.mean() + feature_consistency_loss

        return total_loss

    def sample(self, n_sample, size, device, guide_w = 0.0,refine_steps=2):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        
        c_i = torch.arange(0,self.n_classes).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.

        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        参数:
            root_dir (string): 包含所有分类子文件夹的根目录路径
            transform: 可选的图像转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(os.path.join(root_dir, "images")) if os.path.isdir(os.path.join(root_dir, "images", d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, "images", class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    # 获取对应的XML文件路径
                    xml_path = os.path.join(root_dir, "annotations", img_name.rsplit('.', 1)[0] + '.xml')
                    if os.path.exists(xml_path):
                        self.samples.append((
                            os.path.join(class_dir, img_name),
                            xml_path,
                            self.class_to_idx[class_name]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, xml_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        # 解析XML获取边界框
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bbox = root.find('.//bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # 调整边界框坐标以匹配resize后的图像大小
        orig_w = int(root.find('.//width').text)
        orig_h = int(root.find('.//height').text)
        
        # 创建attention mask - 初始设置所有区域为0.5
        attention_mask = torch.ones((256, 256)) * 0.5
        
        # 图像下半部分设置为1.0
        img_half_height = 256 // 2
        attention_mask[img_half_height:, :] = 1.0
        
        # 边界框区域设置为1.5
        xmin_scaled = max(0, min(255, round(xmin * 256 / orig_w)))
        xmax_scaled = max(0, min(255, round(xmax * 256 / orig_w)))
        ymin_scaled = max(0, min(255, round(ymin * 256 / orig_h)))
        ymax_scaled = max(0, min(255, round(ymax * 256 / orig_h)))
        attention_mask[ymin_scaled:ymax_scaled, xmin_scaled:xmax_scaled] = 1.5

        if self.transform:
            image = self.transform(image)
        
        # h = image.shape[1]
        # top_third = h // 3
        # image[:, :top_third, :] = -1

        return image, label, attention_mask

def train_mnist():

    # hardcoding these here
    n_epoch = 500
    batch_size = 4
    n_T = 500
    device = "cuda:0"
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = True
    save_dir = './output/outputs_mask/'
    ws_test = [2.0, 4.0]

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomDataset("./cropped_images_Japan/", transform=tf)
    n_classes = len(dataset.classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)


    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        accum_steps = 4  # 累积4步相当于批量大小16
        for step, (x, c, attention_mask) in enumerate(pbar):
            optim.zero_grad()
            x = x.to(device)
            c = c.long().to(device)
            attention_mask = attention_mask.to(device)
            with torch.cuda.amp.autocast():
                loss = ddpm(x, c, attention_mask)
            
            loss = loss / accum_steps
            ddpm.scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                # 每累积accum_steps步后更新参数
                ddpm.scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
                ddpm.scaler.step(optim)
                ddpm.scaler.update()
                optim.zero_grad()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.inference_mode():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                if ep%5==0 or ep == int(n_epoch-1):
                    x_gen = ddpm.sample(n_sample, (3, 128, 128), device, guide_w=w)

                    grid = make_grid(x_gen*0.5 + 0.5, nrow=n_classes)
                    save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                    print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()