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

# 优化的 CoordAttention - 修复 BatchNorm 共享和注意力权重问题
class CoordAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (1,w)
        
        # 为水平和垂直方向使用独立的卷积层和BatchNorm
        self.conv1_h = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.conv1_w = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        
        # 分离的 BatchNorm
        self.bn1_h = nn.BatchNorm2d(channel // reduction)
        self.bn1_w = nn.BatchNorm2d(channel // reduction)
        
        self.act = nn.GELU()
        
        # 方向交互卷积（使用1x1卷积避免维度混乱）
        self.h2w_proj = nn.Conv2d(channel // reduction, channel // reduction, kernel_size=1)
        self.w2h_proj = nn.Conv2d(channel // reduction, channel // reduction, kernel_size=1)
        
        # 可学习的交互强度参数
        self.gamma_h = nn.Parameter(torch.zeros(1))
        self.gamma_w = nn.Parameter(torch.zeros(1))
        
        self.conv_h = nn.Conv2d(channel // reduction, channel, kernel_size=1)
        self.conv_w = nn.Conv2d(channel // reduction, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # 初始化为0，通过Sigmoid转为0.5
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        
        # 水平方向池化 (h,1)
        x_h = self.pool_h(x)  # [n,c,h,1]
        # 垂直方向池化 (1,w)
        x_w = self.pool_w(x)  # [n,c,1,w]
        
        # 分别处理各方向，使用独立的BN
        x_h = self.conv1_h(x_h)  # [n,c/r,h,1]
        x_h = self.bn1_h(x_h)
        x_h = self.act(x_h)
        
        x_w = self.conv1_w(x_w)  # [n,c/r,1,w]
        x_w = self.bn1_w(x_w)
        x_w = self.act(x_w)
        
        # 方向交互 - 避免直接插值
        h2w = self.h2w_proj(x_h)  # [n,c/r,h,1]
        w2h = self.w2h_proj(x_w)  # [n,c/r,1,w]
        
        # 转置操作处理维度交换，避免显式插值
        h2w_reshaped = h2w.permute(0, 1, 3, 2)  # [n,c/r,1,h]
        w2h_reshaped = w2h.permute(0, 1, 3, 2)  # [n,c/r,w,1]
        
        # 使用自适应池化将不匹配维度对齐
        h2w_adapted = F.adaptive_avg_pool2d(h2w_reshaped, (1, w))  # [n,c/r,1,w]
        w2h_adapted = F.adaptive_avg_pool2d(w2h_reshaped, (h, 1))  # [n,c/r,h,1]
        
        # 使用可学习参数控制交互强度
        gamma_h = torch.sigmoid(self.gamma_h)
        gamma_w = torch.sigmoid(self.gamma_w)
        
        # 融合交互信息
        x_h = x_h + gamma_h * w2h_adapted
        x_w = x_w + gamma_w * h2w_adapted
        
        # 生成注意力权重
        a_h = self.sigmoid(self.conv_h(x_h))  # [n,c,h,1]
        a_w = self.sigmoid(self.conv_w(x_w))  # [n,c,1,w]
        
        # 使用有界的可学习权重组合注意力图
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        
        # 正规化组合权重，确保和为1
        weight_sum = alpha + beta + 1e-8  # 避免除零
        alpha = alpha / weight_sum
        beta = beta / weight_sum
        
        attention = alpha * a_h + beta * a_w
        
        return identity * attention

# 优化的SE注意力模块
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)  # 正确展平
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block with SE attention
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
        self.se = SEBlock(out_channels) if is_res else None
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # 应用SE注意力增强特征
            if self.se:
                x2 = self.se(x2)
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
    def __init__(self, in_channels, out_channels, compress_ratio=4):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps with channel compression
        '''
        # 通道压缩以减少参数量和计算量
        compressed_channels = in_channels // compress_ratio
        
        self.channel_compressor = nn.Sequential(
            nn.Conv2d(in_channels, compressed_channels, 1),
            nn.BatchNorm2d(compressed_channels),
            nn.GELU()
        )
        
        # 通道调整确保维度匹配
        self.channel_adjust = nn.Conv2d(compressed_channels, out_channels, 1)
        
        # 使用卷积下采样替代MaxPool，保持更多的空间信息
        self.down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            ResidualConvBlock(out_channels, out_channels, is_res=True),
            nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.channel_compressor(x)
        x = self.channel_adjust(x)
        return self.down(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            # 使用更灵活的上采样方式，处理奇数尺寸
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
            ),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
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


class ContextUnet(nn.Module):
    def __init__(self, in_channels=3, n_feat = 192, n_classes=10):  # 增加n_feat默认值
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)
        self.down4 = UnetDown(4 * n_feat, 8 * n_feat)

        # 使用CoordAttention替代CBAM
        self.ca1 = CoordAttention(n_feat)
        self.ca2 = CoordAttention(2 * n_feat)
        self.ca3 = CoordAttention(4 * n_feat)
        self.ca4 = CoordAttention(8 * n_feat)

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


    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        
        down1 = self.down1(x)
        down1 = self.ca1(down1)
        
        down2 = self.down2(down1)
        down2 = self.ca2(down2)
        
        down3 = self.down3(down2)
        down3 = self.ca3(down3)
        
        down4 = self.down4(down3)
        down4 = self.ca4(down4)
        
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

        x_i_store = []
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
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

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
    n_epoch = 400
    batch_size = 1
    n_T = 700
    device = "cuda:0"
    n_feat = 192  # 从128增加到192，平衡表达能力和计算复杂度
    lrate = 1e-4
    save_model = True
    save_dir = './output/outputs_Czech/'  # 更新保存路径
    ws_test = [2.0, 4.0]

    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomDataset("./cropped_images_Czech/", transform=tf)
    n_classes = len(dataset.classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # 使用AdamW优化器并添加权重衰减
    optim = torch.optim.AdamW(ddpm.parameters(), lr=lrate, weight_decay=1e-5)
    
    # 添加余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=10, T_mult=2, eta_min=3e-5)
        
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # 使用调度器替代线性衰减
        # optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c, attention_mask in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.long().to(device)
            attention_mask = attention_mask.to(device)
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                loss = ddpm(x, c, attention_mask)
            
            ddpm.scaler.scale(loss).backward()
            # 添加梯度裁剪
            ddpm.scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            
            ddpm.scaler.step(optim)
            ddpm.scaler.update()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            
        # 在每个epoch结束更新学习率
        scheduler.step()
        
        # 记录当前学习率
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.6f}")

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.inference_mode():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                if ep%5==0 or ep == int(n_epoch-1):
                    x_gen, x_gen_store = ddpm.sample(n_sample, (3, 256, 256), device, guide_w=w)

                    # append some real images at bottom, order by class also
                    # append some real images at bottom, order by class also
                    # x_real = torch.Tensor(x_gen.shape).to(device)
                    # for k in range(n_classes):
                    #     for j in range(int(n_sample/n_classes)):
                    #         try: 
                    #             idx = torch.squeeze((c == k).nonzero())[j]
                    #         except:
                    #             idx = 0
                    #         x_real[k+(j*n_classes)] = x[idx]

                    # x_all = torch.cat([x_gen, x_real])
                    grid = make_grid(x_gen*0.5 + 0.5, nrow=n_classes)
                    save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                    print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                    
                    # fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(16,6))
                    # def animate_diff(i, x_gen_store):
                    #     print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                    #     plots = []
                    #     for row in range(int(n_sample/n_classes)):
                    #         for col in range(n_classes):
                    #             axs[row, col].clear()
                    #             axs[row, col].set_xticks([])
                    #             axs[row, col].set_yticks([])
                    #             img = x_gen_store[i,(row*n_classes)+col].transpose(1,2,0)
                    #             img = img * 0.5 + 0.5
                    #             plots.append(axs[row, col].imshow(img))
                    #     return plots
                    # ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    # ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    # print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()