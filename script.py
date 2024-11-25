from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import json
import random
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

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
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
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
    def __init__(self, in_channels=3, n_feat=256, n_classes=1):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(32), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 32, 32),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        
        # mask out context if context_mask == 1
        context_mask = context_mask.view(-1,1)
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
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

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c[:,0])+self.drop_prob).to(self.device)

        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, target_labels=None, guide_w=0.0):
        """
        简化版的采样方法
        参数:
            n_sample: 生成样本数量
            size: 图像大小 (C, H, W)
            device: 计算设备
            target_labels: 目标类别标签
            guide_w: 引导权重
        """
        # 初始化随机噪声
        x_i = torch.randn(n_sample, *size).to(device)
        
        # 如果没有指定标签，随机生成
        if target_labels is None:
            target_labels = torch.randint(0, 2, (n_sample, self.n_classes)).float().to(device)
        
        # 使用tqdm显示进度
        for i in tqdm(range(self.n_T, 0, -1), desc="生成采样"):
            t_is = torch.tensor([i / self.n_T]).to(device)
            
            # 只在最后一步不添加噪声
            z = torch.randn_like(x_i) if i > 1 else 0
            
            # 条件生成
            if guide_w > 0:
                # 计算有条件和无条件的预测
                with torch.no_grad():
                    # 有条件预测
                    eps = self.nn_model(x_i, target_labels, t_is.repeat(n_sample), 
                                      torch.zeros(n_sample, 1).to(device))
                    # 无条件预测
                    eps_uncond = self.nn_model(x_i, target_labels, t_is.repeat(n_sample), 
                                             torch.ones(n_sample, 1).to(device))
                    # 组合预测结果
                    eps = (1 + guide_w) * eps - guide_w * eps_uncond
            else:
                # 普通预测
                eps = self.nn_model(x_i, target_labels, t_is.repeat(n_sample), 
                                  torch.zeros(n_sample, 1).to(device))
            
            # 更新采样
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        
        return x_i


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, crop_size=256, multi_label=False):
        """
        参数:
            data_dir (str): 数据集根目录路径
            transform: 图像转换操作
            crop_size: 裁剪区域大小
            multi_label: 是否使用多标签模式
        """
        self.data_dir = data_dir
        self.transform = transform
        self.crop_size = crop_size
        self.multi_label = multi_label
        
        # 设置图像和标注目录
        self.img_dir = os.path.join(data_dir, "train/img")
        self.ann_dir = os.path.join(data_dir, "train/ann")
        
        # 创建数据索引
        self.data_index = []
        
        # 加载标注数据和图像列表
        self.annotations, self.classes = self._load_annotations()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self._create_index()
        print(f"找到 {len(self.classes)} 个类别: {self.classes}")
        print(f"总共找到 {len(self.data_index)} 个损坏区域样本")

    def _load_annotations(self):
        """加载所有标注文件并收集类别和边界框信息"""
        annotations = {}
        classes = set()
        
        for ann_file in os.listdir(self.ann_dir):
            if ann_file.endswith('.json'):
                with open(os.path.join(self.ann_dir, ann_file), 'r') as f:
                    try:
                        annotation_data = json.load(f)
                        img_name = ann_file.replace('.json', '')
                        img_path = os.path.join(self.img_dir, img_name)
                        
                        objects_info = []
                        for obj in annotation_data.get('objects', []):
                            class_title = obj.get('classTitle')
                            if class_title and 'points' in obj:
                                points = obj['points']['exterior']
                                x_coords = [p[0] for p in points]
                                y_coords = [p[1] for p in points]
                                bbox = {
                                    'x_min': min(x_coords),
                                    'y_min': min(y_coords),
                                    'x_max': max(x_coords),
                                    'y_max': max(y_coords),
                                    'class': class_title
                                }
                                objects_info.append(bbox)
                                classes.add(class_title)
                        
                        if objects_info and os.path.exists(img_path):
                            annotations[img_name] = {
                                'objects': objects_info,
                                'path': img_path
                            }
                            
                    except json.JSONDecodeError:
                        print(f"警告: 无法解析标注文件 {ann_file}")
                        continue
        
        return annotations, sorted(list(classes))

    def _create_index(self):
        """创建数据索引，每个损坏区域作为一个独立样本"""
        for img_name, ann_data in self.annotations.items():
            for obj_idx, obj in enumerate(ann_data['objects']):
                self.data_index.append({
                    'img_name': img_name,
                    'obj_idx': obj_idx,
                    'bbox': obj,
                    'path': ann_data['path']
                })

    def _get_multi_label(self, img_name, current_bbox, crop_coords):
        """获取裁剪区域内的所有标签
        
        Args:
            img_name: 图片名称
            current_bbox: 当前主要的边界框
            crop_coords: 实际裁剪区域的坐标 (x_min, y_min, x_max, y_max)
        """
        labels = torch.zeros(len(self.classes))
        
        # 将当前类别标记为1
        labels[self.class_to_idx[current_bbox['class']]] = 1
        
        if self.multi_label:
            crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop_coords
            
            # 检查所有边界框与裁剪区域的重叠情况
            for obj in self.annotations[img_name]['objects']:
                # 计算重叠区域
                overlap_x_min = max(crop_x_min, obj['x_min'])
                overlap_y_min = max(crop_y_min, obj['y_min'])
                overlap_x_max = min(crop_x_max, obj['x_max'])
                overlap_y_max = min(crop_y_max, obj['y_max'])
                
                # 计算重叠区域的面积
                if overlap_x_max > overlap_x_min and overlap_y_max > overlap_y_min:
                    overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
                    
                    # 计算原始边界框的面积
                    bbox_area = (obj['x_max'] - obj['x_min']) * (obj['y_max'] - obj['y_min'])
                    
                    # 如果重叠区域超过原始边界框面积的阈值，将该类别标记为1
                    overlap_ratio = overlap_area / bbox_area
                    if overlap_ratio > 0.3:  # 可以调整这个阈值
                        labels[self.class_to_idx[obj['class']]] = 1
        
        return labels

    def _crop_damage_area(self, image, bbox, padding_ratio=1.0):
        """根据边界裁剪损坏区域，并返回裁剪坐标"""
        # 计算边界框的原始尺寸
        width = bbox['x_max'] - bbox['x_min']
        height = bbox['y_max'] - bbox['y_min']
        
        # 计算需要扩展的padding大小
        padding_x = int(width * padding_ratio)
        padding_y = int(height * padding_ratio)
        
        # 确保扩展后的区域不会超出图像边界
        x_min = max(0, bbox['x_min'] - padding_x)
        y_min = max(0, bbox['y_min'] - padding_y)
        x_max = min(image.size[0], bbox['x_max'] + padding_x)
        y_max = min(image.size[1], bbox['y_max'] + padding_y)
        
        # 确保裁剪区域是正方形（取较大的边长）
        width = x_max - x_min
        height = y_max - y_min
        max_size = max(width, height)
        
        # 扩展较小的边以创建正方形
        if width < max_size:
            diff = max_size - width
            x_min = max(0, x_min - diff // 2)
            x_max = min(image.size[0], x_max + diff // 2)
        if height < max_size:
            diff = max_size - height
            y_min = max(0, y_min - diff // 2)
            y_max = min(image.size[1], y_max + diff // 2)
        
        # 最后确保区域仍然是正方形
        final_size = min(x_max - x_min, y_max - y_min)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        x_min = center_x - final_size // 2
        x_max = center_x + final_size // 2
        y_min = center_y - final_size // 2
        y_max = center_y + final_size // 2
        
        # 裁剪区域
        cropped = image.crop((x_min, y_min, x_max, y_max))
        return cropped, (x_min, y_min, x_max, y_max)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        sample_info = self.data_index[idx]
        img_name = sample_info['img_name']
        bbox = sample_info['bbox']
        
        try:
            # 读取原始图像
            image = Image.open(sample_info['path']).convert('RGB')
            
            # 裁剪损坏区域，同时获取裁剪坐标
            cropped_image, crop_coords = self._crop_damage_area(image, bbox)
            
            # 调整裁剪区域大小
            cropped_image = cropped_image.resize((self.crop_size, self.crop_size))
            
            # 获取标签（考虑裁剪区域内有损坏）
            label_tensor = self._get_multi_label(img_name, bbox, crop_coords)
            
            if self.transform:
                cropped_image = self.transform(cropped_image)
                
            return cropped_image, label_tensor
            
        except Exception as e:
            print(f"加载图像 {sample_info['path']} 时出错: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        """返回数据集中的样本总数"""
        return len(self.data_index)


def train_custom_dataset():
    # 修改和优化训练参数
    n_epoch = 400
    batch_size = 2  # 减小批次大小以适应512分辨率
    n_T = 400
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_lr = 2e-4
    save_dir = './data/'
    ws_test = [0.0, 2.0]
    save_model = True
    
    # 数据预处理
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 创建数据集和加载器
    dataset = CustomDataset(
        data_dir="./road-damage-detector-DatasetNinja",
        transform=tf,
        crop_size=512,
        multi_label=True  # 启用多标签模式
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,  # 增加工作进程数
        pin_memory=True,
        drop_last=True,  # 丢弃不完整的批次
        persistent_workers=True  # 保持工作进程存活
    )
    
    # 创建模型实例
    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=256, n_classes=len(dataset.classes)),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1
    )
    ddpm.to(device)
    
    # 使用自定义优化器配置
    optim = torch.optim.AdamW(  # 使用AdamW优化器
        ddpm.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01  # 添加权重衰减
    )
    
    # 使用OneCycleLR而不是CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=base_lr,
        epochs=n_epoch,
        steps_per_epoch=len(dataloader),
        pct_start=0.3
    )
    
    # 混合精度训练设置
    use_amp = True
    scaler = GradScaler(enabled=use_amp)
    
    # 训练循环优化
    accumulation_steps = 4
    
    # 添加损失记录列表
    losses = []
    loss_ema = None
    
    for ep in range(n_epoch):
        print(f'Epoch {ep}')
        ddpm.train()
        epoch_losses = []
        pbar = tqdm(dataloader)
        
        for batch_idx, (x, c) in enumerate(pbar):
            x = x.to(device)
            c = c.to(device)
            
            with autocast(device_type=device, enabled=use_amp):
                loss = ddpm(x, c)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
                
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                
                current_loss = loss.item() * accumulation_steps
                epoch_losses.append(current_loss)
                
                if loss_ema is None:
                    loss_ema = current_loss
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * current_loss
                
                pbar.set_description(f"loss: {loss_ema:.4f}")
        
        # 计算并保存每个epoch的平均损失
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 每10个epoch绘制和保存损失曲线
        if ep % 10 == 0 or ep == n_epoch-1:
            plt.figure(figsize=(10, 5))
            plt.plot(losses, label='Training Loss')
            plt.title('DDPM Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{save_dir}/loss_curve.png')
            plt.close()
        
        # 生成样本和评估
        if ep % 10 == 0 or ep == n_epoch-1:
            ddpm.eval()  # 使用ddpm替代ema_model
            with torch.no_grad():
                with autocast(device_type=device, enabled=False):
                    for w_i, w in enumerate(ws_test):
                        for class_idx, class_name in enumerate(dataset.classes):
                            n_sample = 4
                            target_labels = torch.zeros(n_sample, len(dataset.classes))
                            target_labels[:, class_idx] = 1
                            target_labels = target_labels.to(device)
                            
                            x_gen, _ = ddpm.sample(  # 使用ddpm替代ema_model
                                n_sample, 
                                (3, 512, 512), 
                                device, 
                                target_labels=target_labels, 
                                guide_w=w
                            )
                            
                            grid = make_grid(x_gen * 0.5 + 0.5, nrow=2)
                            save_image(
                                grid, 
                                f"{save_dir}/ep{ep}_w{w}_{class_name}.png"
                            )
            ddpm.train()  # 评估后将模型切回训练模式
    
    # 训练结束后保存最终的损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('DDPM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/final_loss_curve.png')
    plt.close()

if __name__ == "__main__":
    train_custom_dataset()