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
import json

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
        target_labels: [n_sample, n_classes] 指定要生成的目标类别的one-hot编码
        """
        x_i = torch.randn(n_sample, *size).to(device)
        
        if target_labels is None:
            # 如果没有指定标签，随机生成一些标签组合
            target_labels = torch.randint(0, 2, (n_sample, self.n_classes)).float().to(device)
        
        context_mask = torch.zeros(n_sample).to(device)
        
        # 双倍批次大小
        x_i_double = torch.cat([x_i, x_i])  # [2*n_sample, C, H, W]
        c_i_double = torch.cat([target_labels, target_labels])  # [2*n_sample, n_classes]
        context_mask_double = torch.cat([context_mask, torch.ones_like(context_mask)])  # [2*n_sample]
        
        x_i_store = []
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is_double = t_is.repeat(2*n_sample)
            
            # 生成噪声
            z = torch.randn_like(x_i_double) if i > 1 else 0
            
            # 确保context_mask维度正确
            current_context_mask = context_mask_double.view(-1, 1)
            
            # 预测噪声
            eps = self.nn_model(x_i_double, c_i_double, t_is_double, current_context_mask)
            eps1, eps2 = eps.chunk(2)  # 将预测分成两半
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            
            # 更新采样
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * (z[:n_sample] if isinstance(z, torch.Tensor) else z)
            )
            
            # 更新双倍批次
            x_i_double = torch.cat([x_i, x_i])
            
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, max_objects=20):
        self.data_dir = data_dir
        self.transform = transform
        self.max_objects = max_objects
        self.target_size = 512  # 修改目标尺寸为512
        
        # 指定训练数据目录
        self.img_dir = os.path.join(data_dir, "train/img")
        self.ann_dir = os.path.join(data_dir, "train/ann")
        
        # 加载标注数据和图像列表
        self.annotations, self.classes = self._load_annotations()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"找到 {len(self.classes)} 个类别: {self.classes}")
    
    def _load_annotations(self):
        annotations = {}
        classes = set()
        
        for json_file in os.listdir(self.ann_dir):
            if json_file.endswith('.json'):
                with open(os.path.join(self.ann_dir, json_file), 'r') as f:
                    annotation_data = json.load(f)
                    image_name = os.path.join(self.img_dir, json_file.replace('.json', ''))
                    
                    # 存储每个对象的类别、边界框和额外的属性信息
                    objects_info = []
                    for obj in annotation_data.get('objects', []):
                        class_title = obj.get('classTitle')
                        if class_title:
                            points = obj.get('points', {}).get('exterior', [])
                            if points and len(points) >= 2:
                                x1, y1 = points[0]
                                x2, y2 = points[1]
                                
                                # 添加更多的对象属性
                                obj_info = {
                                    'class': class_title,
                                    'bbox': [x1, y1, x2, y2],
                                    'area': abs((x2-x1) * (y2-y1)),  # 添加面积信息
                                    'center': [(x1+x2)/2, (y1+y2)/2],  # 添加中心点信息
                                    'attributes': obj.get('attributes', {})  # 保存其他属性
                                }
                                objects_info.append(obj_info)
                                classes.add(class_title)
                    
                    if objects_info:
                        # 按面积排序,优先关注大目标
                        objects_info.sort(key=lambda x: x['area'], reverse=True)
                        annotations[image_name] = objects_info
        
        return annotations, sorted(list(classes))
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = list(self.annotations.keys())[idx]
            
        try:
            # 读取图像
            image = Image.open(img_path).convert('RGB')
            original_width, original_height = image.size
            
            # 获取标签和边界框信息
            objects_info = self.annotations[img_path]
            
            # 创建增强的标签信息
            label_tensor = torch.zeros(len(self.classes))  # 类别one-hot编码
            position_weights = torch.zeros(len(self.classes))  # 位置权重
            size_weights = torch.zeros(len(self.classes))  # 大小权重
            
            # 创建固定大小的边界框张量
            boxes_tensor = torch.zeros((self.max_objects, 6))  # [max_objects, 6]
            mask_tensor = torch.zeros(self.max_objects)  # 用于标记有效的边界框
            
            total_area = self.target_size * self.target_size  # 使用目标尺寸计算面积
            
            # 限制处理的对象数量
            for obj_idx, obj in enumerate(objects_info[:self.max_objects]):
                class_idx = self.class_to_idx[obj['class']]
                label_tensor[class_idx] = 1
                
                # 计算归一化的边界框坐标
                x1, y1, x2, y2 = obj['bbox']
                area = obj['area']
                center_x, center_y = obj['center']
                
                # 填充边界框信息（保持归一化坐标）
                boxes_tensor[obj_idx] = torch.tensor([
                    x1/original_width,  # 保持归一化坐标
                    y1/original_height,
                    x2/original_width,
                    y2/original_height,
                    float(class_idx),
                    (area/(original_width*original_height))  # 归一化面积
                ])
                mask_tensor[obj_idx] = 1  # 标记为有效边界框
                
                # 更新位置和大小权重（使用归一化坐标）
                position_weights[class_idx] += (center_x/original_width + center_y/original_height) / 2
                size_weights[class_idx] += area/(original_width*original_height)
            
            # 归一化权重
            num_objects = min(len(objects_info), self.max_objects)
            if num_objects > 0:
                position_weights /= num_objects
                size_weights /= num_objects
            
            if self.transform:
                image = self.transform(image)
            
            # 返回增强的信息，所有张量都具有固定大小
            return {
                'image': image,
                'labels': label_tensor,
                'boxes': boxes_tensor,
                'box_mask': mask_tensor,
                'position_weights': position_weights,
                'size_weights': size_weights,
                'num_objects': torch.tensor(num_objects, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"加载图像出错 {img_path}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

def train_custom_dataset():
    # 修改训练参数
    n_epoch = 100
    batch_size = 4
    n_T = 400
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lrate = 1e-4
    save_dir = './data/'
    ws_test = [0.0, 0.5, 2.0]
    save_model = True
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 数据预处理
    tf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 数据集
    dataset = CustomDataset("./road-damage-detector-DatasetNinja", transform=tf)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=5,
        pin_memory=True
    )
    
    # 创建模型实例
    n_total_classes = len(dataset.classes) * 3  # 扩展类别维度以包含位置和大小信息
    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=256, n_classes=n_total_classes),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1
    )
    ddpm.to(device)
    
    # 优化器
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=5, verbose=True
    )
    losses = []
    loss_ema = None
    # 训练循环
    for ep in range(n_epoch):
        print(f'Epoch {ep}')
        ddpm.train()
        
        epoch_losses = []
        pbar = tqdm(dataloader)
        for batch in pbar:
            optim.zero_grad()
            
            with torch.amp.autocast('cuda'):
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)
                position_weights = batch['position_weights'].to(device)
                size_weights = batch['size_weights'].to(device)
                boxes = batch['boxes'].to(device)
                box_mask = batch['box_mask'].to(device)
                num_objects = batch['num_objects'].to(device)
                
                combined_condition = torch.cat([
                    labels,
                    position_weights,
                    size_weights
                ], dim=1)
                
                loss = ddpm(images, combined_condition)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
            
            optim.step()
            epoch_losses.append(loss.item())
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            
            pbar.set_description(f"loss: {loss_ema:.4f}")
        
        # 计算平均损失并更新学习率
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
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
        

        # 每10个epoch保存样本
        if ep % 10 == 0 or ep == n_epoch-1:
            ddpm.eval()
            with torch.no_grad():
                for w_i, w in enumerate(ws_test):
                    # 为每个类别生成样本
                    for class_idx, class_name in enumerate(dataset.classes):
                        n_sample = 4
                        
                        # 创建增强的条件向量
                        target_labels = torch.zeros(n_sample, len(dataset.classes))
                        target_labels[:, class_idx] = 1
                        
                        # 添加默认的位置和大小信息
                        position_info = torch.ones(n_sample, len(dataset.classes)) * 0.5  # 默认中心位置
                        size_info = torch.ones(n_sample, len(dataset.classes)) * 0.3     # 默认中等大小
                        
                        # 组合条件信息
                        combined_target = torch.cat([
                            target_labels,
                            position_info,
                            size_info
                        ], dim=1).to(device)
                        
                        # 生成样本
                        x_gen, _ = ddpm.sample(n_sample, (3, 512, 512), device, 
                                             target_labels=combined_target, guide_w=w)
                        
                        # 保存生成的图片
                        grid = make_grid(x_gen * 0.5 + 0.5, nrow=2)
                        save_image(grid, 
                                 f"{save_dir}/ep{ep}_w{w}_{class_name}.png")
        
        # 保存模型
        if save_model and ep == n_epoch-1:
            torch.save(ddpm.state_dict(), f"{save_dir}/model_ep{ep}.pth")
            print(f'Saved model at {save_dir}/model_ep{ep}.pth')

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