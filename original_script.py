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
import tarfile
import json
import io

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
    def __init__(self, tar_path, transform=None):
        self.tar_path = tar_path
        self.transform = transform
        self.tar = tarfile.open(tar_path, 'r')
        
        # 修改：指定train目录下的文件
        self.img_prefix = "train/img/"
        self.ann_prefix = "train/ann/"
        
        # 加载标注数据和图像列表
        self.annotations, self.classes = self._load_annotations()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
    def _load_annotations(self):
        annotations = {}
        classes = set()
        
        # 遍历tar文件中的标注文件
        for member in self.tar.getmembers():
            if member.name.startswith(self.ann_prefix) and member.name.endswith('.json'):
                f = self.tar.extractfile(member)
                if f is not None:
                    annotation_data = json.load(f)
                    
                    # 获取对应的图像文件名
                    json_name = os.path.basename(member.name)
                    image_name = self.img_prefix + json_name.replace('.json', '')
                    
                    # 收集该图像中的所有对象类别
                    object_classes = []
                    for obj in annotation_data.get('objects', []):
                        class_title = obj.get('classTitle')
                        if class_title:
                            object_classes.append(class_title)
                            classes.add(class_title)
                    
                    if object_classes:  # 只添加有标注的图像
                        annotations[image_name] = object_classes
        
        return annotations, sorted(list(classes))
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name = list(self.annotations.keys())[idx]
        
        # 从tar文件中读取图像
        try:
            img_file = self.tar.extractfile(img_name)
            if img_file is None:
                raise ValueError(f"Cannot find image {img_name} in tar file")
            
            # 读取图像数据
            img_data = img_file.read()
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            
            # 获取标签并创建one-hot编码
            labels = self.annotations[img_name]
            label_tensor = torch.zeros(len(self.classes))
            for label in labels:
                label_tensor[self.class_to_idx[label]] = 1
            
            if self.transform:
                image = self.transform(image)
                
            return image, label_tensor
            
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            # 返回数据集中的下一个样本
            return self.__getitem__((idx + 1) % len(self))
    
    def __del__(self):
        if hasattr(self, 'tar'):
            self.tar.close()


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
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载tar文件中的数据集
    dataset = CustomDataset("/home/usr/ciwh01/road-damage-detector-DatasetNinja.tar", transform=tf)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=5,
        pin_memory=True
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
    
    # 优化器
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    
    # 训练循环
    for ep in range(n_epoch):
        print(f'Epoch {ep}')
        ddpm.train()
        
        # 学习率衰减
        optim.param_groups[0]['lr'] = lrate * (1 - ep/n_epoch)
        
        pbar = tqdm(dataloader)
        loss_ema = None
        
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # 每个epoch保存样本
        if ep % 5 == 0 or ep == n_epoch-1:
            ddpm.eval()
            with torch.no_grad():
                for w_i, w in enumerate(ws_test):
                    # 为每个类别生成样本
                    for class_idx, class_name in enumerate(dataset.classes):
                        n_sample = 4
                        # 创建目标标签
                        target_labels = torch.zeros(n_sample, len(dataset.classes))
                        target_labels[:, class_idx] = 1  # 设置目标类别
                        target_labels = target_labels.to(device)
                        
                        # 生成样本
                        x_gen, _ = ddpm.sample(n_sample, (3, 256, 256), device, 
                                             target_labels=target_labels, guide_w=w)
                        
                        # 保存生成的图片
                        grid = make_grid(x_gen * 0.5 + 0.5, nrow=2)
                        save_image(grid, 
                                 f"{save_dir}/ep{ep}_w{w}_{class_name}.png")
        
        # 保存模型
        if save_model and ep == n_epoch-1:
            torch.save(ddpm.state_dict(), f"{save_dir}/model_ep{ep}.pth")
            print(f'Saved model at {save_dir}/model_ep{ep}.pth')

if __name__ == "__main__":
    train_custom_dataset()