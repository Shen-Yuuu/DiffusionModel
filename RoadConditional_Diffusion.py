from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from diffusers import UNet2DModel
import torch
import numpy as np
from diffusers import DDPMScheduler
from tqdm import tqdm


class RoadDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'potholes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
dataset = RoadDataset(root_dir='C:\\Users\\ASUS\\.cache\\kagglehub\\datasets\\atulyakumar98\\pothole-detection-dataset\\versions\\4', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



class ConditionalUNet2DModel(UNet2DModel):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        time_embed_dim = self.config.block_out_channels[0] * 4
        self.label_emb = torch.nn.Sequential(
            torch.nn.Embedding(num_classes, self.config.block_out_channels[0]),
            torch.nn.Linear(self.config.block_out_channels[0], time_embed_dim)
        )

    def forward(self, x, t, y=None):
        batch_size = x.shape[0]
        
        # 重塑时间步为正确的形状
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        t = t.expand(batch_size)
        
        # 获取时间嵌入
        t_emb = self.time_embedding(t)
        
        # 处理条件标签
        if y is not None:
            y = y.view(-1)
            y_emb = self.label_emb(y)
            t_emb = t_emb + y_emb
            
        # 直接使用修改后的时间嵌入调用 UNet2DModel 的实现
        hidden_states = x
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states, t_emb)

        hidden_states = self.mid_block(hidden_states, t_emb)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states, t_emb)

        return {"sample": self.conv_out(hidden_states)}

# 定义模型
model = ConditionalUNet2DModel(
    num_classes=2,  # 2个类别：正常路面、坑洞
    sample_size=256,  # 图像尺寸
    in_channels=3,    # 输入通道数
    out_channels=3,   # 输出通道数
    layers_per_block=2,  # 每个块的层数
    block_out_channels=(32, 64, 128, 256),  # 每个块的输出通道数
    norm_num_groups=1, 
    down_block_types=(
        "DownBlock2D",  # 下采样块
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # 带注意力机制的下采样块
    ),
    up_block_types=(
        "AttnUpBlock2D",  # 带注意力机制的上采样块
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

model = model.to("cuda") 

# 定义调度器
scheduler = DDPMScheduler(num_train_timesteps=1000)

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(100):
    model.train()
    for batch in tqdm(dataloader):
        images, labels = batch
        images = images.to("cuda")
        labels = labels.to("cuda")
        
        # 生成噪声图像
        noise = torch.randn_like(images)
        timesteps_int = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device="cuda")
        noisy_images = scheduler.add_noise(images, noise, timesteps_int)
        
        timesteps_float = timesteps_int.float()

        # 前向传播
        noise_pred = model(noisy_images, timesteps_float, labels)["sample"]
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


@torch.no_grad()
def generate_image(model, label, device="cuda"):
    model.eval()
    label = torch.tensor([label]).to(device)
    noise = torch.randn(1, 3, 256, 256).to(device)
    
    for t in tqdm(reversed(range(scheduler.num_train_timesteps)), desc='Sampling'):
        timestep = torch.full((1,), t, device=device, dtype=torch.long)
        noise_pred = model(noise, timestep, label)["sample"]
        
        alpha = scheduler.alphas[t]
        alpha_hat = scheduler.alphas_cumprod[t]
        beta = scheduler.betas[t]
        
        if t > 0:
            noise_t = torch.randn_like(noise)
        else:
            noise_t = torch.zeros_like(noise)
        
        noise = 1 / torch.sqrt(alpha) * (
            noise - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise_pred
        ) + torch.sqrt(beta) * noise_t
    
    return noise

# 生成图像
label = 1  # 例如，生成裂纹图像
generated_image = generate_image(model, label)
generated_image = (generated_image.clamp(-1, 1) + 1) / 2
generated_image = generated_image.cpu().permute(0, 2, 3, 1).numpy()

# 显示图像
import matplotlib.pyplot as plt
plt.imshow(generated_image[0])
plt.show()