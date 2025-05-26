import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import copy
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish激活函数: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 单像素测量模型（固定调制模式）
class SinglePixelMeasurementModel(nn.Module):
    def __init__(self, modulation_patterns):
        super(SinglePixelMeasurementModel, self).__init__()
        self.modulation_patterns = nn.Parameter(modulation_patterns, requires_grad=False)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        measurements = torch.matmul(x_flat, self.modulation_patterns.t())
        return measurements

# TV正则化
def tv_regularization(image, weight=1e-4):
    return weight * (torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])) +
                     torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])))

# 定义SE模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            Swish(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 定义空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out) * x

# 定义CBAM模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_gate(x)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_gate(spatial_input)
        
        return x * spatial_att

# 定义密集层
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        out = self.layer(x)
        return torch.cat([x, out], 1)

# 定义密集块
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        current_block_channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(current_block_channels, growth_rate))
            current_block_channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = current_block_channels

    def forward(self, x):
        return self.block(x)

# 编码器
class Encoder(nn.Module):
    def __init__(self, growth_rate=16, block_config=(3, 3, 3, 3)):
        super(Encoder, self).__init__()

        self.init_conv = nn.Conv2d(1, growth_rate, kernel_size=3, padding=1)
        
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        
        current_channels = growth_rate
        for i, num_layers_in_block in enumerate(block_config):
            block = DenseBlock(num_layers_in_block, current_channels, growth_rate)
            self.enc_blocks.append(block)
            current_channels = block.out_channels

            if i < len(block_config) - 1:
                down_conv = nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, stride=2, padding=1)
                self.down_blocks.append(down_conv)
                current_channels *= 2

    def forward(self, x):
        x_init = self.init_conv(x)
        
        skips = []
        current_out = x_init
        for i in range(len(self.enc_blocks)):
            current_out = self.enc_blocks[i](current_out)
            skips.append(current_out)
            if i < len(self.down_blocks):
                current_out = self.down_blocks[i](current_out)
        
        return skips

# 解码器
class Decoder(nn.Module):
    def __init__(self, encoder_skip_channels_deep_to_shallow, growth_rate, block_config=(3, 3, 3)):
        super(Decoder, self).__init__()
        
        self.up_blocks = nn.ModuleList()
        self.dense_blocks_dec = nn.ModuleList()

        current_channels = encoder_skip_channels_deep_to_shallow[0]

        for i in range(len(block_config)):
            target_up_channels = current_channels // 2 if current_channels // 2 > growth_rate else growth_rate
            self.up_blocks.append(
                nn.ConvTranspose2d(current_channels, target_up_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            
            current_channels_after_up = target_up_channels
            
            skip_connection_channel = encoder_skip_channels_deep_to_shallow[i+1] if (i+1) < len(encoder_skip_channels_deep_to_shallow) else 0
            
            concatenated_channels = current_channels_after_up + skip_connection_channel
            
            num_layers_in_block = block_config[i]
            dec_block = DenseBlock(num_layers_in_block, concatenated_channels, growth_rate)
            self.dense_blocks_dec.append(dec_block)
            
            current_channels = dec_block.out_channels

        self.final_conv = nn.Conv2d(current_channels, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, skips_from_encoder_deep_to_shallow):
        current_feature_map = skips_from_encoder_deep_to_shallow[0] 

        for i in range(len(self.up_blocks)):
            x_upsampled = self.up_blocks[i](current_feature_map)
            
            skip_to_concat = skips_from_encoder_deep_to_shallow[i+1] if (i+1) < len(skips_from_encoder_deep_to_shallow) else None

            if skip_to_concat is not None:
                if x_upsampled.size()[2:] != skip_to_concat.size()[2:]:
                    diffY = skip_to_concat.size()[2] - x_upsampled.size()[2]
                    diffX = skip_to_concat.size()[3] - x_upsampled.size()[3]
                    x_upsampled = F.pad(x_upsampled, [diffX // 2, diffX - diffX // 2,
                                                      diffY // 2, diffY - diffY // 2])
                
                concatenated_map = torch.cat([x_upsampled, skip_to_concat], dim=1)
            else:
                concatenated_map = x_upsampled
                
            current_feature_map = self.dense_blocks_dec[i](concatenated_map)
            
        output_image = self.final_conv(current_feature_map)
        output_image = self.final_activation(output_image)
        return output_image

# 创建生成器类
class Generator(nn.Module):
    def __init__(self, M):
        super(Generator, self).__init__()
        self.fc = nn.Linear(M, 1 * 128 * 128)
        
        growth_rate = 16
        block_config_enc = (3, 3, 3, 3) 
        self.encoder = Encoder(growth_rate=growth_rate, block_config=block_config_enc)
        
        _encoder_skips_channels_deep_to_shallow = []
        with torch.no_grad():
            temp_input_enc = torch.randn(1, 1, 128, 128).to(self.fc.weight.device if next(self.parameters(), None) is not None else device)

            temp_skips_shallow_to_deep = self.encoder(temp_input_enc) 
            for skip in reversed(temp_skips_shallow_to_deep): 
                _encoder_skips_channels_deep_to_shallow.append(skip.shape[1])
        
        block_config_dec = block_config_enc[:-1] if len(block_config_enc) > 1 else list(block_config_enc)
        if not block_config_dec and block_config_enc:
             block_config_dec = (block_config_enc[0],)

        self.decoder = Decoder(
            encoder_skip_channels_deep_to_shallow=_encoder_skips_channels_deep_to_shallow, 
            growth_rate=growth_rate, 
            block_config=block_config_dec
        )
        
    def forward(self, x_measurements):
        x_initial_flat = self.fc(x_measurements)
        x_initial_image = x_initial_flat.view(-1, 1, 128, 128)

        sparsity_loss = torch.mean(torch.abs(x_initial_image))

        skips_shallow_to_deep = self.encoder(x_initial_image)
        skips_deep_to_shallow = list(reversed(skips_shallow_to_deep))
        reconstructed = self.decoder(skips_deep_to_shallow)
        
        tv_loss_value = tv_regularization(reconstructed)
        
        return reconstructed, sparsity_loss, tv_loss_value

# 加载图像并预处理
def load_and_preprocess_image(image_path, size=(128, 128)):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    return image_tensor.to(device)

# 数据集类
class ImageFolderDataset(data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, filename)
            for filename in os.listdir(image_dir)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

# 主函数
if __name__ == "__main__":
    image_dir = "datasets/img_align_celeba"
    test_image_dir = "datasets/img_align_celeba"
    test_image_name = "127857.jpg"
    batch_size = 32
    epochs = 600
    N = 128 * 128
    M = 1638
    
    lambda_consistency = 0.01
    sparsity_weight = 5e-4
    tv_weight = 3e-5

    # 检查路径
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录 {image_dir} 不存在，请检查路径。")
    if not os.path.exists(test_image_dir):
        raise FileNotFoundError(f"测试图像目录 {test_image_dir} 不存在，请检查路径。")

    # 加载调制矩阵
    mat_data = loadmat('Adiff.mat')
    if 'A' not in mat_data:
        raise ValueError(".mat 文件中未找到键名 'A'，请检查文件内容。")
    A = mat_data['A']  # 获取调制矩阵
    modulation_patterns = torch.tensor(A[:M, :N], dtype=torch.float32).to(device)

    # 创建测量模型
    measurement_model = SinglePixelMeasurementModel(modulation_patterns).to(device)

    # 创建生成器
    generator = Generator(M).to(device)

    # 初始化损失函数和优化器
    criterion_mse = nn.MSELoss()
    optimizer_G = optim.AdamW(generator.parameters(), lr=0.0001, weight_decay=1e-5)

    # 使用平滑的学习率衰减
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=1e-6)

    # 数据增强与加载
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.ToTensor()
    ])
    dataset = ImageFolderDataset(image_dir, transform=transform)
    if len(dataset) == 0:
        raise ValueError("数据集为空，请检查图像目录。")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型训练
    best_val_loss = float('inf')
    best_model_state = None
    patience = 100
    trigger_times = 0

    for epoch in range(epochs):
        generator.train()
        epoch_loss = 0
        for i, original_images in enumerate(train_loader):
            original_images = original_images.to(device)

            # 计算测量值
            measurements = measurement_model(original_images)

            # 训练生成器
            optimizer_G.zero_grad()
            reconstructed_images, sparsity_loss, tv_loss_value = generator(measurements)

            # 简化损失函数，以MSE为主导
            mse_loss = criterion_mse(reconstructed_images, original_images)
            
            reconstructed_measurements = measurement_model(reconstructed_images)
            consistency_loss = criterion_mse(reconstructed_measurements, measurements)

            # 损失权重平衡，防止过度正则化
            loss_G = mse_loss + \
                     sparsity_weight * sparsity_loss + \
                     tv_weight * tv_loss_value + \
                     lambda_consistency * consistency_loss
            loss_G.backward()
            optimizer_G.step()

            epoch_loss += loss_G.item()

        # 计算训练集平均损失
        epoch_loss /= len(train_loader)

        # 在验证集上评估模型
        generator.eval()
        with torch.no_grad():
            val_loss = 0
            for original_images in val_loader:
                original_images = original_images.to(device)
                measurements = measurement_model(original_images)
                reconstructed_images, _, _ = generator(measurements)
                mse = criterion_mse(reconstructed_images, original_images)
                val_loss += mse.item()
            val_loss /= len(val_loader)

        # 更新学习率
        scheduler_G.step()

        # 检查是否是最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(generator.state_dict())
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"早停触发于第 {epoch+1} 个周期")
                break

        # 打印训练信息
        print(f'周期 [{epoch+1}/{epochs}], 训练损失: {epoch_loss:.4f}, 验证损失: {val_loss:.4f}')

    print('完成训练，正在保存最佳模型...')
    # 保存最佳模型
    torch.save({
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer_G.state_dict(),
    }, 'optimized_model_2.pth')
    print('模型保存完成。')

    # 加载最佳模型
    generator.load_state_dict(best_model_state)

    # 测试模型
    test_image_path = os.path.join(test_image_dir, test_image_name)
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"测试图像 {test_image_path} 不存在，请检查路径。")
    test_image = load_and_preprocess_image(test_image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        measurements = measurement_model(test_image)
        reconstructed_image, _, _ = generator(measurements)

    # 显示原始和重建图像
    reconstructed_image_display = reconstructed_image.clamp(0, 1).cpu().numpy().squeeze() * 255
    reconstructed_image_display = reconstructed_image_display.astype(np.uint8)

    original_image_display = test_image.clamp(0, 1).cpu().numpy().squeeze() * 255
    original_image_display = original_image_display.astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_display, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image_display, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('optimized_reconstruction.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    # 计算评估指标
    mse_value = np.mean((reconstructed_image_display - original_image_display) ** 2)
    ssim_score = ssim(original_image_display, reconstructed_image_display, data_range=255)
    psnr_score = psnr(original_image_display, reconstructed_image_display, data_range=255)
    print(f'MSE: {mse_value:.4f}')
    print(f'SSIM: {ssim_score:.4f}')
    print(f'PSNR: {psnr_score:.4f}')