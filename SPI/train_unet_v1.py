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
import torch.amp


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

# 定义Attention Gate模块
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate
        :param F_g: Gating signal feature channels
        :param F_l: Skip connection feature channels
        :param F_int: Intermediate feature channels
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = Swish() # 使用Swish激活函数

    def forward(self, g, x):
        """
        :param g: Gating signal from the lower decoder layer
        :param x: Skip connection from the corresponding encoder layer
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 对齐空间维度 (如果需要) - 在U-Net中通常g1和x1大小相同
        # 在 Attention U-Net 的标准实现中，输入 g 和 x 的空间维度应该在 AttentionGate 之前对齐
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # 将注意力系数应用到skip connection上
        return x * psi

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
        current_channels = in_channels
        for i in range(num_layers):
            layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, x):
        return self.block(x)

# 定义下采样模块
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.se = SELayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x

# 定义上采样模块 (集成Attention Gate)
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels): # F_g is implicitly in_channels after upsampling
        """
        上采样模块，包含Attention Gate
        :param in_channels: 输入特征通道数 (来自解码器上一层)
        :param skip_channels: 跳跃连接特征通道数 (来自编码器)
        :param out_channels: 输出特征通道数
        """
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 门控信号 (g) 来自上采样后的输入 'x', 其通道数为 in_channels
        # 跳跃连接 (x) 来自编码器, 其通道数为 skip_channels
        self.attention_gate = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=skip_channels // 2)
        self.conv = nn.Sequential(
            # 输入通道数 = in_channels (来自上采样后的 x) + skip_channels (来自加权后的 skip)
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish(),
        )
        self.se = SELayer(out_channels)

    def forward(self, x, skip): # 移除 g 作为显式输入
        """
        :param x: 输入特征图 (来自解码器上一层)
        :param skip: 跳跃连接特征图 (来自编码器)
        """
        # 对来自上一解码器层的输入特征图进行上采样
        g = self.upsample(x) # 这个上采样后的图 'g' 将作为门控信号

        # 应用 Attention Gate
        # 门控信号 g 的空间维度现在与跳跃连接 'skip' 匹配
        # F_g = g.shape[1] = in_channels
        # F_l = skip.shape[1] = skip_channels
        attention_skip = self.attention_gate(g=g, x=skip)

        # 如果 g 的空间维度与 attention_skip 略有不同 (例如，由于编码器中的奇数维度)，则填充 'g'
        # （如果维度是 2 的幂，通常不需要）
        diffY = attention_skip.size()[2] - g.size()[2]
        diffX = attention_skip.size()[3] - g.size()[3]
        g = F.pad(g, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        # 连接上采样后的特征图 'g' 和加权后的跳跃连接
        x = torch.cat([g, attention_skip], dim=1)
        x = self.conv(x)
        x = self.se(x) # 应用SE模块
        return x

# U-Net 编码器
class Encoder(nn.Module):
    def __init__(self, growth_rate=16, block_config=(3, 3, 3, 3)):
        super(Encoder, self).__init__()
        
        self.init_conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        
        channels = 16
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, channels, growth_rate)
            self.enc_blocks.append(block)
            channels = block.out_channels
            if i < len(block_config) - 1:
                down = DownsampleBlock(channels, channels * 2)
                self.down_blocks.append(down)
                channels *= 2

    def forward(self, x):
        x = self.init_conv(x)
        
        skips = []
        for i in range(len(self.enc_blocks)):
            x = self.enc_blocks[i](x)
            skips.append(x)
            if i < len(self.down_blocks):
                x = self.down_blocks[i](x)
        
        return skips[::-1]

# U-Net 解码器 (修改以使用Attention Gate)
class Decoder(nn.Module):
    def __init__(self, encoder_channels, growth_rate=16, block_config=(3, 3, 3)):
        super(Decoder, self).__init__()
        
        self.dec_blocks = nn.ModuleList()
        channels = encoder_channels[0] # 最深层的通道数

        for i in range(len(block_config)):
            skip_channels = encoder_channels[i+1] # 对应编码器的通道数
            out_channels = channels // 2          # 输出通道数减半
            # F_g = channels                        # 门控信号通道数等于当前解码器层的输入通道数 (注释掉，不再需要)

            # UpsampleBlock 现在接收 in_channels, skip_channels, out_channels
            up = UpsampleBlock(channels, skip_channels, out_channels)
            self.dec_blocks.append(up)
            channels = out_channels # 更新当前通道数

        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, skips):
        x = skips[0] # 从编码器最深层开始
        
        for i in range(len(self.dec_blocks)):
            # 输入x作为当前层的门控信号g (注释掉旧逻辑)
            # skips[i+1]是来自编码器的对应跳跃连接
            # x = self.dec_blocks[i](x=x, skip=skips[i+1], g=x) # 旧调用方式

            # 传递特征图 'x' 和对应的跳跃连接给修改后的 UpsampleBlock
            x = self.dec_blocks[i](x=x, skip=skips[i+1])

        x = self.final_conv(x)
        return x

# 创建生成器类
class Generator(nn.Module):
    def __init__(self, M, measurement_model):
        super(Generator, self).__init__()
        self.fc = nn.Linear(M, 1 * 128 * 128)
        
        growth_rate = 16
        block_config_enc = (3, 3, 3, 3)
        self.encoder = Encoder(growth_rate=growth_rate, block_config=block_config_enc)
        
        _encoder_channels = []
        channels = 16
        temp_input = torch.randn(1, 1, 128, 128)
        x = self.encoder.init_conv(temp_input)
        for i in range(len(self.encoder.enc_blocks)):
            x = self.encoder.enc_blocks[i](x)
            _encoder_channels.append(x.shape[1])
            if i < len(self.encoder.down_blocks):
                x = self.encoder.down_blocks[i](x)
        self.encoder_channels = _encoder_channels[::-1]
        
        block_config_dec = (3, 3, 3)
        self.decoder = Decoder(encoder_channels=self.encoder_channels, growth_rate=growth_rate, block_config=block_config_dec)
        
        self.measurement_model = measurement_model

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1, 128, 128)
        
        skips = self.encoder(x)
        reconstructed = self.decoder(skips)
        
        tv_loss_value = tv_regularization(reconstructed)
        sparsity_loss = torch.mean(torch.abs(skips[0]))

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
    image_dir = "datasets/img_align_celeba"  # 训练图像目录
    test_image_dir = "datasets/img_align_celeba"  # 测试图像目录
    test_image_name = "127857.jpg"  # 测试图像名称
    batch_size = 32
    epochs = 600
    N = 128 * 128  # 图像像素总数
    M = 1638  # 调制矩阵行数（测量次数）

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

    # 创建生成器 (U-Net 结构)
    generator = Generator(M, measurement_model).to(device)
    print(f"Generator initialized with U-Net structure.")
    print(f"Encoder output channels (bottleneck to high-res): {generator.encoder_channels}")

    # 初始化损失函数和优化器
    criterion_mse = nn.MSELoss()
    optimizer_G = optim.AdamW(generator.parameters(), lr=0.0001, weight_decay=1e-5)

    # 使用平滑的学习率衰减
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=1e-6)

    # 初始化混合精度训练的 GradScaler
    scaler = torch.amp.GradScaler('cuda')
    gradient_clip_val = 1.0 # 设置梯度裁剪阈值

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

    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 模型训练
    best_val_loss = float('inf')
    best_model_state = None
    patience = 70
    trigger_times = 0

    sparsity_weight = 0.002
    tv_weight = 0.0002

    for epoch in range(epochs):
        generator.train()
        epoch_loss = 0
        for i, original_images in enumerate(train_loader):
            original_images = original_images.to(device)

            # 计算测量值 (在autocast外部或内部皆可，取决于测量模型是否受益于amp)
            # 如果measurement_model内部有大量计算，放入autocast；如果主要是矩阵乘法，影响不大
            measurements = measurement_model(original_images)

            # 训练生成器
            optimizer_G.zero_grad()

            # 使用混合精度训练
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                reconstructed_images, sparsity_loss, tv_loss_value = generator(measurements)
                # 损失函数
                mse_loss = criterion_mse(reconstructed_images, original_images)
                loss_G = mse_loss + sparsity_weight * sparsity_loss + tv_weight * tv_loss_value

            # 反向传播与优化 (使用 GradScaler)
            scaler.scale(loss_G).backward()

            # 在 optimizer.step() 前 unscale 梯度 (重要)
            scaler.unscale_(optimizer_G) 

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(generator.parameters(), gradient_clip_val)

            scaler.step(optimizer_G)
            scaler.update()

            epoch_loss += loss_G.item()

        # 计算训练集平均损失
        epoch_loss /= len(train_loader)

        # 在验证集上评估模型 (不需要autocast，除非显存不足，但一般验证时不需要梯度)
        generator.eval()
        with torch.no_grad():
            val_loss = 0
            for original_images in val_loader:
                original_images = original_images.to(device)
                measurements = measurement_model(original_images)
                # 验证时可以不使用autocast以获得精确评估，或者使用以加速（如果需要）
                # with torch.amp.autocast(device_type='cuda', dtype=torch.float16): # 可选
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
            print(f'周期 [{epoch+1}/{epochs}], 训练损失: {epoch_loss:.4f}, 验证损失: {val_loss:.4f} (保存最佳模型)')
        else:
            trigger_times += 1
            print(f'周期 [{epoch+1}/{epochs}], 训练损失: {epoch_loss:.4f}, 验证损失: {val_loss:.4f} (Patience: {trigger_times}/{patience})')
            if trigger_times >= patience:
                print(f"早停触发于第 {epoch+1} 个周期")
                break

    print('完成训练，正在保存最佳模型...')
    # 保存最佳模型
    torch.save({
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer_G.state_dict(),
        'best_val_loss': best_val_loss,
    }, 'unet_optimized_model.pth')
    print('模型保存完成。')

    # 加载最佳模型进行测试
    if best_model_state is None:
        print("警告：未找到最佳模型状态，可能是训练早期停止或未改善。")
    else:
        generator.load_state_dict(best_model_state)

    # 测试模型
    test_image_path = os.path.join(test_image_dir, test_image_name)
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"测试图像 {test_image_path} 不存在，请检查路径。")
    test_image = load_and_preprocess_image(test_image_path).unsqueeze(0).to(device)

    generator.eval()
    with torch.no_grad():
        # 测试时通常也不需要autocast，除非模型很大或需要加速
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
    plt.title('Reconstructed Image (U-Net)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('unet_optimized_reconstruction.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    # 计算评估指标
    mse_value = np.mean((reconstructed_image_display - original_image_display) ** 2)
    ssim_score = ssim(original_image_display, reconstructed_image_display, data_range=255)
    psnr_score = psnr(original_image_display, reconstructed_image_display, data_range=255)
    print(f'MSE: {mse_value:.4f}')
    print(f'SSIM: {ssim_score:.4f}')
    print(f'PSNR: {psnr_score:.4f}')