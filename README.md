# 扩散模型图像生成项目

## 项目概述

本项目实现了一个增强型扩散模型（Enhanced Diffusion Model），用于条件图像生成。该模型结合了多种先进的注意力机制（CoordAttention、SEBlock等），并针对特定区域进行增强处理，以生成高质量的图像。

## 主要特点

- **多种注意力机制**：结合CoordAttention和SE注意力，增强特征表示
- **局部区域增强**：通过注意力掩码对重要区域进行特殊处理
- **分层数据集划分**：确保训练和验证集的类别分布一致
- **图像质量评估**：集成FID、SSIM和PSNR等指标评估生成图像质量
- **内存优化**：实现高效的内存管理，支持长时间训练
- **早停机制**：基于验证损失自动停止训练，避免过拟合

## 文件结构

- **new_scripy.py**：最新版本的完整实现，包含所有优化和功能
- **custom_dataset.py**：旧版本的数据集加载实现
- **scripy_old.py**：旧版本的扩散模型实现
- **cropped_images/**：数据集目录
  - **images/**：按类别组织的图像
  - **annotations/**：对应的XML标注文件
- **output/**：模型输出和生成图像的保存目录

## 安装依赖

```bash
pip install torch torchvision tqdm numpy pillow matplotlib scikit-learn scipy
```

## 使用方法

### 训练模型

```bash
python new_scripy.py --mode train
```

### 生成图像

```bash
python new_scripy.py --mode generate --checkpoint path/to/checkpoint.pt
```

### 其他选项

```bash
# 指定指导尺度
python new_scripy.py --mode generate --checkpoint path/to/checkpoint.pt --guidance_scales 2.0 4.0 6.0

# 每个类别生成的样本数量
python new_scripy.py --mode generate --checkpoint path/to/checkpoint.pt --samples_per_class 5

# 不评估生成图像质量
python new_scripy.py --mode generate --checkpoint path/to/checkpoint.pt --no_eval

# 禁用内存清理（用于调试）
python new_scripy.py --mode train --no_memory_cleanup
```

## 模型架构

该项目实现了一个基于UNet的条件扩散模型，主要组件包括：

1. **EnhancedContextUnet**：增强型UNet架构，集成多种注意力机制
2. **CoordAttention**：处理方向特征的注意力机制
3. **LocalEnhancementModule**：对高注意力区域进行特殊增强
4. **EnhancedDDPM**：增强型去噪扩散概率模型实现

## 配置参数

所有配置参数都集中在`Config`类中，包括：

- 模型参数（特征数量、时间步等）
- 注意力掩码阈值和权重
- 训练参数（批量大小、学习率等）
- 早停参数
- 数据集参数
- 输出路径
- 采样参数
- 图像参数

## 评估指标

模型使用以下指标评估生成图像质量：

- **FID (Fréchet Inception Distance)**：衡量生成图像与真实图像分布的差异
- **SSIM (结构相似度)**：衡量图像之间的结构相似程度
- **PSNR (峰值信噪比)**：评估图像质量的基本指标

## 注意事项

- 训练需要较大的GPU内存，建议使用至少8GB显存的GPU
- 生成高质量图像时，建议使用较大的指导尺度（4.0-6.0）
- 对于不同的数据集，可能需要调整`Config`类中的参数

## 版本历史

- **v1.0**：基础扩散模型实现（scripy_old.py）
- **v1.5**：添加自定义数据集支持（custom_dataset.py）
- **v2.0**：当前版本，包含所有优化和增强功能（new_scripy.py）
