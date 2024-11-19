# 道路损伤扩散模型生成器

基于去噪扩散概率模型(DDPM)的道路损伤图像生成项目。该项目可以生成各类道路损伤图像,包括纵向裂缝、横向裂缝等。

## 功能特点

- 基于 DDPM 的条件图像生成
- 支持多类别道路损伤生成
- TensorBoard 可视化训练过程
- 支持模型检查点保存和加载
- 提供早停机制避免过拟合
- 灵活的采样和生成控制

## 项目结构

```python
.
├── models/ # 模型定义
│ ├── init.py # 模型组件导出
│ ├── ddpm.py # DDPM 模型实现
│ └── unet.py # U-Net 骨干网络
├── datasets/ # 数据集相关
│ └── custom_dataset.py # 自定义数据集加载
├── trainer/ # 训练相关
│ └── trainer.py # 训练器实现
└── script.py # 主运行脚本
```

## 安装依赖

```bash
python script.py --batch_size 4 --n_epoch 500 --device cuda
```
主要参数:
- batch_size: 批次大小
- n_epoch: 训练轮数
- device: 使用设备(cuda/cpu)

### 生成样本


```python
加载训练好的模型
model.load_state_dict(torch.load('checkpoints/model.pth'))
生成样本
samples = model.sample(
n_sample=4,
size=(3, 256, 256),
device='cuda',
guide_w=0.0
)
```

## 训练可视化

训练过程中会自动记录以下指标:
- 训练损失
- 生成样本质量
- 学习率变化

可通过 TensorBoard 查看:

```bash
tensorboard --logdir runs/diffusion_training
```

## 实现细节

该项目主要基于以下论文实现:
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

核心组件包括:
- U-Net 骨干网络用于噪声预测
- 条件嵌入用于类别控制
- DDPM 采样过程

## 许可证

MIT

## 参考

- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)