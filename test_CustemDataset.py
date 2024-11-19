import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from script import CustomDataset
def test_custom_dataset():
    # 1. 基础数据转换
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 2. 创建数据集实例
    dataset = CustomDataset("./road-damage-detector-DatasetNinja", transform=tf)
    
    # 3. 基本信息检查
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数量: {len(dataset.classes)}")
    print(f"类别映射: {dataset.class_to_idx}")
    
    # 4. 获取单个样本并可视化
    image, label = dataset[0]
    print(f"图像张量形状: {image.shape}")
    print(f"标签张量形状: {label.shape}")
    print(f"标签值: {label}")
    
    # 可视化图像
    plt.figure(figsize=(10, 5))
    
    # 原始图像
    plt.subplot(1, 2, 1)
    img_np = image.permute(1, 2, 0).numpy()
    img_np = (img_np * 0.5 + 0.5).clip(0, 1)  # 反归一化
    plt.imshow(img_np)
    plt.title("图像")
    
    # 标签分布
    plt.subplot(1, 2, 2)
    plt.bar(dataset.classes, label.numpy())
    plt.xticks(rotation=45)
    plt.title("标签分布")
    
    plt.tight_layout()
    plt.show()
    
    # 5. 使用DataLoader测试批处理
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # 获取一个批次
    batch_images, batch_labels = next(iter(dataloader))
    print(f"\n批次图像形状: {batch_images.shape}")
    print(f"批次标签形状: {batch_labels.shape}")
    
    # 可视化批次
    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        img = batch_images[i].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        plt.imshow(img)
        plt.title(f"样本 {i}")
    plt.show()
    
    # 6. 类别分布统计
    all_labels = []
    for _, label in dataset:
        all_labels.append(label.numpy())
    all_labels = np.array(all_labels)
    
    plt.figure(figsize=(10, 5))
    plt.bar(dataset.classes, all_labels.sum(axis=0))
    plt.xticks(rotation=45)
    plt.title("整个数据集的类别分布")
    plt.tight_layout()
    plt.show()
    
    # 7. 错误处理测试
    try:
        # 尝试访问不存在的索引
        _ = dataset[len(dataset) + 1]
    except Exception as e:
        print(f"\n错误处理测试: {str(e)}")

if __name__ == "__main__":
    test_custom_dataset()