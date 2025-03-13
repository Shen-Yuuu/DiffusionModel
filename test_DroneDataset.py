import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from custom_dataset import CustomDataset
def visualize_dataset_samples(dataset, num_samples=5):
    """
    可视化数据集中的样本，显示原始图像、边界框和attention mask
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        
        # 获取一个样本
        image, label, attention_mask = dataset[i+10]
        
        # 获取原始图像路径和XML路径
        img_path, xml_path, _ = dataset.samples[i+10]
        
        # 读取原始图像（不经过transform）
        orig_image = Image.open(img_path).convert('RGB')
        
        # 从XML中获取原始边界框坐标
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bbox = root.find('.//bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # 显示原始图像和边界框
        axes[i, 0].imshow(orig_image)
        rect = patches.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        axes[i, 0].add_patch(rect)
        axes[i, 0].set_title(f'Original Image (Class: {dataset.classes[label]})')
        
        # 显示transformed图像
        img_transformed = image.permute(1, 2, 0) * 0.5 + 0.5  # 反归一化
        axes[i, 1].imshow(img_transformed)
        
        # 计算缩放后的边界框坐标
        orig_w, orig_h = orig_image.size
        xmin_scaled = max(0, min(511, round(xmin * 512 / orig_w)))
        xmax_scaled = max(0, min(511, round(xmax * 512 / orig_w)))
        ymin_scaled = max(0, min(511, round(ymin * 512 / orig_h)))
        ymax_scaled = max(0, min(511, round(ymax * 512 / orig_h)))
        
        # 在transformed图像上显示缩放后的边界框
        rect_transformed = patches.Rectangle(
            (xmin_scaled, ymin_scaled), 
            xmax_scaled-xmin_scaled, 
            ymax_scaled-ymin_scaled, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        axes[i, 1].add_patch(rect_transformed)
        axes[i, 1].set_title('Transformed Image with Scaled Bbox')
        
        # 显示attention mask
        im = axes[i, 2].imshow(attention_mask, cmap='viridis')
        axes[i, 2].set_title('Attention Mask')
        plt.colorbar(im, ax=axes[i, 2])
        
        # 关闭坐标轴
        for ax in axes[i]:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_visualization.png')
    plt.show()

if __name__ == "__main__":
    # 创建数据集实例
    tf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomDataset("./cropped_images_Czech/", transform=tf)
    
    # 验证数据集大小
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Classes: {dataset.classes}")
    
    # 可视化几个样本
    visualize_dataset_samples(dataset, num_samples=5)