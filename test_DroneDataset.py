import unittest
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from custom_dataset import DroneDataset
import numpy as np

class TestDroneDataset(unittest.TestCase):
    def setUp(self):
        self.img_dir = 'Czech/train/images'
        self.anno_dir = 'Czech/train/annotations/xmls'
        self.target_size = 128
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.dataset = DroneDataset(
            img_dir=self.img_dir,
            anno_dir=self.anno_dir,
            target_size=self.target_size,
            transform=self.transform
        )
        
        # 创建输出目录
        self.output_dir = 'test_outputs/comparison'
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_sample_comparison(self, sample_indices):
        """可视化对比原始图像和裁剪后的图像"""
        for idx in sample_indices:
            # 获取样本数据
            sample = self.dataset.samples[idx]
            image_tensor, class_id, mask = self.dataset[idx]
            
            # 读取原始图像
            original_img = Image.open(sample['img_path']).convert('RGB')
            
            # 在原始图像上绘制边界框
            draw = ImageDraw.Draw(original_img)
            xmin, ymin, xmax, ymax = sample['bbox']
            draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
            
            # 获取裁剪后的图像
            cropped_img = self.dataset._crop_and_resize(
                Image.open(sample['img_path']).convert('RGB'),
                sample['bbox']
            )
            
            # 创建图像网格
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # 显示原始图像和边界框
            ax1.imshow(original_img)
            ax1.set_title('Original Image with Bbox')
            ax1.axis('off')
            
            # 显示裁剪后的图像
            ax2.imshow(cropped_img)
            ax2.set_title('Cropped Image')
            ax2.axis('off')
            
            # 显示掩码
            mask_np = mask.squeeze().numpy()
            ax3.imshow(mask_np, cmap='gray')
            ax3.set_title('Region Mask')
            ax3.axis('off')
            
            # 添加总标题
            class_name = [k for k, v in self.dataset.class_map.items() if v == class_id][0]
            plt.suptitle(f'Sample {idx}: Class {class_name}', fontsize=14)
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'comparison_{idx}.png'))
            plt.close()
            
            print(f"已保存样本 {idx} 的对比图像")
            print(f"类别: {class_name}")
            print(f"边界框坐标: {sample['bbox']}")
            print(f"掩码形状: {mask.shape}")
            print("------------------------")

    def test_visualization(self):
        """测试可视化函数"""
        # 随机选择10个样本进行可视化
        num_samples = 10
        total_samples = len(self.dataset)
        sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
        
        self.visualize_sample_comparison(sample_indices)
        
        print(f"\n已生成 {num_samples} 个样本的对比图像")
        print(f"输出目录: {self.output_dir}")
        print(f"类别映射: {self.dataset.class_map}")

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)