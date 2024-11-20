import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
from scripy_old import CustomDataset
def test_custom_dataset():
    # 设置图像大小
    IMAGE_SIZE = 512
    
    # 数据预处理
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 创建数据集实例
    dataset = CustomDataset("./road-damage-detector-DatasetNinja", transform=tf)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True,
        num_workers=0
    )
    
    # 获取一个批次的数据
    batch = next(iter(dataloader))
    
    def visualize_sample(sample_idx=0):
        # 反归一化图像
        img = batch['image'][sample_idx].numpy()
        img = ((img * 0.5 + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        
        # 获取有效的边界框
        valid_boxes = batch['boxes'][sample_idx][batch['box_mask'][sample_idx] == 1]
        
        # 为每个类别分配不同的颜色
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'orange', 'pink']
        
        # 绘制边界框
        for box in valid_boxes:
            x1, y1, x2, y2 = box[:4]
            class_idx = int(box[4])
            confidence = box[5]  # 面积比例
            
            # 将归一化坐标转换为像素坐标
            x1, y1, x2, y2 = [coord * IMAGE_SIZE for coord in [x1, y1, x2, y2]]
            
            # 确保坐标在有效范围内
            x1 = max(0, min(IMAGE_SIZE, x1))
            y1 = max(0, min(IMAGE_SIZE, y1))
            x2 = max(0, min(IMAGE_SIZE, x2))
            y2 = max(0, min(IMAGE_SIZE, y2))
            
            # 获取类别颜色
            color = colors[class_idx % len(colors)]
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # 添加类别标签和置信度
            class_name = dataset.classes[class_idx]
            label = f"{class_name}: {confidence:.2f}"
            draw.text((x1, y1-15), label, fill=color)
            
            # 绘制中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            radius = 3
            draw.ellipse([center_x-radius, center_y-radius, 
                         center_x+radius, center_y+radius], 
                        fill=color)
        
        # 显示图像
        plt.figure(figsize=(15, 15))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Sample {sample_idx}")
        
        # 打印该样本的详细信息
        print(f"\n样本 {sample_idx} 的详细信息:")
        print(f"图像大小: {img.size}")
        print(f"有效目标数量: {int(batch['num_objects'][sample_idx])}")
        print("\n边界框信息:")
        for i, box in enumerate(valid_boxes):
            class_idx = int(box[4])
            class_name = dataset.classes[class_idx]
            print(f"目标 {i+1}:")
            print(f"  类别: {class_name}")
            print(f"  归一化坐标: ({box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f})")
            print(f"  面积比例: {box[5]:.3f}")
        
        # 保存结果
        save_dir = './test_results'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/sample_{sample_idx}_annotated.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    # 可视化所有样本
    for i in range(batch['image'].shape[0]):
        visualize_sample(i)
    
    # 打印数据集统计信息
    print("\n数据集统计信息:")
    print(f"总样本数: {len(dataset)}")
    print(f"类别数量: {len(dataset.classes)}")
    print("类别列表:", dataset.classes)

if __name__ == "__main__":
    test_custom_dataset()