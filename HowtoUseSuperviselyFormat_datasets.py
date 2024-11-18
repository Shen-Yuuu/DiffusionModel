import torch
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from script import CustomDataset

def test_dataset():
    # 数据预处理
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        # 初始化数据集
        print("正在加载数据集...")
        dataset = CustomDataset("./road-damage-detector-DatasetNinja", transform=tf)
        
        # 测试数据加载
        print("\n测试单个样本加载...")
        image, label = dataset[0]
        print(f"图像张量形状: {image.shape}")
        print(f"标签张量形状: {label.shape}")
        
        # 测试数据批处理
        print("\n测试数据批处理...")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        # 获取一个批次
        batch = next(iter(dataloader))
        print(f"批次中图像的形状: {batch[0].shape}")
        print(f"批次中标签的形状: {batch[1].shape}")
        
        print("\n数据集测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")

if __name__ == "__main__":
    test_dataset()