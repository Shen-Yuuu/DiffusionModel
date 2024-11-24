import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from script import CustomDataset


def visualize_sample(dataset, idx):
    """可视化数据集中的样本和所有检测到的标签"""
    image, labels = dataset[idx]
    
    # 转换图像用于显示
    img_display = image * 0.5 + 0.5
    img_display = img_display.permute(1, 2, 0)
    
    # 获取激活的类别
    active_classes = [dataset.classes[i] for i, label in enumerate(labels) if label > 0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_display)
    plt.title(f'检测到的损坏类型:\n{", ".join(active_classes)}')
    plt.axis('off')
    plt.show()

# 测试代码
def test_multi_label_detection(sample_indices=None):
    """
    测试多标签检测并可视化指定的样本
    Args:
        sample_indices: 要查看的样本索引列表，如果为None则默认查看前5个样本
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomDataset(
        data_dir="./road-damage-detector-DatasetNinja",
        transform=transform,
        crop_size=512,
        multi_label=True
    )
    
    # 如果没有指定样本索引，则默认查看前5个样本
    if sample_indices is None:
        sample_indices = range(5)
    
    # 可视化指定的样本
    for i in sample_indices:
        try:
            visualize_sample(dataset, i)
            print(f"\n样本 {i} 的标签分布:")
            image, labels = dataset[i]
            for j, label in enumerate(labels):
                if label > 0:
                    print(f"- {dataset.classes[j]}")
            print("\n" + "="*50 + "\n")
        except IndexError:
            print(f"警告：索引 {i} 超出数据集范围")

if __name__ == "__main__":
    # 可以指定要查看的具体样本索引
    specific_samples = [10, 20, 30, 40, 50,100,110,120,130,140,150]  # 例如查看这些索引的样本
    test_multi_label_detection(specific_samples)
    
    # 或者使用默认的前5个样本
    # test_multi_label_detection()