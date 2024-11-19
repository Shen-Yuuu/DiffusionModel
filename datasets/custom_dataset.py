import torch
from torch.utils.data import Dataset
from PIL import Image
import supervisely as sly
import json
from typing import Tuple, Optional
from config.config import Config


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, config:Config, transform=None):
        self.project_dir = config.project_dir
        self.transform = transform
        self.config = config
        
        self.project = sly.Project(self.project_dir, sly.OpenMode.READ)
        print(f"打开项目: {self.project.name}")
        print(f"项目中的图像总数: {self.project.total_items}")
        
        self.classes = [cls.name for cls in self.project.meta.obj_classes]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"发现 {len(self.classes)} 个类别: {self.classes}")
        
        self.samples = []
        for dataset in self.project.datasets:
            for item_name, image_path, ann_path in dataset.items():
                self.samples.append((image_path, ann_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, ann_path = self.samples[idx]
        
        try:
            image = sly.image.read(image_path)
            image = Image.fromarray(image)
            
            ann_json = json.load(open(ann_path))
            ann = sly.Annotation.from_json(ann_json, self.project.meta)
            
            label_tensor = torch.zeros(len(self.classes))
            for label in ann.labels:
                class_name = label.obj_class.name
                label_tensor[self.class_to_idx[class_name]] = 1
            
            if self.transform:
                image = self.transform(image)
            
            return image, label_tensor
            
        except Exception as e:
            print(f"加载图像 {image_path} 时出错: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))
