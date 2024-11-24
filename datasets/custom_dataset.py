import torch
import json
import os
from PIL import Image
from config.config import Config
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, config:Config, transform=None, max_objects=20):
        self.data_dir = config.project_dir
        self.transform = transform
        self.max_objects = max_objects
        self.target_size = config.image_size[0]
        
        # 指定训练数据目录
        self.img_dir = os.path.join(self.data_dir, "train/img")
        self.ann_dir = os.path.join(self.data_dir, "train/ann")
        
        # 加载标注数据和图像列表
        self.annotations, self.classes = self._load_annotations()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"找到 {len(self.classes)} 个类别: {self.classes}")
    
    def _load_annotations(self):
        annotations = {}
        classes = set()
        
        for json_file in os.listdir(self.ann_dir):
            if json_file.endswith('.json'):
                with open(os.path.join(self.ann_dir, json_file), 'r') as f:
                    annotation_data = json.load(f)
                    image_name = os.path.join(self.img_dir, json_file.replace('.json', ''))
                    
                    # 存储每个对象的类别、边界框和额外的属性信息
                    objects_info = []
                    for obj in annotation_data.get('objects', []):
                        class_title = obj.get('classTitle')
                        if class_title:
                            points = obj.get('points', {}).get('exterior', [])
                            if points and len(points) >= 2:
                                x1, y1 = points[0]
                                x2, y2 = points[1]
                                
                                # 添加更多的对象属性
                                obj_info = {
                                    'class': class_title,
                                    'bbox': [x1, y1, x2, y2],
                                    'area': abs((x2-x1) * (y2-y1)),  # 添加面积信息
                                    'center': [(x1+x2)/2, (y1+y2)/2],  # 添加中心点信息
                                    'attributes': obj.get('attributes', {})  # 保存其他属性
                                }
                                objects_info.append(obj_info)
                                classes.add(class_title)
                    
                    if objects_info:
                        # 按面积排序,优先关注大目标
                        objects_info.sort(key=lambda x: x['area'], reverse=True)
                        annotations[image_name] = objects_info
        
        return annotations, sorted(list(classes))
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = list(self.annotations.keys())[idx]
            
        try:
            # 读取图像
            image = Image.open(img_path).convert('RGB')
            original_width, original_height = image.size
            
            # 获取标签和边界框信息
            objects_info = self.annotations[img_path]
            
            # 创建增强的标签信息
            label_tensor = torch.zeros(len(self.classes))  # 类别one-hot编码
            position_weights = torch.zeros(len(self.classes))  # 位置权重
            size_weights = torch.zeros(len(self.classes))  # 大小权重
            
            # 创建固定大小的边界框张量
            boxes_tensor = torch.zeros((self.max_objects, 6))  # [max_objects, 6]
            mask_tensor = torch.zeros(self.max_objects)  # 用于标记有效的边界框
            
            total_area = self.target_size * self.target_size  # 使用目标尺寸计算面积
            
            # 限制处理的对象数量
            for obj_idx, obj in enumerate(objects_info[:self.max_objects]):
                class_idx = self.class_to_idx[obj['class']]
                label_tensor[class_idx] = 1
                
                # 计算归一化的边界框坐标
                x1, y1, x2, y2 = obj['bbox']
                area = obj['area']
                center_x, center_y = obj['center']
                
                # 填充边界框信息（保持归一化坐标）
                boxes_tensor[obj_idx] = torch.tensor([
                    x1/original_width,  # 保持归一化坐标
                    y1/original_height,
                    x2/original_width,
                    y2/original_height,
                    float(class_idx),
                    (area/(original_width*original_height))  # 归一化面积
                ])
                mask_tensor[obj_idx] = 1  # 标记为有效边界框
                
                # 更新位置和大小权重（使用归一化坐标）
                position_weights[class_idx] += (center_x/original_width + center_y/original_height) / 2
                size_weights[class_idx] += area/(original_width*original_height)
            
            # 归一化权重
            num_objects = min(len(objects_info), self.max_objects)
            if num_objects > 0:
                position_weights /= num_objects
                size_weights /= num_objects
            
            if self.transform:
                image = self.transform(image)
            
            # 返回增强的信息，所有张量都具有固定大小
            return {
                'image': image,
                'labels': label_tensor,
                'boxes': boxes_tensor,
                'box_mask': mask_tensor,
                'position_weights': position_weights,
                'size_weights': size_weights,
                'num_objects': torch.tensor(num_objects, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"加载图像出错 {img_path}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))
