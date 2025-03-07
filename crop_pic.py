import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
class DroneImageCropper(Dataset):
    def __init__(self, img_dir, anno_dir, target_size=128, save_dir='./cropped_images'):
        """
        初始化数据集处理器
        
        Args:
            img_dir (str): 原始图像目录路径
            anno_dir (str): XML标注文件目录路径
            target_size (int): 输出图像的目标尺寸
            save_dir (str): 裁剪图像保存的根目录
        """
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.target_size = target_size
        self.save_dir = save_dir
        
        # 初始化类别映射和样本列表
        self.samples = []
        self.class_map = {}
        self.num_classes = 0
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建标注文件保存目录
        self.anno_save_dir = os.path.join(save_dir, 'annotations')
        os.makedirs(self.anno_save_dir, exist_ok=True)
        
        # 扫描所有XML文件并提取样本信息
        print("正在解析标注文件...")
        for xml_file in tqdm(os.listdir(anno_dir)):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(anno_dir, xml_file)
                img_name = xml_file.replace('.xml', '.jpg')
                img_path = os.path.join(img_dir, img_name)
                
                if os.path.exists(img_path):
                    objects = self._parse_xml(xml_path)
                    # 将所有对象信息添加到样本中
                    self.samples.append({
                        'img_path': img_path,
                        'img_name': img_name,
                        'objects': objects
                    })
                    
                    # 更新类别映射
                    for obj in objects:
                        if obj['name'] not in self.class_map:
                            self.class_map[obj['name']] = self.num_classes
                            # 创建类别对应的保存目录
                            class_dir = os.path.join(save_dir, f"{obj['name']}_{self.num_classes}")
                            os.makedirs(class_dir, exist_ok=True)
                            self.num_classes += 1

    def _parse_xml(self, xml_path):
        """解析XML文件并提取边界框信息"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            objects.append({
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax]
            })
            
        return objects

    def _crop_and_resize(self, image, bbox, expand_ratio=10.0):
        """裁剪边界框并调整大小为正方形，返回裁剪后的图像和新的边界框坐标"""
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        
        # 计算正方形边长
        size = max(width, height) * expand_ratio
        
        # 计算中心点
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        
        # 计算扩充后的边界框
        new_xmin = max(0, center_x - size/2)
        new_ymin = max(0, center_y - size/2)
        new_xmax = min(image.size[0], center_x + size/2)
        new_ymax = min(image.size[1], center_y + size/2)
        
        # 计算原始目标在裁剪图像中的相对位置
        rel_xmin = xmin - new_xmin
        rel_ymin = ymin - new_ymin
        rel_xmax = xmax - new_xmin
        rel_ymax = ymax - new_ymin
        
        # 计算缩放比例
        scale = self.target_size / (new_ymax - new_ymin)
        
        # 计算缩放后的坐标
        scaled_xmin = int(rel_xmin * scale)
        scaled_ymin = int(rel_ymin * scale)
        scaled_xmax = int(rel_xmax * scale)
        scaled_ymax = int(rel_ymax * scale)
        
        # 确保坐标在有效范围内
        scaled_xmin = max(0, min(scaled_xmin, self.target_size-1))
        scaled_ymin = max(0, min(scaled_ymin, self.target_size-1))
        scaled_xmax = max(0, min(scaled_xmax, self.target_size-1))
        scaled_ymax = max(0, min(scaled_ymax, self.target_size-1))
        
        # 裁剪图像
        cropped = image.crop((new_xmin, new_ymin, new_xmax, new_ymax))
        
        # 调整大小
        resized = cropped.resize((self.target_size, self.target_size), Image.LANCZOS)
        # 去除上半部分（将上半部分像素设为黑色）
        resized_array = np.array(resized)
        resized_array[:self.target_size//3, :, :] = 0  # 将上半部分设为黑色
        resized = Image.fromarray(resized_array)
        return resized, [scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax]

    def _create_xml_annotation(self, img_name, img_size, obj_name, bbox):
        """
        创建XML格式的标注文件
        
        Args:
            img_name (str): 图像文件名
            img_size (tuple): 图像尺寸 (width, height)
            obj_name (str): 目标类别名称
            bbox (list): 目标边界框坐标 [xmin, ymin, xmax, ymax]
        """
        root = ET.Element('annotation')
        
        # 添加文件名信息
        filename = ET.SubElement(root, 'filename')
        filename.text = img_name
        
        # 添加图像大小信息
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(img_size[0])
        height = ET.SubElement(size, 'height')
        height.text = str(img_size[1])
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'  # RGB图像
        
        # 添加目标对象信息
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = obj_name
        
        # 添加边界框信息
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(bbox[0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(bbox[1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(bbox[2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(bbox[3])
        
        # 创建XML树
        tree = ET.ElementTree(root)
        return tree

    def process_all_images(self):
        """处理并保存所有图像及其标注信息"""
        print(f"\n开始处理图像...")
        print(f"共发现 {len(self.samples)} 个图像")
        print(f"类别信息: {self.class_map}")
        
        for sample in tqdm(self.samples):
            # 读取图像
            image = Image.open(sample['img_path']).convert('RGB')
            
            # 对每个对象单独处理
            for obj_index, obj in enumerate(sample['objects']):
                cropped_image, new_bbox = self._crop_and_resize(image, obj['bbox'])
                
                # 构建图像保存路径
                class_dir = os.path.join(self.save_dir, f"{obj['name']}_{self.class_map[obj['name']]}")
                base_name = os.path.splitext(sample['img_name'])[0]
                img_save_name = f"{base_name}_obj{obj_index}_crop.jpg"
                img_save_path = os.path.join(class_dir, img_save_name)
                
                # 构建XML保存路径
                xml_save_name = f"{base_name}_obj{obj_index}_crop.xml"
                xml_save_path = os.path.join(self.anno_save_dir, xml_save_name)
                
                # 如果文件都已存在则跳过
                if os.path.exists(img_save_path) and os.path.exists(xml_save_path):
                    continue
                
                # 保存图像
                cropped_image.save(img_save_path, quality=95)
                
                # 创建并保存XML标注
                xml_tree = self._create_xml_annotation(
                    img_name=img_save_name,
                    img_size=(self.target_size, self.target_size),
                    obj_name=obj['name'],
                    bbox=new_bbox
                )
                xml_tree.write(xml_save_path, encoding='utf-8', xml_declaration=True)

def main():
    # 设置路径
    img_dir = 'Czech/train/images'
    anno_dir = 'Czech/train/annotations/xmls'
    save_dir = './data/cropped_images1'
    target_size = 512  # 调整目标尺寸
    
    # 创建处理器并执行处理
    cropper = DroneImageCropper(
        img_dir=img_dir,
        anno_dir=anno_dir,
        target_size=target_size,
        save_dir=save_dir
    )
    
    # 处理所有图像
    cropper.process_all_images()
    
    print("\n处理完成!")
    print(f"裁剪后的图像已保存至: {save_dir}")
    print(f"标注文件已保存至: {os.path.join(save_dir, 'annotations')}")
    print(f"共处理 {len(cropper.samples)} 张图像")
    print(f"类别信息:")
    for class_name, class_id in cropper.class_map.items():
        class_dir = os.path.join(save_dir, f"{class_name}_{class_id}")
        num_images = len(os.listdir(class_dir))
        print(f"  - {class_name} (ID: {class_id}): {num_images} 张图像")

if __name__ == "__main__":
    main()