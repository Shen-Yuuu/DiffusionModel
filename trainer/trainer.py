import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from typing import Optional, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
import numpy as np
from models.ddpm import DDPM

class DDPMTrainer:
    def __init__(self, config, model, dataset, logger=None):
        self.config = config
        self.dataset = dataset
        self.device = config.training.device
        
        # 创建 DDPM 模型
        n_total_classes = len(dataset.classes) * 3  # 扩展类别维度以包含位置和大小信息
        self.model = DDPM(
            nn_model=model,
            betas=(1e-4, 0.02),
            n_T=config.model.n_T,
            device=self.device,
            drop_prob=0.1
        ).to(self.device)
        
        # 创建优化器和学习率调度器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 创建数据加载器
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=True
        )
        
        # 设置保存目录
        self.save_dir = Path(config.training.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'samples').mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = logger or logging.getLogger(__name__)
        self.writer = SummaryWriter(self.save_dir / 'logs')
        
        # 训练状态
        self.current_epoch = 0
        self.losses = []
        self.loss_ema = None

    def train_epoch(self):
        self.model.train()
        epoch_losses = []
        pbar = tqdm(self.dataloader)
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                position_weights = batch['position_weights'].to(self.device)
                size_weights = batch['size_weights'].to(self.device)
                
                # 组合条件信息
                combined_condition = torch.cat([
                    labels,
                    position_weights,
                    size_weights
                ], dim=1)
                
                loss = self.model(images, combined_condition)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if self.loss_ema is None:
                self.loss_ema = loss.item()
            else:
                self.loss_ema = 0.95 * self.loss_ema + 0.05 * loss.item()
            
            pbar.set_description(f"loss: {self.loss_ema:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        return avg_loss

    def sample_images(self):
        self.model.eval()
        with torch.no_grad():
            for w_i, w in enumerate(self.config.training.ws_test):
                for class_idx, class_name in enumerate(self.dataset.classes):
                    n_sample = 4
                    
                    # 创建增强的条件向量
                    target_labels = torch.zeros(n_sample, len(self.dataset.classes))
                    target_labels[:, class_idx] = 1
                    
                    # 添加默认的位置和大小信息
                    position_info = torch.ones(n_sample, len(self.dataset.classes)) * 0.5
                    size_info = torch.ones(n_sample, len(self.dataset.classes)) * 0.3
                    
                    # 组合条件信息
                    combined_target = torch.cat([
                        target_labels,
                        position_info,
                        size_info
                    ], dim=1).to(self.device)
                    
                    # 生成样本
                    x_gen, _ = self.model.sample(
                        n_sample, 
                        (3, self.config.data.image_size[0], self.config.data.image_size[1]),
                        self.device,
                        target_labels=combined_target,
                        guide_w=w
                    )
                    
                    # 保存生成的图片
                    grid = make_grid(x_gen * 0.5 + 0.5, nrow=2)
                    save_image(grid, self.save_dir / f'samples/ep{self.current_epoch}_w{w}_{class_name}.png')

    def train(self):
        self.logger.info("开始训练...")
        
        for epoch in range(self.config.training.n_epoch):
            self.current_epoch = epoch
            self.logger.info(f"开始 Epoch {epoch}/{self.config.training.n_epoch-1}")
            
            # 训练一个 epoch
            avg_loss = self.train_epoch()
            self.losses.append(avg_loss)
            
            # 更新学习率
            self.scheduler.step(avg_loss)
            
            # 保存损失曲线
            if epoch % 10 == 0 or epoch == self.config.training.n_epoch - 1:
                plt.figure(figsize=(10, 5))
                plt.plot(self.losses, label='Training Loss')
                plt.title('DDPM Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.savefig(self.save_dir / 'loss_curve.png')
                plt.close()
            
            # 生成样本
            if epoch % 10 == 0 or epoch == self.config.training.n_epoch - 1:
                self.sample_images()
            
            # 保存模型
            if self.config.training.save_model and epoch == self.config.training.n_epoch - 1:
                torch.save(
                    self.model.state_dict(),
                    self.save_dir / f'model_ep{epoch}.pth'
                )
                self.logger.info(f'保存模型到 {self.save_dir}/model_ep{epoch}.pth')
        
        self.logger.info("训练结束")