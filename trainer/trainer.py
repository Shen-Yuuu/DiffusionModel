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

class DDPMTrainer:
    def __init__(
        self,
        config: Any,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.logger = logger or self._setup_logger()
        
        self.setup_training()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('DDPMTrainer')
        logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        log_dir = Path(self.config.training.save_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / 'training.log')
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
        
        return logger

    def setup_training(self):
        self.save_dir = Path(self.config.training.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )
        
        self.device = torch.device(self.config.training.device)
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.training.scheduler_factor,
            patience=self.config.training.scheduler_patience,
            verbose=True
        )
        
        self.writer = SummaryWriter(self.save_dir / 'runs')
        
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        self.load_checkpoint()

    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter
        }
        
        checkpoint_path = self.save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_model_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"保存最佳模型到 {best_model_path}")

    def load_checkpoint(self):
        checkpoint_path = self.save_dir / 'latest_checkpoint.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.patience_counter = checkpoint['patience_counter']
            
            self.logger.info(f"从epoch {self.current_epoch} 恢复训练")

    def train_epoch(self) -> float:
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            loss = self.model(images, labels)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{np.mean(epoch_losses):.4f}"
            })
            
            global_step = self.current_epoch * len(self.dataloader) + batch_idx
            self.writer.add_scalar('Loss/train_step', loss.item(), global_step)
            
        avg_loss = np.mean(epoch_losses)
        return avg_loss

    def sample_images(self):
        self.model.eval()
        with torch.no_grad():
            for w_i, w in enumerate(self.config.training.ws_test):
                for class_idx, class_name in enumerate(self.dataset.classes):
                    n_sample = 4
                    target_labels = torch.zeros(n_sample, len(self.dataset.classes))
                    target_labels[:, class_idx] = 1
                    target_labels = target_labels.to(self.device)
                    
                    x_gen, x_gen_store = self.model.sample(
                        n_sample,
                        (3, 256, 256),
                        self.device,
                        target_labels=target_labels,
                        guide_w=w
                    )
                    
                    grid = make_grid(x_gen * 0.5 + 0.5, nrow=2)
                    save_image(
                        grid,
                        self.save_dir / f'samples/ep{self.current_epoch}_w{w}_{class_name}.png'
                    )
                    
                    self.writer.add_image(
                        f'Samples/w{w}_{class_name}',
                        grid,
                        self.current_epoch
                    )

    def train(self):
        self.logger.info("开始训练...")
        n_epochs = self.config.training.n_epoch
        
        try:
            for epoch in range(self.current_epoch, n_epochs):
                self.current_epoch = epoch
                self.logger.info(f"开始 Epoch {epoch}/{n_epochs-1}")
                
                avg_loss = self.train_epoch()
                
                self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
                
                self.scheduler.step(avg_loss)
                
                is_best = avg_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                self.save_checkpoint(is_best)
                
                if epoch % 10 == 0 or epoch == n_epochs - 1:
                    self.sample_images()
                
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    self.logger.info(f"触发早停: {self.patience_counter} epochs未改善")
                    break
                
                self.logger.info(
                    f"Epoch {epoch} 完成: "
                    f"avg_loss={avg_loss:.4f}, "
                    f"best_loss={self.best_loss:.4f}, "
                    f"patience={self.patience_counter}"
                )
                
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        
        finally:
            self.save_checkpoint()
            self.writer.close()
            self.logger.info("训练结束")

    def plot_loss_curve(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.writer.scalar_dict['Loss/train_epoch'], label='Training Loss')
        plt.title('DDPM Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / 'loss_curve.png')
        plt.close()