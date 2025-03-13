from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import xml.etree.ElementTree as ET
from sklearn.model_selection import StratifiedShuffleSplit
import gc
import time
import torchvision.models as models
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

# Config class for parameters
class Cfg:
    # Model params
    N_FEAT = 192
    IN_CH = 3
    N_T = 700
    BETAS = (1e-4, 0.02)
    DROP_PROB = 0.1
    
    # Attention mask thresholds
    HIGH_THRESH = 1.2
    MID_THRESH = 0.8
    HIGH_WEIGHT = 3.0
    MID_WEIGHT = 1.0
    LOW_WEIGHT = 0.5
    FEAT_CONSIST_WEIGHT = 2.0
    
    # Training params
    BATCH_SIZE = 4
    ACCUM_STEPS = 4
    LR = 1e-4
    WD = 1e-5
    N_EPOCH = 400
    SAVE_FREQ = 50
    MIN_SAVE_EP = 200
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Dataset params
    VAL_SPLIT = 0.1
    NUM_WORKERS = 5
    PIN_MEM = True
    
    # Output paths
    SAVE_DIR = './output/diffusion/'
    SAMPLE_DIR = './output/samples/'
    
    # Sampling params
    GUIDE_SCALES = [2.0, 4.0]
    SAMPLES_PER_CLASS = 3
    
    # Image params
    IMG_SIZE = 256
    NORM_MEAN = (0.5, 0.5, 0.5)
    NORM_STD = (0.5, 0.5, 0.5)

# Coordinate Attention module
class CoordAttn(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CoordAttn, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1_h = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.conv1_w = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        
        self.bn1_h = nn.BatchNorm2d(channel // reduction)
        self.bn1_w = nn.BatchNorm2d(channel // reduction)
        
        self.act = nn.GELU()
        
        self.h2w_proj = nn.Conv2d(channel // reduction, channel // reduction, kernel_size=1)
        self.w2h_proj = nn.Conv2d(channel // reduction, channel // reduction, kernel_size=1)
        
        self.gamma_h = nn.Parameter(torch.zeros(1))
        self.gamma_w = nn.Parameter(torch.zeros(1))
        
        self.conv_h = nn.Conv2d(channel // reduction, channel, kernel_size=1)
        self.conv_w = nn.Conv2d(channel // reduction, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        
        x_h = self.conv1_h(x_h)
        x_h = self.bn1_h(x_h)
        x_h = self.act(x_h)
        
        x_w = self.conv1_w(x_w)
        x_w = self.bn1_w(x_w)
        x_w = self.act(x_w)
        
        h2w = self.h2w_proj(x_h)
        w2h = self.w2h_proj(x_w)
        
        h2w_reshaped = h2w.permute(0, 1, 3, 2)
        w2h_reshaped = w2h.permute(0, 1, 3, 2)
        
        h2w_adapted = F.adaptive_avg_pool2d(h2w_reshaped, (1, w))
        w2h_adapted = F.adaptive_avg_pool2d(w2h_reshaped, (h, 1))
        
        gamma_h = torch.sigmoid(self.gamma_h)
        gamma_w = torch.sigmoid(self.gamma_w)
        
        x_h = x_h + gamma_h * w2h_adapted
        x_w = x_w + gamma_w * h2w_adapted
        
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        
        weight_sum = alpha + beta + 1e-8
        alpha = alpha / weight_sum
        beta = beta / weight_sum
        
        attention = alpha * a_h + beta * a_w
        
        return identity * attention

# SE Block module
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Local enhancement module
class LocalEnhancer(nn.Module):
    def __init__(self, in_ch, high_thresh=Cfg.HIGH_THRESH):
        super(LocalEnhancer, self).__init__()
        self.high_thresh = high_thresh
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
        )
        
    def forward(self, x, mask):
        high_attn = (mask > self.high_thresh).float().unsqueeze(1)
        return x + self.conv(x) * high_attn

class ResConvBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.same_ch = in_ch == out_ch
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.se = SEBlock(out_ch) if is_res else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.se:
                x2 = self.se(x2)
            if self.same_ch:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_ch, out_ch, compress_ratio=4):
        super(UnetDown, self).__init__()
        compressed_ch = in_ch // compress_ratio
        
        self.channel_compress = nn.Sequential(
            nn.Conv2d(in_ch, compressed_ch, 1),
            nn.BatchNorm2d(compressed_ch),
            nn.GELU()
        )
        
        self.ch_adjust = nn.Conv2d(compressed_ch, out_ch, 1)
        
        self.down = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            ResConvBlock(out_ch, out_ch, is_res=True),
            nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.channel_compress(x)
        x = self.ch_adjust(x)
        return self.down(x)

class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetUp, self).__init__()
        layers = [
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, 3, padding=1)
            ),
            ResConvBlock(out_ch, out_ch),
            ResConvBlock(out_ch, out_ch),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, in_ch=3, n_feat=192, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_ch = in_ch
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResConvBlock(in_ch, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)
        self.down4 = UnetDown(4 * n_feat, 8 * n_feat)

        self.ca1 = CoordAttn(n_feat)
        self.ca2 = CoordAttn(2 * n_feat)
        self.ca3 = CoordAttn(4 * n_feat)
        self.ca4 = CoordAttn(8 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(8), nn.GELU())

        self.time_emb1 = EmbedFC(1, 8*n_feat)
        self.time_emb2 = EmbedFC(1, 4*n_feat)
        self.ctx_emb1 = EmbedFC(n_classes, 8*n_feat)
        self.ctx_emb2 = EmbedFC(n_classes, 4*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(8 * n_feat, 8 * n_feat, 8, 8),
            nn.GroupNorm(8, 8 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(16 * n_feat, 4 * n_feat)
        self.up2 = UnetUp(8 * n_feat, 2 * n_feat)
        self.up3 = UnetUp(4 * n_feat, n_feat)
        self.up4 = UnetUp(2 * n_feat, n_feat)
        
        self.local_enhance = LocalEnhancer(n_feat)
        
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_ch, 3, 1, 1),
        )

    def forward(self, x, c, t, ctx_mask):
        x = self.init_conv(x)
        
        down1 = self.down1(x)
        down1 = self.ca1(down1)
        
        down2 = self.down2(down1)
        down2 = self.ca2(down2)
        
        down3 = self.down3(down2)
        down3 = self.ca3(down3)
        
        down4 = self.down4(down3)
        down4 = self.ca4(down4)
        
        hidden = self.to_vec(down4)

        c = c.long()
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        ctx_mask = ctx_mask[:, None]
        ctx_mask = ctx_mask.repeat(1, self.n_classes)
        
        c = c * ctx_mask
        
        cemb1 = self.ctx_emb1(c).view(-1, self.n_feat * 8, 1, 1)
        temb1 = self.time_emb1(t).view(-1, self.n_feat * 8, 1, 1)
        cemb2 = self.ctx_emb2(c).view(-1, self.n_feat * 4, 1, 1)
        temb2 = self.time_emb2(t).view(-1, self.n_feat * 4, 1, 1)

        up1 = self.up0(hidden)
        up2 = self.up1(cemb1*up1 + temb1, down4)
        up3 = self.up2(cemb2*up2 + temb2, down3)
        up4 = self.up3(up3, down2)
        up5 = self.up4(up4, down1)
        
        up5 = self.local_enhance(up5, ctx_mask)
        
        out = self.out(torch.cat((up5, x), 1))
        return out

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        self.scaler = torch.cuda.amp.GradScaler()

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.n_classes = self.nn_model.n_classes

    def forward(self, x, c, attn_mask):
        """
        Used for training - samples t and noise randomly
        """
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        ctx_mask = torch.bernoulli(torch.ones_like(c, dtype=torch.float) * (1 - self.drop_prob)).to(self.device)
        
        pred_noise = self.nn_model(x_t, c, _ts / self.n_T, ctx_mask)
        
        attn_mask = attn_mask.to(self.device)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        
        weighted_mask = torch.where(attn_mask > Cfg.HIGH_THRESH, 
                    torch.tensor(Cfg.HIGH_WEIGHT).to(self.device),  
                    torch.where(attn_mask > Cfg.MID_THRESH, 
                        torch.tensor(Cfg.MID_WEIGHT).to(self.device), 
                        torch.tensor(Cfg.LOW_WEIGHT).to(self.device)))
        
        loss = (noise - pred_noise)**2
        weighted_loss = loss * weighted_mask
        
        high_attn = (attn_mask > Cfg.HIGH_THRESH).float()
        feat_consist_loss = torch.mean(
            torch.abs(
                pred_noise * high_attn - 
                noise * high_attn
            )
        ) * Cfg.FEAT_CONSIST_WEIGHT

        total_loss = weighted_loss.mean() + feat_consist_loss

        return total_loss

    def sample(self, n_sample, size, device, guide_w=0.0, refine_steps=2):
        """
        Sample from the diffusion model with classifier-free guidance
        """
        x_i = torch.randn(n_sample, *size).to(device)
        
        c_i = torch.arange(0, self.n_classes).to(device)
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        ctx_mask = torch.zeros_like(c_i).to(device)

        c_i = c_i.repeat(2)
        ctx_mask = ctx_mask.repeat(2)
        ctx_mask[n_sample:] = 1.

        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling step {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.nn_model(x_i, c_i, t_is, ctx_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
        return x_i

class CrackDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, attn_cfg=None):
        """
        Args:
            root_dir (string): Directory with all image class folders
            transform: Optional image transformations
            attn_cfg: Attention mask configuration
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.mask_cfg = attn_cfg or {
            'high_thresh': Cfg.HIGH_THRESH,
            'mid_thresh': Cfg.MID_THRESH,
            'default_val': Cfg.LOW_WEIGHT
        }
        
        self.classes = sorted([d for d in os.listdir(os.path.join(root_dir, "images")) 
                              if os.path.isdir(os.path.join(root_dir, "images", d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, "images", class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    xml_path = os.path.join(root_dir, "annotations", img_name.rsplit('.', 1)[0] + '.xml')
                    if os.path.exists(xml_path):
                        self.samples.append((
                            os.path.join(class_dir, img_name),
                            xml_path,
                            self.class_to_idx[class_name]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, xml_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        # Parse XML for bounding box
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bbox = root.find('.//bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Scale coords to match resize
        orig_w = int(root.find('.//width').text)
        orig_h = int(root.find('.//height').text)
        
        # Create attention mask
        attn_mask = torch.ones((Cfg.IMG_SIZE, Cfg.IMG_SIZE)) * self.mask_cfg['default_val']
        
        # Middle attention for lower half
        img_half = Cfg.IMG_SIZE // 2
        attn_mask[img_half:, :] = Cfg.MID_WEIGHT
        
        # High attention for bbox area
        xmin_scaled = max(0, min(Cfg.IMG_SIZE-1, round(xmin * Cfg.IMG_SIZE / orig_w)))
        xmax_scaled = max(0, min(Cfg.IMG_SIZE-1, round(xmax * Cfg.IMG_SIZE / orig_w)))
        ymin_scaled = max(0, min(Cfg.IMG_SIZE-1, round(ymin * Cfg.IMG_SIZE / orig_h)))
        ymax_scaled = max(0, min(Cfg.IMG_SIZE-1, round(ymax * Cfg.IMG_SIZE / orig_h)))
        attn_mask[ymin_scaled:ymax_scaled, xmin_scaled:xmax_scaled] = Cfg.HIGH_WEIGHT

        if self.transform:
            image = self.transform(image)

        return image, label, attn_mask

# Image utility functions
def save_samples(images, save_path, nrow=None, denorm=True):
    """Save generated images as grid"""
    if denorm:
        images = images * 0.5 + 0.5
    
    grid = make_grid(images, nrow=nrow)
    save_image(grid, save_path)
    return save_path

def gen_and_save(model, n_samples, img_size, device, 
               guide_scales, save_dir, classes=None, 
               denorm=True, samples_per_class=None):
    """Generate and save samples"""
    results = {}
    n_classes = len(classes) if classes else 1
    
    with torch.no_grad():
        for guide_scale in guide_scales:
            print(f"\nGenerating samples with guidance scale {guide_scale}")
            n_sample = samples_per_class * n_classes if samples_per_class else n_samples
            x_gen = model.sample(n_sample, (Cfg.IN_CH, *img_size), device, guide_w=guide_scale)
            
            # Save grid
            grid_path = os.path.join(save_dir, f"samples_g{guide_scale}.png")
            save_samples(x_gen, grid_path, nrow=samples_per_class, denorm=denorm)
            
            results[guide_scale] = {
                'samples': x_gen,
                'grid_path': grid_path
            }
            
    return results

class EarlyStop:
    """Early stopping when validation loss stops improving"""
    def __init__(self, patience=Cfg.PATIENCE, min_delta=Cfg.MIN_DELTA, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state = None
        
    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            # Found better model
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Val loss improved to {val_loss:.6f}")
            # Save best state
            self.best_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f"Val loss not improved, patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered! Training halted.")
            return False
            
def create_loaders(dataset, batch_size, val_split=Cfg.VAL_SPLIT, 
                 num_workers=Cfg.NUM_WORKERS, 
                 pin_mem=Cfg.PIN_MEM):
    """Create training and validation data loaders with stratified sampling"""
    # Get labels for stratification
    labels = [dataset.samples[i][2] for i in range(len(dataset))]
    
    # Create stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    
    # Create subsets
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    
    # Report split info
    print(f"Dataset split - Train: {len(train_set)} samples, Val: {len(val_set)} samples")
    
    # Create loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    
    return train_loader, val_loader

def train_model():
    # Ensure save directory exists
    os.makedirs(Cfg.SAVE_DIR, exist_ok=True)

    # Create metrics directory
    metrics_dir = os.path.join(Cfg.SAVE_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize results tracking
    metrics_log = {
        'train_loss': [],
        'val_loss': [],
        'img_metrics': [],
        'lr': []
    }

    # Training setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    guide_scales = Cfg.GUIDE_SCALES
    
    # Initialize image quality metrics
    img_metrics = ImageMetrics(device=device)
    
    # Image transformations
    tf = transforms.Compose([
        transforms.Resize((Cfg.IMG_SIZE, Cfg.IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(Cfg.NORM_MEAN, Cfg.NORM_STD)
    ])
    
    # Create dataset
    dataset = CrackDataset("./cropped_images/", transform=tf)
    n_classes = len(dataset.classes)
    
    # Create train and validation loaders with stratified sampling
    train_loader, val_loader = create_loaders(
        dataset, 
        batch_size=Cfg.BATCH_SIZE
    )
    
    # Initialize model
    ddpm = DDPM(
        nn_model=ContextUnet(
            in_ch=Cfg.IN_CH, 
            n_feat=Cfg.N_FEAT, 
            n_classes=n_classes
        ),
        betas=Cfg.BETAS,
        n_T=Cfg.N_T,
        device=device,
        drop_prob=Cfg.DROP_PROB
    )
    ddpm.to(device)

    # Initialize optimizer with weight decay
    optim = torch.optim.AdamW(
        ddpm.parameters(), 
        lr=Cfg.LR, 
        weight_decay=Cfg.WD
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=10, T_mult=2, eta_min=3e-5
    )
    
    # Setup early stopping
    early_stop = EarlyStop()
    
    # Helper function to save checkpoints
    def save_ckpt(model, optim, scheduler, epoch, loss, is_best=False):
        filename = f"ckpt_ep{epoch}.pt"
        if is_best:
            filename = "best_model.pt"
        
        filepath = os.path.join(Cfg.SAVE_DIR, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'metrics': metrics_log
        }, filepath)
        print(f'Saved {"best " if is_best else ""}checkpoint: {filepath}')
    
    # Collect validation samples for evaluation
    eval_samples = []
    eval_sample_count = min(32, len(val_loader.dataset))
    
    # Samples per class for evaluation
    samples_per_class = max(2, eval_sample_count // n_classes)
    class_counts = {i: 0 for i in range(n_classes)}
    
    with torch.no_grad():
        for x, c, _ in val_loader:
            for i in range(len(c)):
                class_idx = c[i].item()
                if class_counts[class_idx] < samples_per_class and len(eval_samples) < eval_sample_count:
                    eval_samples.append((x[i].to(device), class_idx))
                    class_counts[class_idx] += 1
            
            if sum(class_counts.values()) >= eval_sample_count:
                break
    
    print(f"Collected {len(eval_samples)} samples for evaluation")
        
    for ep in range(Cfg.N_EPOCH):
        print(f'Epoch {ep+1}/{Cfg.N_EPOCH}')
        epoch_start_time = time.time()
        
        # Training phase
        ddpm.train()
        pbar = tqdm(train_loader)
        train_loss_ema = None
        train_losses = []
        
        for step, (x, c, attn_mask) in enumerate(pbar):
            # Move data to device
            x = x.to(device)
            c = c.long().to(device)
            attn_mask = attn_mask.to(device)
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                loss = ddpm(x, c, attn_mask)
                loss = loss / Cfg.ACCUM_STEPS  # Normalize loss for gradient accumulation
            
            # Record loss
            train_losses.append(loss.item() * Cfg.ACCUM_STEPS)
            
            # Backprop
            ddpm.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % Cfg.ACCUM_STEPS == 0 or (step + 1 == len(train_loader)):
                # Gradient clipping
                ddpm.scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
                
                # Optimizer step
                ddpm.scaler.step(optim)
                ddpm.scaler.update()
                optim.zero_grad()

            # Update progress bar
            if train_loss_ema is None:
                train_loss_ema = loss.item() * Cfg.ACCUM_STEPS
            else:
                train_loss_ema = 0.95 * train_loss_ema + 0.05 * loss.item() * Cfg.ACCUM_STEPS
            pbar.set_description(f"Train loss: {train_loss_ema:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        metrics_log['train_loss'].append(avg_train_loss)
        metrics_log['lr'].append(scheduler.get_last_lr()[0])
            
        # Validation phase
        ddpm.eval()
        val_losses = []
        
        with torch.no_grad():
            for x, c, attn_mask in val_loader:
                x = x.to(device)
                c = c.long().to(device)
                attn_mask = attn_mask.to(device)
                
                with torch.cuda.amp.autocast():
                    loss = ddpm(x, c, attn_mask)
                    
                val_losses.append(loss.item())
        
        # Calculate average validation loss
        val_loss = sum(val_losses) / len(val_losses)
        metrics_log['val_loss'].append(val_loss)
        print(f"Validation loss: {val_loss:.4f}")
        
        # Check for early stopping
        is_best = early_stop(val_loss, ddpm, ep)
        if early_stop.early_stop:
            print("Early stopping triggered. Training halted.")
            # Save best model
            if early_stop.best_state:
                torch.save(early_stop.best_state, os.path.join(Cfg.SAVE_DIR, "best_model_early.pt"))
                print(f"Saved best model to {Cfg.SAVE_DIR}best_model_early.pt")
            break
            
        # Update learning rate
        scheduler.step()
        
        # Generate samples and evaluate quality periodically
        if ep % 5 == 0 or ep == Cfg.N_EPOCH - 1:
            ddpm.eval()
            with torch.no_grad():
                # Generate and evaluate samples
                if eval_samples:
                    # Prepare real images and classes
                    real_imgs = torch.stack([s[0] for s in eval_samples])
                    real_classes = [s[1] for s in eval_samples]
                    
                    # Generate images
                    for w in guide_scales:
                        # Generate class-conditional samples
                        c_i = torch.tensor(real_classes).to(device)
                        ctx_mask = torch.zeros_like(c_i).to(device)
                        
                        # Generate matching samples
                        x_gen = ddpm.sample(
                            n_sample=len(eval_samples), 
                            size=(3, Cfg.IMG_SIZE, Cfg.IMG_SIZE), 
                            device=device, 
                            guide_w=w
                        )
                        
                        # Save generated images
                        grid = make_grid(x_gen * 0.5 + 0.5, nrow=4)
                        grid_path = os.path.join(Cfg.SAVE_DIR, f"img_ep{ep}_w{w}.png")
                        save_image(grid, grid_path)
                        print(f'Saved generated images: {grid_path}')
                        
                        # Calculate quality metrics
                        try:
                            quality_metrics = img_metrics.evaluate_batch(real_imgs, x_gen)
                            quality_metrics['guide_scale'] = w
                            quality_metrics['epoch'] = ep
                            
                            print(f"Image quality metrics (w={w}):")
                            for metric_name, metric_value in quality_metrics.items():
                                if metric_name not in ['guide_scale', 'epoch']:
                                    print(f"  {metric_name.upper()}: {metric_value:.4f}")
                            
                            metrics_log['img_metrics'].append(quality_metrics)
                        except Exception as e:
                            print(f"Quality assessment failed: {e}")
        
        # Save model checkpoints
        if ((ep + 1) % Cfg.SAVE_FREQ == 0 or ep == Cfg.N_EPOCH - 1) and ep >= Cfg.MIN_SAVE_EP:
            save_ckpt(ddpm, optim, scheduler, ep, train_loss_ema)
        
        # Save best model separately
        if is_best:
            save_ckpt(ddpm, optim, scheduler, ep, val_loss, is_best=True)
            
        # Save metrics log
        with open(os.path.join(metrics_dir, f"metrics_ep{ep}.json"), 'w') as f:
            import json
            # Convert numpy values to serializable Python types
            serializable_log = {}
            for k, v in metrics_log.items():
                if isinstance(v, list):
                    serializable_log[k] = [
                        {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                         for kk, vv in item.items()} if isinstance(item, dict) else 
                        float(item) if isinstance(item, (np.floating, np.integer)) else item
                        for item in v
                    ]
                else:
                    serializable_log[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
            
            json.dump(serializable_log, f, indent=2)
            
        # Show epoch time
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch time: {epoch_time:.2f} seconds")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Save final model
    save_ckpt(ddpm, optim, scheduler, Cfg.N_EPOCH-1, train_loss_ema)
    
    # Load best model if available
    if early_stop.best_state:
        ddpm.load_state_dict(early_stop.best_state['model_state_dict'])
        print(f"Loaded best model, val loss: {early_stop.best_loss:.6f}")
    
    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return ddpm

def gen_samples(ckpt_path, n_samples_per_class=Cfg.SAMPLES_PER_CLASS, 
              guide_scales=Cfg.GUIDE_SCALES, denorm=True, eval_quality=True):
    """
    Generate samples from trained model
    
    Args:
        ckpt_path: Model checkpoint path
        n_samples_per_class: Number of samples per class to generate
        guide_scales: Guidance scales to use
        denorm: Whether to denormalize output
        eval_quality: Whether to evaluate image quality
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load dataset for class info
    dataset = CrackDataset("./cropped_images/")
    n_classes = len(dataset.classes)
    
    # Load model
    print(f"Loading checkpoint: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None
    
    # Initialize model
    model = ContextUnet(in_ch=Cfg.IN_CH, n_feat=Cfg.N_FEAT, n_classes=n_classes)
    ddpm = DDPM(nn_model=model, betas=Cfg.BETAS, n_T=Cfg.N_T, device=device, drop_prob=0.0)
    
    try:
        ddpm.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded model weights")
        
        # Check for training metrics
        if 'metrics' in checkpoint:
            print("Checkpoint contains training metrics")
    except:
        print("Trying alternative loading method...")
        try:
            # Try direct state dict loading
            ddpm.load_state_dict(checkpoint)
            print("Successfully loaded model weights")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
        
    ddpm.to(device)
    ddpm.eval()
    
    # Create output directory
    samples_dir = os.path.join(Cfg.SAMPLE_DIR, f"samples_{int(time.time())}")
    os.makedirs(samples_dir, exist_ok=True)
    print(f"Samples will be saved to: {samples_dir}")
    
    # Collect real images for comparison
    if eval_quality:
        print("Collecting real images for quality assessment...")
        img_metrics = ImageMetrics(device=device)
        
        # Create validation loader
        val_loader = DataLoader(
            dataset, 
            batch_size=n_samples_per_class,
            shuffle=True,
            num_workers=Cfg.NUM_WORKERS
        )
        
        # Get small set of real images
        real_images = []
        real_classes = []
        samples_needed = n_samples_per_class * min(n_classes, 4)
        
        with torch.no_grad():
            for x, c, _ in val_loader:
                for i in range(len(x)):
                    if len(real_images) < samples_needed:
                        real_images.append(x[i])
                        real_classes.append(c[i].item())
                    else:
                        break
                if len(real_images) >= samples_needed:
                    break
                    
        real_images = torch.stack(real_images).to(device)
    
    # Generate and save samples
    results = {}
    quality_metrics = {}
    
    with torch.no_grad():
        for guide_scale in guide_scales:
            print(f"\nGenerating samples with guidance scale {guide_scale}")
            n_sample = n_samples_per_class * n_classes
            
            # Generate samples
            x_gen = ddpm.sample(n_sample, (Cfg.IN_CH, Cfg.IMG_SIZE, Cfg.IMG_SIZE), device, guide_w=guide_scale)
            
            # Save grid
            grid_path = os.path.join(samples_dir, f"samples_g{guide_scale}.png")
            denormalized_x_gen = x_gen * 0.5 + 0.5 if denorm else x_gen
            grid = make_grid(denormalized_x_gen, nrow=n_samples_per_class)
            save_image(grid, grid_path)
            print(f"Saved grid image to {grid_path}")
            
            # Save individual class samples
            for i in range(len(x_gen)):
                class_idx = i // n_samples_per_class
                sample_idx = i % n_samples_per_class
                class_name = dataset.classes[class_idx]
                
                img = x_gen[i]
                if denorm:
                    img = img * 0.5 + 0.5
                
                img_path = os.path.join(samples_dir, f"{class_name}_s{sample_idx}_g{guide_scale}.png")
                save_image(img, img_path)
            
            # Evaluate quality
            if eval_quality and len(real_images) > 0:
                try:
                    print("Evaluating image quality...")
                    metrics = img_metrics.evaluate_batch(real_images, x_gen[:len(real_images)])
                    quality_metrics[guide_scale] = metrics
                    
                    print(f"Image quality metrics (w={guide_scale}):")
                    for metric_name, metric_value in metrics.items():
                        print(f"  {metric_name.upper()}: {metric_value:.4f}")
                except Exception as e:
                    print(f"Quality assessment failed: {e}")
            
            results[guide_scale] = {
                'samples': x_gen,
                'grid_path': grid_path
            }
    
    # Save quality metrics
    if eval_quality and quality_metrics:
        metrics_path = os.path.join(samples_dir, "quality_metrics.json")
        try:
            with open(metrics_path, 'w') as f:
                import json
                
                # Convert numpy values to standard Python types
                serializable_metrics = {}
                for k, v in quality_metrics.items():
                    serializable_metrics[str(k)] = {
                        kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                        for kk, vv in v.items()
                    }
                
                json.dump(serializable_metrics, f, indent=2)
                print(f"Quality metrics saved to: {metrics_path}")
        except Exception as e:
            print(f"Failed to save quality metrics: {e}")
    
    print(f"All samples generated and saved to {samples_dir}")
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return results

# Image quality metrics
class ImageMetrics:
    """Class for evaluating generated image quality"""
    
    def __init__(self, device='cuda'):
        self.device = device
        # Load inception model for FID calculation
        self.inception_model = None
        self.inception_loaded = False
        
    def _load_inception(self):
        """Lazy loading of Inception model"""
        if not self.inception_loaded:
            self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
            self.inception_model.fc = torch.nn.Identity()  # Remove final FC layer
            self.inception_model.eval()
            self.inception_model.to(self.device)
            self.inception_loaded = True
    
    @torch.no_grad()
    def _extract_features(self, images):
        """Extract features using Inception model"""
        self._load_inception()
        
        # Ensure images are normalized to [0,1]
        if images.min() < 0:
            images = (images + 1) / 2
            
        # Resize to Inception input size
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Extract features
        features = self.inception_model(images)
        return features
    
    def calc_fid(self, real_images, gen_images, batch_size=8):
        """
        Calculate FID (Fréchet Inception Distance)
        
        Args:
            real_images: Real images tensor [N, C, H, W]
            gen_images: Generated images tensor [N, C, H, W]
            batch_size: Batch size to reduce memory usage
        
        Returns:
            fid_score: FID score (lower is better)
        """
        # Extract features
        real_feats = []
        gen_feats = []
        
        for i in range(0, len(real_images), batch_size):
            batch_real = real_images[i:i+batch_size].to(self.device)
            real_feats.append(self._extract_features(batch_real).cpu())
            
        for i in range(0, len(gen_images), batch_size):
            batch_gen = gen_images[i:i+batch_size].to(self.device)
            gen_feats.append(self._extract_features(batch_gen).cpu())
            
        real_feats = torch.cat(real_feats, dim=0).numpy()
        gen_feats = torch.cat(gen_feats, dim=0).numpy()
        
        # Calculate mean and covariance
        mu_real = np.mean(real_feats, axis=0)
        sigma_real = np.cov(real_feats, rowvar=False)
        
        mu_gen = np.mean(gen_feats, axis=0)
        sigma_gen = np.cov(gen_feats, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_gen
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
        return fid_score
    
    @staticmethod
    def calc_ssim(img1, img2):
        """
        Calculate Structural Similarity Index (SSIM)
        
        Args:
            img1, img2: Image tensors [C, H, W] in range [0,1]
            
        Returns:
            ssim_value: SSIM value (higher is better)
        """
        # Convert to [0,1] range
        if img1.min() < 0:
            img1 = (img1 + 1) / 2
        if img2.min() < 0:
            img2 = (img2 + 1) / 2
            
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Convert to NumPy for calculation
        img1 = img1.cpu().numpy()
        img2 = img2.cpu().numpy()
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                    ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))
        
        return ssim_value
    
    @staticmethod
    def calc_psnr(img1, img2):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        
        Args:
            img1, img2: Image tensors [C, H, W] in range [0,1]
            
        Returns:
            psnr_value: PSNR value (higher is better)
        """
        # Convert to [0,1] range
        if img1.min() < 0:
            img1 = (img1 + 1) / 2
        if img2.min() < 0:
            img2 = (img2 + 1) / 2
            
        # Calculate MSE
        mse = torch.mean((img1 - img2) ** 2).item()
        if mse == 0:
            return float('inf')
            
        # Calculate PSNR
        psnr_value = 20 * np.log10(1.0 / np.sqrt(mse))
        return psnr_value
    
    def evaluate_batch(self, real_images, gen_images):
        """
        Evaluate batch image quality
        
        Args:
            real_images: Real images tensor [N, C, H, W]
            gen_images: Generated images tensor [N, C, H, W]
            
        Returns:
            metrics: Dictionary with quality metrics
        """
        metrics = {}
        
        # Calculate FID (if enough samples)
        if len(real_images) >= 10 and len(gen_images) >= 10:
            try:
                metrics['fid'] = self.calc_fid(real_images, gen_images)
            except Exception as e:
                print(f"FID calculation failed: {e}")
                metrics['fid'] = float('nan')
        
        # Calculate SSIM and PSNR (needs one-to-one comparison)
        if len(real_images) == len(gen_images):
            ssim_values = []
            psnr_values = []
            
            for i in range(len(real_images)):
                try:
                    ssim_values.append(self.calc_ssim(real_images[i], gen_images[i]))
                    psnr_values.append(self.calc_psnr(real_images[i], gen_images[i]))
                except Exception as e:
                    print(f"SSIM/PSNR calculation failed: {e}")
            
            if ssim_values:
                metrics['ssim'] = np.mean(ssim_values)
            if psnr_values:
                metrics['psnr'] = np.mean(psnr_values)
        
        return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Diffusion Model Training/Generation")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"], 
                      help="Mode: 'train' for training, 'generate' for sample generation")
    parser.add_argument("--ckpt", type=str, default=None,
                      help="Checkpoint path for generation mode")
    parser.add_argument("--guide_scales", type=float, nargs='+', default=Cfg.GUIDE_SCALES,
                      help="Guidance scales for generation")
    parser.add_argument("--samples", type=int, default=Cfg.SAMPLES_PER_CLASS,
                      help="Number of samples per class")
    parser.add_argument("--no_eval", action="store_true",
                      help="Skip image quality evaluation")

    args = parser.parse_args()
    
    if args.mode == "train":
        train_model()
    elif args.mode == "generate":
        if args.ckpt is None:
            print("Error: Checkpoint path required for generation mode")
            parser.print_help()
            exit(1)
        gen_samples(
            args.ckpt, 
            n_samples_per_class=args.samples, 
            guide_scales=args.guide_scales,
            eval_quality=not args.no_eval
        )