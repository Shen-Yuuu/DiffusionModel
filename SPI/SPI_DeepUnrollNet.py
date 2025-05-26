import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import copy
import torch.nn.utils as nn_utils # For gradient clipping
from torch.cuda.amp import autocast, GradScaler # For mixed precision
from tqdm import tqdm

# --- Activation Function ---
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# --- Building Blocks ---
class StandardResBlock(nn.Module):
    """标准的ResBlock: Conv -> BN -> Activation -> Conv -> BN -> residual + Activation"""
    def __init__(self, channels, use_swish=True):
        super(StandardResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = Swish() if use_swish else nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.final_act = Swish() if use_swish else nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual # Add residual before final activation
        out = self.final_act(out)
        return out

class ResPriorNet(nn.Module):
    """增强的残差先验网络，用于DeepUnrollFISTANet"""
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=64, num_resblocks=5, use_swish=True):
        super(ResPriorNet, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.initial_act = Swish() if use_swish else nn.ReLU(inplace=True)
        
        res_blocks_list = []
        for _ in range(num_resblocks):
            res_blocks_list.append(StandardResBlock(hidden_channels, use_swish=use_swish))
        self.res_blocks = nn.Sequential(*res_blocks_list)
        
        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_act(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        return x # Learns the denoised image directly

# --- Main Unrolling Network ---
class DeepUnrollFISTANet(nn.Module):
    def __init__(self, num_stages, measurement_matrix_A, img_shape=(128, 128),
                 prior_in_channels=1, prior_out_channels=1, 
                 prior_hidden_channels=64, prior_num_resblocks=5,
                 learnable_step_sizes=True, learnable_momentum=True):
        super(DeepUnrollFISTANet, self).__init__()
        self.num_stages = num_stages
        self.img_shape = img_shape
        self.N_pixels = img_shape[0] * img_shape[1]
        self._is_first_batch_debug_run = True # For detailed logging on the first batch
        self._current_debug_stage = 0 # For _print_stats method

        # Register A and A_T as buffers so they are moved to the correct device with the model
        self.register_buffer('A', measurement_matrix_A) # measurement_matrix_A is [M, N]
        self.register_buffer('At', measurement_matrix_A.t()) # A.t() is [N, M]

        # Debug: Print shapes of A and At upon initialization
        if self._is_first_batch_debug_run:
            print(f"[DEBUG Init] self.A shape: {self.A.shape}")
            print(f"[DEBUG Init] self.At shape: {self.At.shape}")

        self.prior_nets = nn.ModuleList()
        for _ in range(num_stages):
            self.prior_nets.append(
                ResPriorNet(
                    in_channels=prior_in_channels, 
                    out_channels=prior_out_channels,
                    hidden_channels=prior_hidden_channels, 
                    num_resblocks=prior_num_resblocks,
                    use_swish=True # Consistently use Swish or LeakyReLU
                )
            )

        self.learnable_step_sizes = learnable_step_sizes
        if learnable_step_sizes:
            # Initialize step sizes (lambda_k in ISTA, or 1/L in POGM)
            # Small initial values, positive
            self.step_size_params = nn.Parameter(torch.empty(num_stages))
            nn.init.uniform_(self.step_size_params, 1e-5, 1e-4) # Adjusted range, was 1e-5, 1e-4
        else:
            # Fixed step sizes (e.g., 1.0 or based on Lipschitz constant if known)
            # For simplicity, using a small constant if not learnable
            step_sizes_tensor = torch.full((num_stages,), 1e-4, dtype=torch.float32)
            self.register_buffer('step_size_scale', step_sizes_tensor)

        self.learnable_momentum = learnable_momentum
        if learnable_momentum:
            # Parameters for FISTA-like momentum (beta_k)
            # Initialize near zero for sigmoid to be around 0.5, or small for direct use
            self.momentum_params = nn.Parameter(torch.rand(self.num_stages) * 0.1) # Initialize momentum parameters (beta_k)
        
        self.final_activation = nn.Sigmoid() # To ensure output is in [0,1] range

    def _print_stats(self, tensor_name, tensor_val):
        if self._is_first_batch_debug_run:
            if tensor_val is None:
                print(f"[DEBUG Stage {self._current_debug_stage}] {tensor_name} is None")
                return
            if not isinstance(tensor_val, torch.Tensor):
                print(f"[DEBUG Stage {self._current_debug_stage}] {tensor_name} is not a tensor: {tensor_val}")
                return

            has_nan_flag = torch.isnan(tensor_val).any().item()
            has_inf_flag = torch.isinf(tensor_val).any().item()
            # Ensure tensor is on CPU for numpy operations if any, or handle potential errors for min/max/mean/std on empty or all-nan/inf tensors
            try:
                min_val = tensor_val.min().item()
                max_val = tensor_val.max().item()
                mean_val = tensor_val.mean().item()
                std_val = tensor_val.std().item()
            except RuntimeError: # Handles cases like all-NaN tensor for min/max
                min_val, max_val, mean_val, std_val = float('nan'), float('nan'), float('nan'), float('nan')

            print(f"[DEBUG Stage {self._current_debug_stage}] {tensor_name} stats: "
                  f"min={min_val:.4e}, max={max_val:.4e}, "
                  f"mean={mean_val:.4e}, std={std_val:.4e}, "
                  f"has_nan={has_nan_flag}, has_inf={has_inf_flag}, shape={tensor_val.shape}")

    def forward(self, y_measurements):
        B = y_measurements.shape[0] # Batch size
        # y_measurements shape: [B, M] (M is number of measurements)

        if self._is_first_batch_debug_run:
            self._current_debug_stage = 0 # Reset for each forward pass during debug

        # Initial estimate for x_k. Traditionally A^T @ y.
        # Old: x_k_flat = torch.matmul(y_measurements, self.At.T) # This was y @ A
        # Corrected to A^T @ y:
        # (self.At [N,M]) @ (y_measurements.transpose(-1,-2) [M,B]) -> [N,B] -> transpose -> [B,N]
        x_k_flat = torch.matmul(self.At, y_measurements.transpose(-1,-2)).transpose(-1,-2)

        if self._is_first_batch_debug_run:
            self._print_stats("Init y_measurements", y_measurements)
            self._print_stats("Init self.At", self.At)
            self._print_stats("Init x_k (A^T y)", x_k_flat)

        # For FISTA-like momentum
        t_k_plus_1 = torch.tensor(1.0, device=y_measurements.device)
        
        # y_k_iter is the point where the gradient (data consistency) is evaluated
        y_k_iter_flat = x_k_flat.clone() # Start with x_0

        if self._is_first_batch_debug_run:
            print(f"[DEBUG Stage 0 Init] y_measurements stats: min={y_measurements.min().item():.4e}, max={y_measurements.max().item():.4e}, mean={y_measurements.mean().item():.4e}, has_nan={'True' if torch.isnan(y_measurements).any() else 'False'}")
            print(f"[DEBUG Stage 0 Init] self.At stats: min={self.At.min().item():.4e}, max={self.At.max().item():.4e}, mean={self.At.mean().item():.4e}, has_nan={'True' if torch.isnan(self.At).any() else 'False'}")
            print(f"[DEBUG Stage 0 Init] x_k (A^T y) stats: min={x_k_flat.min().item():.4e}, max={x_k_flat.max().item():.4e}, mean={x_k_flat.mean().item():.4e}, has_nan={'True' if torch.isnan(x_k_flat).any() else 'False'}")

        # Iterations for unfolding
        for k in range(self.num_stages):
            if self._is_first_batch_debug_run:
                self._current_debug_stage = k

            # grad_calc_point is where we evaluate the gradient of the data term: A^T (A x - y)
            # For FISTA, this is y_k_iter_flat
            grad_calc_point_flat = y_k_iter_flat
            self._print_stats(f"grad_calc_point_flat (y_k_iter_flat)", grad_calc_point_flat)
            
            # CLAMP grad_calc_point_flat to prevent matmul overflow
            # Values observed were up to 1e4, which is too high for image estimates (typically [-1,1] or similar)
            # Let's try a clamp value. If NaN persists, this might need to be more aggressive or other normalization.
            CLIPPING_VAL_GRAD_POINT = 50.0 
            grad_calc_point_flat = torch.clamp(grad_calc_point_flat, min=-CLIPPING_VAL_GRAD_POINT, max=CLIPPING_VAL_GRAD_POINT)
            
            if self._is_first_batch_debug_run and k == 0:
                print(f"[DEBUG Stage {k}] grad_calc_point_flat (after clamp to +/-{CLIPPING_VAL_GRAD_POINT})")
                print(f"[DEBUG Stage {k}] Before Ay_k matmul")

            # Calculate A @ grad_calc_point_flat
            # Old: Ay_k = torch.matmul(self.A, grad_calc_point_flat) # Incorrect matmul [M,N] @ [B,N]
            # Corrected to A @ x (for each batch item):
            # (self.A [M,N]) @ (grad_calc_point_flat.transpose(-1,-2) [N,B]) -> [M,B] -> transpose -> [B,M]
            Ay_k = torch.matmul(self.A, grad_calc_point_flat.transpose(-1,-2)).transpose(-1,-2)

            self._print_stats(f"Ay_k (A@grad_calc_point)", Ay_k)

            # y_measurements_expanded_for_A = y_measurements.unsqueeze(-1) # Shape [B, M, 1] # No longer needed
            diff_Ay_y = Ay_k - y_measurements # Ay_k is [B,M], y_measurements is [B,M]

            if self._is_first_batch_debug_run:
                print(f"[DEBUG Stage {k}] (Ay_k - y_measurements) stats: min={diff_Ay_y.min().item():.4e}, max={diff_Ay_y.max().item():.4e}, mean={diff_Ay_y.mean().item():.4e}, has_nan={'True' if torch.isnan(diff_Ay_y).any() else 'False'}")

            # Calculate data consistency gradient: A^T @ (Ay_k - y_measurements)
            # Old: grad_data_term_flat = torch.matmul(diff_Ay_y.unsqueeze(1), self.At).squeeze(1) # Incorrect
            # Corrected to A^T @ diff (for each batch item):
            # (self.At [N,M]) @ (diff_Ay_y.transpose(-1,-2) [M,B]) -> [N,B] -> transpose -> [B,N]
            grad_data_term_flat = torch.matmul(self.At, diff_Ay_y.transpose(-1,-2)).transpose(-1,-2)
            
            grad_data_term_flat = grad_data_term_flat.squeeze(-1) # (B, N)
            self._print_stats(f"grad_data_term_flat (A^T(Ax-y))", grad_data_term_flat)

            # Learnable step size (lambda_k)
            # Sigmoid ensures positivity and a bounded range, then scaled.
            step_size_k_val = torch.sigmoid(self.step_size_params[k]) * 1e-4 # Scale to a smaller range initially
            if self._is_first_batch_debug_run:
                 print(f"[DEBUG Stage {k}] step_size_k_val: {step_size_k_val.item():.4e}")

            # Gradient descent update part
            x_intermediate_flat = y_k_iter_flat - step_size_k_val * grad_data_term_flat
            
            if self._is_first_batch_debug_run:
                print(f"[DEBUG Stage {k}] x_intermediate (before prior) stats: min={x_intermediate_flat.min().item():.4e}, max={x_intermediate_flat.max().item():.4e}, mean={x_intermediate_flat.mean().item():.4e}, has_nan={'True' if torch.isnan(x_intermediate_flat).any() else 'False'}")

            # Apply prior network (proximal mapping)
            # Reshape to image dimensions [B, C, H, W] for the prior network
            x_intermediate_img = x_intermediate_flat.view(B, 1, self.img_shape[0], self.img_shape[1])
            x_k_plus_1_img = self.prior_nets[k](x_intermediate_img)
            x_k_plus_1_flat = x_k_plus_1_img.view(B, -1) # Flatten back

            if self._is_first_batch_debug_run:
                 print(f"[DEBUG Stage {k}] x_k_plus_1 (after prior) stats: min={x_k_plus_1_flat.min().item():.4e}, max={x_k_plus_1_flat.max().item():.4e}, mean={x_k_plus_1_flat.mean().item():.4e}, has_nan={'True' if torch.isnan(x_k_plus_1_flat).any() else 'False'}")
            
            # FISTA-like momentum update
            if self.learnable_momentum:
                # beta_k_params are raw parameters, sigmoid maps them to (0,1)
                beta_k_learned = torch.sigmoid(self.momentum_params[k]) 
            else: # Original FISTA momentum
                t_k = t_k_plus_1
                t_k_plus_1 = (1 + torch.sqrt(1 + 4 * t_k**2)) / 2
                beta_k_learned = (t_k - 1) / t_k_plus_1
            
            if self._is_first_batch_debug_run:
                print(f"[DEBUG Stage {k}] beta_k_learned: {beta_k_learned.item():.4e}")

            # Check for NaN before momentum difference
            if self._is_first_batch_debug_run and (torch.isnan(x_k_plus_1_flat).any() or torch.isnan(x_k_flat).any()):
                print(f"[DEBUG Stage {k}] NaN in x_k_plus_1 or x_k before momentum diff.")

            momentum_term = x_k_plus_1_flat - x_k_flat # Can be large if x_k_plus_1_flat is very different
            y_k_iter_flat = x_k_plus_1_flat + beta_k_learned * momentum_term

            # Update x_k for the next iteration's momentum calculation
            x_k_flat = x_k_plus_1_flat

            if torch.isnan(y_k_iter_flat).any() or torch.isinf(y_k_iter_flat).any():
                print(f"NaN or Inf detected in y_k_iter at stage {k}!")
                if self._is_first_batch_debug_run: # If still in first batch debug, turn it off
                    self._is_first_batch_debug_run = False
                # Return a zero tensor or current x_k to prevent full crash, loss will be NaN/Inf
                return torch.zeros_like(x_k_flat.view(B, 1, self.img_shape[0], self.img_shape[1]))

        if self._is_first_batch_debug_run: # If successfully completed first batch debug
            self._is_first_batch_debug_run = False 
            
        # Final output reshaped to image dimensions
        final_reconstruction = x_k_flat.view(B, 1, self.img_shape[0], self.img_shape[1])
        return final_reconstruction

# --- Utility Classes and Functions (from original) ---
class SinglePixelMeasurementModel(nn.Module):
    def __init__(self, modulation_patterns):
        super(SinglePixelMeasurementModel, self).__init__()
        self.modulation_patterns = nn.Parameter(modulation_patterns, requires_grad=False)

    def forward(self, x): # Input x is an image tensor [B, C, H, W]
        x_flat = x.view(x.size(0), -1) # Flatten to [B, N]
        # Ensure patterns is [N, M] if measurement is y = x_flat * P.T
        # Or [M, N] if y = P * x_flat.T
        # Original was y = x_flat @ P.T, so P should be [M,N], P.T is [N,M]
        # modulation_patterns loaded is [M,N], so self.modulation_patterns.t() is [N,M]
        measurements = torch.matmul(x_flat, self.modulation_patterns.t())
        return measurements

def load_and_preprocess_image(image_path, size=(128, 128), device_to_use='cpu'):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)) # Optional: Normalize if network expects it
    ])
    image_tensor = transform(image)
    return image_tensor.to(device_to_use)

class ImageFolderDataset(data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, filename)
            for filename in os.listdir(image_dir)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    # Paths
    image_dir = "datasets/img_align_celeba"
    test_image_dir = "datasets/img_align_celeba"
    test_image_name = "127857.jpg" # Example image
    output_model_name = "deep_unroll_fista_net.pth"
    output_image_name = "deep_unroll_fista_reconstruction.png"

    # Device
    manual_seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seeding for reproducibility
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
        # torch.backends.cudnn.deterministic = True # Can slow down, use if needed
        # torch.backends.cudnn.benchmark = False   # Can slow down

    # Model Hyperparameters
    img_height, img_width = 128, 128
    N_pixels = img_height * img_width
    M_measurements = 1638 # Number of measurements

    num_stages = 10          # Number of unrolling stages
    prior_hidden_ch = 64    # Hidden channels in ResPriorNet
    prior_resblocks = 4     # Number of ResBlocks in each ResPriorNet
    learnable_steps = True
    learnable_momentum_flag = True # 保持FISTA特性

    # Training Hyperparameters
    batch_size = 16         # Adjust based on GPU memory (16, 32 are common)
    epochs = 400            # Total training epochs
    learning_rate = 1e-6    # Initial learning rate (1e-4 to 5e-5 often good for AdamW)
    weight_decay = 1e-5     # Weight decay for AdamW
    use_mixed_precision = True # Enable AMP
    gradient_clip_value = 1.0  # Max norm for gradient clipping

    # Early stopping
    patience = 50
    
    # --- Data Loading ---
    if not os.path.exists(image_dir): raise FileNotFoundError(f"Image dir {image_dir} not found.")
    if not os.path.exists(test_image_dir): raise FileNotFoundError(f"Test image dir {test_image_dir} not found.")

    mat_data = loadmat('Adiff.mat') 
    if 'A' not in mat_data: raise ValueError("Key 'A' not found in Adiff.mat.")
    A_matrix_full = mat_data['A']
    
    if A_matrix_full.shape[0] < M_measurements or A_matrix_full.shape[1] < N_pixels:
        raise ValueError(f"Adiff.mat 'A' shape {A_matrix_full.shape} is too small for M={M_measurements}, N={N_pixels}")
    
    A_matrix_raw = torch.tensor(A_matrix_full[:M_measurements, :N_pixels], dtype=torch.float32)
    
    # <--- 添加 A_matrix 规范化 --->
    max_abs_A = A_matrix_raw.abs().max()
    if max_abs_A > 1e-9: # Avoid division by zero or very small numbers
        A_matrix_norm = A_matrix_raw / max_abs_A
    else:
        A_matrix_norm = A_matrix_raw # Should not happen with typical matrices
    print(f"A_matrix original max absolute value: {max_abs_A.item()}")
    print(f"Normalized A_matrix_norm stats: min={A_matrix_norm.min():.4e}, max={A_matrix_norm.max():.4e}, mean={A_matrix_norm.mean():.4e}")
    # <--- 结束规范化 --->

    # Decide which A to use for measurement_op. 
    # For now, to keep things simple and focus on model stability,
    # let's assume the measurements y are generated with the *original* A,
    # and the model internally uses a *normalized* A.
    # This means y_measurements might have a different scale than if A_norm was used for generation.
    measurement_op = SinglePixelMeasurementModel(A_matrix_raw).to(device) # Use original A for measurement generation

    data_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)), # 稍加强数据增强
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # CelebA是人脸，颜色抖动可能帮助不大，但灰度图上亮度对比度可能有用
        transforms.ToTensor()
    ])
    full_dataset = ImageFolderDataset(image_dir, transform=data_transform)
    if len(full_dataset) == 0: raise ValueError("Dataset is empty. Check image_dir.")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = data.random_split(full_dataset, [train_size, val_size], 
                                                 generator=torch.Generator().manual_seed(manual_seed)) # For reproducible split

    num_workers = min(os.cpu_count() // 2, 4) if device.type == 'cuda' else 0 # 调整num_workers
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False, drop_last=True) # drop_last
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val images.")

    # --- Model, Loss, Optimizer ---
    model = DeepUnrollFISTANet(
        num_stages=num_stages,
        measurement_matrix_A=A_matrix_norm, # Pass the NORMALIZED A_matrix to the model
        img_shape=(img_height, img_width),
        prior_hidden_channels=prior_hidden_ch,
        prior_num_resblocks=prior_resblocks,
        learnable_step_sizes=learnable_steps,
        learnable_momentum=learnable_momentum_flag
    ).to(device)
    
    print(f"Model initialized: DeepUnrollFISTANet with {num_stages} stages.")
    # You can uncomment to print model summary if torchsummary is installed
    # from torchsummary import summary
    # summary(model, input_size=(M_measurements,), device=device.type)


    criterion_reconstruct = nn.L1Loss() 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99)) # betas微调
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.01) # eta_min相对当前lr
    
    # Initialize GradScaler for mixed precision
    # scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision and torch.cuda.is_available()) # Old
    # scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision and torch.cuda.is_available()) # Still old
    # scaler = torch.amp.GradScaler(device='cuda', enabled=use_mixed_precision and torch.cuda.is_available()) # Incorrect: use device_type
    scaler = torch.amp.GradScaler(device='cuda', enabled=use_mixed_precision and torch.cuda.is_available())

    if use_mixed_precision and torch.cuda.is_available():
        print("Using GradScaler for mixed precision.")
    
    # print("Initial lambda_k:", model.lambda_k.data)
    # print("Initial beta_k_params:", model.beta_k_params.data)


    # --- Training Loop ---
    best_val_loss = float('inf')
    best_model_state_dict = None
    trigger_times = 0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch_idx, original_images in enumerate(progress_bar):
            original_images = original_images.to(device)
            
            # Simulate measurements
            measurements_y = measurement_op(original_images)

            optimizer.zero_grad(set_to_none=True) # More efficient

            # Forward pass with Automatic Mixed Precision
            # with torch.cuda.amp.autocast(enabled=use_mixed_precision and torch.cuda.is_available()): # Old
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16 if use_mixed_precision else torch.float32, enabled=use_mixed_precision and torch.cuda.is_available()):
                reconstructed_images = model(measurements_y)
                loss = criterion_reconstruct(reconstructed_images, original_images)
            
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx+1}! Skipping batch.")
                # Potentially log more info or break if it happens часто
                # optimizer.zero_grad() # Clear any NaN gradients
                continue 

            scaler.scale(loss).backward()
            
            # Gradient Clipping
            if gradient_clip_value > 0:
                scaler.unscale_(optimizer) # Unscale before clipping
                nn_utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
        
        avg_epoch_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        scheduler.step()

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} Validation", leave=False)
            for original_images_val in val_progress_bar:
                original_images_val = original_images_val.to(device)
                measurements_y_val = measurement_op(original_images_val)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16 if use_mixed_precision else torch.float32, enabled=use_mixed_precision and torch.cuda.is_available()):
                    reconstructed_images_val = model(measurements_y_val)
                    val_loss_item = criterion_reconstruct(reconstructed_images_val, original_images_val)
                epoch_val_loss += val_loss_item.item()
        
        avg_epoch_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_train_loss:.5f}, Val Loss: {avg_epoch_val_loss:.5f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            trigger_times = 0
            print(f"    New best model saved with Val Loss: {best_val_loss:.5f}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': { 
                    'num_stages': num_stages, 'prior_hidden_channels': prior_hidden_ch,
                    'prior_num_resblocks': prior_resblocks, 'learnable_step_sizes': learnable_steps,
                    'learnable_momentum': learnable_momentum_flag, 'img_shape': (img_height, img_width),
                    'M_measurements': M_measurements, 'N_pixels': N_pixels
                }
            }, output_model_name)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} after {patience} epochs of no improvement.")
                break
        
        # Optional: Add a small delay if CUDA memory is an issue or for monitoring
        # if device.type == 'cuda': torch.cuda.empty_cache()
        # import time; time.sleep(0.1)

    print("Training finished.")

    # --- Testing ---
    if best_model_state_dict is not None:
        print(f"Loading best model for testing (Val Loss: {best_val_loss:.5f})...")
        # Need to re-initialize model with same architecture as saved if not loading from checkpoint dict directly
        # For simplicity, assume we can load the state dict into the current model instance
        model.load_state_dict(best_model_state_dict)
    else:
        print("No best model found to test. Testing with last state if available, or exiting.")
        # Optionally, try to load the last saved model if an error occurred mid-training
        if os.path.exists(output_model_name):
            try:
                checkpoint = torch.load(output_model_name, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded last saved model from '{output_model_name}' for testing.")
            except Exception as e:
                print(f"Could not load last model: {e}. Exiting testing.")
                exit()
        else:
            exit()


    model.eval()
    test_image_full_path = os.path.join(test_image_dir, test_image_name)
    if not os.path.exists(test_image_full_path):
        print(f"Test image {test_image_full_path} not found! Skipping test.")
    else:
        test_original_image = load_and_preprocess_image(test_image_full_path, 
                                                      size=(img_height, img_width), 
                                                      device_to_use=device).unsqueeze(0) # Add batch dim
        
        with torch.no_grad():
            test_measurements_y = measurement_op(test_original_image)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16 if use_mixed_precision else torch.float32, enabled=use_mixed_precision and torch.cuda.is_available()):
                test_reconstructed_image = model(test_measurements_y)

        # Convert to displayable format (numpy uint8)
        original_display = test_original_image.squeeze().cpu().numpy() * 255
        original_display = original_display.astype(np.uint8)
        
        reconstructed_display = test_reconstructed_image.clamp(0, 1).squeeze().cpu().numpy() * 255
        reconstructed_display = reconstructed_display.astype(np.uint8)

        # Calculate metrics
        mse_val = np.mean((original_display.astype(np.float32) - reconstructed_display.astype(np.float32)) ** 2)
        ssim_val = ssim(original_display, reconstructed_display, data_range=255)
        psnr_val = psnr(original_display, reconstructed_display, data_range=255)

        print(f"\nTest Results for {test_image_name}:")
        print(f"  MSE: {mse_val:.4f}")
        print(f"  SSIM: {ssim_val:.4f}")
        print(f"  PSNR: {psnr_val:.4f} dB")

        # Display images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_display, cmap='gray')
        plt.title(f"Original: {test_image_name}")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_display, cmap='gray')
        plt.title(f"Reconstructed (DeepUnrollFISTANet - {num_stages} stages)\nPSNR: {psnr_val:.2f} dB")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_image_name, bbox_inches='tight', pad_inches=0)
        print(f"Reconstruction saved to {output_image_name}")
        # plt.show() # Uncomment to display inline if running in an interactive environment