import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchvision
from PIL import Image
import os

class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, pred, target, weighted = 1.0):
        loss = self._loss(pred, target)
        weighted_loss = (loss * weighted).mean()
        return weighted_loss
    
class WeightedL1Loss(WeightedLoss):
    def _loss(self, pred, target):
        return torch.abs(pred - target)
    
class WeightedL2Loss(WeightedLoss):
    def _loss(self, pred, target):
        return F.mse_loss(pred, target, reduction="none")
    
Losses = {
    "l1": WeightedL1Loss,
    "l2": WeightedL2Loss,
}

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class SinusoidalPosEmb(nn.Module):
    #位置编码
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim=16):
        super(MLP, self).__init__()

        self.device = device
        self.t_dim = t_dim
        self.a_dim = action_dim

        input_dim = state_dim + action_dim + t_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        self.final_layer = nn.Linear(hidden_dim, action_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, time, state):
        t_emb = self.time_mlp(time)
        x = torch.cat([x, state, t_emb], dim=-1)

        x = self.mid_layer(x)
        x = self.final_layer(x)

        return x

class Diffusion(nn.Module):
    def __init__(self, loss_type, beta_schedule="linear", clip_denoised=True, predict_epsilon=True, **kwargs):
        super(Diffusion, self).__init__()
        self.state_dim = kwargs["obs_dim"]
        self.action_dim = kwargs["act_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.device = torch.device(kwargs["device"])
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.T = kwargs["T"]
        self.model = MLP(self.state_dim, self.action_dim, self.hidden_dim, self.device)

        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, self.T, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))

        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    def q_posterior(self, x_start, x, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x, t, pred_noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x 
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise)

    def p_mean_variance(self, x, t, state):
        pred_noise = self.model(x, t, state)
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        x_recon = x_recon.clamp_(-1, 1)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)

        return model_mean, posterior_log_variance
        
    def p_sample(self, x, t, state):
        b, *_, device = *x.shape, x.device
        model_mean, model_log_variance = self.p_mean_variance(x, t, state)
        noise = torch.randn_like(x)

        nonzero_mask = (1-(t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state, shape, *args, **kwargs):
        device = self.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device, requires_grad=False)

        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, state)

        return x

    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = [batch_size, self.action_dim]
        action = self.p_sample_loop(state, shape, *args, **kwargs)

        return action.clamp_(-1, 1)

    def q_sample(self, x_start, t, noise):
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weight=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        x_recon = self.model(x_noisy, t, state)

        loss = self.loss_fn(x_recon, noise, weight)

    def loss(self, x, state, weight=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()

        return self.p_losses(x, state, t, weight)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)
    

