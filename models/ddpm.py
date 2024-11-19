import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np
from config.config import ModelConfig

def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    生成DDPM采样所需的预计算调度
    参数:
        beta1: 初始beta值
        beta2: 最终beta值
        T: 时间步数
    返回:
        包含各种预计算值的字典
    """
    assert beta1 < beta2 < 1.0, "beta1 和 beta2 必须在 (0, 1) 范围内"

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
    def __init__(
        self, 
        config: ModelConfig,
        nn_model: nn.Module,
    ):
        """
        初始化 DDPM 模型
        参数:
            config: 模型配置对象
            nn_model: 神经网络模型（ContextUnet）
        """
        super().__init__()
        self.nn_model = nn_model.to(config.device)  #从配置中获取device

        # 注册 DDPM 调度参数
        for k, v in ddpm_schedules(config.betas[0], config.betas[1], config.n_T).items():
            self.register_buffer(k, v)

        self.n_T = config.n_T
        self.device = config.device
        self.drop_prob = config.drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        训练时的前向传播
        参数:
            x: 输入图像
            c: 条件标签
        返回:
            损失值
        """
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        # 随机丢弃上下文
        context_mask = torch.bernoulli(
            torch.zeros_like(c[:,0]) + self.drop_prob
        ).to(self.device)

        # 返回预测噪声和实际噪声之间的MSE损失
        return self.loss_mse(
            noise, 
            self.nn_model(x_t, c, _ts / self.n_T, context_mask)
        )

    def sample(self, n_sample: int, size: Tuple[int, ...], device: str,
               target_labels: Optional[torch.Tensor] = None,
               guide_w: float = 0.0) -> Tuple[torch.Tensor, np.ndarray]:
        """
        生成样本
        参数:
            n_sample: 样本数量
            size: 样本大小
            device: 设备
            target_labels: 目标标签
            guide_w: 引导权重
        返回:
            生成的样本和采样过程中的中间状态
        """
        x_i = torch.randn(n_sample, *size).to(device)
        
        if target_labels is None:
            target_labels = torch.randint(0, 2, (n_sample, self.n_classes)).float().to(device)
        
        context_mask = torch.zeros(n_sample).to(device)
        
        # 双倍批次大小用于分类器引导
        x_i_double = torch.cat([x_i, x_i])
        c_i_double = torch.cat([target_labels, target_labels])
        context_mask_double = torch.cat([context_mask, torch.ones_like(context_mask)])

        x_i_store = []

        for i in range(self.n_T, 0, -2):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is_double = t_is.repeat(2*n_sample)

            # 生成噪声
            z = torch.randn_like(x_i_double) if i > 1 else 0

            # 预测并应用分类器引导
            eps = self.nn_model(x_i_double, c_i_double, t_is_double, context_mask_double)
            eps1, eps2 = eps.chunk(2)
            eps = (1 + guide_w) * eps1 - guide_w * eps2

            # 更新样本
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z[:n_sample]
            )
            
            # 更新双倍批次
            x_i_double = torch.cat([x_i, x_i])

            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store