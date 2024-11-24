from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    in_channels: int = 3
    n_feat: int = 128
    n_classes: int = 7
    drop_prob: float = 0.1
    betas: tuple = (1e-4, 0.02)
    n_T: int = 1000
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainingConfig:
    n_epoch: int = 500
    batch_size: int = 1
    learning_rate: float = 2e-4
    save_dir: str = './data'
    ws_test: list = (0.0, 2.0)
    save_model: bool = True
    num_workers: int = 5
    device: str = "cuda:0"
    early_stopping_patience: int = 20
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

@dataclass
class DataConfig:
    project_dir: str = "./road-damage-detector-DatasetNinja"
    image_size: tuple = (512, 512)
    augmentation_params = {
        'brightness': 0.2,
        'contrast': 0.2,
        'rotation_degrees': 10,
        'flip_prob': 0.5
    }

class Config:
    model = ModelConfig()
    training = TrainingConfig()
    data = DataConfig()