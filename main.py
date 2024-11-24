from config.config import Config
from models.ddpm import DDPM
from models.unet import ContextUnet
from datasets.custom_dataset import CustomDataset
from trainer.trainer import DDPMTrainer
from torchvision import transforms

def main():
    config = Config()
    
    transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomDataset(config.data, transform=transform)
    
    context_unet = ContextUnet(config.model)
    ddpm = DDPM(config.model, context_unet)
    
    trainer = DDPMTrainer(config, ddpm, dataset)
    
    trainer.train()

if __name__ == "__main__":
    main()

