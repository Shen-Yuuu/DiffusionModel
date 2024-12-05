from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
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
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        # 嵌入处理
        cemb1 = self.contextembed1(c)
        temb1 = self.timeembed1(t)
        cemb2 = self.contextembed2(c)
        temb2 = self.timeembed2(t)
        
        # 调整维度
        cemb1 = cemb1.view(-1, self.n_feat * 2, 1, 1).expand(-1, -1, down2.shape[2], down2.shape[3])
        temb1 = temb1.view(-1, self.n_feat * 2, 1, 1).expand(-1, -1, down2.shape[2], down2.shape[3])
        cemb2 = cemb2.view(-1, self.n_feat, 1, 1).expand(-1, -1, down1.shape[2], down1.shape[3])
        temb2 = temb2.view(-1, self.n_feat, 1, 1).expand(-1, -1, down1.shape[2], down1.shape[3])

        up1 = self.up0(hiddenvec)
        up1 = F.interpolate(up1, size=down2.shape[2:])
        up2 = self.up1(cemb1*up1 + temb1, down2)
        up2 = F.interpolate(up2, size=down1.shape[2:])
        up3 = self.up2(cemb2*up2 + temb2, down1)
        up3 = F.interpolate(up3, size=x.shape[2:])
        out = self.out(torch.cat((up3, x), 1))
        
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
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0):
        """
        修改采样函数以确保维度匹配
        """
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        
        # 确保类别数量正确
        c_i = torch.arange(0, min(n_sample, self.nn_model.n_classes)).to(device)
        c_i = c_i.repeat(int(np.ceil(n_sample/len(c_i))))[:n_sample]
        
        # 创建上下文掩码
        context_mask = torch.zeros_like(c_i).to(device)
        
        # 双倍批次大小
        x_i_store = []
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1)  # 修改时间步的形状
            
            # 双倍批次
            x_i_double = x_i.repeat(2,1,1,1)
            t_is_double = t_is.repeat(2,1)
            c_i_double = c_i.repeat(2)
            context_mask_double = context_mask.repeat(2)
            context_mask_double[n_sample:] = 1.
            
            # 预测噪声
            eps = self.nn_model(x_i_double, c_i_double, t_is_double, context_mask_double)
            
            # 分离预测结果
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            
            # 更新采样
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

class DroneDataset(Dataset):
    def __init__(self, img_dir, anno_dir, target_size=128, transform=None):
        """
        参数:
            img_dir (str): 图像目录路径
            anno_dir (str): XML标注文件目录路径
            target_size (int): 输出图像的目标尺寸
            transform (callable, optional): 可选的图像转换
        """
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.target_size = target_size
        self.transform = transform
        
        # 获取所有图像和标注文件
        self.samples = []
        self.class_map = {}
        self.num_classes = 0
        
        # 扫描所有XML文件并提取样本
        for xml_file in os.listdir(anno_dir):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(anno_dir, xml_file)
                img_name = xml_file.replace('.xml', '.jpg')
                img_path = os.path.join(img_dir, img_name)
                
                if os.path.exists(img_path):
                    objects = self._parse_xml(xml_path)
                    for obj in objects:
                        # 将类别名称映射到数字索引
                        if obj['name'] not in self.class_map:
                            self.class_map[obj['name']] = self.num_classes
                            self.num_classes += 1
                        
                        self.samples.append({
                            'img_path': img_path,
                            'bbox': obj['bbox'],
                            'class': self.class_map[obj['name']]
                        })

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

    def _crop_and_resize(self, image, bbox, expand_ratio=1.5):
        """裁剪边界框并调整大小为正方形
        Args:
            image: PIL Image对象
            bbox: [xmin, ymin, xmax, ymax] 边界框坐标
            expand_ratio: 扩充系数，>1 表示扩大裁剪区域
        """
        # 扩展边界框为正方形
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
        
        # 裁剪图像
        cropped = image.crop((new_xmin, new_ymin, new_xmax, new_ymax))
        
        # 调整大小
        resized = cropped.resize((self.target_size, self.target_size), Image.LANCZOS)
        return resized

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取图像
        image = Image.open(sample['img_path']).convert('RGB')
        
        # 裁剪并调整大小
        image = self._crop_and_resize(image, sample['bbox'])
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, sample['class']
def train_drone():
    # 修改参数
    n_epoch = 200
    batch_size = 2  # 根据您的GPU内存调整
    n_T = 400
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_feat = 128
    lrate = 1e-4
    save_model = True
    save_dir = './data/diffusion_outputs_drone/'
    ws_test = [0.0, 2.0]
    image_size = 128  # 目标图像大小

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 数据预处理
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 创建数据集
    dataset = DroneDataset(
        img_dir='China_Drone/train/images',
        anno_dir='China_Drone/train/annotations/xmls',
        target_size=image_size,
        transform=tf
    )
    
    # 修改模型以支持RGB图像
    ddpm = DDPM(
        nn_model=ContextUnet(
            in_channels=3,  # 修改为3通道
            n_feat=n_feat,
            n_classes=dataset.num_classes
        ),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1
    )
    ddpm.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # 其余训练逻辑保持不变
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*dataset.num_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (3, image_size, image_size), device, guide_w=w)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(dataset.num_classes):
                    for j in range(int(n_sample/dataset.num_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*dataset.num_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep%5==0 or ep == int(n_epoch-1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/dataset.num_classes), ncols=dataset.num_classes,sharex=True,sharey=True,figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/dataset.num_classes)):
                            for col in range(dataset.num_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*dataset.num_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_drone() 