import torch
import torch.nn as nn

#神经网络MiniUnet框架搭建

# MiniUnet MNIST 28*28 4090 3G左右显存
class DownLayer(nn.Module):
    """MiniUnet的下采样层，基于ResNet架构设计。
    
    该层用于特征图的下采样，并结合时间编码信息（time embedding）进行处理。
    特征图通过两个卷积层和批标准化层处理，并在必要时应用残差连接。
    如果设置了downsample参数，则会在最后对输出进行最大池化操作以减小空间尺寸。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        time_emb_dim (int, optional): 时间编码维度，默认是16。
        downsample (bool, optional): 是否执行下采样操作，默认不执行。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim=16,
                 downsample=False):
        super(DownLayer, self).__init__()

        # 第一个卷积层，保持输入大小不变
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 第二个卷积层，同样保持输入大小不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 批量归一化层，加速训练并稳定网络
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 激活函数ReLU，引入非线性
        self.act = nn.ReLU()

        # 线性变换层，将时间编码转换为与输入通道数匹配的形状
        self.fc = nn.Linear(time_emb_dim, in_channels)

        # 如果输入和输出通道数不同，则创建一个1x1卷积层作为捷径连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

        # 下采样标志位和对应的池化层
        self.downsample = downsample
        if downsample:
            # 最大池化层，用来减小特征图的空间尺寸
            self.pool = nn.MaxPool2d(2)

        self.in_channels = in_channels

    def forward(self, x, temb):
        """前向传播方法
        
        Args:
            x (Tensor): 输入张量，形状为 [B, C, H, W]。
            temb (Tensor): 时间编码张量，形状为 [B, dim]。
        
        Returns:
            Tensor: 处理后的输出张量。
        """
        res = x  # 保存输入作为残差分支
        # 将时间编码通过线性层转换后加到输入上，增加时间信息
        x += self.fc(temb)[:, :, None, None]  # 扩展时间编码至 [B, in_channels, 1, 1]
        # 第一次卷积 + 批量归一化 + 激活
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        # 第二次卷积 + 批量归一化 + 激活
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        # 如果存在捷径连接，则应用于残差分支
        if self.shortcut is not None:
            res = self.shortcut(res)

        # 将主分支结果与残差分支相加
        x = x + res

        # 如果需要下采样，则应用最大池化
        if self.downsample:
            x = self.pool(x)

        return x


class UpLayer(nn.Module):
    """MiniUnet的上采样层。
    
    该层用于特征图的上采样，并结合时间编码信息（time embedding）进行处理。
    特征图通过两个卷积层和批标准化层处理，并在必要时应用残差连接。
    如果设置了upsample参数，则会在开始对输入进行上采样操作以增大空间尺寸。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        time_emb_dim (int, optional): 时间编码维度，默认是16。
        upsample (bool, optional): 是否执行上采样操作，默认不执行。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim=16,
                 upsample=False):
        super(UpLayer, self).__init__()

        # 第一个卷积层，保持输入大小不变
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 第二个卷积层，同样保持输入大小不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 批量归一化层，加速训练并稳定网络
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 激活函数ReLU，引入非线性
        self.act = nn.ReLU()

        # 线性变换层，将时间编码转换为与输入通道数匹配的形状
        self.fc = nn.Linear(time_emb_dim, in_channels)

        # 如果输入和输出通道数不同，则创建一个1x1卷积层作为捷径连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

        # 上采样标志位和对应的上采样层
        self.upsample_layer = nn.Upsample(scale_factor=2) if upsample else None

    def forward(self, x, temb):
        """前向传播方法
        
        Args:
            x (Tensor): 输入张量，形状为 [B, C, H, W]。
            temb (Tensor): 时间编码张量，形状为 [B, dim]。
        
        Returns:
            Tensor: 处理后的输出张量。
        """
        # 如果需要上采样，则先应用上采样层增大特征图的空间尺寸
        if self.upsample_layer is not None:
            x = self.upsample_layer(x)

        res = x  # 保存输入作为残差分支
        # 将时间编码通过线性层转换后加到输入上，增加时间信息
        x += self.fc(temb)[:, :, None, None]  # 扩展时间编码至 [B, in_channels, 1, 1]
        # 第一次卷积 + 批量归一化 + 激活
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        # 第二次卷积 + 批量归一化 + 激活
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        # 如果存在捷径连接，则应用于残差分支
        if self.shortcut is not None:
            res = self.shortcut(res)

        # 将主分支结果与残差分支相加
        x = x + res

        return x


class MiddleLayer(nn.Module):
    """MiniUnet的中间层。
    
    该层位于网络的中部，用于处理特征图而不改变其空间尺寸，
    同时结合时间编码信息（time embedding）进行处理。
    特征图通过两个卷积层和批标准化层处理，并应用残差连接。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        time_emb_dim (int, optional): 时间编码维度，默认是16。
    """

    def __init__(self, in_channels, out_channels, time_emb_dim=16):
        super(MiddleLayer, self).__init__()

        # 第一个卷积层，保持输入大小不变
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 第二个卷积层，同样保持输入大小不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 批量归一化层，加速训练并稳定网络
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 激活函数ReLU，引入非线性
        self.act = nn.ReLU()

        # 线性变换层，将时间编码转换为与输入通道数匹配的形状
        self.fc = nn.Linear(time_emb_dim, in_channels)

        # 如果输入和输出通道数不同，则创建一个1x1卷积层作为捷径连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x, temb):
        """前向传播方法
        
        Args:
            x (Tensor): 输入张量，形状为 [B, C, H, W]。
            temb (Tensor): 时间编码张量，形状为 [B, dim]。
        
        Returns:
            Tensor: 处理后的输出张量。
        """
        res = x  # 保存输入作为残差分支
        # 将时间编码通过线性层转换后加到输入上，增加时间信息
        x += self.fc(temb)[:, :, None, None]  # 扩展时间编码至 [B, in_channels, 1, 1]
        # 第一次卷积 + 批量归一化 + 激活
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        # 第二次卷积 + 批量归一化 + 激活
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        # 如果存在捷径连接，则应用于主分支而不是残差分支
        if self.shortcut is not None:
            x = self.shortcut(x)

        # 将主分支结果与原始输入相加（注意这里的修改：原代码中的残差连接应该作用于未经修改的输入）
        x = x + res

        return x


class MiniUnet(nn.Module):
    """采用MiniUnet架构，对MNIST数据进行生成任务。
    
    该网络结构包含两个下采样block、一个中间block和两个上采样block。
    它结合时间编码（time embedding）和标签编码（label embedding），使得模型能够感知到输入的时间信息和类别条件。

    Args:
        base_channels (int, optional): 基础通道数，默认是32。
        time_emb_dim (int, optional): 时间编码维度，默认与基础通道数相同。
        num_layers_per_block (int, optional): 每个block中的层数，默认是3。
    """

    def __init__(self, base_channels=32, time_emb_dim=None, num_layers_per_block=3):
        super(MiniUnet, self).__init__()

        if time_emb_dim is None:
            self.time_emb_dim = base_channels
        else:
            self.time_emb_dim = time_emb_dim

        self.base_channels = base_channels

        # 输入卷积层，将单通道的MNIST图像转换为基础通道数
        self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        # 下采样模块1
        self.down1 = nn.ModuleList([
            DownLayer(
                base_channels if i == 0 else base_channels * 2,
                base_channels * 2,
                time_emb_dim=self.time_emb_dim,
                downsample=False) for i in range(num_layers_per_block)
        ])
        self.maxpool1 = nn.MaxPool2d(2)

        # 下采样模块2
        self.down2 = nn.ModuleList([
            DownLayer(
                base_channels * 2 if i == 0 else base_channels * 4,
                base_channels * 4,
                time_emb_dim=self.time_emb_dim,
                downsample=False) for i in range(num_layers_per_block)
        ])
        self.maxpool2 = nn.MaxPool2d(2)

        # 中间模块
        self.middle = nn.ModuleList([
            MiddleLayer(
                base_channels * 4,
                base_channels * 4,
                time_emb_dim=self.time_emb_dim) for _ in range(num_layers_per_block)
        ])

        # 上采样模块1
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.up1 = nn.ModuleList([
            UpLayer(
                base_channels * 8 if i == 0 else base_channels * 2,  # concat
                base_channels * 2,
                time_emb_dim=self.time_emb_dim,
                upsample=False) for i in range(num_layers_per_block)
        ])

        # 上采样模块2
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.up2 = nn.ModuleList([
            UpLayer(
                base_channels * 4 if i == 0 else base_channels,
                base_channels,
                time_emb_dim=self.time_emb_dim,
                upsample=False) for i in range(num_layers_per_block)
        ])

        # 输出卷积层，将特征图转换回单通道
        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)

    def time_emb(self, t, dim):
        """对时间进行正弦函数的编码，使模型能够感知到输入x_t的时刻t。
        
        Args:
            t (float): 时间，维度为[B]
            dim (int): 编码的维度

        Returns:
            torch.Tensor: 编码后的时间，维度为[B, dim]
        """
        t = t * 1000
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def label_emb(self, y, dim):
        """对类别标签进行编码，同样采用正弦编码。
        
        Args:
            y (torch.Tensor): 图像标签，维度为[B] 或者 [B, L] 对于文本
            dim (int): 编码的维度

        Returns:
            torch.Tensor: 编码后的标签，维度为[B, dim]
        """
        y = y * 1000
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(y.device)
        sin_emb = torch.sin(y[:, None] / freqs)
        cos_emb = torch.cos(y[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def forward(self, x, t, y=None):
        """前向传播函数
        
        Args:
            x (torch.Tensor): 输入数据，维度为[B, C, H, W]
            t (torch.Tensor): 时间，维度为[B]
            y (torch.Tensor, optional): 数据标签或文本，维度为[B] 或 [B, L]。默认为None。
        """
        x = self.conv_in(x)
        temb = self.time_emb(t, self.base_channels)

        if y is not None:
            if len(y.shape) == 1:
                yemb = self.label_emb(y, self.base_channels)
                yemb[y == -1] = 0.0  # 不对-1进行编码
                temb += yemb
            else:
                pass  # 文字版本暂不支持

        # 下采样阶段
        for layer in self.down1:
            x = layer(x, temb)
        x1 = x
        x = self.maxpool1(x)

        for layer in self.down2:
            x = layer(x, temb)
        x2 = x
        x = self.maxpool2(x)

        # 中间阶段
        for layer in self.middle:
            x = layer(x, temb)

        # 上采样阶段
        x = torch.cat([self.upsample1(x), x2], dim=1)
        for layer in self.up1:
            x = layer(x, temb)

        x = torch.cat([self.upsample2(x), x1], dim=1)
        for layer in self.up2:
            x = layer(x, temb)

        x = self.conv_out(x)
        return x
