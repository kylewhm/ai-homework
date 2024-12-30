import torch
import torch.nn.functional as F

#改进扩散模型代码（优于Denoising Diffusion Probabilistic Models），整个模型的大框架。

#三要素：路线、车、司机
#定义了Euler采样方法、分布变换路线、loss函数

class RectifiedFlow:
    # 车：图像生成一个迭代公式 ODE f(t+dt) = f(t) + dt*f'(t)
    def euler(self, x_t, v, dt):
        """ 使用欧拉方法计算下一个时间步长的值
            欧拉法求常微分方程，相当于可以用近似的方法求常微分方程的近似解。(几何意义：相当于用一条近似的折线去近似实际曲线)
        
        Args:
            x_t: 当前的一批图像的像素值，维度为 [B, C, H, W]
            v: 当前的像素值变化速度，维度为 [B, C, H, W]
            dt: 时间步长
        """
        x_t = x_t + v * dt

        return x_t

    # 路线
    def create_flow(self, x_1, t):
        """ 使用x_t = t * x_1 + (1 - t) * x_0公式构建x_0到x_1的流

            X_1是原始图像 X_0是噪声图像（服从标准高斯分布）
            
        Args:
            x_1: 原始图像，维度为 [B, C, H, W]，这里假设为[batch_size,单通道,28长,28宽]
            t: 一个标量，表示时间，时间范围为 [0, 1]，维度为 [B]
            
        Returns:
            x_t: 在时间t的图像，维度为 [B, C, H, W]
            x_0: 噪声图像，维度为 [B, C, H, W]
        
        """

        # 需要一个x0，x0服从高斯噪声
        x_0 = torch.randn_like(x_1)

        t = t[:, None, None, None]  # [B, 1, 1, 1]

        # 获得xt的值
        x_t = t * x_1 + (1 - t) * x_0

        return x_t, x_0

    # 司机
    def mse_loss(self, v, x_1, x_0):
        """ 计算RectifiedFlow的损失函数
        L = MSE(x_1 - x_0 - v(t))  匀速直线运动

        Args:
            v: 速度，维度为 [B, C, H, W]
            x_1: 原始图像，维度为 [B, C, H, W]
            x_0: 噪声图像，维度为 [B, C, H, W]
        """

        # 求loss函数，是一个MSE，最后维度是[B]

        loss = F.mse_loss(x_1 - x_0, v)

        return loss