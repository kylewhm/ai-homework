import torch
import random
from model import MiniUnet
from rectified_flow import RectifiedFlow
import cv2
import os

#用模型生成图片

def infer(
        checkpoint_path,   # 模型路径
        base_channels=64,  # 基础通道数应与训练时设置的保持一致
        step=50,           # 采样步数，决定了Euler方法迭代次数，影响生成质量
        num_imgs=5,        # 每次推理想要生成的图片数量
        y=None,            # 条件变量，用于有条件生成；如果为None，则进行无条件生成
        cfg_scale=7.0,     # Classifier-free Guidance缩放因子，控制生成图像的条件符合程度
        save_path='./results',  # 图像保存路径
        device='cuda'):    # 推理设备，默认是GPU ('cuda'),否则用CPU（'cpu'）
    """flow matching模型推理函数

    Args:
        参数已在函数签名中详细说明。
    """

    # 创建保存结果的目录，如果不存在的话
    os.makedirs(save_path, exist_ok=True)

    # 如果提供了条件y，检查其形状是否正确，并根据需要重复以匹配num_imgs
    if y is not None:
        assert len(y.shape) == 1 or len(y.shape) == 2, '条件张量y必须是一维或二维'
        assert y.shape[0] == num_imgs or y.shape[0] == 1, '条件张量的第一个维度大小必须等于num_imgs或为1'
        if y.shape[0] == 1:
            # 如果条件只有一个，则重复它来为每一张图提供相同的条件
            y = y.repeat(num_imgs, 1).reshape(num_imgs)
        # 将条件移动到指定的设备上
        y = y.to(device)

    # 实例化模型，并加载预训练权重
    model = MiniUnet(base_channels=base_channels)
    model.to(device)
    model.eval()  # 设置模型为评估模式

    # 加载RectifiedFlow实例，用于实现Euler法更新
    rf = RectifiedFlow()

    # 加载检查点并更新模型参数
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # 关闭梯度计算以节省显存和加速推理过程
    with torch.no_grad():
        for i in range(num_imgs):
            print(f'正在生成第{i}张图像...')
            dt = 1.0 / step  # Euler法的时间间隔

            # 初始x_t是标准高斯噪声，作为初始输入
            x_t = torch.randn(1, 1, 64, 64).to(device)

            # 对于有条件生成，提取对应的条件
            if y is not None:
                y_i = y[i].unsqueeze(0)

            # 进行step步的Euler法迭代
            for j in range(step):
                if j % 10 == 0:
                    print(f'生成第{i}张图像，步骤{j}...')

                t = j * dt
                t = torch.tensor([t]).to(device)

                # 根据是否有条件以及cfg_scale应用classifier-free guidance
                if y is not None:
                    v_pred_uncond = model(x=x_t, t=t)  # 无条件预测
                    v_pred_cond = model(x=x_t, t=t, y=y_i)  # 有条件预测
                    v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)  # 应用CFG
                else:
                    v_pred = model(x=x_t, t=t)  # 无条件生成

                # 使用Euler法更新x_t到下一个时间点
                x_t = rf.euler(x_t, v_pred, dt)

            # 最终输出的x_t转换成图像格式并保存
            x_t = x_t[0].clamp(0, 1)  # 归一化到[0, 1]
            img = x_t.detach().cpu().numpy()[0] * 255  # 转换为numpy数组并缩放到[0, 255]
            img = img.astype('uint8')  # 转换为8位整数类型
            cv2.imwrite(os.path.join(save_path, f'{i}.png'), img)  # 保存图像




if __name__ == '__main__':
    # 用户交互入口点。当此脚本作为主程序运行时，以下代码块会被执行。

    # 提示用户输入想要的图片生成方式，并读取输入。
    label = int(input("请输入要生成图片的方式（0-3）(0为指定生成、1为顺序生成、2为随机生成、3为逻辑随机生成)："))
    
    # 检查用户输入是否在合法范围内，如果不是，则抛出异常。
    if label < 0 or label > 3:
        raise ValueError("标签必须在0到3之间")

    # 根据用户选择的生成方式调用 infer 函数

    # 方式0: 指定生成图片
    if(label == 0):
        # 用户指定要生成的图片的类别标签，并进行合法性检查。
        labels = int(input("请输入要生成的图片的标签（0-9）(0为苹果、1为香蕉、2为胡萝卜、3为葡萄、4为橙子、5为桃子、6为梨、7为菠萝、8为草莓、9为西瓜): "))
        if labels < 0 or labels > 9:
            raise ValueError("标签必须在0到9之间")
        
        # 创建一个包含相同标签的列表，用于生成10张相同类别的图片。
        y = [labels] * 10
        
        # 调用infer函数，使用指定条件y生成图像，并保存到指定路径。
        infer(checkpoint_path='C:/Users/Lenovo/Desktop/ai-homework/checkpoint/b_s_4.pth',
              base_channels=128,
              step=50,
              num_imgs=10,
              y=torch.tensor(y),
              cfg_scale=9.0,
              save_path='C:/Users/Lenovo/Desktop/ai-homework/results/infer_fig',
              device='cuda')

    # 方式1: 顺序生成图片
    elif(label == 1):
        # 对于每一个可能的标签值，生成3张对应的图像，总共生成30张图片。
        y = []
        for i in range(10):
            y.extend([i] * 3)
        
        # 调用infer函数，使用顺序条件y生成图像，并保存到指定路径。
        infer(checkpoint_path='C:/Users/Lenovo/Desktop/ai-homework/checkpoint/b_s_4.pth',
              base_channels=128,
              step=50,
              num_imgs=30,
              y=torch.tensor(y),
              cfg_scale=9.0,
              save_path='C:/Users/Lenovo/Desktop/ai-homework/results/fig30',
              device='cuda')

    # 方式2: 随机生成图片
    elif(label == 2):
        # 不提供条件y，即无条件生成，生成的图片不基于任何特定的标签。
        y = None
        
        # 调用infer函数，无条件地生成30张随机图像，并保存到指定路径。
        infer(checkpoint_path='C:/Users/Lenovo/Desktop/ai-homework/checkpoint/b_s_4.pth',
              base_channels=128,
              step=50,
              num_imgs=30,
              y=y,
              cfg_scale=9.0,
              save_path='C:/Users/Lenovo/Desktop/ai-homework/results/fig_random',
              device='cuda')

    # 方式3: 逻辑随机生成图片
    elif(label == 3):
        import random  # 导入random模块，以便可以使用随机选择功能。
        
        # 使用随机选择逻辑，从0到9中随机选取30个标签，用于有条件地生成图像。
        y = [random.choice(list(range(10))) for _ in range(30)]
        
        # 调用infer函数，使用随机条件y生成图像，并保存到指定路径。
        infer(checkpoint_path='C:/Users/Lenovo/Desktop/ai-homework/checkpoint/b_s_4.pth',
              base_channels=128,
              step=50,
              num_imgs=30,
              y=torch.tensor(y),
              cfg_scale=9.0,
              save_path='C:/Users/Lenovo/Desktop/ai-homework/results/fig_logic_random',
              device='cuda')