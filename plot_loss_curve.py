import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块用于绘图
import torch  # 导入PyTorch库

# 主函数入口
if __name__ == '__main__':
    # 加载模型检查点（checkpoint）文件，此文件通常包含训练过程中保存的模型状态和其它有用信息。
    # 注意：这里需要根据实际情况更改pth文件路径。
    checkpoint_path = 'C:/Users/Lenovo/Desktop/ai-homework/checkpoint/b_s_4.pth'
    checkpoint = torch.load(checkpoint_path)  # 使用torch.load()加载.pth格式的模型检查点文件

    # 从检查点中提取损失值列表（loss_list），这通常是用来追踪模型训练过程中的损失变化情况。
    loss_list = checkpoint['loss_list']  # 假设'loss_list'是存储在检查点中的损失值列表的键名

    # 绘制损失曲线以观察模型的收敛情况
    plt.plot(loss_list)  # 绘制损失值列表，横轴为迭代次数，纵轴为损失值
    plt.xlabel('Iteration')  # 设置x轴标签为"Iteration"（迭代次数）
    plt.ylabel('Loss')  # 设置y轴标签为"Loss"（损失值）
    plt.title('Loss Curve')  # 设置图表标题为"Loss Curve"（损失曲线）
    plt.tight_layout()  # 调整图表各元素布局以防止重叠或裁剪
    plt.show()  # 显示绘制好的图表