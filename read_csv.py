import numpy as np
import torch

#从csv训练集中提取训练数据并转换为指定格式输出

def read_mnist(file_name):                              
    with open(file_name, encoding='utf-8') as f:        
        lines = f.readlines()                           #使用 with open 语句打开传入的数据集，并以UTF-8编码读取所有行到 lines 列表中
        
    # 跳过标题行（如果有）
    if lines and lines[0].startswith('label'):          #检查 lines 是否有东西且检查lines第一行是否以字符串 "label" 开头。
        lines = lines[1:]                               #如果都符合则跳过第一行（通常是列名）

    rows = len(lines)                                   #计算 lines 中的行数，并将其存储在变量 rows 中
    
    # 初始化空列表来存储像素值和标签
    pixels = []
    labels = []

    for line in lines:                                  #遍历 lines 中的每一行：
        parts = line.strip().split(',')                 #去除每行首尾的空白字符，并按逗号分割成多个部分，存储在 parts 列表中。
        
        if len(parts) < 2:
            raise ValueError("Line does not contain enough elements.")      #检查 parts 的长度是否至少为2。如果不足2个元素，则抛出一个 ValueError 异常，提示该行数据不完整。

        label = int(parts[0])                                               #将 parts 的第一个元素（即标签）转换为整数并存储在 label 变量中。
        # 转换像素值为整数，并忽略非数字项
        pixel_values = [int(item) for item in parts[1:] if item.isdigit()]  #对于 parts 中从第二个元素开始的所有元素，尝试将其转换为整数，并只保留那些可以成功转换的值（即忽略非数字项），结果存储在 pixel_values 列表中

        if pixel_values:                                                    #如果 pixel_values 不为空，则将 pixel_values 添加到 pixels 列表中，并将 label 添加到 labels 列表中。
            pixels.append(pixel_values)
            labels.append(label)
    
    images_array = np.array(pixels)                                         #将 pixels 和 labels 列表转换为NumPy数组 images_array 和 labels_array。
    labels_array = np.array(labels)
    
    images_array = images_array.reshape(rows, -1)                           #将 images_array 重塑为 (rows, 784) 形状，假设每个图像有784个像素（例如28x28的灰度图像）。
    
    images_tensor = torch.from_numpy(images_array).float() / 255.0
    labels_tensor = torch.from_numpy(labels_array)                          #将 images_array 转换为PyTorch张量，并将其归一化到 [0, 1] 区间。将 labels_array 转换为PyTorch张量。
    
    return images_tensor, labels_tensor                                     #返回处理后的图像张量和标签张量。