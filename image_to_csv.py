# -*- coding:utf-8 -*-
import csv
import os
import cv2

# 定义函数将图片数据转化为CSV数据

# 图片所在的根目录
IMG_DIR = r"C:/Users/Lenovo/Desktop/0"

def convert_img_to_csv(img_dir):
    """
    将指定目录下的所有图片转换为CSV格式的数据文件。
    
    参数:
        img_dir (str): 包含图片的根目录路径。
        
    返回:
        None: 此函数不返回任何值，它会创建一个CSV文件。
    """
    # 输出CSV文件的位置
    output_file_path = r'C:/Users/Lenovo/Desktop/0/train_data.csv'
    
    # 打开输出文件，使用'w'模式表示我们将写入文件，并且启用newline=''来避免在不同操作系统上产生额外的空行
    with open(output_file_path, 'w', newline='') as f:
        # CSV文件的列名初始化，首列为'label'用于存储标签信息
        column_name = ['label']
        
        # 设置column_name列表内容为[label1，label2，...，label64*64]
        column_name.extend('pixel%d' % i for i in range(64 * 64))
        
        # 创建CSV写入器对象
        writer = csv.writer(f)
        
        # 写入CSV文件的表头（即列名）
        writer.writerow(column_name)
 
        # 遍历10个不同的类别或标签
        for i in range(10):
            # 构建当前类别的图片所在子目录的路径
            img_file_path = os.path.join(img_dir, str(i))
            
            # 列出该子目录下所有的文件名（即图片名称）
            img_list = os.listdir(img_file_path)
            print(img_file_path)  # 打印当前处理的子目录
            
            # 遍历子目录下的每一张图片
            for img_name in img_list:
                # 构建完整的图片路径
                img_path = os.path.join(img_file_path, img_name)
                
                # 使用OpenCV读取图片，以灰度模式加载
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # 获取图片的高度和宽度（128*128）
                x, y = img.shape[0:2]
                
                # 将图片缩小为原来的一半（64*64）
                img_test1 = cv2.resize(img, (int(y / 2), int(x / 2)))

                # 初始化一行数据，首先添加标签信息
                image_data = [i]
                
                # 将图片展平为一维数组，并将其附加到image_data列表中
                image_data.extend(img_test1.flatten())
                
                # 写入一行数据到CSV文件
                writer.writerow(image_data)

if __name__ == "__main__":
    # 如果此脚本作为主程序运行，则调用convert_img_to_csv函数并传入图片目录路径
    convert_img_to_csv(IMG_DIR)