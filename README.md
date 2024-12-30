# ai-homework
## 代码说明
* 代码基于MNIST数据集实现改进版的扩散模型算法的模型训练与推理，可实现有条件或无条件生成水果，目前有条件生成仅支持使用类别label
* 数据集为自己的数据集
* 模型结构为改进MiniUnet，也可以使用其他更复杂的模型，比如Unet、DiT等。

部分代码参考来源:
Double童发发 
通义千问2.5

部分代码实现原理参考论文:
Denoising Diffusion Probabilistic Models
Classifier-Free Diffusion Guidance
Flow Matching for Generative Modeling
Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

代码环境：
代码环境要求很低，甚至不需要GPU都可以, 如果没有安装Cuba训练环境建议使用cpu训练和推理
  Python 3.8+
  Pytorch 2.0+ 
  Numpy
  Matplotlib
  opencv-python
  其他的就缺啥装啥

代码运行方式:
  * 如果需要训练代码请务必先查看train_config.yaml文件，并根据实际情况修改相关参数，是否使用GPU等，设置好了再开始训练
  * 训练：`python train.py` 注意设置相关参数(在train_config.yaml文件设置),没有安装Cuba训练环境则用cpu代替Cuba,此外在主函数有一个地方要自己改路径
  * 推理(生成图片)：`python infer.py` 没有安装Cuba训练环境则用cpu代替Cuba,此外在主函数有八个地方要自己改路径
  * 画loss曲线：`python plot_loss_curve.py`在主函数有一个地方要自己改路径
  
其他：
  * `__pycache__`不用管
  * `checkpoint`存放训练好的模型的文件夹
  * `results`存放模型生成图片的文件夹
  * `model.py`改进miniunet神经网络模块
  * `rectified_flow.py`改进扩散模型算法
  * `read_csv.py`从csv训练集中提取训练数据模块
  * `train_config.yaml`训练参数配置
  * `train-data.csv`csv训练数据集（以标签和单像素点数值为存储方式）
  * `image_to_csv.py`灰度图片数据转csv训练数据集程序
