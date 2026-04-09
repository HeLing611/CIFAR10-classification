# CIFAR-10 图像分类实验

## 一、项目简介
本项目基于 PyTorch 深度学习框架，搭建简单卷积神经网络（CNN），完成 CIFAR-10 数据集的图像分类任务。实现了从数据加载、模型训练、测试评估到结果可视化的完整流程，并支持模型保存与继续训练。

## 二、运行环境
- Python 3.x
- PyTorch / torchvision
- matplotlib
- numpy

## 三、环境依赖
- Python 3.8+
- PyTorch、torchvision、matplotlib
- 安装命令：`pip install torch torchvision matplotlib`

## 三、项目结构

├── model.py      # 卷积神经网络模型定义
├── train.py      # 训练、测试、预测可视化主程序
├── cifar10_model.pth  # 训练好的模型权重
├── test_result.png    # 测试集预测结果图
└── README.md         # 项目说明文档

## 四、运行方法
1. 训练模型
在终端进入项目文件夹，执行：
python train.py
程序会自动下载数据集、开始训练，并输出每轮训练损失。
2. 继续训练（断点续训）
如需加载已训练好的模型继续训练，在 train.py 中添加：
python
运行
model.load_state_dict(torch.load('cifar10_model.pth'))
再次运行即可接着之前的进度训练。
3. 测试与可视化
训练完成后，程序自动在测试集上评估，并输出测试集准确率，同时保存预测结果图 test_result.png。

## 五、模型结构
采用两层卷积 + 两层全连接的 CNN 结构：
卷积层 + ReLU + 最大池化
卷积层 + ReLU + 最大池化
全连接层 + ReLU
输出层（10 分类）
优化器：Adam
损失函数：交叉熵损失 CrossEntropyLoss
## 六、实验结果
训练完成后模型保存为 cifar10_model.pth
测试集准确率可在控制台查看
生成预测结果图 test_result.png，直观展示模型分类效果
## 七、使用说明
首次运行会自动下载 CIFAR-10 数据集到 data 文件夹
训练过程可通过调整 EPOCHS、BATCH_SIZE、学习率等参数优化效果
运行前确保终端路径与代码文件路径一致

「作者」→ 贺玲玲
「日期」→ 2026.4.7
「准确率」→ 98.48%