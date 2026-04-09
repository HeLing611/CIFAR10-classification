import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 从训练文件中导入模型结构（注意文件名要和你的训练脚本一致）
from cifar10_train import SimpleCNN
 
# ---------------------- 配置部分 ----------------------
# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理（和训练时保持一致）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ---------------------- 加载模型 ----------------------
# 加载训练好的模型
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('cifar10_simple_cnn.pth', map_location=device))
model.eval()  # 切换到评估模式，关闭 dropout 和 batch norm

# ---------------------- 加载测试数据 ----------------------
test_dataset = datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=False, 
    transform=transform
)

# ---------------------- 随机测试函数 ----------------------
def predict_random_image():
    # 随机选一张图
    idx = random.randint(0, len(test_dataset)-1)
    img, label = test_dataset[idx]
    
    # 模型预测
    with torch.no_grad():
        img_tensor = img.unsqueeze(0).to(device)  # 增加 batch 维度
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item() * 100

    # 显示图片和结果
    plt.figure(figsize=(6, 6))
    # 反归一化，让图片恢复正常显示
    img = img * 0.5 + 0.5
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'真实类别: {classes[label]}\n预测类别: {classes[predicted.item()]} (置信度: {confidence:.2f}%)')
    plt.show()

    print(f"✅ 预测结果: {classes[predicted.item()]} | 真实答案: {classes[label]}")
    print(f"📊 模型置信度: {confidence:.2f}%")
    print("-" * 30)

# ---------------------- 运行测试 ----------------------
if __name__ == "__main__":
    print("开始随机测试模型...")
    print("按提示查看图片，关闭图片窗口即可继续下一次测试")
    print("输入 'exit' 可退出程序\n")
    
    while True:
        predict_random_image()
        user_input = input("按回车继续测试下一张，输入 'exit' 退出：")
        if user_input.lower() == 'exit':
            break