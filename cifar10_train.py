import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# 设备配置（优先用GPU，没有则用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数设置
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

# 数据预处理与加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 搭建简单的卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    return test_loss, test_acc

# 主训练流程
if __name__ == "__main__":
    print("开始训练...")
    start_time = time.time()
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(EPOCHS):
        print(f"\n第 {epoch+1}/{EPOCHS} 轮训练")
        t_loss, t_acc = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(t_loss)
        train_accs.append(t_acc)

        te_loss, te_acc = test(model, test_loader, criterion, device)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        print(f"训练集损失: {t_loss:.4f}, 准确率: {t_acc:.2f}%")
        print(f"测试集损失: {te_loss:.4f}, 准确率: {te_acc:.2f}%")

    end_time = time.time()
    print(f"\n训练完成，总耗时: {end_time - start_time:.2f} 秒")

    # 训练结束后，先保存模型，再绘图显示！
    # 保存模型
    torch.save(model.state_dict(), r"D:\VScode\cifar10_simple_cnn.pth")
    print("✅ 模型已保存到 D:\\VScode\\cifar10_simple_cnn.pth")

    # 绘制准确率曲线（放到保存模型之后）
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.show()
# ====================== 下面是新加的：测试 + 画图 ======================
import matplotlib.pyplot as plt
import torchvision

# 1. 定义画图函数
def imshow(img):
    img = img / 2 + 0.5   # 反归一化
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.axis('off')

# 2. 拿一批测试图
dataiter = iter(test_loader)
images, labels = next(dataiter)

# 3. 预测
model.eval()
with torch.no_grad():
    outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

# 4. 显示图片和预测结果
plt.figure(figsize=(10, 4))
imshow(torchvision.utils.make_grid(images[:8]))
class_names = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
plt.title('Predicted: ' + ' '.join(f'{class_names[p]}' for p in predicted[:8]))
plt.savefig('test_result.png')
plt.show()

# 5. 输出测试集准确率
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'测试集准确率: {100 * correct / total:.2f}%')
# ====================== 新增：预测结果可视化 ======================
import matplotlib.pyplot as plt
import torchvision

# 1. 定义反归一化+画图函数
def imshow(img):
    img = img / 2 + 0.5  # 把归一化到[-1,1]的图片还原到[0,1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 把(C,H,W)转为(H,W,C)适配matplotlib
    plt.axis('off')  # 隐藏坐标轴

# 2. 从测试集取一批图片
dataiter = iter(test_loader)
images, labels = next(dataiter)

# 3. 用模型做预测
model.eval()
with torch.no_grad():
    outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

# 4. CIFAR-10类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 5. 画图+显示预测结果
plt.figure(figsize=(12, 6))
# 展示前8张测试图
imshow(torchvision.utils.make_grid(images[:8]))
# 标注预测结果
pred_labels = ' '.join(f'{classes[predicted[j]]}' for j in range(8))
plt.title(f'模型预测结果:\n{pred_labels}', fontsize=14)
# 保存图片到项目文件夹
plt.savefig('cifar10_pred_result.png', dpi=300, bbox_inches='tight')
print("✅ 预测结果图已保存为 cifar10_pred_result.png")
# 弹出图片窗口（可选，不想弹可以注释掉这行）
plt.show()