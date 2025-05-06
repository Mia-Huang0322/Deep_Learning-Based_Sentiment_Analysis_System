import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 读取训练集和验证集
train_dataset = datasets.ImageFolder(root="./AffectNet/train", transform=transform)
val_dataset = datasets.ImageFolder(root="./AffectNet/test", transform=transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(val_dataset.class_to_idx)
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
# from torch.utils.data import DataLoader
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import transforms
class ResNetModel(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetModel, self).__init__()
        # 加载预训练的 ResNet50
        self.resnet = models.resnet18(pretrained=False)
        # 修改全连接层输出为7类情感
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Step 6: 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# model = ResNetModel(num_classes=3).to(device)
# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)  # 使用L2正则化
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in train_loader:
        

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # print(outputs.argmax(1))
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {correct/len(train_dataset):.4f}")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        # for i in range(5):
        #     img = images[i].cpu().numpy().transpose(1, 2, 0)
        #     img = (img * 0.5) + 0.5  # 反归一化
        #     axes[i].imshow(img)
        #     axes[i].set_title(f"Pred: {predicted[i].item()} | Label: {labels[i].item()}")
        #     axes[i].axis("off")
        # plt.show()

print(correct,total)
test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

from PIL import Image
import cv2
# 读取图片
class ResNetModel(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetModel, self).__init__()
        # 加载预训练的 ResNet50
        self.resnet = models.resnet18(pretrained=False)
        # 修改全连接层输出为7类情感
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)



image_path = "C:/Users/30813\Desktop/f01f53f9dc169601e62ac6fbf61b881.jpg"  # 你的图片路径
image = cv2.imread(image_path)
model=torch.load('./sentiment_web/model-64-unpretrained.pth',map_location=torch.device('cpu'))
model.eval()

face_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
image = transform(face_pil).unsqueeze(0).to(device)  # 增加 batch 维度
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)  # 获取最大概率的类别索引

class_labels = ["happy", "sad","natrual"]  # 修改为你的实际类别
predicted_label = class_labels[predicted.item()]

print(f"预测结果: {predicted_label}")

torch.save(model, "sentiment_web/model-64-unpretrained.pth")

