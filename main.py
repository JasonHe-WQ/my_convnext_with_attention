import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from timm import create_model
import os
import numpy as np
import time

torch.set_float32_matmul_precision('high')


# 1. 准备数据集
class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = []
        self.labels = []
        self.transform = transform
        self.class_to_int = {
            "err1": 1,
            "err2": 2,
            "err3": 3,
            "err4": 4,
            "err5": 5,
            "err6": 6,
            "right": 0,
        }
        for class_name in self.class_to_int.keys():
            class_dir = os.path.join(root, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.paths.append(img_path)
                self.labels.append(self.class_to_int[class_name])

    def __getitem__(self, index):
        img_path = self.paths[index]
        label = self.labels[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.paths)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize image
])

train_dataset = MyDataset("./train", transform=transform)
test_dataset = MyDataset("./test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 创建模型实例
model = create_model('convnext_base', num_classes=7)
# model = torch.compile(model)

device = torch.device("cuda")
model.to(device)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


past_losses = []
# 4. 在训练数据上训练模型
for epoch in range(1000):  # 迭代1000轮
    if epoch % 100 == 0:
        start_time = time.time()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每100个epoch，保存模型参数
        if epoch % 100 == 99:
            torch.save(model.state_dict(),f'./pthlib/model_{epoch+1}.pth')

        # 如果past_losses列表已满，删除最旧的损失
        if len(past_losses) >= 5:
            past_losses.pop(0)

        # 将当前损失添加到past_losses列表
        past_losses.append(loss.item())

        # 如果past_losses列表已满，检查当前损失是否大于过去5个epoch的平均损失加3倍标准差
        if len(past_losses) == 5:
            mean_loss = np.mean(past_losses)
            std_loss = np.std(past_losses)
            if loss.item() > mean_loss + 3 * std_loss:
                if epoch >= 100:  # 确保有一个先前的模型状态可供加载
                    model.load_state_dict(torch.load(f'./pthlib/model_{epoch-10+1}.pth'))
        if epoch % 100 == 99:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'Time elapsed for 100 epochs: {elapsed_time} seconds.')
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 5. 对模型进行评估
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')
