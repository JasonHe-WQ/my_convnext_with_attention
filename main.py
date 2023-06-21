import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from timm import create_model
import os
import numpy as np
import json
import matplotlib.pyplot as plt  # 引入matplotlib
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

torch.set_float32_matmul_precision('high')


# 准备数据集
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


def prepare_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize image
    ])

    train_dataset = MyDataset("./train", transform=transform)
    test_dataset = MyDataset("./test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def train_model(model, device, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 设置学习率热身
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    min_loss = np.inf
    losses_record = {}

    for epoch in range(1000):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 更新学习率
            scheduler.step(epoch + images.size(0) / len(train_loader))

        epoch_loss = running_loss / len(train_loader)
        losses_record[epoch + 1] = epoch_loss
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

        # 仅在损失达到新的最低值时保存模型状态
        if epoch_loss < min_loss:
            torch.save(model.state_dict(), f'./pthlib/best_model.pth')
            min_loss = epoch_loss

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

    with open("./train/losses_record.json", 'w') as f:
        json.dump(losses_record, f)

    # 绘制loss变化曲线
    plt.plot(list(losses_record.keys()), list(losses_record.values()))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


def eval_model(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model('convnext_xlarge', num_classes=7)
model = torch.compile(model)
model.to(device)

train_loader, test_loader = prepare_data()
train_model(model, device, train_loader)
eval_model(model, device, test_loader)
