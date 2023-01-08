import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from timm import create_model
import os


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

train_dataset = MyDataset("/home/convnext/train", transform=transform)
test_dataset = MyDataset("/home/convnext/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 创建模型实例
model = create_model('convnext_base', num_classes=7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 在训练数据上训练模型
for epoch in range(100):  # 迭代100轮
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
