import torch
from torchvision.transforms import transforms
from timm import create_model
from PIL import Image
import os
import json

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize image
])

# 创建模型并加载权重
model = create_model('convnext_xlarge', num_classes=7)
model.load_state_dict(torch.load('./pthlib/best_model.pth'))  # 加载最后保存的权重文件

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 映射类别到名称
int_to_class = {
    0: "right",
    1: "err1",
    2: "err2",
    3: "err3",
    4: "err4",
    5: "err5",
    6: "err6"
}

# 获取图片列表
image_list = os.listdir('./cur')

# 创建数据集和数据加载器
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_names, transform=None):
        self.img_names = img_names
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join('./cur', img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return img_name, image

dataset = CustomDataset(image_list, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

# 进行预测并保存结果
results = {}
with torch.no_grad():
    for batch in data_loader:
        img_names, images = batch
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for img_name, pred in zip(img_names, predicted):
            results[img_name] = int_to_class[pred.item()]

# 将结果保存到 json 文件
with open('./jud/results.json', 'w') as f:
    json.dump(results, f)
