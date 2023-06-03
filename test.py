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
model = create_model('convnext_base', num_classes=7)
model.load_state_dict(torch.load('./pthlib/model_1000.pth'))  # 加载最后保存的权重文件

device = torch.device("cuda")
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

# 进行预测并保存结果
results = {}
for img_name in os.listdir('./cur'):
    img_path = os.path.join('./cur', img_name)
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        results[img_name] = int_to_class[predicted.item()]

# 将结果保存到 json 文件
with open('./jud/results.json', 'w') as f:
    json.dump(results, f)
