import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import os
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# ------------------------
# CONFIG
# ------------------------
# 你可以将这些图片换成你本地的图片路径
training_samples = {
    "banana": ["./images/banana1.png", "./images/banana2.png"],
    "cat": ["./images/cat1.png", "./images/cat2.png"],
    "apple": ["./images/apple1.png"],
}

test_image_path = "./images/test_banana.png"  # 用于测试的图片

# ------------------------
# SETUP MODEL
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 ResNet18 提特征（去掉最后分类层）
model = resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉最后一层fc
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

def extract_embedding(img_path):
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().cpu().numpy()
    return embedding

# ------------------------
# TRAIN KNN
# ------------------------
X, y = [], []
for label, paths in training_samples.items():
    for path in paths:
        emb = extract_embedding(path)
        X.append(emb)
        y.append(label)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# ------------------------
# TEST
# ------------------------
test_emb = extract_embedding(test_image_path)
pred = knn.predict([test_emb])[0]
proba = knn.predict_proba([test_emb])[0]

print(f"✅ Predicted: {pred}")
print(f"Confidence: {dict(zip(knn.classes_, proba))}")

