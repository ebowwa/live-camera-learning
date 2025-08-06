import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import pickle


class OnlineKNNClassifier:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = self._load_cnn_feature_extractor()
        self.transform = self._get_transform()
        self.embeddings = []
        self.labels = []
        self.knn = None  # Will initialize on first fit

    def _load_cnn_feature_extractor(self):
        model = resnet18(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # remove FC layer
        model.eval().to(self.device)
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _extract_embedding(self, img_path: str):
        image = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_tensor).squeeze().cpu().numpy()
        return embedding

    def add_sample(self, label: str, img_path: str):
        emb = self._extract_embedding(img_path)
        self.embeddings.append(emb)
        self.labels.append(label)
        self._fit_knn()

    def _fit_knn(self):
        if len(self.labels) >= 1:
            self.knn = KNeighborsClassifier(n_neighbors=1)
            self.knn.fit(self.embeddings, self.labels)

    def predict(self, img_path: str, threshold: float = 0.5):
        if not self.knn:
            return "Model not trained yet", {}
        emb = self._extract_embedding(img_path)
        pred = self.knn.predict([emb])[0]
        proba = self.knn.predict_proba([emb])[0]
        pred_conf = dict(zip(self.knn.classes_, proba))
        
        if pred_conf[pred] < threshold:
            return "unknown", pred_conf
        return pred, pred_conf


    def list_classes(self):
        return sorted(set(self.labels))

    def reset_model(self):
        self.embeddings = []
        self.labels = []
        self.knn = None


    def save_model(self, path="knn_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "labels": self.labels
            }, f)

    def load_model(self, path="knn_model.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.labels = data["labels"]
            self._fit_knn()
            
    def _extract_embedding_from_image(self, image: Image.Image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_tensor).squeeze().cpu().numpy()
        return embedding

    def add_sample_from_image(self, label: str, image: Image.Image):
        emb = self._extract_embedding_from_image(image)
        self.embeddings.append(emb)
        self.labels.append(label)
        self._fit_knn()

    def predict_from_image(self, image: Image.Image):
        emb = self._extract_embedding_from_image(image)
        pred = self.knn.predict([emb])[0]
        proba = self.knn.predict_proba([emb])[0]
        return pred, dict(zip(self.knn.classes_, proba))

if __name__ == "__main__":
    clf = OnlineKNNClassifier()
    # List learned labels
    print("Known classes:", clf.list_classes())

    # Teach the model
    clf.add_sample("banana", "./images/banana1.png")
    clf.add_sample("cat", "./images/cat1.png")

    # Teach the model
    pred, conf = clf.predict("./images/test_apple.png")
    print(f"Prediction: {pred}")
    print(f"Confidence: {conf}")

    clf.add_sample("apple", "./images/apple1.png")

    # Teach the model
    pred, conf = clf.predict("./images/test_apple.png")
    print(f"Prediction: {pred}")
    print(f"Confidence: {conf}")


