# smart_camera.py

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2

import sounddevice as sd
import scipy.io.wavfile as wavfile
import tempfile
import os
import asyncio
import subprocess

# at the very top of smart_camera.py, before `import whisper`:
import sys
import types
# create a dummy timing module so whisper.timing import is skipped
sys.modules['whisper.timing'] = types.SimpleNamespace(add_word_timestamps=lambda *a, **k: None)

import whisper


# RTSP CAMERA URL
RTSP_URL = "rtsp://admin:admin@192.168.86.28:554/live"

# ---------- CNN + KNN ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final FC
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

X_train, y_train = [], []
knn = None

def extract_embedding(image: np.ndarray) -> np.ndarray:
    image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(torch.from_numpy(image_pil).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(image_tensor).squeeze().cpu().numpy()
    return emb

def reset_model():
    global X_train, y_train, knn
    X_train, y_train = [], []
    knn = None
    print("✅ Model reset.")

def list_known_objects():
    return list(set(y_train))

# ---------- CAMERA ----------
def capture_image() -> np.ndarray:
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError("❌ Failed to open RTSP stream.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("❌ Failed to read frame.")
    return frame

# ---------- TTS ----------
async def speak(text: str):
    cmd = ["edge-tts", "--text", text, "--write-media", "output.mp3"]
    await asyncio.create_subprocess_exec(*cmd)
    subprocess.run(["ffplay", "-nodisp", "-autoexit", "output.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ---------- Whisper ----------
def record_and_transcribe(seconds=3) -> str:
    print("🎤 Listening...")
    fs = 16000
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, fs, recording)
        audio_path = f.name

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="en")
    os.remove(audio_path)
    print(f"📝 Transcript: {result['text']}")
    return result['text']

# ---------- Main Teaching ----------
def teach_object():
    text = record_and_transcribe()
    label = extract_label(text)
    if not label:
        print("cannot find the label")
        return

    image = capture_image()
    emb = extract_embedding(image)

    X_train.append(emb)
    y_train.append(label)

    global knn
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    print(f"✅ I've learnt: {label}")
    asyncio.run(speak(f"i remember now. this is {label}"))

def extract_label(text: str) -> str:
    if "is" in text:
        return text.split("this is")[-1].strip()
    return text.strip()

# ---------- Detection ----------
def detect_once():
    if not knn:
        print("I haven't learnt anything")
        return

    image = capture_image()
    emb = extract_embedding(image)
    pred = knn.predict([emb])[0]
    proba = knn.predict_proba([emb])[0].max()

    print(f"🔍 I have seen {pred}, confidence:{proba:.2f}")
    if proba > 0.6:
        asyncio.run(speak(f"I'm seeing {pred}"))
    else:
        asyncio.run(speak(f"I'm not sure what is this"))

