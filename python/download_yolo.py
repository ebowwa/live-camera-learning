#!/usr/bin/env python3
"""Download YOLO model for the project."""

from ultralytics import YOLO
import os

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

print("Downloading YOLOv11 nano model...")

# Download and export YOLOv8 nano model (YOLOv11 might not be available yet)
model = YOLO('yolov8n.pt')  # This will auto-download if not present

# Export to ONNX format
print("Exporting to ONNX format...")
model.export(format='onnx', imgsz=640, simplify=True)

# Move the exported model to assets folder
import shutil
if os.path.exists('yolov8n.onnx'):
    shutil.move('yolov8n.onnx', 'assets/yolo11n.onnx')
    print("✅ Model saved to assets/yolo11n.onnx")
else:
    print("❌ Failed to export model")

print("\nYou can now run the application with:")
print("  cd python && uv run python main.py --url 0")
