# EdaxShifu - Intelligent Edge AI Camera System

An AI-powered smart camera system built on the Seeed Studio reCamera platform, featuring real-time object detection, voice-controlled teaching capabilities, and adaptive learning through human feedback.

## 🎯 Project Vision

The system creates an intelligent feedback loop where:
1. RTSP stream provides real-time video feed
2. YOLO performs initial hand/object detection
3. When detection triggers, the system captures a photo
4. KNN classifier attempts to identify the object
5. On failed detection, Gemini API provides human annotation
6. Annotations feed back into the labeled dataset for continuous learning

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│    RSTP     │────▶│ (Yolo) hand-     │────▶│        KNN          │────▶│  GEMINI API  │
│             │     │  detection       │     │                     │     │              │
└─────────────┘     └──────────────────┘     └─────────────────────┘     └──────────────┘
                            │                           │                         │
                            │ detection is a           │                         │
                            │ hand holding             │                         │ Failed detection
                            │ something ==>            │                         │
                            │ Take a photo             │                         ▼
                            ▼                           │                  ┌──────────────────┐
                                                       │                  │ Human Annotation │
                                                       │                  └──────────────────┘
                                                       │                         │
                                                       │ Real-time              │ Formatting the
                                                       │ training               │ data
                                                       ▼                         ▼
                                                ┌─────────────────────┐
                                                │  Labeled DataSet    │
                                                └─────────────────────┘
```

## 🚀 Quick Start

### New Refactored Version
```bash
# Run with default RTSP URL
uv run main.py

# Run with custom camera URL
uv run main.py --url "rtsp://your-camera-url" --window-name "My Camera"
```

### Project Structure
```
edaxshifu/
├── main.py                 # Main entry point with CLI arguments
├── src/
│   └── rtsp_stream.py     # Refactored RTSP streaming classes
├── deprecated/            # Previous implementations (see below)
├── images/               # Training samples
├── snapshots/           # Captured frames (80+ images)
├── flows.json           # Node-RED visual programming
├── yolo11n.onnx        # YOLO v11 model
└── recamera-user-manual.pdf
```

## 🔧 Core Components

### 1. **RTSP Streaming Module** (`src/rtsp_stream.py`)
- `RTSPStream` class: Handles connection, frame reading, reconnection
- `RTSPViewer` class: Manages display and user interaction
- Auto-reconnection on stream failure
- Performance statistics tracking

### 2. **Deprecated Modules** (Historical Evolution)

#### **smart_camera.py** - Voice-Controlled AI Camera
The most sophisticated implementation featuring:
- **Voice Commands**: "This is a [object]" for teaching
- **ResNet18 + KNN**: Few-shot learning system
- **Whisper STT**: Speech recognition
- **Edge TTS**: Voice feedback
- **Real-time Recognition**: With confidence scoring

#### **demo.py** - CLI Interface
Command-line tool with modes:
- `teach`: Train new objects
- `detect`: Recognize objects
- `reset`: Clear learned objects
- `list`: Show known objects

#### **knn_test.py** - Classifier Testing
Standalone K-NN classifier using ResNet18 embeddings for fruit classification

#### **gemini_vision.py** - Cloud AI Integration
Google Gemini Vision API for single-word object descriptions

#### **preview_stream.py** - Remote Control Server
Flask server providing:
- HTTP endpoints for snapshots
- Audio recording from RTSP
- Remote trigger capabilities

#### **yolo_finetune.py** - Model Training
Template for fine-tuning YOLO models with Ultralytics

## 🎛️ Hardware Setup

### Seeed Studio reCamera
- **Default IP**: 192.168.42.1
- **RTSP URL**: rtsp://admin:admin@192.168.42.1:554/live
- **Node-RED**: http://192.168.42.1:1880
- **Web Interface**: http://192.168.42.1/#/workplace
- **Password**: asdf1234!

### Network Configuration
- **USB-C Connection**: CDC-NCM networking
- **Alternative Access**: http://192.168.86.28 (when on same network)

## 🤖 AI Models & Technologies

### Computer Vision
- **YOLO v11**: Real-time object detection (ONNX format)
- **ResNet18**: Feature extraction for KNN classifier
- **OpenCV**: Video processing and display

### Machine Learning
- **scikit-learn**: K-Nearest Neighbors classifier
- **PyTorch**: Deep learning framework
- **Google Gemini**: Vision API for annotations

### Audio & Voice
- **Whisper**: Speech-to-text recognition
- **edge-tts**: Text-to-speech synthesis
- **FFmpeg**: Audio/video processing

### Integration
- **Node-RED**: Visual flow programming
- **Flask**: HTTP API server
- **RTSP Protocol**: Real-time streaming

## 📊 Data Pipeline

1. **Capture**: RTSP stream from reCamera
2. **Detection**: YOLO identifies hands/objects
3. **Classification**: KNN attempts recognition
4. **Annotation**: Gemini provides labels on failure
5. **Training**: Real-time model updates
6. **Storage**: Labeled dataset accumulation

## 🎯 Use Cases

- **Smart Home Monitoring**: Object recognition and alerts
- **Educational Tool**: Teaching AI about new objects
- **Research Platform**: Few-shot learning experiments
- **Security System**: Person and object detection
- **Interactive Assistant**: Voice-controlled camera

## 📚 Model Conversion

For deploying custom models to reCamera:
https://wiki.seeedstudio.com/recamera_model_conversion#convert-and-quantize-ai-models-to-the-cvimodel-format

## 🔑 Original Setup Information

### Access Credentials
- **Password**: asdf1234!

### Direct Access URLs
- **Local Network Access**: http://192.168.86.28
- **reCamera Web Interface**: http://192.168.42.1/#/workplace

## 🔄 Development Timeline

The project shows evolution from simple RTSP testing to a sophisticated AI system:
1. Basic RTSP streaming tests
2. YOLO integration for object detection
3. KNN classifier for custom objects
4. Voice control implementation
5. Cloud AI integration (Gemini)
6. Continuous learning pipeline

## 📝 Recent Activity

- **August 2025**: Active development with 80+ test snapshots
- Ongoing experimentation with voice commands and object teaching
- Integration of multiple AI models for robust detection

## 🚦 Future Enhancements

Based on the architecture diagram, planned features include:
- Automated dataset labeling pipeline
- Real-time model retraining
- Improved hand gesture recognition
- Expanded object categories
- Enhanced human-in-the-loop annotation

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

---

*EdaxShifu combines edge AI with continuous learning to create an intelligent camera system that gets smarter with every interaction.*