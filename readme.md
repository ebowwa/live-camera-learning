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

### Main Application (Recommended)
```bash
# Run with webcam (most common)
uv run main.py --url 0

# Run with RTSP camera
uv run main.py --url "rtsp://your-camera-url"

# Examples:
uv run main.py --url "rtsp://admin:admin@192.168.42.1:554/live"  # reCamera
uv run main.py --url "rtsp://192.168.1.100:554/stream1"         # Generic IP cam
```

### Enhanced AI+Human Annotation Features
- **🤖 AI Suggestions**: Gemini Vision provides automatic annotations (requires GEMINI_API_KEY)
- **👤 Human Interface**: Web-based annotation at http://localhost:7860
- **📊 Dual Statistics**: Track AI vs human annotations
- **🔄 Real-time Learning**: Model updates immediately with new annotations

#### Setup AI Features (Optional)
```bash
# Set Gemini API key for AI suggestions
export GEMINI_API_KEY="your-gemini-api-key-here"

# Run with AI annotations enabled
uv run main.py --url 0
```

Without the API key, the system falls back to human-only annotation mode.

### Legacy Components (Alternative Methods)
```bash
# Run RTSP-focused version
uv run python run_rtsp.py --url "rtsp://your-camera-url"

# Run annotation interface separately
uv run annotate.py

# Using launcher script
./run_with_annotation.sh "rtsp://your-camera-url"
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

### AI Annotation System
- **Google Gemini Vision**: Automatic AI annotations with confidence scoring
- **Abstract Annotator Architecture**: Pluggable annotation system supporting:
  - Human annotation via web interface
  - AI-powered suggestions
  - Consensus between multiple annotators
  - Fallback strategies (AI → Human)
  - Weighted combinations of annotator results

### Audio & Voice
- **Whisper**: Speech-to-text recognition
- **edge-tts**: Text-to-speech synthesis
- **FFmpeg**: Audio/video processing

### Integration
- **Node-RED**: Visual flow programming
- **Flask**: HTTP API server
- **RTSP Protocol**: Real-time streaming

## 📊 Data Pipeline

### RTSP Integration Workflow
1. **RTSP Stream Input**: 
   - Connects to reCamera via `rtsp://admin:admin@192.168.42.1:554/live`
   - Falls back to webcam if RTSP unavailable
   - Handles reconnection automatically

2. **Real-time Detection**:
   - YOLO v11 processes each frame from RTSP stream
   - Identifies hands and objects in real-time
   - Triggers capture based on detection confidence

3. **Intelligent Classification**:
   - KNN classifier with ResNet18 embeddings
   - Attempts to recognize captured objects
   - Routes to success/failure paths based on confidence

4. **Enhanced AI+Human Annotation**:
   - Failed recognitions saved to `captures/failed/`
   - Enhanced Gradio interface at http://localhost:7860 with:
     - 🤖 **AI Suggestions**: Gemini Vision provides instant annotations
     - 👤 **Human Override**: Users can accept/reject/modify AI suggestions
     - 📊 **Dual Statistics**: Track AI vs human annotation performance
     - 🔄 **Multiple Strategies**: Consensus, fallback, or weighted combinations

5. **Intelligent Continuous Learning**:
   - AI annotations provide fast initial labeling
   - Human annotations ensure high-quality training data
   - Model improves from both AI and human feedback
   - Dataset grows in `captures/dataset/` with source tracking
   - Real-time model updates after each annotation

### Directory Structure
```
captures/
├── successful/    # Recognized objects
├── failed/        # Unknown objects for annotation
└── dataset/       # Growing labeled dataset
```

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