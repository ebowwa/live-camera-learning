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
cd python && uv run python main.py --url 0

# Run with RTSP camera
cd python && uv run python main.py --url "rtsp://your-camera-url"

# Examples:
cd python && uv run python main.py --url "rtsp://admin:admin@192.168.42.1:554/live"  # reCamera
cd python && uv run python main.py --url "rtsp://192.168.1.100:554/stream1"         # Generic IP cam
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
cd python && uv run python main.py --url 0
```

Without the API key, the system falls back to human-only annotation mode.


### Project Structure
```
edaxshifu/
├── python/               # Python implementation
│   ├── main.py           # Main entry point with CLI arguments
│   ├── api_server.py     # API server
│   ├── unified_interface.py  # Web interface
│   ├── edaxshifu/        # Core package
│   ├── examples/         # Python examples
│   ├── tests/            # Python tests
│   ├── deprecated/       # Previous implementations
│   ├── assets/           # Python-specific assets
│   │   ├── images/       # Training samples
│   │   └── snapshots/    # Captured frames (80+ images)
│   ├── data/             # Runtime data
│   │   ├── captures/     # Live capture data
│   │   ├── intelligent_captures/  # Intelligent capture metadata
│   │   └── flows.json    # Node-RED visual programming
│   └── models/           # KNN model storage
├── swift/                # Swift implementation
└── assets/               # Shared documentation
    └── recamera-user-manual.pdf
```



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
   - Failed recognitions saved to `python/data/captures/failed/`
   - Enhanced Gradio interface at http://localhost:7860 with:
     - 🤖 **AI Suggestions**: Gemini Vision provides instant annotations
     - 👤 **Human Override**: Users can accept/reject/modify AI suggestions
     - 📊 **Dual Statistics**: Track AI vs human annotation performance
     - 🔄 **Multiple Strategies**: Consensus, fallback, or weighted combinations

5. **Intelligent Continuous Learning**:
   - AI annotations provide fast initial labeling
   - Human annotations ensure high-quality training data
   - Model improves from both AI and human feedback
   - Dataset grows in `python/data/captures/dataset/` with source tracking
   - Real-time model updates after each annotation

### Directory Structure
```
python/data/captures/
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



---

*EdaxShifu combines edge AI with continuous learning to create an intelligent camera system that gets smarter with every interaction.*
