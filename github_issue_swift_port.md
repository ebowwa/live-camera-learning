# Swift Port: Complete Python Main File Analysis & Architecture Mapping

## 🎯 Issue Summary

This issue documents the comprehensive analysis of the Python EdaxShifu system's main entry point (`main.py`) and all its dependencies to facilitate the Swift port development. It provides detailed mapping from Python modules to Swift equivalents and outlines the complete architecture translation.

## 📋 Python Main File Structure

### Entry Point Analysis
- **Root Entry**: `/main.py` (wrapper) → `/python/main.py` (actual implementation)
- **Main Function**: `main()` with unified argument parsing using `argparse`
- **Operation Modes**: 4 distinct modes (stream, detect, hand, intelligent)
- **Interface Selection**: CLI vs Web UI routing based on `--cli` flag

### Core Import Dependencies

#### Standard Library Imports → Swift Equivalents
```python
import argparse      # → ArgumentParser framework
import sys          # → Foundation (exit, arguments)  
import os           # → Foundation (FileManager)
import time         # → Foundation (Date, Timer)
import logging      # → os.log or custom logging
import threading    # → DispatchQueue, async/await
import json         # → Codable, JSONEncoder/Decoder
from datetime import datetime  # → Foundation (Date)
from typing import Optional, List, Dict, Callable  # → Swift optionals, arrays, dictionaries, closures
```

#### Computer Vision & ML Stack
```python
import cv2          # → Vision framework + AVFoundation
import numpy as np  # → Accelerate framework (vDSP, BNNS)
```

## 🏗️ Core EdaxShifu Modules Analysis

### 1. RTSP Streaming (`rtsp_stream.py`)
**Python Classes:**
- `RTSPStream`: Camera capture and stream management
- `RTSPViewer`: Basic video display with OpenCV

**Key Features:**
- RTSP URL handling with webcam fallback (`rtsp_url` or integer for webcam)
- Frame reading with automatic reconnection logic
- FPS tracking and performance statistics
- OpenCV-based window display

**Swift Translation:**
```swift
import AVFoundation
import SwiftUI

class CameraManager: ObservableObject {
    private var captureSession: AVCaptureSession
    private var videoOutput: AVCaptureVideoDataOutput
    // For RTSP: AVPlayer integration
    
    func connect() -> Bool
    func readFrame() -> CVPixelBuffer?
    func reconnect() -> Bool
    func getStats() -> CameraStats
}

struct CameraPreview: UIViewRepresentable {
    // AVCaptureVideoPreviewLayer wrapper
}
```

### 2. Object Detection (`integrated_detector.py`)
**Python Classes:**
- `IntegratedDetector`: YOLO + RTSP integration with auto-capture
- `HandObjectTrigger`: Advanced trigger logic for hand gestures

**Key Features:**
- YOLO model inference on video frames
- Hand-object detection with confidence thresholds
- Automatic capture with cooldown periods
- Metadata saving for captured frames

**Swift Translation:**
```swift
import Vision
import CoreML

class ObjectDetector: ObservableObject {
    private var yoloModel: VNCoreMLModel
    private var handPoseRequest: VNDetectHumanHandPoseRequest
    
    func detectObjects(in image: CVPixelBuffer) -> [VNRecognizedObjectObservation]
    func detectHandPoses(in image: CVPixelBuffer) -> [VNHumanHandPoseObservation]
    func shouldCapture(detections: [Detection]) -> Bool
}
```

### 3. Intelligent Capture System (`intelligent_capture.py`)
**Python Classes:**
- `IntelligentCaptureSystem`: Main pipeline orchestrator (595 lines)
- `ObjectRecognition`: Data structure for recognition results

**Complete Pipeline:**
```
RTSP → YOLO → Trigger → Capture → KNN → Success/Failure → Gemini/Dataset
```

**Key Features:**
- Complete intelligent feedback loop
- Live learning with automatic model reloading
- Multiple trigger types (keyboard, object detection)
- Statistics tracking and performance monitoring
- Gemini API integration for failed recognitions
- Real-time KNN classification updates

**Swift Translation:**
```swift
import Combine

class CaptureEngine: ObservableObject {
    private var cameraManager: CameraManager
    private var objectDetector: ObjectDetector
    private var knnClassifier: KNNClassifier
    private var triggerSystem: TriggerSystem
    
    @Published var stats: CaptureStats
    @Published var recognitions: [ObjectRecognition]
    
    func processFrame(_ frame: CVPixelBuffer) -> ProcessingResults
    func captureAndClassify(_ frame: CVPixelBuffer, detections: [Detection]) -> [ObjectRecognition]
}

struct ObjectRecognition {
    let yoloClass: String
    let yoloConfidence: Float
    let yoloBoundingBox: CGRect
    let knnResult: ClassificationResult
    let croppedImage: UIImage
}
```

## 🔧 Additional Core Dependencies

### KNN Classification
```python
from .knn_classifier import AdaptiveKNNClassifier, Recognition
```
**Features:** Adaptive learning, confidence thresholds, model persistence

**Swift Equivalent:**
```swift
import CoreML
import Accelerate

class KNNClassifier {
    private var model: MLModel?
    private var trainingData: [MLFeatureProvider]
    
    func predict(_ image: UIImage) -> ClassificationResult
    func addTrainingSample(_ image: UIImage, label: String)
    func saveModel() / loadModel()
}
```

### Trigger System
```python
from .trigger_system import TriggerManager, KeyboardTrigger, ObjectDetectionTrigger
```
**Features:** Multiple trigger types, cooldown management, event handling

**Swift Equivalent:**
```swift
import Combine

protocol Trigger {
    var publisher: AnyPublisher<TriggerEvent, Never> { get }
}

class TriggerSystem {
    private var triggers: [Trigger] = []
    private var cancellables = Set<AnyCancellable>()
    
    func addTrigger(_ trigger: Trigger)
    func checkAll(frame: UIImage, detections: [Detection]) -> [TriggerEvent]
}
```

### Live Learning System
```python
from .live_model_reloader import PollingModelReloader
```
**Features:** File system monitoring, automatic model reloading, polling-based updates

**Swift Equivalent:**
```swift
import Foundation

class ModelReloader {
    private var fileWatcher: DispatchSourceFileSystemObject?
    private let reloadCallback: () -> Void
    
    func startWatching(modelPath: URL)
    func stopWatching()
}
```

## 🎮 Operation Modes Translation

### 1. Stream Mode
**Python:** Basic RTSP streaming without detection
**Swift:** Simple camera preview with AVFoundation

### 2. Detection Mode
**Python:** YOLO detection with auto-capture
**Swift:** Vision framework + Core ML inference

### 3. Hand Mode  
**Python:** MediaPipe hand detection and gesture recognition
**Swift:** `VNDetectHumanHandPoseRequest` + custom gesture classification

### 4. Intelligent Mode (Default)
**Python:** Full pipeline with KNN classification and learning
**Swift:** Complete `CaptureEngine` with all components integrated

## 🖥️ User Interface Mapping

### CLI Mode (OpenCV Windows)
```python
cv2.imshow(window_name, frame)
key = cv2.waitKey(1) & 0xFF
```

**Swift Equivalent:**
```swift
// macOS: NSWindow with custom view
// iOS: Full-screen SwiftUI view
struct CameraView: View {
    @StateObject private var captureEngine = CaptureEngine()
    
    var body: some View {
        CameraPreview()
            .overlay(DetectionOverlay())
            .onTapGesture { captureEngine.manualCapture() }
            .onKeyPress(.space) { captureEngine.manualCapture() }
    }
}
```

### Web UI Mode (Gradio)
```python
from unified_interface import UnifiedEdaxShifu
app.launch(server_port=7860, share=False)
```

**Swift Equivalent:**
```swift
// Option 1: Pure SwiftUI app (recommended)
// Option 2: SwiftUI + embedded web server for browser access
import Swifter

class WebInterface {
    private let server = HttpServer()
    
    func startServer(port: Int) {
        server["/"] = { request in
            // Serve SwiftUI-generated interface
        }
    }
}
```

## 📁 Proposed Swift File Structure

```
swift/LiveLearningCamera/
├── LiveLearningCameraApp.swift     # App entry point (replaces main.py)
├── Core/
│   ├── CameraManager.swift         # RTSPStream equivalent
│   ├── ObjectDetector.swift        # YOLODetector + Vision
│   ├── CaptureEngine.swift         # IntelligentCaptureSystem
│   ├── KNNClassifier.swift         # AdaptiveKNNClassifier
│   ├── TriggerSystem.swift         # TriggerManager
│   └── ModelReloader.swift         # PollingModelReloader
├── UI/
│   ├── ContentView.swift           # Main interface
│   ├── CameraPreview.swift         # Video display
│   ├── AnnotationView.swift        # Labeling interface
│   └── StatsView.swift             # Performance monitoring
├── Models/
│   └── YOLOv11.mlmodel            # Converted from ONNX
└── Resources/
    ├── Info.plist                  # Camera permissions
    └── Assets.xcassets             # App assets
```

## 🔄 Data Flow Architecture

```
Camera Input → Object Detection → Trigger System → Capture → KNN Classification
     ↓              ↓                ↓              ↓            ↓
SwiftUI Preview → Vision Results → Combine Events → Core Data → Learning Updates
```

## 📦 Required Swift Dependencies

### Core Frameworks
```swift
import SwiftUI              // UI framework
import AVFoundation         // Camera capture  
import Vision               // Object detection
import CoreML               // Machine learning
import Combine              // Reactive programming
import ArgumentParser       // CLI arguments (macOS)
import Foundation           // File system, networking
```

### Potential Third-Party
```swift
import Alamofire           // HTTP networking (Gemini API)
import SwiftCSV            // Data export
import Charts              // Statistics visualization
```

## 🚀 Implementation Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Camera System: AVFoundation-based capture
- [x] Basic UI: SwiftUI camera preview  
- [ ] Object Detection: Vision framework integration
- [ ] Model Conversion: ONNX → Core ML

### Phase 2: ML Pipeline
- [ ] Core ML Integration: YOLO inference
- [ ] KNN Classifier: Custom or Core ML implementation
- [ ] Model Persistence: Local storage system
- [ ] Basic Recognition: Object classification

### Phase 3: Intelligent Features  
- [ ] Trigger System: Combine-based reactive triggers
- [ ] Live Learning: File watching and model updates
- [ ] Annotation Interface: SwiftUI labeling UI
- [ ] Statistics Tracking: Performance metrics

### Phase 4: Advanced Features
- [ ] Hand Detection: Vision framework hand poses
- [ ] Gesture Recognition: Custom gesture classification  
- [ ] API Integration: Gemini API Swift client
- [ ] Export/Import: Model and data sharing

## ⚠️ Key Challenges & Solutions

### 1. ONNX Model Conversion
**Challenge:** Python uses `yolo11n.onnx` model
**Solution:** Convert to Core ML using `coremltools` or find equivalent models

### 2. NumPy Array Handling  
**Challenge:** Heavy NumPy usage for image processing
**Solution:** Use Accelerate framework or `MLMultiArray`

### 3. OpenCV Dependencies
**Challenge:** Extensive OpenCV usage for image operations
**Solution:** Replace with Vision framework and Core Graphics

### 4. Real-time Performance
**Challenge:** Maintaining 30+ FPS inference
**Solution:** Optimize Core ML models, use GPU acceleration

### 5. Live Learning Implementation
**Challenge:** Dynamic model updates during runtime
**Solution:** File system monitoring + Core ML model reloading

## 📊 Current Status

- **Python Analysis**: ✅ Complete
- **Swift Architecture**: ✅ Designed  
- **iOS App Structure**: ✅ Basic implementation exists
- **Core ML Integration**: 🔄 In progress
- **Full Pipeline**: ❌ Not implemented

## 🎯 Next Steps

1. **Convert YOLO Model**: ONNX → Core ML conversion
2. **Implement CameraManager**: AVFoundation integration
3. **Build ObjectDetector**: Vision + Core ML inference
4. **Create TriggerSystem**: Combine-based reactive system
5. **Develop KNNClassifier**: On-device learning implementation

## 📚 References

- **Python Main File**: `/python/main.py` (420 lines)
- **Core Modules**: `/python/edaxshifu/` directory
- **Swift Implementation**: `/swift/LiveLearningCamera/` directory
- **Documentation**: This analysis document

---

**Labels:** `enhancement`, `swift`, `ios`, `macos`, `computer-vision`, `machine-learning`, `architecture`
**Assignees:** @ebowwa
**Projects:** Swift Port Development
