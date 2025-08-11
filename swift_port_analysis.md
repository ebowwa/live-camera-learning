# Swift Port Analysis: Python Main File and Dependencies

## Overview

This document provides a comprehensive analysis of the Python EdaxShifu system's main entry point and all its dependencies, mapped to Swift equivalents for the iOS/macOS port.

## Main Entry Point Structure

### Root Files
- **Entry Point**: `/main.py` (wrapper) → `/python/main.py` (actual implementation)
- **Main Function**: `main()` with unified argument parsing
- **Interface Modes**: CLI vs Web UI routing based on `--cli` flag

### Core Imports Analysis

#### Standard Library Dependencies
```python
import argparse      # → Swift: ArgumentParser framework
import sys          # → Swift: Foundation (exit, arguments)
import os           # → Swift: Foundation (FileManager)
import time         # → Swift: Foundation (Date, Timer)
import logging      # → Swift: os.log or custom logging
import threading    # → Swift: DispatchQueue, async/await
import json         # → Swift: Codable, JSONEncoder/Decoder
from datetime import datetime  # → Swift: Foundation (Date)
from typing import Optional, List, Dict, Callable  # → Swift: Optional, Array, Dictionary, closures
```

#### Computer Vision & ML Stack
```python
import cv2          # → Swift: Vision framework + AVFoundation
import numpy as np  # → Swift: Accelerate framework (vDSP, BNNS)
```

## EdaxShifu Core Modules

### 1. RTSP Streaming (`rtsp_stream.py`)
**Python Classes:**
- `RTSPStream`: Camera capture and stream management
- `RTSPViewer`: Basic video display

**Key Features:**
- RTSP URL handling with webcam fallback
- Frame reading with reconnection logic
- FPS tracking and statistics
- OpenCV-based display

**Swift Equivalents:**
```swift
// Camera capture
import AVFoundation
class CameraManager {
    private var captureSession: AVCaptureSession
    private var videoOutput: AVCaptureVideoDataOutput
    // RTSP support via AVPlayer
}

// Display
import SwiftUI
struct CameraPreview: UIViewRepresentable {
    // AVCaptureVideoPreviewLayer wrapper
}
```

### 2. Object Detection (`integrated_detector.py`)
**Python Classes:**
- `IntegratedDetector`: YOLO + RTSP integration
- `HandObjectTrigger`: Advanced trigger logic for hand gestures

**Key Features:**
- YOLO model inference
- Hand-object detection
- Automatic capture with cooldown
- Metadata saving

**Swift Equivalents:**
```swift
import Vision
import CoreML

class ObjectDetector {
    private var yoloModel: VNCoreMLModel
    private var handPoseRequest: VNDetectHumanHandPoseRequest
    
    func detectObjects(in image: CVPixelBuffer) -> [VNRecognizedObjectObservation]
    func detectHandPoses(in image: CVPixelBuffer) -> [VNHumanHandPoseObservation]
}
```

### 3. Intelligent Capture System (`intelligent_capture.py`)
**Python Classes:**
- `IntelligentCaptureSystem`: Main pipeline orchestrator
- `ObjectRecognition`: Recognition result data structure

**Key Features:**
- Complete RTSP → YOLO → Trigger → Capture → KNN pipeline
- Live learning with model reloading
- Statistics tracking
- Gemini API integration (optional)
- Multiple trigger types (keyboard, object detection)

**Swift Equivalents:**
```swift
import Combine

class CaptureEngine: ObservableObject {
    private var cameraManager: CameraManager
    private var objectDetector: ObjectDetector
    private var knnClassifier: KNNClassifier
    private var triggerSystem: TriggerSystem
    
    @Published var stats: CaptureStats
    @Published var recognitions: [ObjectRecognition]
}

struct ObjectRecognition {
    let yoloClass: String
    let yoloConfidence: Float
    let yoloBoundingBox: CGRect
    let knnResult: ClassificationResult
    let croppedImage: UIImage
}
```

## Additional Core Dependencies

### KNN Classification
```python
from .knn_classifier import AdaptiveKNNClassifier, Recognition
```
**Swift Equivalent:**
```swift
import CoreML
import Accelerate

class KNNClassifier {
    private var model: MLModel?
    private var trainingData: [MLFeatureProvider]
    
    func predict(_ image: UIImage) -> ClassificationResult
    func addTrainingSample(_ image: UIImage, label: String)
    func saveModel()
    func loadModel()
}
```

### Trigger System
```python
from .trigger_system import TriggerManager, KeyboardTrigger, ObjectDetectionTrigger
```
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

### Live Learning
```python
from .live_model_reloader import PollingModelReloader
```
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

## Operation Modes

### 1. Stream Mode
- Basic RTSP streaming without detection
- **Swift**: Simple camera preview with AVFoundation

### 2. Detection Mode  
- YOLO detection with auto-capture
- **Swift**: Vision framework + Core ML inference

### 3. Hand Mode
- Hand detection and gesture recognition
- **Swift**: VNDetectHumanHandPoseRequest + custom gesture classification

### 4. Intelligent Mode (Default)
- Full pipeline with KNN classification and learning
- **Swift**: Complete CaptureEngine with all components

## User Interface Mapping

### CLI Mode (OpenCV Windows)
```python
cv2.imshow(window_name, frame)
cv2.waitKey(1)
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
    }
}
```

### Web UI Mode (Gradio)
```python
from unified_interface import UnifiedEdaxShifu
app.launch(server_port=7860)
```
**Swift Equivalent:**
```swift
// Option 1: Pure SwiftUI app
// Option 2: SwiftUI + embedded web server
import Swifter

class WebInterface {
    private let server = HttpServer()
    
    func startServer(port: Int) {
        server["/"] = { request in
            // Serve SwiftUI-generated HTML or native interface
        }
    }
}
```

## Data Flow Architecture

```
Camera Input → Object Detection → Trigger System → Capture → KNN Classification
     ↓              ↓                ↓              ↓            ↓
SwiftUI Preview → Vision Results → Combine Events → Core Data → Learning Updates
```

## File Structure Mapping

### Python Structure
```
python/
├── main.py                    # Main entry point
├── edaxshifu/
│   ├── rtsp_stream.py        # Camera handling
│   ├── integrated_detector.py # YOLO integration
│   ├── intelligent_capture.py # Main pipeline
│   ├── yolo_detector.py      # YOLO wrapper
│   ├── knn_classifier.py     # KNN implementation
│   └── trigger_system.py     # Trigger management
└── assets/
    └── yolo11n.onnx          # YOLO model
```

### Proposed Swift Structure
```
swift/LiveLearningCamera/
├── LiveLearningCameraApp.swift     # App entry point
├── Core/
│   ├── CameraManager.swift         # AVFoundation wrapper
│   ├── ObjectDetector.swift        # Vision + Core ML
│   ├── CaptureEngine.swift         # Main pipeline
│   ├── KNNClassifier.swift         # ML classification
│   └── TriggerSystem.swift         # Reactive triggers
├── UI/
│   ├── ContentView.swift           # Main interface
│   ├── CameraPreview.swift         # Camera display
│   └── AnnotationView.swift        # Labeling interface
└── Resources/
    └── YOLOv11.mlmodel            # Core ML model
```

## Required Swift Dependencies

### Core Frameworks
```swift
import SwiftUI              // UI framework
import AVFoundation         // Camera capture
import Vision               // Object detection
import CoreML               // Machine learning
import Combine              // Reactive programming
import ArgumentParser       // CLI argument parsing (macOS)
```

### Potential Third-Party
```swift
import Alamofire           // HTTP networking (Gemini API)
import SwiftCSV            // Data export
import Charts              // Statistics visualization
```

## Platform-Specific Considerations

### iOS Adaptations
- **Background Processing**: Limited by app lifecycle
- **File System**: Sandboxed document directory
- **Camera Permissions**: Info.plist camera usage description
- **Model Size**: Optimize Core ML models for mobile
- **Memory Management**: Careful image buffer handling

### macOS Adaptations
- **Full File System Access**: More flexible storage
- **Multiple Windows**: Native multi-window support
- **Menu Bar Integration**: System tray functionality
- **External Cameras**: USB/network camera support
- **CLI Mode**: Full command-line interface support

## Implementation Phases

### Phase 1: Core Infrastructure
1. **Camera System**: AVFoundation-based capture
2. **Object Detection**: Vision framework integration  
3. **Basic UI**: SwiftUI camera preview
4. **Model Conversion**: ONNX → Core ML

### Phase 2: ML Pipeline
1. **Core ML Integration**: YOLO inference
2. **KNN Classifier**: Custom or Core ML implementation
3. **Model Persistence**: Local storage system
4. **Basic Recognition**: Object classification

### Phase 3: Intelligent Features
1. **Trigger System**: Combine-based reactive triggers
2. **Live Learning**: File watching and model updates
3. **Annotation Interface**: SwiftUI labeling UI
4. **Statistics Tracking**: Performance metrics

### Phase 4: Advanced Features
1. **Hand Detection**: Vision framework hand poses
2. **Gesture Recognition**: Custom gesture classification
3. **API Integration**: Gemini API Swift client
4. **Export/Import**: Model and data sharing

## Key Challenges & Solutions

### 1. ONNX Model Conversion
**Challenge**: Python uses ONNX models
**Solution**: Convert to Core ML using coremltools or find equivalent models

### 2. NumPy Array Handling
**Challenge**: Heavy NumPy usage in Python
**Solution**: Use Accelerate framework or MLMultiArray

### 3. OpenCV Dependencies
**Challenge**: Extensive OpenCV usage
**Solution**: Replace with Vision framework and Core Graphics

### 4. Real-time Performance
**Challenge**: Maintaining real-time inference
**Solution**: Optimize Core ML models, use GPU acceleration

### 5. Live Learning
**Challenge**: Dynamic model updates
**Solution**: File system monitoring + Core ML model reloading

This analysis provides a comprehensive roadmap for porting the Python EdaxShifu system to Swift while maintaining all core functionality and leveraging native iOS/macOS capabilities.
