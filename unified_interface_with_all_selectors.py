#!/usr/bin/env python3
"""
Unified EdaxShifu Interface - Everything in one Gradio app
- Live RTSP/webcam stream
- Real-time YOLO detection
- Capture and annotate
- Teach new objects
- All in one interface!
"""

import gradio as gr
import cv2
import numpy as np
import time
import os
import json
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from PIL import Image
import threading
import queue
import requests
import base64
import io

from src.intelligent_capture import IntelligentCaptureSystem
from src.knn_classifier import AdaptiveKNNClassifier
from src.annotators import AnnotatorFactory, AnnotationRequest
from src.annotators.bbox_utils import draw_bounding_boxes, crop_object_from_bbox, crop_all_objects
from src.hand_detector import HandDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedEdaxShifu:
    """Single interface for everything - stream, capture, annotate, teach."""
    
    def __init__(self, rtsp_url: str = "0", model_path: str = "python/models/knn_classifier.npz", 
                 cloud_api_url: str = "http://localhost:8000"):
        """Initialize the unified system."""
        self.rtsp_url = rtsp_url
        self.cloud_api_url = cloud_api_url
        self.use_cloud_knn = False  # Default to local predictions
        
        # Initialize capture system
        self.system = IntelligentCaptureSystem(
            rtsp_url=rtsp_url,
            yolo_model_path="python/assets/yolo11n.onnx",
            capture_dir="python/data/captures",
            confidence_threshold=0.5
        )
        
        # Load training samples
        if os.path.exists("python/assets/images"):
            self.system.knn.add_samples_from_directory("python/assets/images")
            logger.info(f"Loaded training samples")
        
        # Stream state
        self.streaming = False
        self.current_frame = None
        self.last_capture = None
        self.capture_queue = queue.Queue()
        
        # Connect stream
        if not self.system.stream.connect():
            logger.error("Failed to connect to stream")
        else:
            logger.info(f"Connected to stream: {rtsp_url}")
        
        # Stats
        self.stats = {
            'captures': 0,
            'taught': 0,
            'annotations': 0,
            'ai_annotations': 0,
            'hands_detected': 0,
            'gestures_recognized': 0
        }
        
        # Initialize hand detector
        self.hand_detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Detection methods (what objects are)
        self.detection_modes = {
            'yolo': True,  # YOLO object detection
            'knn': True,   # KNN classification
        }
        
        # Triggers/Controls (actions to take)
        self.triggers = {
            'hand_gestures': False,  # Hand gesture triggers
            'auto_capture': False,   # Auto capture on detection
        }
        
        # Gesture mappings
        self.gesture_actions = {
            'peace': 'capture',
            'thumbs_up': 'teach',
            'fist': 'stop',
            'open_palm': 'start',
            'pointing': 'select'
        }
        
        logger.info("Detection and trigger systems initialized")
        
        # Initialize Gemini annotator
        self.gemini_annotator = None
        try:
            self.gemini_annotator = AnnotatorFactory.create_gemini_annotator()
            if self.gemini_annotator.is_available():
                logger.info("Gemini annotator initialized and available")
            else:
                logger.info("Gemini annotator initialized but API key not available")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini annotator: {e}")
    
    def predict_cloud(self, image: np.ndarray) -> Optional[Any]:
        """Make prediction using cloud KNN API."""
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image
            pil_image = Image.fromarray(img_rgb.astype('uint8'))
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Make API request
            response = requests.post(
                f"{self.cloud_api_url}/predict",
                json={"image_base64": img_base64},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                # Create a Recognition-like object for compatibility
                from dataclasses import dataclass
                @dataclass
                class CloudRecognition:
                    label: str
                    confidence: float
                    all_scores: Dict[str, float]
                    is_known: bool
                    
                return CloudRecognition(
                    label=data['label'],
                    confidence=data['confidence'],
                    all_scores=data.get('all_scores', {}),
                    is_known=data['is_known']
                )
            else:
                logger.error(f"Cloud API error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Cloud API connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"Cloud prediction error: {e}")
            return None
    
    def check_cloud_api_status(self) -> bool:
        """Check if cloud API is available."""
        try:
            response = requests.get(f"{self.cloud_api_url}/", timeout=2)
            return response.status_code == 200
        except:
            return False
        
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame with YOLO detections and KNN recognition."""
        # Read frame from stream
        ret, frame = self.system.stream.read_frame()
        if not ret or frame is None:
            return None
        
        # Store original frame
        self.current_frame = frame
        display_frame = frame.copy()
        
        # Detect hands if gesture triggers are enabled
        if self.triggers.get('hand_gestures', False):
            hand_detections = self.hand_detector.detect_hands(frame)
            
            # Draw hand landmarks
            for hand in hand_detections:
                # Draw skeleton
                self.hand_detector.draw_hand_landmarks(display_frame, hand)
                
                # Recognize gesture
                gesture = self.hand_detector.recognize_gesture(hand)
                if gesture:
                    self.stats['gestures_recognized'] += 1
                    action = self.gesture_actions.get(gesture)
                    
                    # Display gesture info
                    x, y, w, h = hand.bounding_box or (0, 0, 100, 100)
                    cv2.putText(display_frame, f"{gesture} -> {action}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 0, 255), 2)
            
            if hand_detections:
                self.stats['hands_detected'] += len(hand_detections)
        
        # Run YOLO detection if enabled
        detections = []
        if self.detection_modes.get('yolo', True):
            detections = self.system.yolo.detect(frame)
        
        # If KNN is enabled but YOLO is not, run KNN on full frame
        if self.detection_modes.get('knn', True) and not self.detection_modes.get('yolo', True):
            # Run KNN on the entire frame (cloud or local)
            if self.use_cloud_knn:
                full_frame_recognition = self.predict_cloud(frame)
            else:
                full_frame_recognition = self.system.knn.predict(frame)
            
            # Display result in corner
            if full_frame_recognition and full_frame_recognition.is_known:
                text = f"Frame: {full_frame_recognition.label} ({full_frame_recognition.confidence:.2f})"
                color = (0, 255, 0)
            else:
                text = "Frame: Unknown"
                color = (0, 0, 255)
            
            # Add background for text
            cv2.rectangle(display_frame, (10, 60), (400, 90), (0, 0, 0), -1)
            cv2.putText(display_frame, text, (15, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Process each YOLO detection with KNN if both are enabled
        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = [int(v) for v in bbox]
                
                # Crop the detected object for KNN
                crop = frame[y:y+h, x:x+w]
                if crop.size > 0 and self.detection_modes.get('knn', True):
                    # Run KNN classification on the crop (cloud or local)
                    if self.use_cloud_knn:
                        recognition = self.predict_cloud(crop)
                    else:
                        recognition = self.system.knn.predict(crop)
                else:
                    recognition = None
                    
                # Determine color and label based on recognition
                if recognition and recognition.is_known:
                    # Known object - green box
                    color = (0, 255, 0)
                    label = f"{recognition.label} ({recognition.confidence:.2f})"
                else:
                    # Unknown object - red box
                    color = (0, 0, 255)
                    yolo_class = det.get('class_name', 'object')
                    conf = det.get('confidence', 0)
                    label = f"Unknown: {yolo_class} ({conf:.2f})"
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label with background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(display_frame, (x, y-20), (x+label_size[0], y), color, -1)
                cv2.putText(display_frame, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add status overlay
        classes = self.system.knn.get_known_classes()
        active_detection = [name.upper() for name, enabled in self.detection_modes.items() if enabled]
        active_triggers = [name.replace('_', ' ').title() for name, enabled in self.triggers.items() if enabled]
        
        status_text = f"Detection: {', '.join(active_detection) if active_detection else 'None'}"
        if active_triggers:
            status_text += f" | Triggers: {', '.join(active_triggers)}"
        
        # Add KNN mode indicator
        knn_mode_text = " (Cloud)" if self.use_cloud_knn else " (Local)"
        if 'KNN' in active_detection:
            status_text = status_text.replace('KNN', f'KNN{knn_mode_text}')
        
        cv2.putText(display_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show some known classes
        if classes:
            classes_text = f"Classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}"
            cv2.putText(display_frame, classes_text, (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display_frame
    
    def capture_frame(self) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
        """Capture current frame for annotation."""
        if self.current_frame is None:
            return None, None, "No frame available"
        
        # Save capture
        self.last_capture = self.current_frame.copy()
        self.stats['captures'] += 1
        
        # Run KNN classification on the captured frame (cloud or local)
        if self.use_cloud_knn:
            recognition = self.predict_cloud(self.last_capture)
        else:
            recognition = self.system.knn.predict(self.last_capture)
        
        # Save to captures directory based on recognition
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if recognition and recognition.is_known:
            # Save to successful
            save_dir = "python/data/captures/successful"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{timestamp}_{recognition.label}.jpg"
            msg = f"✅ Recognized: {recognition.label} ({recognition.confidence:.2f})"
        else:
            # Unknown object - try AI annotation if available
            ai_suggestion = ""
            if self.gemini_annotator and self.gemini_annotator.is_available():
                try:
                    # Create annotation request for AI
                    request = AnnotationRequest(
                        image=self.last_capture,
                        image_path="",
                        metadata={},
                        yolo_detections=[],
                        knn_prediction=recognition.label if recognition else None,
                        knn_confidence=recognition.confidence if recognition else 0.0,
                        timestamp=timestamp
                    )
                    
                    # Get AI annotation
                    ai_result = self.gemini_annotator.annotate(request)
                    if ai_result.success:
                        ai_suggestion = f" | 🤖 AI suggests: {ai_result.label} ({ai_result.confidence:.2f})"
                        
                        # If we have bounding boxes, draw them and mention count
                        if ai_result.bounding_boxes:
                            self.last_ai_bounding_boxes = ai_result.bounding_boxes
                            bbox_count = len(ai_result.bounding_boxes)
                            ai_suggestion += f" | {bbox_count} object{'s' if bbox_count > 1 else ''} detected"
                            
                            # Draw bounding boxes on the image for saving
                            self.last_capture = draw_bounding_boxes(self.last_capture, ai_result.bounding_boxes)
                        
                        self.stats['ai_annotations'] += 1
                except Exception as e:
                    logger.error(f"AI annotation failed: {e}")
            
            # Save to failed for annotation
            save_dir = "python/data/captures/failed"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{timestamp}_unknown.jpg"
            msg = f"❌ Unknown object{ai_suggestion} - teach below or use AI suggestion"
        
        # Save the image
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, self.last_capture)
        
        # Convert to PIL for Gradio display
        img_rgb = cv2.cvtColor(self.last_capture, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        # Also create an annotated version if we have AI annotations with bounding boxes
        annotated_image = None
        if hasattr(self, 'last_ai_bounding_boxes') and self.last_ai_bounding_boxes:
            # Create annotated version with bounding boxes
            annotated_img_bgr = draw_bounding_boxes(
                cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), 
                self.last_ai_bounding_boxes
            )
            annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(annotated_img_rgb)
        
        return pil_image, annotated_image, msg
    
    def teach_object(self, image: Optional[Image.Image], label: str) -> str:
        """Teach the system a new object."""
        if image is None:
            return "No image to teach"
        
        if not label:
            return "Please provide a label"
        
        # Convert PIL to numpy
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Check if we have stored bounding boxes from a recent AI annotation
        cropped_objects = 0
        if hasattr(self, 'last_ai_bounding_boxes') and self.last_ai_bounding_boxes:
            # Try to match the label with detected objects
            for bbox in self.last_ai_bounding_boxes:
                if bbox.get('label', '').lower() == label.lower():
                    # Crop this specific object
                    cropped = crop_object_from_bbox(img_array, bbox, padding=20)
                    if cropped is not None:
                        self.system.teach_object(cropped, label)
                        cropped_objects += 1
                        logger.info(f"Taught cropped {label} object from bounding box")
        
        # If no specific objects were cropped, teach the whole image
        if cropped_objects == 0:
            self.system.teach_object(img_array, label)
        
        self.stats['taught'] += cropped_objects if cropped_objects > 0 else 1
        
        # Save model immediately
        self.system.knn.save_model()
        
        # Force reload to ensure the model is updated
        self.system.reload_model()
        
        classes = self.system.knn.get_known_classes()
        crop_info = f" ({cropped_objects} cropped objects)" if cropped_objects > 0 else ""
        return f"✅ Taught: {label}{crop_info} (Total: {len(classes)} classes: {', '.join(classes)})"
    
    def get_last_capture_as_pil(self) -> Optional[Image.Image]:
        """Get the last capture as a PIL image for the teaching interface."""
        if self.last_capture is None:
            return None
        
        # Convert BGR to RGB and then to PIL
        img_rgb = cv2.cvtColor(self.last_capture, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    def get_stats(self) -> str:
        """Get current statistics."""
        classes = self.system.knn.get_known_classes()
        ai_status = "🟢 Available" if (self.gemini_annotator and self.gemini_annotator.is_available()) else "🔴 Not available"
        
        # Show active detection and triggers
        active_detection = [name.upper() for name, enabled in self.detection_modes.items() if enabled]
        active_triggers = [name.replace('_', ' ').title() for name, enabled in self.triggers.items() if enabled]
        
        # Add KNN mode if KNN is active
        knn_mode_info = ""
        if 'knn' in self.detection_modes and self.detection_modes['knn']:
            knn_mode = "☁️ Cloud" if self.use_cloud_knn else "💻 Local"
            knn_mode_info = f"\n• KNN Mode: {knn_mode}"
            
            # If using cloud, try to get cloud stats
            if self.use_cloud_knn:
                try:
                    response = requests.get(f"{self.cloud_api_url}/model/stats", timeout=1)
                    if response.status_code == 200:
                        cloud_stats = response.json()
                        knn_mode_info += f" | Cloud: {len(cloud_stats.get('known_classes', []))} classes"
                except:
                    knn_mode_info += " | Cloud: offline"
        
        return f"""📊 Statistics:
• Detection: {', '.join(active_detection) if active_detection else 'None'}
• Triggers: {', '.join(active_triggers) if active_triggers else 'None'}{knn_mode_info}
• Known classes (local): {len(classes)}
• Total samples (local): {len(self.system.knn.X_train) if self.system.knn.X_train is not None else 0}
• Captures: {self.stats['captures']}
• Taught: {self.stats['taught']}
• AI annotations: {self.stats['ai_annotations']}
• AI status: {ai_status}
• Hands detected: {self.stats['hands_detected']}
• Gestures: {self.stats['gestures_recognized']}
• Classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}"""
    
    def create_interface(self) -> gr.Blocks:
        """Create the unified Gradio interface."""
        
        with gr.Blocks(title="EdaxShifu - Unified Interface", theme=gr.themes.Soft()) as interface:
            gr.Markdown(
                """
                # 🎯 EdaxShifu - Intelligent Camera System
                ### Live Stream • Capture • Annotate • Teach - All in One!
                """
            )
            
            with gr.Row():
                # Left side - Live stream and capture
                with gr.Column(scale=2):
                    gr.Markdown("### 📹 Live Stream")
                    
                    # Video stream
                    video = gr.Image(
                        label="Live Feed",
                        sources=None,
                        interactive=False,
                        streaming=True,
                        height=480
                    )
                    
                    # Stream controls
                    with gr.Row():
                        stream_btn = gr.Button("🎥 Start Stream", variant="primary")
                        capture_btn = gr.Button("📸 Capture", variant="secondary")
                        stop_btn = gr.Button("⏹ Stop")
                    
                    # Detection controls
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**🔍 Detection Modes:**")
                            detection_selector = gr.CheckboxGroup(
                                choices=[
                                    ("YOLO Object Detection", "yolo"),
                                    ("KNN Classification", "knn"),
                                ],
                                value=["yolo", "knn"],  # Default: both enabled
                                label="Object Detection",
                                info="How to identify objects"
                            )
                        
                        with gr.Column():
                            gr.Markdown("**🎮 Triggers & Controls:**")
                            trigger_selector = gr.CheckboxGroup(
                                choices=[
                                    ("Hand Gestures", "hand_gestures"),
                                    ("Auto Capture", "auto_capture"),
                                ],
                                value=[],  # Default: no triggers
                                label="Action Triggers",
                                info="Automatic actions"
                            )
                    
                    # Captured image displays
                    with gr.Row():
                        captured_img = gr.Image(
                            label="Last Capture",
                            interactive=False,
                            height=240,
                            scale=1
                        )
                        annotated_img = gr.Image(
                            label="AI Analysis (with bounding boxes)",
                            interactive=False,
                            height=240,
                            scale=1
                        )
                    
                    capture_status = gr.Textbox(
                        label="Capture Status",
                        interactive=False
                    )
                
                # Right side - Teaching and Settings
                with gr.Column(scale=1):
                    gr.Markdown("### 🏷️ Teach Objects")
                    
                    # Single teaching interface
                    with gr.Group():
                        gr.Markdown("**Teach from capture or upload:**")
                        
                        # Image source - either last capture or upload
                        teach_img = gr.Image(
                            label="Image to teach (upload or use last capture)",
                            sources=["upload", "clipboard"],
                            type="pil",
                            height=200
                        )
                        
                        teach_label = gr.Textbox(
                            label="Object name",
                            placeholder="e.g., apple, phone, cup, banana"
                        )
                        
                        with gr.Row():
                            use_capture_btn = gr.Button("📸 Use Last Capture", variant="secondary")
                            ai_suggest_btn = gr.Button("🤖 Ask AI", variant="secondary")
                            teach_btn = gr.Button("✅ Teach", variant="primary")
                        
                        teach_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                    
                    gr.Markdown("---")
                    
                    # KNN Mode Selector
                    with gr.Group():
                        gr.Markdown("### 🧠 KNN Prediction Mode")
                        knn_mode = gr.Radio(
                            choices=["Local", "Cloud"],
                            value="Local",
                            label="Prediction Mode",
                            info="Choose between local or cloud KNN predictions"
                        )
                        
                        # Cloud API Configuration
                        with gr.Group(visible=False) as cloud_config_group:
                            cloud_url_input = gr.Textbox(
                                label="Cloud API URL",
                                value=self.cloud_api_url,
                                placeholder="http://localhost:8000",
                                info="URL of the KNN API server"
                            )
                            cloud_status = gr.Textbox(
                                label="Cloud API Status",
                                value="🔴 Not connected",
                                interactive=False
                            )
                            test_cloud_btn = gr.Button("Test Connection", variant="secondary", size="sm")
                    
                    gr.Markdown("---")
                    
                    # API Key Configuration
                    with gr.Accordion("🔑 API Configuration", open=False):
                        gr.Markdown("Configure Gemini API for AI-powered features")
                        api_key_input = gr.Textbox(
                            label="Gemini API Key",
                            placeholder="Enter your Gemini API key",
                            type="password",
                            value="",
                            info="Get key from Google AI Studio"
                        )
                        api_status = gr.Textbox(
                            label="Status",
                            value="🔴 No API key configured",
                            interactive=False
                        )
                        update_api_btn = gr.Button("Update API Key", variant="secondary", size="sm")
                    
                    gr.Markdown("---")
                    
                    # Statistics
                    stats_display = gr.Textbox(
                        label="System Stats",
                        value=self.get_stats(),
                        interactive=False,
                        lines=8
                    )
                    
                    refresh_stats = gr.Button("🔄 Refresh Stats")
            
            # Footer controls
            with gr.Row():
                gr.Markdown(
                    """
                    **Controls:** 
                    • Click 'Start Stream' to begin
                    • 'Capture' when you see an object
                    • Annotate unknown objects
                    • Teach new objects anytime
                    • Toggle detection modes and triggers
                    """
                )
            
            # Event handlers
            def stream_frames():
                """Generator for streaming frames."""
                self.streaming = True
                while self.streaming:
                    frame = self.get_frame()
                    if frame is not None:
                        # Convert to RGB for Gradio
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        yield frame_rgb
                    else:
                        # Return a placeholder if no frame
                        yield np.zeros((480, 640, 3), dtype=np.uint8)
                    time.sleep(0.03)  # ~30 FPS
            
            def start_stream():
                """Start the video stream."""
                self.streaming = True
                return gr.update(visible=True), "Stream started"
            
            def stop_stream():
                """Stop the video stream."""
                self.streaming = False
                return gr.update(visible=False), "Stream stopped"
            
            # Connect events
            stream_btn.click(
                start_stream,
                outputs=[video, capture_status]
            ).then(
                stream_frames,
                outputs=video
            )
            
            stop_btn.click(
                stop_stream,
                outputs=[video, capture_status]
            )
            
            capture_btn.click(
                self.capture_frame,
                outputs=[captured_img, annotated_img, capture_status]
            )
            
            # Detection mode selector handler
            def update_detection_modes(selected_modes):
                """Update active detection methods."""
                self.detection_modes['yolo'] = 'yolo' in selected_modes
                self.detection_modes['knn'] = 'knn' in selected_modes
                
                active = [m.upper() for m in selected_modes]
                status = f"🔍 Detection: {', '.join(active)}" if active else "⚠️ No detection enabled"
                return status, self.get_stats()
            
            detection_selector.change(
                update_detection_modes,
                inputs=[detection_selector],
                outputs=[capture_status, stats_display]
            )
            
            # Trigger selector handler
            def update_triggers(selected_triggers):
                """Update active triggers."""
                self.triggers['hand_gestures'] = 'hand_gestures' in selected_triggers
                self.triggers['auto_capture'] = 'auto_capture' in selected_triggers
                
                active = [t.replace('_', ' ').title() for t in selected_triggers]
                status = f"🎮 Triggers: {', '.join(active)}" if active else "No triggers active"
                return status, self.get_stats()
            
            trigger_selector.change(
                update_triggers,
                inputs=[trigger_selector],
                outputs=[capture_status, stats_display]
            )
            
            # Use last capture button - loads captured image into teach interface
            use_capture_btn.click(
                self.get_last_capture_as_pil,
                outputs=[teach_img]
            )
            
            # AI suggestion function
            def get_ai_suggestion(image):
                """Get AI suggestion for the current image and return annotated image."""
                if not image:
                    return image, "", "No image to analyze"
                
                if not self.gemini_annotator or not self.gemini_annotator.is_available():
                    return image, "", "❌ AI not available (check API key)"
                
                try:
                    # Convert PIL to numpy
                    img_array = np.array(image)
                    if len(img_array.shape) == 2:  # Grayscale
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                    elif img_array.shape[2] == 4:  # RGBA
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                    else:  # RGB
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Create annotation request
                    request = AnnotationRequest(
                        image=img_array,
                        image_path="",
                        metadata={},
                        yolo_detections=[],
                        knn_prediction=None,
                        knn_confidence=0.0,
                        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
                    )
                    
                    # Get AI annotation
                    result = self.gemini_annotator.annotate(request)
                    if result.success:
                        self.stats['ai_annotations'] += 1
                        
                        # Store bounding boxes for potential cropping during teaching
                        annotated_image = image  # Default to original image
                        
                        if result.bounding_boxes:
                            self.last_ai_bounding_boxes = result.bounding_boxes
                            bbox_count = len(result.bounding_boxes)
                            bbox_info = f" | Found {bbox_count} object{'s' if bbox_count > 1 else ''} with bounding boxes"
                            
                            # List detected objects
                            if bbox_count > 1:
                                object_labels = [bbox.get('label', 'object') for bbox in result.bounding_boxes]
                                bbox_info += f": {', '.join(object_labels)}"
                            
                            # Draw bounding boxes on the image
                            img_with_boxes = draw_bounding_boxes(img_array, result.bounding_boxes)
                            # Convert back to PIL RGB for Gradio
                            img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                            annotated_image = Image.fromarray(img_with_boxes_rgb)
                        else:
                            self.last_ai_bounding_boxes = None
                            bbox_info = ""
                        
                        return annotated_image, result.label, f"🤖 AI suggests: {result.label} (confidence: {result.confidence:.2f}){bbox_info}"
                    else:
                        return image, "", f"❌ AI annotation failed: {result.error_message}"
                        
                except Exception as e:
                    logger.error(f"AI suggestion error: {e}")
                    return image, "", f"❌ Error: {str(e)}"
            
            # Connect AI suggestion button
            ai_suggest_btn.click(
                get_ai_suggestion,
                inputs=[teach_img],
                outputs=[teach_img, teach_label, teach_status]
            )
            
            teach_btn.click(
                self.teach_object,
                inputs=[teach_img, teach_label],
                outputs=[teach_status]
            ).then(
                lambda: self.get_stats(),
                outputs=[stats_display]
            ).then(
                lambda: (None, ""),  # Clear inputs
                outputs=[teach_img, teach_label]
            )
            
            refresh_stats.click(
                lambda: self.get_stats(),
                outputs=[stats_display]
            )
            
            # API Key update handler
            def update_api_key(api_key):
                """Update the Gemini API key for AI features."""
                if not api_key or api_key.strip() == "":
                    return "🔴 No API key configured", "Please enter an API key", self.get_stats()
                
                try:
                    import os
                    # Set the API key in environment
                    os.environ['GEMINI_API_KEY'] = api_key.strip()
                    
                    # Reinitialize the Gemini annotator
                    self.gemini_annotator = AnnotatorFactory.create_gemini_annotator()
                    
                    if self.gemini_annotator.is_available():
                        logger.info("API key updated and validated successfully")
                        return "🟢 API key configured successfully!", "✅ AI features enabled", self.get_stats()
                    else:
                        return "🟡 API key set but validation failed", "Warning: Key may be invalid", self.get_stats()
                        
                except Exception as e:
                    logger.error(f"Error updating API key: {e}")
                    return "🔴 Error updating key", f"Error: {str(e)[:100]}", self.get_stats()
            
            update_api_btn.click(
                update_api_key,
                inputs=[api_key_input],
                outputs=[api_status, teach_status, stats_display]
            )
            
            # KNN Mode switching handlers
            def switch_knn_mode(mode):
                """Switch between local and cloud KNN modes."""
                self.use_cloud_knn = (mode == "Cloud")
                cloud_visible = (mode == "Cloud")
                
                if self.use_cloud_knn:
                    # Test cloud connection when switching to cloud mode
                    cloud_available = self.check_cloud_api_status()
                    if cloud_available:
                        status_msg = "🟢 Switched to Cloud KNN"
                        cloud_status = "🟢 Connected to cloud API"
                    else:
                        status_msg = "🟡 Switched to Cloud KNN (API not available)"
                        cloud_status = "🔴 Cloud API not available"
                else:
                    status_msg = "🟢 Switched to Local KNN"
                    cloud_status = "🔴 Not connected"
                
                return gr.update(visible=cloud_visible), cloud_status, status_msg, self.get_stats()
            
            knn_mode.change(
                switch_knn_mode,
                inputs=[knn_mode],
                outputs=[cloud_config_group, cloud_status, capture_status, stats_display]
            )
            
            # Test cloud connection handler
            def test_cloud_connection(url):
                """Test connection to cloud KNN API."""
                self.cloud_api_url = url
                if self.check_cloud_api_status():
                    try:
                        # Get more info about the cloud model
                        response = requests.get(f"{self.cloud_api_url}/model/stats", timeout=2)
                        if response.status_code == 200:
                            stats = response.json()
                            classes_count = len(stats.get('known_classes', []))
                            samples_count = stats.get('total_samples', 0)
                            return f"🟢 Connected! Model has {classes_count} classes, {samples_count} samples"
                    except:
                        pass
                    return "🟢 Connected to cloud API"
                else:
                    return "🔴 Failed to connect to cloud API"
            
            test_cloud_btn.click(
                test_cloud_connection,
                inputs=[cloud_url_input],
                outputs=[cloud_status]
            )
            
            # Load initial stats
            interface.load(
                lambda: self.get_stats(),
                outputs=[stats_display]
            )
        
        return interface


def main():
    """Run the unified interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='EdaxShifu Unified Interface')
    
    parser.add_argument(
        '--url',
        type=str,
        default='0',
        help='RTSP URL or webcam index (default: 0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port for interface (default: 7860)'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public share link'
    )
    
    parser.add_argument(
        '--cloud-api',
        type=str,
        default='http://localhost:8000',
        help='URL for cloud KNN API (default: http://localhost:8000)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎯 EdaxShifu - Unified Interface")
    print("="*60)
    print(f"📹 Video source: {args.url}")
    print(f"🌐 Interface: http://localhost:{args.port}")
    print(f"☁️  Cloud API: {args.cloud_api}")
    print("="*60)
    print("\nEverything in one place:")
    print("• Live stream with YOLO detection")
    print("• Capture objects with one click")
    print("• Annotate unknown objects")
    print("• Teach new objects")
    print("• Real-time learning")
    print("• Cloud/Local KNN switching")
    print("• Hand gesture controls\n")
    
    # Create and launch
    app = UnifiedEdaxShifu(rtsp_url=args.url, cloud_api_url=args.cloud_api)
    interface = app.create_interface()
    # Try multiple ports if the requested one is in use
    for port_attempt in range(args.port, args.port + 10):
        try:
            interface.launch(
                server_port=port_attempt,
                share=args.share,
                server_name="0.0.0.0"
            )
            break
        except OSError as e:
            if "address already in use" in str(e).lower() and port_attempt < args.port + 9:
                print(f"Port {port_attempt} in use, trying {port_attempt + 1}")
                continue
            else:
                raise e


if __name__ == "__main__":
    main()
