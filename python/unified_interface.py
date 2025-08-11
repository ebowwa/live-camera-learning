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

from edaxshifu.intelligent_capture import IntelligentCaptureSystem
from edaxshifu.knn_classifier import AdaptiveKNNClassifier
from edaxshifu.annotators import AnnotatorFactory, AnnotationRequest
from edaxshifu.annotators.bbox_utils import draw_bounding_boxes, crop_object_from_bbox, crop_all_objects
from edaxshifu.hand_detector import HandDetector

# Import distributed training (optional)
try:
    from edaxshifu.distributed import MODAL_AVAILABLE, is_distributed_available
    from edaxshifu.distributed.client import DistributedTrainingClient
    from edaxshifu.distributed.modal_config import ModalConfig
    DISTRIBUTED_AVAILABLE = is_distributed_available()
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    DistributedTrainingClient = None
    ModalConfig = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedEdaxShifu:
    """Single interface for everything - stream, capture, annotate, teach."""
    
    def __init__(self, rtsp_url: str = "0", model_path: str = "python/models/knn_classifier.npz"):
        """Initialize the unified system."""
        self.rtsp_url = rtsp_url
        
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
            'gestures_recognized': 0,
            'distributed_submissions': 0,
            'model_syncs': 0
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
        
        # Initialize distributed training client (optional)
        self.distributed_client = None
        self.distributed_enabled = False
        if DISTRIBUTED_AVAILABLE:
            try:
                config = ModalConfig.from_env()
                self.distributed_client = DistributedTrainingClient(
                    config=config,
                    local_knn=self.system.knn
                )
                logger.info("Distributed training client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed training: {e}")
        
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
        
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame with YOLO detections, KNN recognition, and hand detection."""
        # Read frame from stream
        ret, frame = self.system.stream.read_frame()
        if not ret or frame is None:
            return None
        
        # Store original frame
        self.current_frame = frame
        display_frame = frame.copy()
        
        # Process hand gestures as triggers if enabled
        if self.triggers.get('hand_gestures', False):
            hand_detections = self.hand_detector.detect(frame)
            
            for hand in hand_detections:
                # Draw hand landmarks and connections
                display_frame = self.hand_detector.draw_landmarks(
                    display_frame, hand,
                    draw_connections=True,
                    draw_landmarks=True,
                    draw_bounding_box=True,
                    landmark_color=(0, 255, 0),
                    connection_color=(0, 255, 255),
                    bbox_color=(255, 0, 255)
                )
                
                # Detect gesture and trigger action
                gesture = self.hand_detector.detect_gesture(hand)
                if gesture:
                    self.stats['gestures_recognized'] += 1
                    
                    # Get action for this gesture
                    action = self.gesture_actions.get(gesture, None)
                    
                    # Display gesture and action
                    if hand.bounding_box:
                        x, y, w, h = hand.bounding_box
                        if action:
                            text = f"Gesture: {gesture} → {action.upper()}"
                            color = (0, 255, 255)  # Yellow for actionable
                        else:
                            text = f"Gesture: {gesture}"
                            color = (255, 255, 0)  # Cyan for info only
                        cv2.putText(display_frame, text, 
                                   (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, color, 2)
                    
                    # Execute trigger action (we'll implement this later)
                    if action == 'capture':
                        # Would trigger capture here
                        pass
            
            if hand_detections:
                self.stats['hands_detected'] += len(hand_detections)
        
        # Run YOLO detection if enabled
        detections = []
        if self.detection_modes.get('yolo', True):
            detections = self.system.yolo.detect(frame)
        
        # If KNN is enabled but YOLO is not, run KNN on full frame
        if self.detection_modes.get('knn', True) and not self.detection_modes.get('yolo', True):
            # Run KNN on the entire frame
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
                    # Run KNN classification on the crop
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
        
        # Check if hand gestures are being tracked
        hand_info = ""
        if self.triggers.get('hand_gestures', False):
            hand_detections = self.hand_detector.detect(self.last_capture)
            if hand_detections:
                gestures = []
                for hand in hand_detections:
                    gesture = self.hand_detector.detect_gesture(hand)
                    if gesture:
                        gestures.append(f"{hand.handedness} hand: {gesture}")
                if gestures:
                    hand_info = f" | 🖐 {', '.join(gestures)}"
        
        # Run KNN classification on the captured frame
        recognition = self.system.knn.predict(self.last_capture)
        
        # Save to captures directory based on recognition
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if recognition and recognition.is_known:
            # Save to successful
            save_dir = "python/data/captures/successful"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{timestamp}_{recognition.label}.jpg"
            msg = f"✅ Recognized: {recognition.label} ({recognition.confidence:.2f}){hand_info}"
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
            msg = f"❌ Unknown object{ai_suggestion}{hand_info} - teach below or use AI suggestion"
        
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
        
        # Submit to distributed training if enabled
        distributed_info = ""
        if self.distributed_enabled and self.distributed_client:
            try:
                success = self.distributed_client.submit_annotation(
                    image=img_array,
                    label=label,
                    confidence=1.0,
                    metadata={"source": "manual_teaching"}
                )
                if success:
                    self.stats['distributed_submissions'] += 1
                    distributed_info = " 🌐 Shared"
            except Exception as e:
                logger.error(f"Failed to submit to distributed training: {e}")
        
        classes = self.system.knn.get_known_classes()
        crop_info = f" ({cropped_objects} cropped objects)" if cropped_objects > 0 else ""
        return f"✅ Taught: {label}{crop_info}{distributed_info} (Total: {len(classes)} classes: {', '.join(classes)})"
    
    def get_last_capture_as_pil(self) -> Optional[Image.Image]:
        """Get the last capture as a PIL image for the teaching interface."""
        if self.last_capture is None:
            return None
        
        # Convert BGR to RGB and then to PIL
        img_rgb = cv2.cvtColor(self.last_capture, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    def toggle_distributed_training(self, enable: bool) -> str:
        """Enable or disable distributed training."""
        if not DISTRIBUTED_AVAILABLE:
            return "❌ Distributed training not available (Modal not installed)"
        
        if enable:
            if not self.distributed_client:
                try:
                    config = ModalConfig.from_env()
                    self.distributed_client = DistributedTrainingClient(
                        config=config,
                        local_knn=self.system.knn
                    )
                except Exception as e:
                    return f"❌ Failed to initialize: {e}"
            
            self.distributed_enabled = True
            # Try initial sync
            if self.distributed_client:
                self.distributed_client.flush_pending_annotations()
            return "✅ Distributed training enabled"
        else:
            self.distributed_enabled = False
            if self.distributed_client:
                self.distributed_client.disconnect()
            return "⏹ Distributed training disabled"
    
    def sync_distributed_model(self) -> str:
        """Manually sync with distributed model."""
        if not self.distributed_enabled or not self.distributed_client:
            return "❌ Distributed training not enabled"
        
        try:
            success = self.distributed_client.sync_model(force=True)
            if success:
                self.stats['model_syncs'] += 1
                # Reload local model to reflect changes
                self.system.reload_model()
                return f"✅ Model synced successfully (sync #{self.stats['model_syncs']})"
            else:
                return "⚠️ No updates available"
        except Exception as e:
            return f"❌ Sync failed: {e}"
    
    def get_distributed_stats(self) -> str:
        """Get distributed training statistics."""
        if not self.distributed_enabled or not self.distributed_client:
            return "Distributed training not enabled"
        
        try:
            user_stats = self.distributed_client.get_contribution_stats()
            global_stats = self.distributed_client.get_global_stats()
            
            return f"""🌐 Distributed Training Stats:
            
**Your Contributions:**
• User ID: {user_stats.get('user_id', 'Unknown')}
• Contributions: {user_stats.get('contributions', 0)}
• Pending: {user_stats.get('pending', 0)}
• Last sync: {user_stats.get('last_sync', 'Never')}

**Global Network:**
• Total users: {global_stats.get('total_users', 0)}
• Total annotations: {global_stats.get('total_annotations', 0)}
• Model samples: {global_stats.get('model_samples', 0)}
• Model classes: {global_stats.get('model_classes', 0)}
• Last aggregation: {global_stats.get('last_aggregation', 'Unknown')}
"""
        except Exception as e:
            return f"Failed to get stats: {e}"
    
    def get_stats(self) -> str:
        """Get current statistics."""
        classes = self.system.knn.get_known_classes()
        ai_status = "🟢 Available" if (self.gemini_annotator and self.gemini_annotator.is_available()) else "🔴 Not available"
        
        # Show active detection and triggers
        active_detection = [name.upper() for name, enabled in self.detection_modes.items() if enabled]
        active_triggers = [name.replace('_', ' ').title() for name, enabled in self.triggers.items() if enabled]
        
        # Distributed training status
        distributed_status = ""
        if DISTRIBUTED_AVAILABLE:
            if self.distributed_enabled and self.distributed_client:
                distributed_status = f"\n• 🌐 Distributed: ON | Shared: {self.stats['distributed_submissions']} | Syncs: {self.stats['model_syncs']}"
            else:
                distributed_status = "\n• 🌐 Distributed: OFF (available)"
        
        return f"""📊 Statistics:
• Detection: {', '.join(active_detection) if active_detection else 'None'}
• Triggers: {', '.join(active_triggers) if active_triggers else 'None'}
• Known classes: {len(classes)}
• Total samples: {len(self.system.knn.X_train) if self.system.knn.X_train is not None else 0}
• Captures: {self.stats['captures']}
• Taught: {self.stats['taught']}
• AI annotations: {self.stats['ai_annotations']}
• AI status: {ai_status}
• Hands detected: {self.stats['hands_detected']}
• Gestures: {self.stats['gestures_recognized']}{distributed_status}
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
                            
                            # Show gesture mappings
                            gr.Markdown("""
                            **Gesture Actions:**
                            - ✌️ Peace → Capture
                            - 👍 Thumbs Up → Teach
                            - ✊ Fist → Stop
                            - 🖐 Open Palm → Start
                            - 👉 Pointing → Select
                            """)
                    
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
                
                # Right side - Teaching
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
                    
                    # Distributed Training Controls (if available)
                    if DISTRIBUTED_AVAILABLE:
                        with gr.Accordion("🌐 Distributed Training", open=False):
                            gr.Markdown("Share your training data with the network")
                            
                            with gr.Row():
                                distributed_toggle = gr.Checkbox(
                                    label="Enable Distributed Training",
                                    value=self.distributed_enabled,
                                    info="Share annotations with network"
                                )
                                sync_btn = gr.Button("🔄 Sync Model", size="sm")
                            
                            distributed_status = gr.Textbox(
                                label="Network Status",
                                value="Not connected",
                                interactive=False,
                                lines=2
                            )
                            
                            distributed_stats = gr.Textbox(
                                label="Network Statistics",
                                value=self.get_distributed_stats() if self.distributed_enabled else "Disabled",
                                interactive=False,
                                lines=8
                            )
                            
                            refresh_distributed = gr.Button("🔄 Refresh Network Stats", size="sm")
                    
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
                        lines=6
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
            
            # Distributed training event handlers (if available)
            if DISTRIBUTED_AVAILABLE:
                # Toggle distributed training
                def toggle_distributed(enable):
                    status = self.toggle_distributed_training(enable)
                    stats = self.get_distributed_stats() if enable else "Disabled"
                    return status, stats, self.get_stats()
                
                distributed_toggle.change(
                    toggle_distributed,
                    inputs=[distributed_toggle],
                    outputs=[distributed_status, distributed_stats, stats_display]
                )
                
                # Sync model button
                def sync_model():
                    status = self.sync_distributed_model()
                    return status, self.get_distributed_stats(), self.get_stats()
                
                sync_btn.click(
                    sync_model,
                    outputs=[distributed_status, distributed_stats, stats_display]
                )
                
                # Refresh distributed stats
                refresh_distributed.click(
                    lambda: (self.get_distributed_stats(), self.get_stats()),
                    outputs=[distributed_stats, stats_display]
                )
            
            # Load initial stats
            interface.load(
                lambda: self.get_stats(),
                outputs=[stats_display]
            )
        
        return interface
    
    def __del__(self):
        """Cleanup resources when object is destroyed."""
        if hasattr(self, 'hand_detector'):
            self.hand_detector.release()


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
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎯 EdaxShifu - Unified Interface")
    print("="*60)
    print(f"📹 Video source: {args.url}")
    print(f"🌐 Interface: http://localhost:{args.port}")
    print("="*60)
    print("\nEverything in one place:")
    print("• Live stream with YOLO detection")
    print("• Capture objects with one click")
    print("• Annotate unknown objects")
    print("• Teach new objects")
    print("• Real-time learning\n")
    
    # Create and launch
    app = UnifiedEdaxShifu(rtsp_url=args.url)
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
