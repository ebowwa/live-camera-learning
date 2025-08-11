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

from .intelligent_capture import IntelligentCaptureSystem
from .knn_classifier import AdaptiveKNNClassifier
from .annotators import AnnotatorFactory, AnnotationRequest
from .annotators.bbox_utils import draw_bounding_boxes, crop_object_from_bbox, crop_all_objects

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
            'ai_annotations': 0
        }
        
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
        """Get the current frame with YOLO detections and KNN recognition."""
        # Read frame from stream
        ret, frame = self.system.stream.read_frame()
        if not ret or frame is None:
            return None
        
        # Store original frame
        self.current_frame = frame
        display_frame = frame.copy()
        
        # Run YOLO detection
        detections = self.system.yolo.detect(frame)
        
        # Process each detection
        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = [int(v) for v in bbox]
                
                # Crop the detected object for KNN
                crop = frame[y:y+h, x:x+w]
                if crop.size > 0:
                    # Run KNN classification on the crop
                    recognition = self.system.knn.predict(crop)
                    
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
        status_text = f"Known: {len(classes)} classes | Objects detected: {len(detections)}"
        cv2.putText(display_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show some known classes
        if classes:
            classes_text = f"Classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}"
            cv2.putText(display_frame, classes_text, (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display_frame
    
    def capture_frame(self) -> Tuple[Optional[Image.Image], str]:
        """Capture current frame for annotation."""
        if self.current_frame is None:
            return None, "No frame available"
        
        # Save capture
        self.last_capture = self.current_frame.copy()
        self.stats['captures'] += 1
        
        # Run KNN classification on the captured frame
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
        
        return f"""📊 Statistics:
• Known classes: {len(classes)}
• Total samples: {len(self.system.knn.X_train) if self.system.knn.X_train is not None else 0}
• Captures: {self.stats['captures']}
• Taught: {self.stats['taught']}
• AI annotations: {self.stats['ai_annotations']}
• AI status: {ai_status}
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
