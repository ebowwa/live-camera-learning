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

from src.intelligent_capture import IntelligentCaptureSystem
from src.knn_classifier import AdaptiveKNNClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedEdaxShifu:
    """Single interface for everything - stream, capture, annotate, teach."""
    
    def __init__(self, rtsp_url: str = "0", model_path: str = "models/knn_classifier.pkl"):
        """Initialize the unified system."""
        self.rtsp_url = rtsp_url
        
        # Initialize capture system
        self.system = IntelligentCaptureSystem(
            rtsp_url=rtsp_url,
            yolo_model_path="assets/yolo11n.onnx",
            capture_dir="captures",
            confidence_threshold=0.5
        )
        
        # Load training samples
        if os.path.exists("assets/images"):
            self.system.knn.add_samples_from_directory("assets/images")
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
            'annotations': 0
        }
        
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame with detections."""
        # Read frame from stream
        ret, frame = self.system.stream.read_frame()
        if not ret or frame is None:
            return None
        
        # Process frame with YOLO and KNN
        results = self.system.process_frame(frame)
        
        # Store original frame
        self.current_frame = frame
        
        # Return display frame if available, otherwise return original
        if results and 'display_frame' in results and results['display_frame'] is not None:
            return results['display_frame']  # Return frame with detections
        
        # If no display frame, draw detections ourselves
        if results and 'detections' in results and results['detections']:
            display_frame = frame.copy()
            for det in results['detections']:
                bbox = det.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(x) for x in bbox]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det.get('class_name', 'object')}: {det.get('confidence', 0):.2f}"
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return display_frame
        
        return frame
    
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
            save_dir = "captures/successful"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{timestamp}_{recognition.label}.jpg"
            msg = f"✅ Recognized: {recognition.label} ({recognition.confidence:.2f})"
        else:
            # Save to failed for annotation
            save_dir = "captures/failed"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{timestamp}_unknown.jpg"
            msg = f"❌ Unknown object - please annotate below"
        
        # Save the image
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, self.last_capture)
        
        # Convert to PIL for Gradio display
        img_rgb = cv2.cvtColor(self.last_capture, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        return pil_image, msg
    
    def teach_object(self, image: Optional[Image.Image], label: str) -> str:
        """Teach the system a new object."""
        if image is None:
            return "No image to teach"
        
        if not label:
            return "Please provide a label"
        
        # Convert PIL to numpy
        img_array = np.array(image)
        if img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Teach the system
        self.system.teach_object(img_array, label)
        self.stats['taught'] += 1
        
        # Save model
        self.system.knn.save_model()
        
        return f"✅ Taught: {label} (Total: {len(self.system.knn.get_known_classes())} classes)"
    
    def annotate_capture(self, label: str) -> str:
        """Annotate the last captured frame."""
        if self.last_capture is None:
            return "No capture to annotate"
        
        if not label:
            return "Please provide a label"
        
        # Teach using last capture
        self.system.teach_object(self.last_capture, label)
        self.stats['annotations'] += 1
        
        # Save model
        self.system.knn.save_model()
        
        return f"✅ Annotated as: {label}"
    
    def get_stats(self) -> str:
        """Get current statistics."""
        classes = self.system.knn.get_known_classes()
        return f"""📊 Statistics:
• Known classes: {len(classes)}
• Total samples: {len(self.system.knn.X_train) if self.system.knn.X_train is not None else 0}
• Captures: {self.stats['captures']}
• Annotations: {self.stats['annotations']}
• Taught: {self.stats['taught']}
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
                    
                    # Captured image display
                    captured_img = gr.Image(
                        label="Last Capture",
                        interactive=False,
                        height=240
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
                            teach_btn = gr.Button("✅ Teach", variant="primary")
                        
                        teach_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                    
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
                outputs=[captured_img, capture_status]
            )
            
            annotate_btn.click(
                self.annotate_capture,
                inputs=[annotate_label],
                outputs=[annotate_status]
            ).then(
                lambda: self.get_stats(),
                outputs=[stats_display]
            ).then(
                lambda: "",  # Clear the label input
                outputs=[annotate_label]
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
    interface.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()