import cv2
import numpy as np
import time
import os
from datetime import datetime
from typing import Optional, List, Dict, Callable
import logging

from .rtsp_stream import RTSPStream
from .yolo_detector import YOLODetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedDetector:
    """Integrates RTSP streaming with YOLO detection and automatic capture."""
    
    def __init__(self, 
                 rtsp_url: str,
                 yolo_model_path: str = "assets/yolo11n.onnx",
                 capture_dir: str = "captures",
                 conf_threshold: float = 0.5,
                 capture_cooldown: float = 2.0):
        """
        Initialize integrated detector.
        
        Args:
            rtsp_url: RTSP stream URL
            yolo_model_path: Path to YOLO model
            capture_dir: Directory to save captured frames
            conf_threshold: Confidence threshold for detections
            capture_cooldown: Minimum seconds between captures
        """
        self.stream = RTSPStream(rtsp_url)
        self.detector = YOLODetector(yolo_model_path, conf_threshold)
        self.capture_dir = capture_dir
        self.capture_cooldown = capture_cooldown
        self.last_capture_time = 0
        self.capture_callback: Optional[Callable] = None
        
        # Create capture directory
        os.makedirs(capture_dir, exist_ok=True)
        
    def set_capture_callback(self, callback: Callable[[np.ndarray, List[Dict]], None]):
        """
        Set callback function for when objects are captured.
        
        Args:
            callback: Function that receives (frame, detections)
        """
        self.capture_callback = callback
        
    def should_capture(self, detections: List[Dict]) -> bool:
        """
        Determine if we should capture based on detections.
        
        Args:
            detections: List of YOLO detections
            
        Returns:
            True if capture conditions are met
        """
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_capture_time < self.capture_cooldown:
            return False
            
        # Check for hands holding objects
        for detection in detections:
            if detection.get('holding_gesture', False):
                return True
                
        return False
        
    def capture_frame(self, frame: np.ndarray, detections: List[Dict]) -> str:
        """
        Capture and save frame with metadata.
        
        Args:
            frame: Frame to capture
            detections: Associated detections
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename with detected objects
        detected_objects = "_".join([d['class_name'].replace(' ', '') 
                                    for d in detections[:3]])  # Max 3 objects in filename
        
        filename = f"capture_{timestamp}_{detected_objects}.jpg"
        filepath = os.path.join(self.capture_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, frame)
        
        # Save metadata
        metadata_file = filepath.replace('.jpg', '_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Detections:\n")
            for det in detections:
                f.write(f"  - {det['class_name']}: {det['confidence']:.3f}\n")
                f.write(f"    BBox: {det['bbox']}\n")
                f.write(f"    Holding: {det.get('holding_gesture', False)}\n")
                
        self.last_capture_time = time.time()
        logger.info(f"Captured frame: {filename}")
        
        # Call callback if set
        if self.capture_callback:
            self.capture_callback(frame, detections)
            
        return filepath
        
    def run(self, display: bool = True, window_name: str = "YOLO Detection"):
        """
        Run integrated detection loop.
        
        Args:
            display: Whether to display video window
            window_name: Name of display window
        """
        if not self.stream.connect():
            logger.error("Failed to connect to stream")
            return
            
        self.stream.start_time = time.time()
        frame_count = 0
        detection_count = 0
        capture_count = 0
        
        try:
            while True:
                ret, frame = self.stream.read_frame()
                
                if not ret:
                    logger.warning("Frame grab failed — attempting reconnection")
                    if not self.stream.reconnect():
                        break
                    continue
                    
                frame_count += 1
                
                # Run YOLO detection
                detections = self.detector.detect_hands_with_objects(frame)
                
                if detections:
                    detection_count += 1
                    
                    # Check if we should capture
                    if self.should_capture(detections):
                        self.capture_frame(frame, detections)
                        capture_count += 1
                
                # Draw detections on frame
                if display:
                    display_frame = self.detector.draw_detections(frame, detections)
                    
                    # Add status text
                    status = f"Frames: {frame_count} | Detections: {detection_count} | Captures: {capture_count}"
                    cv2.putText(display_frame, status, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, display_frame)
                    
                    if cv2.waitKey(1) == 27:  # ESC key
                        break
                        
        finally:
            stats = self.stream.get_stats()
            logger.info(f"Session stats:")
            logger.info(f"  Frames processed: {frame_count}")
            logger.info(f"  Objects detected: {detection_count}")
            logger.info(f"  Frames captured: {capture_count}")
            logger.info(f"  Average FPS: {stats['avg_fps']:.2f}")
            
            self.stream.release()
            if display:
                cv2.destroyAllWindows()
                

class HandObjectTrigger:
    """Advanced trigger logic for detecting hands holding objects."""
    
    def __init__(self, 
                 min_confidence: float = 0.6,
                 min_holding_frames: int = 3,
                 gesture_timeout: float = 5.0):
        """
        Initialize trigger logic.
        
        Args:
            min_confidence: Minimum confidence for valid detection
            min_holding_frames: Consecutive frames needed to trigger
            gesture_timeout: Timeout for gesture detection
        """
        self.min_confidence = min_confidence
        self.min_holding_frames = min_holding_frames
        self.gesture_timeout = gesture_timeout
        
        self.holding_buffer = []
        self.last_holding_time = 0
        
    def update(self, detections: List[Dict]) -> bool:
        """
        Update trigger state with new detections.
        
        Args:
            detections: Current frame detections
            
        Returns:
            True if trigger conditions are met
        """
        current_time = time.time()
        
        # Check for high-confidence holding gesture
        holding_detected = any(
            d.get('holding_gesture', False) and 
            d['confidence'] >= self.min_confidence 
            for d in detections
        )
        
        if holding_detected:
            self.holding_buffer.append(current_time)
            self.last_holding_time = current_time
            
            # Clean old entries
            self.holding_buffer = [
                t for t in self.holding_buffer 
                if current_time - t < 1.0
            ]
            
            # Check if we have enough consecutive detections
            if len(self.holding_buffer) >= self.min_holding_frames:
                self.holding_buffer = []  # Reset after trigger
                return True
                
        else:
            # Clear buffer if no holding detected
            if current_time - self.last_holding_time > 0.5:
                self.holding_buffer = []
                
        return False