import cv2
import time
from typing import Optional, Callable, List
import logging
from edaxshifu.rtsp_stream import RTSPStream
from edaxshifu.hand_detector import HandDetector, HandDetection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HandStreamViewer:
    """
    RTSP stream viewer with integrated hand detection and gesture recognition.
    """
    
    def __init__(
        self,
        stream: RTSPStream,
        window_name: str = "Hand Detection Stream",
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5
    ):
        """
        Initialize the hand detection stream viewer.
        
        Args:
            stream: RTSPStream instance
            window_name: Name of the display window
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum confidence for detection
            tracking_confidence: Minimum confidence for tracking
        """
        self.stream = stream
        self.window_name = window_name
        
        # Initialize hand detector
        self.hand_detector = HandDetector(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            static_image_mode=False
        )
        
        # Callbacks
        self.gesture_callback: Optional[Callable[[str, HandDetection], None]] = None
        self.hand_callback: Optional[Callable[[List[HandDetection]], None]] = None
        
        # Stats
        self.total_detections = 0
        self.current_gesture = None
        self.gesture_start_time = None
        self.gesture_duration_threshold = 1.0  # seconds
        
    def set_gesture_callback(self, callback: Callable[[str, HandDetection], None]):
        """Set callback for when a gesture is detected."""
        self.gesture_callback = callback
        
    def set_hand_callback(self, callback: Callable[[List[HandDetection]], None]):
        """Set callback for when hands are detected."""
        self.hand_callback = callback
        
    def process_frame(self, frame) -> tuple:
        """
        Process a single frame for hand detection.
        
        Returns:
            Tuple of (annotated_frame, detections)
        """
        # Detect hands
        detections = self.hand_detector.detect(frame)
        
        # Draw annotations
        annotated_frame = frame.copy()
        gestures = []
        
        for detection in detections:
            # Draw hand landmarks and connections
            annotated_frame = self.hand_detector.draw_landmarks(
                annotated_frame,
                detection,
                draw_connections=True,
                draw_landmarks=True,
                draw_bounding_box=True
            )
            
            # Detect gesture
            gesture = self.hand_detector.detect_gesture(detection)
            if gesture:
                gestures.append(gesture)
                
                # Check for pinching
                if self.hand_detector.is_pinching(detection):
                    gesture = "pinch"
                    
                # Draw gesture label
                if detection.bounding_box:
                    x, y, _, _ = detection.bounding_box
                    cv2.putText(
                        annotated_frame,
                        f"Gesture: {gesture}",
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )
            
            # Get palm center
            palm_center = self.hand_detector.get_palm_center(detection)
            if palm_center:
                height, width = frame.shape[:2]
                px, py = palm_center.to_pixel_coords(width, height)
                cv2.circle(annotated_frame, (px, py), 10, (255, 0, 255), -1)
        
        # Handle gesture callbacks
        if gestures and self.gesture_callback:
            for gesture, detection in zip(gestures, detections):
                self.handle_gesture(gesture, detection)
        
        # Handle hand detection callback
        if detections and self.hand_callback:
            self.hand_callback(detections)
            
        self.total_detections += len(detections)
        
        return annotated_frame, detections
    
    def handle_gesture(self, gesture: str, detection: HandDetection):
        """Handle detected gestures with duration tracking."""
        current_time = time.time()
        
        if gesture == self.current_gesture:
            # Same gesture continuing
            if self.gesture_start_time:
                duration = current_time - self.gesture_start_time
                if duration >= self.gesture_duration_threshold:
                    # Gesture held long enough, trigger callback
                    if self.gesture_callback:
                        self.gesture_callback(gesture, detection)
                    # Reset to avoid repeated triggers
                    self.gesture_start_time = current_time
        else:
            # New gesture detected
            self.current_gesture = gesture
            self.gesture_start_time = current_time
    
    def draw_stats(self, frame, fps: float):
        """Draw statistics on the frame."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw stats text
        stats = [
            f"FPS: {fps:.1f}",
            f"Hands detected: {self.total_detections}",
            f"Current gesture: {self.current_gesture or 'None'}"
        ]
        
        y_offset = 35
        for stat in stats:
            cv2.putText(
                frame,
                stat,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_offset += 25
        
        return frame
    
    def run(self, show_stats: bool = True):
        """
        Run the hand detection stream viewer.
        
        Args:
            show_stats: Whether to display statistics overlay
        """
        if not self.stream.connect():
            logger.error("Failed to connect to stream")
            return
            
        fps = 0
        frame_time = time.time()
        
        logger.info("Hand detection stream started. Press ESC to quit.")
        logger.info("Gestures: open_palm, fist, peace, pointing, thumbs_up, pinch")
        
        try:
            while True:
                ret, frame = self.stream.read_frame()
                
                if not ret:
                    logger.warning("Frame grab failed — attempting reconnection")
                    if not self.stream.reconnect():
                        break
                    continue
                
                # Process frame for hand detection
                annotated_frame, detections = self.process_frame(frame)
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - frame_time) if current_time != frame_time else 0
                frame_time = current_time
                
                # Draw stats if enabled
                if show_stats:
                    annotated_frame = self.draw_stats(annotated_frame, fps)
                
                # Display frame
                cv2.imshow(self.window_name, annotated_frame)
                
                # Update stream frame count
                self.stream.frame_count += 1
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key == ord('s'):  # Save snapshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"hand_capture_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    logger.info(f"Snapshot saved: {filename}")
                elif key == ord('r'):  # Reset detector
                    self.hand_detector = HandDetector(
                        max_num_hands=self.hand_detector.max_num_hands
                    )
                    logger.info("Hand detector reset")
                    
        finally:
            # Clean up
            stats = self.stream.get_stats()
            logger.info(f"Session stats: {stats['frame_count']} frames, "
                       f"{stats['avg_fps']:.2f} avg FPS")
            logger.info(f"Total hand detections: {self.total_detections}")
            
            self.hand_detector.release()
            self.stream.release()
            cv2.destroyAllWindows()


def demo_gesture_handler(gesture: str, detection: HandDetection):
    """Example gesture handler for demonstration."""
    logger.info(f"Gesture detected: {gesture} ({detection.handedness} hand, "
               f"confidence: {detection.confidence:.2f})")
    
    # You can add custom actions based on gestures
    if gesture == "thumbs_up":
        logger.info("👍 Thumbs up detected!")
    elif gesture == "peace":
        logger.info("✌️ Peace sign detected!")
    elif gesture == "fist":
        logger.info("✊ Fist detected!")
    elif gesture == "open_palm":
        logger.info("✋ Open palm detected!")
    elif gesture == "pointing":
        logger.info("👉 Pointing detected!")
    elif gesture == "pinch":
        logger.info("🤏 Pinch detected!")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Hand Detection RTSP Stream Viewer')
    parser.add_argument(
        '--url',
        type=str,
        default='rtsp://admin:admin@192.168.42.1:554/live',
        help='RTSP stream URL'
    )
    parser.add_argument(
        '--max-hands',
        type=int,
        default=2,
        help='Maximum number of hands to detect'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold'
    )
    
    args = parser.parse_args()
    
    # Create stream and viewer
    stream = RTSPStream(args.url)
    viewer = HandStreamViewer(
        stream,
        max_hands=args.max_hands,
        detection_confidence=args.confidence
    )
    
    # Set up gesture callback
    viewer.set_gesture_callback(demo_gesture_handler)
    
    # Run viewer
    viewer.run(show_stats=True)
