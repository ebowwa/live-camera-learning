import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HandLandmark:
    """Represents a single hand landmark with its position and confidence."""
    x: float
    y: float
    z: float
    visibility: float = 1.0
    
    def to_pixel_coords(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return int(self.x * width), int(self.y * height)


@dataclass
class HandDetection:
    """Represents a detected hand with all its landmarks and properties."""
    landmarks: List[HandLandmark]
    handedness: str  # 'Left' or 'Right'
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    
    def get_landmark(self, landmark_id: int) -> Optional[HandLandmark]:
        """Get a specific landmark by its ID (0-20 for hand landmarks)."""
        if 0 <= landmark_id < len(self.landmarks):
            return self.landmarks[landmark_id]
        return None
    
    def get_fingertip_positions(self) -> Dict[str, HandLandmark]:
        """Get the positions of all fingertips."""
        fingertips = {
            'thumb': self.get_landmark(4),
            'index': self.get_landmark(8),
            'middle': self.get_landmark(12),
            'ring': self.get_landmark(16),
            'pinky': self.get_landmark(20)
        }
        return {k: v for k, v in fingertips.items() if v is not None}
    
    def calculate_bounding_box(self, width: int, height: int, padding: int = 20) -> Tuple[int, int, int, int]:
        """Calculate bounding box for the hand with optional padding."""
        if not self.landmarks:
            return 0, 0, 0, 0
            
        x_coords = [lm.x * width for lm in self.landmarks]
        y_coords = [lm.y * height for lm in self.landmarks]
        
        min_x = max(0, int(min(x_coords)) - padding)
        min_y = max(0, int(min(y_coords)) - padding)
        max_x = min(width, int(max(x_coords)) + padding)
        max_y = min(height, int(max(y_coords)) + padding)
        
        return min_x, min_y, max_x - min_x, max_y - min_y


class HandDetector:
    """
    A comprehensive hand detection utility using MediaPipe.
    
    This class provides methods for detecting hands, extracting landmarks,
    recognizing gestures, and visualizing results.
    """
    
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_complexity: int = 1
    ):
        """
        Initialize the HandDetector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            static_image_mode: Whether to treat each image independently
            model_complexity: Model complexity (0 or 1)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        
        self.max_num_hands = max_num_hands
        logger.info(f"HandDetector initialized with max_hands={max_num_hands}, "
                   f"detection_conf={min_detection_confidence}, "
                   f"tracking_conf={min_tracking_confidence}")
    
    def detect(self, image: np.ndarray, rgb: bool = False) -> List[HandDetection]:
        """
        Detect hands in an image.
        
        Args:
            image: Input image as numpy array
            rgb: Whether the image is already in RGB format (default: False, assumes BGR)
            
        Returns:
            List of HandDetection objects
        """
        if not rgb:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        results = self.hands.process(image_rgb)
        
        detections = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                landmarks = [
                    HandLandmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=getattr(lm, 'visibility', 1.0)
                    )
                    for lm in hand_landmarks.landmark
                ]
                
                detection = HandDetection(
                    landmarks=landmarks,
                    handedness=handedness.classification[0].label,
                    confidence=handedness.classification[0].score
                )
                
                height, width = image.shape[:2]
                detection.bounding_box = detection.calculate_bounding_box(width, height)
                
                detections.append(detection)
                
        return detections
    
    def draw_landmarks(
        self,
        image: np.ndarray,
        detection: HandDetection,
        draw_connections: bool = True,
        draw_landmarks: bool = True,
        draw_bounding_box: bool = False,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 0, 0),
        bbox_color: Tuple[int, int, int] = (0, 0, 255)
    ) -> np.ndarray:
        """
        Draw hand landmarks and connections on an image.
        
        Args:
            image: Input image
            detection: HandDetection object
            draw_connections: Whether to draw connections between landmarks
            draw_landmarks: Whether to draw individual landmarks
            draw_bounding_box: Whether to draw a bounding box around the hand
            landmark_color: Color for landmarks (BGR)
            connection_color: Color for connections (BGR)
            bbox_color: Color for bounding box (BGR)
            
        Returns:
            Image with drawn annotations
        """
        height, width = image.shape[:2]
        annotated_image = image.copy()
        
        # Draw connections and landmarks manually to avoid MediaPipe object issues
        if draw_connections or draw_landmarks:
            # Define hand connections (MediaPipe hand connections)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (0, 17)  # Palm connection
            ]
            
            # Draw connections
            if draw_connections:
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(detection.landmarks) and end_idx < len(detection.landmarks):
                        start_lm = detection.landmarks[start_idx]
                        end_lm = detection.landmarks[end_idx]
                        
                        start_x, start_y = start_lm.to_pixel_coords(width, height)
                        end_x, end_y = end_lm.to_pixel_coords(width, height)
                        
                        cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), 
                                connection_color, 2)
            
            # Draw landmarks
            if draw_landmarks:
                for lm in detection.landmarks:
                    x, y = lm.to_pixel_coords(width, height)
                    cv2.circle(annotated_image, (x, y), 4, landmark_color, -1)
                    cv2.circle(annotated_image, (x, y), 4, (255, 255, 255), 1)
        
        if draw_bounding_box and detection.bounding_box:
            x, y, w, h = detection.bounding_box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), bbox_color, 2)
            
            # Add handedness label
            label = f"{detection.handedness} ({detection.confidence:.2f})"
            cv2.putText(
                annotated_image, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2
            )
        
        return annotated_image
    
    def detect_gesture(self, detection: HandDetection) -> Optional[str]:
        """
        Detect common hand gestures based on landmark positions.
        
        Args:
            detection: HandDetection object
            
        Returns:
            Gesture name or None if no gesture detected
        """
        if not detection.landmarks:
            return None
            
        # Get relevant landmarks
        thumb_tip = detection.get_landmark(self.THUMB_TIP)
        index_tip = detection.get_landmark(self.INDEX_TIP)
        middle_tip = detection.get_landmark(self.MIDDLE_TIP)
        ring_tip = detection.get_landmark(self.RING_TIP)
        pinky_tip = detection.get_landmark(self.PINKY_TIP)
        
        index_mcp = detection.get_landmark(self.INDEX_MCP)
        middle_mcp = detection.get_landmark(self.MIDDLE_MCP)
        ring_mcp = detection.get_landmark(self.RING_MCP)
        pinky_mcp = detection.get_landmark(self.PINKY_MCP)
        
        if not all([thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                   index_mcp, middle_mcp, ring_mcp, pinky_mcp]):
            return None
        
        # Count raised fingers
        raised_fingers = 0
        
        # Check thumb (special case - horizontal comparison)
        if thumb_tip.x > index_mcp.x * 1.1:  # Right hand
            raised_fingers += 1
        elif thumb_tip.x < index_mcp.x * 0.9:  # Left hand
            raised_fingers += 1
            
        # Check other fingers (vertical comparison)
        if index_tip.y < index_mcp.y:
            raised_fingers += 1
        if middle_tip.y < middle_mcp.y:
            raised_fingers += 1
        if ring_tip.y < ring_mcp.y:
            raised_fingers += 1
        if pinky_tip.y < pinky_mcp.y:
            raised_fingers += 1
        
        # Recognize gestures based on raised fingers
        if raised_fingers == 0:
            return "fist"
        elif raised_fingers == 1:
            if index_tip.y < index_mcp.y and middle_tip.y > middle_mcp.y:
                return "pointing"
            elif thumb_tip.x != index_mcp.x:
                return "thumbs_up"
        elif raised_fingers == 2:
            if index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y:
                return "peace"
        elif raised_fingers == 3:
            return "three"
        elif raised_fingers == 4:
            return "four"
        elif raised_fingers == 5:
            return "open_palm"
            
        return f"fingers_{raised_fingers}"
    
    def calculate_distance(
        self,
        point1: HandLandmark,
        point2: HandLandmark,
        image_shape: Tuple[int, int]
    ) -> float:
        """
        Calculate Euclidean distance between two landmarks in pixels.
        
        Args:
            point1: First landmark
            point2: Second landmark
            image_shape: Shape of the image (height, width)
            
        Returns:
            Distance in pixels
        """
        height, width = image_shape
        x1, y1 = point1.to_pixel_coords(width, height)
        x2, y2 = point2.to_pixel_coords(width, height)
        
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def is_pinching(self, detection: HandDetection, threshold: float = 30.0) -> bool:
        """
        Detect if the hand is making a pinching gesture.
        
        Args:
            detection: HandDetection object
            threshold: Distance threshold in pixels
            
        Returns:
            True if pinching gesture detected
        """
        thumb_tip = detection.get_landmark(self.THUMB_TIP)
        index_tip = detection.get_landmark(self.INDEX_TIP)
        
        if thumb_tip and index_tip:
            # Assuming a default image size, adjust as needed
            distance = self.calculate_distance(thumb_tip, index_tip, (480, 640))
            return distance < threshold
            
        return False
    
    def get_palm_center(self, detection: HandDetection) -> Optional[HandLandmark]:
        """
        Calculate the center of the palm.
        
        Args:
            detection: HandDetection object
            
        Returns:
            HandLandmark representing the palm center
        """
        key_points = [
            detection.get_landmark(self.WRIST),
            detection.get_landmark(self.INDEX_MCP),
            detection.get_landmark(self.MIDDLE_MCP),
            detection.get_landmark(self.RING_MCP),
            detection.get_landmark(self.PINKY_MCP)
        ]
        
        key_points = [p for p in key_points if p is not None]
        
        if not key_points:
            return None
            
        avg_x = sum(p.x for p in key_points) / len(key_points)
        avg_y = sum(p.y for p in key_points) / len(key_points)
        avg_z = sum(p.z for p in key_points) / len(key_points)
        
        return HandLandmark(x=avg_x, y=avg_y, z=avg_z)
    
    def release(self):
        """Release resources."""
        self.hands.close()
        logger.info("HandDetector resources released")