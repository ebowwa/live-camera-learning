import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO object detection wrapper for ONNX models."""
    
    def __init__(self, model_path: str = "assets/yolo11n.onnx", 
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize YOLO detector with ONNX model.
        
        Args:
            model_path: Path to YOLO ONNX model
            conf_threshold: Confidence threshold for detections
            nms_threshold: Non-max suppression threshold
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.classes = self._load_coco_classes()
        self.load_model()
        
    def _load_coco_classes(self) -> List[str]:
        """Load COCO class names."""
        # Standard COCO classes
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
    def load_model(self) -> bool:
        """Load YOLO ONNX model."""
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            logger.info(f"YOLO model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
            
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Perform object detection on a frame.
        
        Args:
            frame: Input image/frame as numpy array
            
        Returns:
            List of detections, each containing:
            - class_name: Name of detected object
            - confidence: Detection confidence
            - bbox: Bounding box (x, y, width, height)
            - class_id: Class ID
        """
        if self.net is None:
            logger.error("Model not loaded")
            return []
            
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run inference
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        # Process outputs
        detections = self._process_outputs(outputs, frame.shape)
        
        return detections
        
    def _process_outputs(self, outputs: List, frame_shape: Tuple) -> List[Dict]:
        """Process YOLO outputs to get detections."""
        height, width = frame_shape[:2]
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Handle different YOLO output formats
        if len(outputs) == 1 and len(outputs[0].shape) == 3:
            # YOLOv8/v11 format: [1, num_detections, 84] or similar
            output = outputs[0][0]  # Remove batch dimension
            
            # Transpose if needed (some models output [84, num_detections])
            if output.shape[0] == 84 or output.shape[0] == 85:
                output = output.T
                
            for detection in output:
                # First 4 values are box coordinates
                x_center, y_center, w_val, h_val = detection[:4]
                
                # Remaining values are class scores
                scores = detection[4:]
                
                # Get best class
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                
                if confidence > self.conf_threshold and class_id < len(self.classes):
                    # YOLOv11 outputs are in pixels relative to 640x640
                    # Need to scale to actual image dimensions
                    scale_x = width / 640.0
                    scale_y = height / 640.0
                    
                    x = int((x_center - w_val/2) * scale_x)
                    y = int((y_center - h_val/2) * scale_y)
                    w = int(w_val * scale_x)
                    h = int(h_val * scale_y)
                    
                    # Clamp to image boundaries
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)
        else:
            # Classic YOLO format
            for output in outputs:
                for detection in output:
                    scores = detection[4:]
                    class_id = int(np.argmax(scores))
                    confidence = float(scores[class_id])
                    
                    if confidence > self.conf_threshold and class_id < len(self.classes):
                        # Scale bounding box to frame size
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Calculate top-left corner
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(confidence)
                        class_ids.append(class_id)
        
        # Apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else 'unknown',
                    'confidence': confidences[i],
                    'bbox': boxes[i]
                })
                
        return detections
    
    def detect_hands_with_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect if hands are holding objects.
        
        This method looks for:
        1. Person detection (which includes visible hands)
        2. Objects near the person/hand area
        3. Specific objects commonly held (phone, cup, bottle, etc.)
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections where hands might be holding objects
        """
        detections = self.detect(frame)
        
        # Objects commonly held in hands
        handheld_objects = {
            'cell phone', 'bottle', 'cup', 'wine glass', 'fork', 'knife', 'spoon',
            'banana', 'apple', 'sandwich', 'orange', 'donut', 'cake', 'book',
            'remote', 'scissors', 'toothbrush', 'sports ball', 'frisbee',
            'baseball bat', 'tennis racket', 'handbag', 'umbrella', 'tie'
        }
        
        # Filter for person and handheld objects
        person_detections = [d for d in detections if d['class_name'] == 'person']
        object_detections = [d for d in detections if d['class_name'] in handheld_objects]
        
        # Analyze spatial relationships
        hand_object_detections = []
        
        for obj in object_detections:
            # Check if object is in typical "holding" position
            # (upper portion of frame, suggesting raised hand)
            obj_bbox = obj['bbox']
            obj_center_y = obj_bbox[1] + obj_bbox[3] / 2
            
            # If object is in upper 60% of frame (likely being held up)
            if obj_center_y < frame.shape[0] * 0.6:
                obj['holding_gesture'] = True
                hand_object_detections.append(obj)
            else:
                obj['holding_gesture'] = False
                hand_object_detections.append(obj)
                
        return hand_object_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input frame
            detections: List of detections from detect()
            
        Returns:
            Frame with drawn detections
        """
        for detection in detections:
            x, y, w, h = detection['bbox']
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            
            # Color based on whether it's a holding gesture
            color = (0, 255, 0) if detection.get('holding_gesture', False) else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return frame