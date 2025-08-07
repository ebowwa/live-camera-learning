"""
Intelligent capture system that integrates:
- YOLO object detection
- KNN classification  
- Trigger system
- Gemini API fallback (placeholder)
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable
import logging
import json

from .rtsp_stream import RTSPStream
from .yolo_detector import YOLODetector
from .knn_classifier import AdaptiveKNNClassifier, Recognition
from dataclasses import dataclass
from .trigger_system import TriggerManager, KeyboardTrigger, ObjectDetectionTrigger, TriggerEvent
from .live_model_reloader import PollingModelReloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ObjectRecognition:
    """Recognition result for a specific detected object."""
    yolo_class: str
    yolo_confidence: float
    yolo_bbox: List[int]
    knn_result: Recognition
    cropped_image: np.ndarray


class IntelligentCaptureSystem:
    """
    Complete intelligent capture system implementing the feedback loop:
    RTSP -> YOLO -> Trigger -> Capture -> KNN -> Success/Failure -> Gemini/Dataset
    """
    
    def __init__(self,
                 rtsp_url: str,
                 yolo_model_path: str = "assets/yolo11n.onnx",
                 knn_model_path: str = "models/knn_classifier.pkl",
                 capture_dir: str = "intelligent_captures",
                 confidence_threshold: float = 0.7,
                 enable_live_learning: bool = True):
        """
        Initialize the intelligent capture system.
        
        Args:
            rtsp_url: RTSP stream URL or webcam index
            yolo_model_path: Path to YOLO model
            knn_model_path: Path to save/load KNN model
            capture_dir: Directory for captured frames
            confidence_threshold: Threshold for KNN confidence
            enable_live_learning: Enable automatic model reloading for live learning
        """
        # Initialize components
        self.stream = RTSPStream(rtsp_url)
        self.yolo = YOLODetector(yolo_model_path, conf_threshold=0.3)  # Lower threshold for better detection
        self.knn = AdaptiveKNNClassifier(
            confidence_threshold=confidence_threshold,
            model_path=knn_model_path
        )
        
        # Setup triggers
        self.trigger_manager = TriggerManager()
        self._setup_triggers()
        
        # Capture settings
        self.capture_dir = capture_dir
        os.makedirs(capture_dir, exist_ok=True)
        os.makedirs(os.path.join(capture_dir, "successful"), exist_ok=True)
        os.makedirs(os.path.join(capture_dir, "failed"), exist_ok=True)
        os.makedirs(os.path.join(capture_dir, "dataset"), exist_ok=True)
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'triggers_fired': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'gemini_queries': 0,
            'knn_updates': 0
        }
        
        # Callbacks
        self.gemini_callback: Optional[Callable] = None
        self.capture_callback: Optional[Callable] = None
        
        # Live learning setup
        self.enable_live_learning = enable_live_learning
        self.model_reloader = None
        if enable_live_learning:
            self._setup_live_learning()
        
    def _setup_triggers(self):
        """Setup default triggers."""
        # Keyboard trigger (press 's' to capture)
        keyboard = KeyboardTrigger(key='s')
        keyboard.set_cooldown(1.0)
        self.trigger_manager.add_trigger(keyboard)
        
        # Object detection trigger (handheld objects)
        handheld_objects = [
            'cell phone', 'bottle', 'cup', 'book', 'banana', 'apple',
            'sandwich', 'orange', 'donut', 'scissors', 'toothbrush',
            'remote', 'tennis racket', 'baseball bat'
        ]
        
        object_trigger = ObjectDetectionTrigger(
            target_objects=handheld_objects,
            min_confidence=0.6,
            require_motion=True
        )
        object_trigger.set_cooldown(3.0)
        self.trigger_manager.add_trigger(object_trigger)
        
        # Set trigger callback
        self.trigger_manager.set_capture_callback(self._on_trigger)
        
    def _on_trigger(self, event: TriggerEvent):
        """Handle trigger events."""
        self.stats['triggers_fired'] += 1
        logger.info(f"Trigger fired: {event.description}")
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame
            
        Returns:
            Processing results including detections and recognitions
        """
        self.stats['frames_processed'] += 1
        
        # Run YOLO detection
        yolo_detections = self.yolo.detect(frame)
        handheld_detections = self.yolo.detect_hands_with_objects(frame)
        
        # Check triggers
        trigger_events = self.trigger_manager.check_all(
            frame=frame,
            detections=handheld_detections
        )
        
        # Process captures if triggered
        results = {
            'yolo_detections': yolo_detections,
            'handheld_detections': handheld_detections,
            'triggers': trigger_events,
            'object_recognitions': [],
            'captured': False
        }
        
        if trigger_events:
            # Capture and classify each detected object
            object_recognitions = self._capture_and_classify(frame, handheld_detections)
            results['object_recognitions'] = object_recognitions
            results['captured'] = True
            
        return results
        
    def _capture_and_classify(self, frame: np.ndarray, 
                             detections: List[Dict]) -> List[ObjectRecognition]:
        """
        Capture frame and run KNN classification on each detected object.
        
        Args:
            frame: Frame to capture
            detections: YOLO detections
            
        Returns:
            List of ObjectRecognition results, one per detected object
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_recognitions = []
        
        if not detections:
            logger.warning("No objects detected for classification")
            return []
        
        # Process each detected object
        for i, detection in enumerate(detections):
            # Extract object bounding box
            x, y, w, h = detection['bbox']
            
            # Ensure bounding box is within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                logger.warning(f"Invalid bounding box for {detection['class_name']}: {detection['bbox']}")
                continue
                
            # Crop the object from the frame
            cropped_obj = frame[y:y+h, x:x+w].copy()
            
            # Skip if crop is too small
            if cropped_obj.shape[0] < 10 or cropped_obj.shape[1] < 10:
                logger.warning(f"Cropped object too small: {cropped_obj.shape}")
                continue
            
            # Run KNN classification on cropped object
            knn_result = self.knn.predict(cropped_obj)
            
            # Create ObjectRecognition result
            obj_recognition = ObjectRecognition(
                yolo_class=detection['class_name'],
                yolo_confidence=detection['confidence'],
                yolo_bbox=[x, y, w, h],
                knn_result=knn_result,
                cropped_image=cropped_obj
            )
            
            object_recognitions.append(obj_recognition)
            
            # Log the result
            status = "✅" if knn_result.is_known else "❌"
            logger.info(f"{status} Object {i+1}/{len(detections)}: YOLO={detection['class_name']} → KNN={knn_result.label} ({knn_result.confidence:.2f})")
        
        # Process results for saving
        self._process_object_recognitions(frame, object_recognitions, timestamp)
        
        return object_recognitions
    
    def _process_object_recognitions(self, frame: np.ndarray, 
                                   object_recognitions: List[ObjectRecognition], 
                                   timestamp: str):
        """
        Process and save object recognition results.
        
        Args:
            frame: Original frame
            object_recognitions: List of recognition results
            timestamp: Timestamp for file naming
        """
        successful_objects = [obj for obj in object_recognitions if obj.knn_result.is_known]
        failed_objects = [obj for obj in object_recognitions if not obj.knn_result.is_known]
        
        # Update statistics
        self.stats['successful_recognitions'] += len(successful_objects)
        self.stats['failed_recognitions'] += len(failed_objects)
        
        # Save successful recognitions
        for i, obj in enumerate(successful_objects):
            # Save cropped object to successful directory
            filename = f"success_{timestamp}_{obj.knn_result.label}_obj{i+1}.jpg"
            filepath = os.path.join(self.capture_dir, "successful", filename)
            cv2.imwrite(filepath, obj.cropped_image)
            
            # Add to dataset for reinforcement
            dataset_dir = os.path.join(self.capture_dir, "dataset", obj.knn_result.label)
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_file = os.path.join(dataset_dir, f"{timestamp}_obj{i+1}.jpg")
            cv2.imwrite(dataset_file, obj.cropped_image)
            
            # Save metadata
            self._save_object_metadata(filepath, obj, frame.shape)
        
        # Save failed recognitions
        for i, obj in enumerate(failed_objects):
            # Save cropped object to failed directory
            filename = f"failed_{timestamp}_{obj.yolo_class}_obj{i+1}.jpg"
            filepath = os.path.join(self.capture_dir, "failed", filename)
            cv2.imwrite(filepath, obj.cropped_image)
            
            # Save metadata in format expected by annotation interface
            metadata_file = filepath.replace('.jpg', '_metadata.json')
            metadata = {
                'timestamp': timestamp,
                'yolo_class': obj.yolo_class,
                'yolo_confidence': float(obj.yolo_confidence),
                'yolo_bbox': obj.yolo_bbox,
                'knn_prediction': obj.knn_result.label if obj.knn_result.label else "unknown",
                'knn_confidence': float(obj.knn_result.confidence),
                'all_scores': {k: float(v) for k, v in obj.knn_result.all_scores.items()} if obj.knn_result.all_scores else {},
                'original_frame_shape': list(frame.shape)
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        # Save original frame with all objects for reference
        if object_recognitions:
            frame_filename = f"frame_{timestamp}_with_{len(object_recognitions)}_objects.jpg"
            frame_filepath = os.path.join(self.capture_dir, frame_filename)
            cv2.imwrite(frame_filepath, frame)
        
        # Call capture callback with updated signature
        if self.capture_callback:
            self.capture_callback(frame, object_recognitions)
    
    def _save_object_metadata(self, filepath: str, obj_recognition: ObjectRecognition, frame_shape: tuple):
        """Save metadata for a successfully recognized object."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'filepath': filepath,
            'yolo_detection': {
                'class_name': obj_recognition.yolo_class,
                'confidence': float(obj_recognition.yolo_confidence),
                'bbox': obj_recognition.yolo_bbox
            },
            'knn_recognition': {
                'label': obj_recognition.knn_result.label,
                'confidence': float(obj_recognition.knn_result.confidence),
                'is_known': bool(obj_recognition.knn_result.is_known),
                'all_scores': {k: float(v) for k, v in obj_recognition.knn_result.all_scores.items()}
            },
            'original_frame_shape': list(frame_shape),
            'cropped_object_shape': list(obj_recognition.cropped_image.shape)
        }
        
        metadata_file = filepath.replace('.jpg', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
    def _query_gemini(self, frame: np.ndarray, filepath: str):
        """
        Query Gemini API for annotation (placeholder).
        
        In production, this would:
        1. Send image to Gemini Vision API
        2. Get object label/description
        3. Add to KNN training data
        4. Move from failed to dataset directory
        """
        self.stats['gemini_queries'] += 1
        
        if self.gemini_callback:
            # Call external Gemini handler
            self.gemini_callback(frame, filepath)
        else:
            logger.info("🤖 Would query Gemini API for annotation...")
            
            # Simulate Gemini response for demo
            # In production, this would be the actual API response
            simulated_label = "unknown_object"
            
            # Add to KNN with Gemini's label
            self.knn.add_feedback_sample(
                frame,
                predicted_label="unknown",
                correct_label=simulated_label,
                source="gemini"
            )
            self.stats['knn_updates'] += 1
            
            
    def teach_object(self, frame: np.ndarray, label: str):
        """
        Manually teach the system a new object.
        
        Args:
            frame: Image of the object
            label: Label for the object
        """
        self.knn.add_sample(frame, label)
        self.stats['knn_updates'] += 1
        
        # Save to dataset
        dataset_dir = os.path.join(self.capture_dir, "dataset", label)
        os.makedirs(dataset_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(dataset_dir, f"taught_{timestamp}.jpg")
        cv2.imwrite(filepath, frame)
        
        logger.info(f"📚 Taught new object: {label}")
    
    def _setup_live_learning(self):
        """Setup live model reloading for continuous learning."""
        try:
            logger.info("Setting up live learning with polling approach...")
            self.model_reloader = PollingModelReloader(
                model_path=self.knn.model_path,
                reload_callback=self._reload_model_callback,
                poll_interval=2.0  # Check every 2 seconds
            )
            
            if self.model_reloader.start():
                logger.info("🔄 Live learning enabled - model will auto-reload when annotations are added")
            else:
                logger.warning("❌ Could not enable live learning")
                self.enable_live_learning = False
                
        except Exception as e:
            logger.warning(f"Live learning setup failed: {e}, continuing without it")
            self.enable_live_learning = False

    def _reload_model_callback(self):
        """Callback for model reloading."""
        old_count = len(self.knn.X_train) if self.knn.X_train is not None else 0
        self.knn.load_model()
        new_count = len(self.knn.X_train) if self.knn.X_train is not None else 0
        
        if new_count > old_count:
            self.stats['knn_updates'] += 1
            logger.info(f"🎯 Live learning: +{new_count - old_count} samples (total: {new_count})")
            return True
        return False

    def stop_live_learning(self):
        """Stop the live learning system."""
        if self.model_reloader and self.model_reloader.is_running():
            self.model_reloader.stop()
            logger.info("🛑 Live learning stopped")
            
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model_reloader'):
            self.stop_live_learning()
        
    def reload_model(self):
        """Reload the KNN model to get latest annotations."""
        try:
            if os.path.exists(self.knn.model_path):
                old_count = len(self.knn.X_train) if self.knn.X_train is not None else 0
                self.knn.load_model()
                new_count = len(self.knn.X_train) if self.knn.X_train is not None else 0
                if new_count > old_count:
                    logger.info(f"🔄 Reloaded KNN model: {new_count} samples ({new_count - old_count} new)")
                    return True
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
        return False
    
    def run(self, display: bool = True, max_frames: Optional[int] = None):
        """
        Run the intelligent capture system.
        
        Args:
            display: Whether to display video window
            max_frames: Maximum frames to process (None for infinite)
        """
        if not self.stream.connect():
            logger.error("Failed to connect to stream")
            return
            
        window_name = "Intelligent Capture System"
        frame_count = 0
        last_model_reload = time.time()
        model_reload_interval = 10  # Reload model every 10 seconds
        
        try:
            while True:
                ret, frame = self.stream.read_frame()
                
                if not ret:
                    logger.warning("Frame grab failed - reconnecting...")
                    if not self.stream.reconnect():
                        break
                    continue
                    
                # Periodically reload model to get new annotations
                if time.time() - last_model_reload > model_reload_interval:
                    self.reload_model()
                    last_model_reload = time.time()
                    
                # Process frame
                results = self.process_frame(frame)
                
                # Display if enabled
                if display:
                    display_frame = self._create_display_frame(frame, results)
                    cv2.imshow(window_name, display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == ord('r'):  # Reset KNN
                        self.knn.reset()
                        logger.info("KNN classifier reset")
                    elif key == ord('i'):  # Show info
                        self._print_stats()
                    elif key == ord('l'):  # Force reload model
                        self.reload_model()
                        logger.info("Manual model reload")
                        
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
                    
        finally:
            self.stream.release()
            if display:
                cv2.destroyAllWindows()
                
            # Save KNN model
            self.knn.save_model()
            
            # Print final stats
            self._print_stats()
            
    def _create_display_frame(self, frame: np.ndarray, 
                             results: Dict[str, Any]) -> np.ndarray:
        """Create annotated frame for display."""
        display_frame = frame.copy()
        
        # If we have object recognitions, show them with KNN results
        if results.get('object_recognitions'):
            for i, obj_rec in enumerate(results['object_recognitions']):
                x, y, w, h = obj_rec.yolo_bbox
                
                # Color based on KNN recognition success
                if obj_rec.knn_result.is_known:
                    color = (0, 255, 0)  # Green for successful recognition
                    status_symbol = "✓"  # Checkmark
                else:
                    color = (0, 0, 255)  # Red for unknown
                    status_symbol = "?"
                    
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Create combined label: YOLO class → KNN result
                yolo_label = f"{obj_rec.yolo_class}: {obj_rec.yolo_confidence:.2f}"
                knn_label = f"{status_symbol} {obj_rec.knn_result.label} ({obj_rec.knn_result.confidence:.2f})"
                
                # Draw YOLO label (top)
                label_y = y - 30 if y > 40 else y + h + 20
                cv2.putText(display_frame, yolo_label, (x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                           
                # Draw KNN label (below YOLO label)
                knn_y = label_y + 15 if y > 40 else label_y + 15
                cv2.putText(display_frame, knn_label, (x, knn_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                           
                # Draw object number
                cv2.circle(display_frame, (x + 10, y + 10), 8, color, -1)
                cv2.putText(display_frame, str(i+1), (x + 6, y + 14),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # Fallback: Draw basic YOLO detections if no object recognitions
            for det in results['handheld_detections']:
                x, y, w, h = det['bbox']
                color = (0, 255, 0) if det.get('holding_gesture') else (255, 0, 0)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(display_frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        # Show recognition summary at top of screen
        if results.get('object_recognitions'):
            total_objects = len(results['object_recognitions'])
            successful = sum(1 for obj in results['object_recognitions'] if obj.knn_result.is_known)
            failed = total_objects - successful
            
            summary_text = f"Objects: {total_objects} | Recognized: {successful} | Unknown: {failed}"
            cv2.putText(display_frame, summary_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        # Show global stats at bottom
        stats_text = f"Total - Frames: {self.stats['frames_processed']} | " \
                    f"Success: {self.stats['successful_recognitions']} | " \
                    f"Failed: {self.stats['failed_recognitions']}"
        cv2.putText(display_frame, stats_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                   
        # Show trigger status
        if results['triggers']:
            cv2.putText(display_frame, "TRIGGERED!", (frame.shape[1]//2 - 50, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                       
        return display_frame
        
    def _print_stats(self):
        """Print system statistics."""
        print("\n" + "="*50)
        print("Intelligent Capture System Statistics")
        print("="*50)
        for key, value in self.stats.items():
            print(f"{key}: {value}")
            
        # KNN stats
        known_classes = self.knn.get_known_classes()
        sample_counts = self.knn.get_sample_count()
        accuracy_stats = self.knn.get_accuracy_stats()
        
        print(f"\nKNN Classifier:")
        print(f"  Known classes: {known_classes}")
        print(f"  Samples per class: {sample_counts}")
        if accuracy_stats:
            print(f"  Accuracy: {accuracy_stats.get('accuracy', 0):.2%}")
        print("="*50)