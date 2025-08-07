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
from .trigger_system import TriggerManager, KeyboardTrigger, ObjectDetectionTrigger, TriggerEvent
from .live_model_reloader import PollingModelReloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            'knn_result': None,
            'captured': False
        }
        
        if trigger_events:
            # Capture and classify
            knn_result = self._capture_and_classify(frame, handheld_detections)
            results['knn_result'] = knn_result
            results['captured'] = True
            
        return results
        
    def _capture_and_classify(self, frame: np.ndarray, 
                             detections: List[Dict]) -> Optional[Recognition]:
        """
        Capture frame and run KNN classification.
        
        Args:
            frame: Frame to capture
            detections: YOLO detections
            
        Returns:
            KNN recognition result
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Run KNN classification
        recognition = self.knn.predict(frame)
        
        # Determine success/failure path
        if recognition.is_known:
            # Successful recognition
            self.stats['successful_recognitions'] += 1
            
            # Save to successful directory
            filename = f"success_{timestamp}_{recognition.label}.jpg"
            filepath = os.path.join(self.capture_dir, "successful", filename)
            cv2.imwrite(filepath, frame)
            
            # Add to dataset for reinforcement
            dataset_dir = os.path.join(self.capture_dir, "dataset", recognition.label)
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_file = os.path.join(dataset_dir, f"{timestamp}.jpg")
            cv2.imwrite(dataset_file, frame)
            
            logger.info(f"✅ Recognized: {recognition.label} ({recognition.confidence:.2f})")
            
        else:
            # Failed recognition - need annotation
            self.stats['failed_recognitions'] += 1
            
            # Save to failed directory
            filename = f"failed_{timestamp}_unknown.jpg"
            filepath = os.path.join(self.capture_dir, "failed", filename)
            cv2.imwrite(filepath, frame)
            
            # Save metadata in format expected by annotation interface
            metadata_file = filepath.replace('.jpg', '_metadata.json')
            metadata = {
                'timestamp': timestamp,
                'knn_prediction': recognition.label if recognition.label else "unknown",
                'knn_confidence': float(recognition.confidence),
                'yolo_detections': [
                    {
                        'class_name': d['class_name'],
                        'confidence': float(d['confidence']),
                        'bbox': [int(x) for x in d['bbox']]
                    }
                    for d in detections
                ],
                'all_scores': {k: float(v) for k, v in recognition.all_scores.items()} if recognition.all_scores else {}
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"❌ Unknown object saved for annotation: {filepath}")
            
            # Trigger Gemini API (placeholder - now handled by annotation interface)
            # self._query_gemini(frame, filepath)
            
        # Save general metadata
        self._save_metadata(filepath, recognition, detections)
        
        # Call capture callback
        if self.capture_callback:
            self.capture_callback(frame, recognition, detections)
            
        return recognition
        
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
            
    def _save_metadata(self, filepath: str, 
                      recognition: Recognition,
                      detections: List[Dict]):
        """Save metadata for captured frame."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'filepath': filepath,
            'recognition': {
                'label': recognition.label,
                'confidence': float(recognition.confidence),
                'is_known': bool(recognition.is_known),  # Convert numpy bool to Python bool
                'all_scores': {k: float(v) for k, v in recognition.all_scores.items()}
            },
            'yolo_detections': [
                {
                    'class_name': d['class_name'],
                    'confidence': float(d['confidence']),
                    'bbox': [int(x) for x in d['bbox']]  # Ensure bbox values are JSON serializable
                }
                for d in detections
            ],
            'stats': dict(self.stats)
        }
        
        metadata_file = filepath.replace('.jpg', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
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
        
        # Draw YOLO detections
        for det in results['handheld_detections']:
            x, y, w, h = det['bbox']
            color = (0, 255, 0) if det.get('holding_gesture') else (255, 0, 0)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(display_frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        # Show KNN result if available
        if results['knn_result']:
            knn_result = results['knn_result']
            text = f"KNN: {knn_result.label} ({knn_result.confidence:.2f})"
            color = (0, 255, 0) if knn_result.is_known else (0, 0, 255)
            cv2.putText(display_frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                       
        # Show stats
        stats_text = f"Frames: {self.stats['frames_processed']} | " \
                    f"Success: {self.stats['successful_recognitions']} | " \
                    f"Failed: {self.stats['failed_recognitions']}"
        cv2.putText(display_frame, stats_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
        # Show trigger status
        if results['triggers']:
            cv2.putText(display_frame, "TRIGGERED!", (frame.shape[1]//2 - 50, 50),
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