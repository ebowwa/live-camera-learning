"""
Human annotator implementation using Gradio interface.
"""

import time
import queue
import threading
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from PIL import Image
import logging

from .base_annotator import BaseAnnotator, AnnotationResult, AnnotationRequest, AnnotationSource
from ..annotation_interface import HumanAnnotationInterface, AnnotationTask
from ..knn_classifier import AdaptiveKNNClassifier

logger = logging.getLogger(__name__)


class HumanAnnotator(BaseAnnotator):
    """
    Human annotator that uses the existing Gradio interface.
    
    This annotator can work in two modes:
    1. Interactive mode: Blocks until human provides annotation
    2. Queue mode: Adds to annotation queue and returns when ready
    """
    
    def __init__(self, 
                 name: str = "human",
                 knn_classifier: Optional[AdaptiveKNNClassifier] = None,
                 failed_dir: str = "captures/failed",
                 dataset_dir: str = "captures/dataset",
                 interactive_mode: bool = False,
                 timeout_seconds: float = 300.0):  # 5 minute timeout
        """
        Initialize human annotator.
        
        Args:
            name: Annotator name
            knn_classifier: KNN classifier to update with annotations
            failed_dir: Directory for failed recognitions
            dataset_dir: Directory for annotated dataset
            interactive_mode: If True, blocks until annotation received
            timeout_seconds: Timeout for waiting for annotations
        """
        super().__init__(name, AnnotationSource.HUMAN)
        
        self.knn_classifier = knn_classifier
        self.interactive_mode = interactive_mode
        self.timeout_seconds = timeout_seconds
        
        # Initialize the Gradio interface
        self.gradio_interface = HumanAnnotationInterface(
            knn_classifier=knn_classifier,
            failed_dir=failed_dir,
            dataset_dir=dataset_dir
        )
        
        # For non-interactive mode, we'll use a result queue
        self.result_queue = queue.Queue()
        self.pending_requests = {}  # request_id -> AnnotationRequest
        self.gradio_launched = False
        self.gradio_thread = None
        
    def _launch_gradio_if_needed(self):
        """Launch Gradio interface if not already running."""
        if not self.gradio_launched:
            def launch_gradio():
                try:
                    self.gradio_interface.launch(
                        share=False,
                        port=7860,
                        prevent_thread_lock=True
                    )
                    self.gradio_launched = True
                except Exception as e:
                    logger.error(f"Failed to launch Gradio interface: {e}")
            
            self.gradio_thread = threading.Thread(target=launch_gradio, daemon=True)
            self.gradio_thread.start()
            time.sleep(2)  # Give it time to start up
    
    def annotate(self, request: AnnotationRequest) -> AnnotationResult:
        """
        Annotate an image using human input.
        
        Args:
            request: Annotation request
            
        Returns:
            AnnotationResult with human annotation
        """
        start_time = time.time()
        
        try:
            if self.interactive_mode:
                return self._interactive_annotate(request)
            else:
                return self._queue_annotate(request)
        except Exception as e:
            processing_time = time.time() - start_time
            result = AnnotationResult(
                label="error",
                confidence=0.0,
                source=self.source,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
            self._update_stats(result, processing_time)
            return result
    
    def _interactive_annotate(self, request: AnnotationRequest) -> AnnotationResult:
        """Interactive annotation using command line."""
        print(f"\n{'='*60}")
        print(f"HUMAN ANNOTATION REQUIRED")
        print(f"{'='*60}")
        
        if request.yolo_detections:
            print(f"YOLO detected: {', '.join(request.yolo_detections)}")
        
        if request.knn_prediction:
            print(f"KNN prediction: {request.knn_prediction} ({request.knn_confidence:.2f})")
        
        if request.image_path:
            print(f"Image path: {request.image_path}")
            
        print(f"{'='*60}")
        
        # Simple command line input
        while True:
            label = input("Enter object label (or 'skip' to skip): ").strip()
            
            if label.lower() == 'skip':
                return AnnotationResult(
                    label="skipped",
                    confidence=0.0,
                    source=self.source,
                    success=False,
                    processing_time=time.time()
                )
            
            if label:
                # Ask for confidence
                try:
                    conf_input = input("Confidence (0.0-1.0, default 0.9): ").strip()
                    confidence = float(conf_input) if conf_input else 0.9
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.9
                
                processing_time = time.time()
                result = AnnotationResult(
                    label=label,
                    confidence=confidence,
                    source=self.source,
                    metadata={
                        'method': 'interactive_cli',
                        'yolo_detections': request.yolo_detections,
                        'knn_prediction': request.knn_prediction
                    },
                    processing_time=processing_time
                )
                
                # Update KNN if available
                if self.knn_classifier and request.image is not None:
                    self.knn_classifier.add_feedback_sample(
                        request.image,
                        predicted_label=request.knn_prediction or "unknown",
                        correct_label=label,
                        source="human"
                    )
                    logger.info(f"Updated KNN classifier with label: {label}")
                
                self._update_stats(result, processing_time)
                return result
            
            print("Please enter a valid label or 'skip'")
    
    def _queue_annotate(self, request: AnnotationRequest) -> AnnotationResult:
        """
        Queue-based annotation using Gradio interface.
        
        This method adds the request to the Gradio annotation queue
        and waits for a human to annotate it through the web interface.
        """
        # Ensure Gradio is running
        self._launch_gradio_if_needed()
        
        if not self.gradio_launched:
            return AnnotationResult(
                label="error",
                confidence=0.0,
                source=self.source,
                success=False,
                error_message="Could not launch Gradio interface"
            )
        
        # Create annotation task
        task = AnnotationTask(
            image_path=request.image_path or "temp_annotation.jpg",
            image_array=request.image,
            metadata=request.metadata or {},
            timestamp=request.timestamp or str(time.time()),
            yolo_detections=request.yolo_detections or [],
            knn_prediction=request.knn_prediction,
            knn_confidence=request.knn_confidence
        )
        
        # Save image temporarily if no path provided
        if not request.image_path and request.image is not None:
            temp_path = f"/tmp/annotation_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(temp_path, request.image)
            task.image_path = temp_path
        
        # Add to Gradio annotation queue
        self.gradio_interface.annotation_queue.queue.put(task)
        
        # For now, return a pending result
        # In a full implementation, you'd wait for the annotation to be completed
        return AnnotationResult(
            label="pending",
            confidence=0.0,
            source=self.source,
            metadata={'status': 'queued_for_human_annotation'},
            success=True,
            processing_time=0.0
        )
    
    def is_available(self) -> bool:
        """Check if human annotator is available."""
        return self.enabled
    
    def get_queue_size(self) -> int:
        """Get number of pending annotations."""
        if hasattr(self.gradio_interface, 'annotation_queue'):
            return self.gradio_interface.annotation_queue.task_count()
        return 0
    
    def launch_interface(self, port: int = 7860, share: bool = False):
        """
        Launch the Gradio interface.
        
        Args:
            port: Port to run on
            share: Whether to create public link
        """
        logger.info(f"Launching human annotation interface on port {port}")
        return self.gradio_interface.launch(share=share, port=port)
    
    def stop_interface(self):
        """Stop the Gradio interface."""
        if self.gradio_thread and self.gradio_thread.is_alive():
            # Gradio doesn't have a clean shutdown method in older versions
            # This is a limitation of the current implementation
            logger.info("Stopping human annotation interface")
            self.gradio_launched = False