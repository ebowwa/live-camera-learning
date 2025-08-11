"""
Human annotator implementation for CLI interaction.
"""

import time
import logging
from typing import Optional, Dict, Any
import numpy as np

from .base_annotator import BaseAnnotator, AnnotationResult, AnnotationRequest, AnnotationSource

logger = logging.getLogger(__name__)


class HumanAnnotator(BaseAnnotator):
    """
    Human annotator for interactive command-line annotation.
    
    Provides simple CLI-based annotation without Gradio dependencies.
    """
    
    def __init__(self, 
                 name: str = "human",
                 interactive_mode: bool = True,
                 **kwargs):
        """
        Initialize human annotator.
        
        Args:
            name: Annotator name
            interactive_mode: If True, uses command line interaction
            **kwargs: Additional parameters (ignored for compatibility)
                - knn_classifier: KNN classifier (not used in CLI mode)
                - failed_dir: Failed captures directory
                - dataset_dir: Dataset directory
                - timeout_seconds: Timeout for annotation
        """
        super().__init__(name, AnnotationSource.HUMAN)
        self.interactive_mode = interactive_mode
        # Store optional parameters but don't use them in CLI mode
        self.knn_classifier = kwargs.get('knn_classifier')
        self.failed_dir = kwargs.get('failed_dir', 'python/data/captures/failed')
        self.dataset_dir = kwargs.get('dataset_dir', 'python/data/captures/dataset')
        self.timeout_seconds = kwargs.get('timeout_seconds', 300.0)
    
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
                # Non-interactive mode returns pending
                return AnnotationResult(
                    label="pending",
                    confidence=0.0,
                    source=self.source,
                    metadata={'status': 'queued_for_human_annotation'},
                    success=True,
                    processing_time=0.0
                )
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
                
                self._update_stats(result, processing_time)
                return result
            
            print("Please enter a valid label or 'skip'")
    
    def is_available(self) -> bool:
        """Check if human annotator is available."""
        return self.enabled
