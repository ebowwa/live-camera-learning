"""
Base annotator interface and data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
from enum import Enum
import time


class AnnotationSource(Enum):
    """Source of annotation."""
    HUMAN = "human"
    GEMINI = "gemini"
    CLAUDE = "claude"
    GPT4_VISION = "gpt4_vision"
    MULTI = "multi"
    CONSENSUS = "consensus"


@dataclass
class AnnotationRequest:
    """Request for annotation."""
    image: np.ndarray
    image_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    yolo_detections: Optional[List[str]] = None
    knn_prediction: Optional[str] = None
    knn_confidence: Optional[float] = None
    timestamp: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass 
class AnnotationResult:
    """Result of annotation."""
    label: str
    confidence: float
    source: AnnotationSource
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'label': self.label,
            'confidence': self.confidence,
            'source': self.source.value,
            'metadata': self.metadata or {},
            'processing_time': self.processing_time,
            'success': self.success,
            'error_message': self.error_message
        }


class BaseAnnotator(ABC):
    """
    Abstract base class for all annotators.
    
    Defines the interface that all annotators must implement,
    whether they be human-based, AI-based, or composite.
    """
    
    def __init__(self, name: str, source: AnnotationSource):
        """
        Initialize annotator.
        
        Args:
            name: Human-readable name for this annotator
            source: Source type of annotations
        """
        self.name = name
        self.source = source
        self.enabled = True
        self.stats = {
            'total_requests': 0,
            'successful_annotations': 0,
            'failed_annotations': 0,
            'average_confidence': 0.0,
            'total_processing_time': 0.0
        }
    
    @abstractmethod
    def annotate(self, request: AnnotationRequest) -> AnnotationResult:
        """
        Annotate an image.
        
        Args:
            request: Annotation request with image and context
            
        Returns:
            AnnotationResult with label and metadata
        """
        pass
    
    def batch_annotate(self, requests: List[AnnotationRequest]) -> List[AnnotationResult]:
        """
        Annotate multiple images.
        
        Default implementation processes sequentially.
        Subclasses can override for batch optimization.
        
        Args:
            requests: List of annotation requests
            
        Returns:
            List of annotation results
        """
        results = []
        for request in requests:
            result = self.annotate(request)
            results.append(result)
        return results
    
    def is_available(self) -> bool:
        """
        Check if annotator is currently available.
        
        Returns:
            True if annotator can process requests
        """
        return self.enabled
    
    def get_stats(self) -> Dict[str, Any]:
        """Get annotator statistics."""
        stats = self.stats.copy()
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_annotations'] / stats['total_requests']
            stats['average_processing_time'] = stats['total_processing_time'] / stats['total_requests']
        else:
            stats['success_rate'] = 0.0
            stats['average_processing_time'] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_annotations': 0,
            'failed_annotations': 0,
            'average_confidence': 0.0,
            'total_processing_time': 0.0
        }
    
    def _update_stats(self, result: AnnotationResult, processing_time: float):
        """Update internal statistics."""
        self.stats['total_requests'] += 1
        self.stats['total_processing_time'] += processing_time
        
        if result.success:
            self.stats['successful_annotations'] += 1
            # Update rolling average confidence
            total_success = self.stats['successful_annotations']
            old_avg = self.stats['average_confidence']
            self.stats['average_confidence'] = (old_avg * (total_success - 1) + result.confidence) / total_success
        else:
            self.stats['failed_annotations'] += 1
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', source={self.source.value})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.__class__.__name__}(name='{self.name}', source={self.source.value}, enabled={self.enabled})"