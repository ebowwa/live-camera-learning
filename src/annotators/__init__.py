"""
Abstract annotator system for EdaxShifu.

This module provides a flexible annotation system that supports multiple
annotation methods (human, AI-based) and allows for composition and
orchestration of different annotators.
"""

from .base_annotator import BaseAnnotator, AnnotationResult, AnnotationRequest
from .human_annotator import HumanAnnotator
from .gemini_annotator import GeminiAnnotator
from .multi_annotator import MultiAnnotator, ConsensusAnnotator, FallbackAnnotator
from .annotator_factory import AnnotatorFactory, create_dual_annotator

__all__ = [
    'BaseAnnotator',
    'AnnotationResult', 
    'AnnotationRequest',
    'HumanAnnotator',
    'GeminiAnnotator', 
    'MultiAnnotator',
    'ConsensusAnnotator',
    'FallbackAnnotator',
    'AnnotatorFactory',
    'create_dual_annotator'
]