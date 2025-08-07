"""
Multi-annotator implementations for combining different annotation sources.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Callable
from abc import abstractmethod
from collections import Counter
import numpy as np

from .base_annotator import BaseAnnotator, AnnotationResult, AnnotationRequest, AnnotationSource

logger = logging.getLogger(__name__)


class MultiAnnotator(BaseAnnotator):
    """
    Base class for multi-annotator systems.
    
    Combines multiple annotators using different strategies.
    """
    
    def __init__(self, 
                 name: str,
                 annotators: List[BaseAnnotator],
                 source: AnnotationSource = AnnotationSource.MULTI):
        """
        Initialize multi-annotator.
        
        Args:
            name: Annotator name
            annotators: List of annotators to combine
            source: Source type for combined results
        """
        super().__init__(name, source)
        self.annotators = annotators
        self.individual_results = []  # Store individual results for analysis
        
    @abstractmethod
    def _combine_results(self, results: List[AnnotationResult], request: AnnotationRequest) -> AnnotationResult:
        """
        Combine multiple annotation results into a single result.
        
        Args:
            results: List of annotation results from different annotators
            request: Original annotation request
            
        Returns:
            Combined annotation result
        """
        pass
    
    def annotate(self, request: AnnotationRequest) -> AnnotationResult:
        """
        Annotate using multiple annotators and combine results.
        
        Args:
            request: Annotation request
            
        Returns:
            Combined annotation result
        """
        start_time = time.time()
        results = []
        
        # Get annotations from all available annotators
        for annotator in self.annotators:
            if annotator.is_available():
                try:
                    result = annotator.annotate(request)
                    results.append(result)
                    logger.debug(f"{annotator.name} result: {result.label} ({result.confidence:.2f})")
                except Exception as e:
                    logger.error(f"Error with annotator {annotator.name}: {e}")
                    # Create error result for this annotator
                    error_result = AnnotationResult(
                        label="error",
                        confidence=0.0,
                        source=annotator.source,
                        success=False,
                        error_message=str(e)
                    )
                    results.append(error_result)
            else:
                logger.debug(f"Annotator {annotator.name} not available")
        
        # Store individual results for analysis
        self.individual_results.append(results.copy())
        if len(self.individual_results) > 100:  # Keep last 100
            self.individual_results.pop(0)
        
        if not results:
            # No annotators available
            processing_time = time.time() - start_time
            result = AnnotationResult(
                label="error",
                confidence=0.0,
                source=self.source,
                success=False,
                error_message="No annotators available",
                processing_time=processing_time
            )
            self._update_stats(result, processing_time)
            return result
        
        # Combine results using strategy
        combined_result = self._combine_results(results, request)
        combined_result.processing_time = time.time() - start_time
        
        # Add metadata about individual results
        if not combined_result.metadata:
            combined_result.metadata = {}
        combined_result.metadata['individual_results'] = [
            {
                'annotator': annotator.name,
                'label': result.label,
                'confidence': result.confidence,
                'success': result.success
            }
            for annotator, result in zip(self.annotators, results)
        ]
        
        self._update_stats(combined_result, combined_result.processing_time)
        return combined_result
    
    def is_available(self) -> bool:
        """Check if any annotators are available."""
        return self.enabled and any(annotator.is_available() for annotator in self.annotators)
    
    def get_annotator_availability(self) -> Dict[str, bool]:
        """Get availability status of all annotators."""
        return {annotator.name: annotator.is_available() for annotator in self.annotators}


class ConsensusAnnotator(MultiAnnotator):
    """
    Multi-annotator that uses consensus/voting to combine results.
    
    Chooses the label that appears most frequently among successful annotations.
    """
    
    def __init__(self, 
                 annotators: List[BaseAnnotator],
                 name: str = "consensus",
                 min_consensus_ratio: float = 0.5,
                 confidence_weighting: bool = True):
        """
        Initialize consensus annotator.
        
        Args:
            annotators: List of annotators
            name: Annotator name
            min_consensus_ratio: Minimum ratio of annotators that must agree
            confidence_weighting: Whether to weight votes by confidence
        """
        super().__init__(name, annotators, AnnotationSource.CONSENSUS)
        self.min_consensus_ratio = min_consensus_ratio
        self.confidence_weighting = confidence_weighting
    
    def _combine_results(self, results: List[AnnotationResult], request: AnnotationRequest) -> AnnotationResult:
        """
        Combine results using consensus voting.
        
        Args:
            results: Individual annotation results
            request: Original request
            
        Returns:
            Consensus result
        """
        # Filter successful results
        successful_results = [r for r in results if r.success and r.label != "error"]
        
        if not successful_results:
            return AnnotationResult(
                label="error",
                confidence=0.0,
                source=self.source,
                success=False,
                error_message="No successful annotations available for consensus"
            )
        
        # Count votes
        if self.confidence_weighting:
            # Weight votes by confidence
            vote_weights = {}
            for result in successful_results:
                label = result.label
                weight = result.confidence
                vote_weights[label] = vote_weights.get(label, 0) + weight
        else:
            # Simple counting
            labels = [result.label for result in successful_results]
            vote_counts = Counter(labels)
            vote_weights = dict(vote_counts)
        
        if not vote_weights:
            return AnnotationResult(
                label="unknown",
                confidence=0.0,
                source=self.source,
                success=False,
                error_message="No valid labels for consensus"
            )
        
        # Find most voted label
        winning_label = max(vote_weights.keys(), key=lambda k: vote_weights[k])
        winning_weight = vote_weights[winning_label]
        total_weight = sum(vote_weights.values())
        
        # Check if consensus threshold is met
        consensus_ratio = winning_weight / total_weight
        if consensus_ratio < self.min_consensus_ratio:
            return AnnotationResult(
                label="uncertain",
                confidence=consensus_ratio,
                source=self.source,
                success=False,
                error_message=f"Insufficient consensus: {consensus_ratio:.2f} < {self.min_consensus_ratio}",
                metadata={
                    'vote_weights': vote_weights,
                    'consensus_ratio': consensus_ratio
                }
            )
        
        # Calculate combined confidence
        # Average confidence of annotators that voted for winning label
        winning_confidences = [
            r.confidence for r in successful_results 
            if r.label == winning_label
        ]
        combined_confidence = np.mean(winning_confidences) if winning_confidences else 0.0
        
        return AnnotationResult(
            label=winning_label,
            confidence=combined_confidence,
            source=self.source,
            success=True,
            metadata={
                'vote_weights': vote_weights,
                'consensus_ratio': consensus_ratio,
                'voters': len(successful_results),
                'total_annotators': len(results)
            }
        )


class FallbackAnnotator(MultiAnnotator):
    """
    Multi-annotator that uses fallback strategy.
    
    Tries annotators in order of preference until one succeeds.
    """
    
    def __init__(self, 
                 annotators: List[BaseAnnotator],
                 name: str = "fallback",
                 min_confidence: float = 0.5):
        """
        Initialize fallback annotator.
        
        Args:
            annotators: List of annotators in order of preference
            name: Annotator name
            min_confidence: Minimum confidence to accept result
        """
        super().__init__(name, annotators, AnnotationSource.MULTI)
        self.min_confidence = min_confidence
    
    def _combine_results(self, results: List[AnnotationResult], request: AnnotationRequest) -> AnnotationResult:
        """
        Use first successful result that meets confidence threshold.
        
        Args:
            results: Individual annotation results
            request: Original request
            
        Returns:
            First acceptable result or error
        """
        # Try results in order of annotator preference
        for i, result in enumerate(results):
            if result.success and result.confidence >= self.min_confidence:
                # Use this result
                result.metadata = result.metadata or {}
                result.metadata['fallback_position'] = i
                result.metadata['tried_annotators'] = i + 1
                return result
        
        # No acceptable results
        best_result = None
        best_confidence = -1
        
        for result in results:
            if result.success and result.confidence > best_confidence:
                best_result = result
                best_confidence = result.confidence
        
        if best_result:
            # Return best result even if below threshold
            best_result.metadata = best_result.metadata or {}
            best_result.metadata['below_threshold'] = True
            best_result.metadata['threshold'] = self.min_confidence
            return best_result
        
        # All failed
        return AnnotationResult(
            label="error",
            confidence=0.0,
            source=self.source,
            success=False,
            error_message="All annotators failed",
            metadata={
                'failed_annotators': len(results),
                'error_messages': [r.error_message for r in results if r.error_message]
            }
        )


class WeightedAnnotator(MultiAnnotator):
    """
    Multi-annotator that combines results using weighted averaging.
    
    Weights can be based on annotator reliability, confidence, or custom weights.
    """
    
    def __init__(self, 
                 annotators: List[BaseAnnotator],
                 weights: Optional[List[float]] = None,
                 name: str = "weighted",
                 reliability_weighting: bool = True):
        """
        Initialize weighted annotator.
        
        Args:
            annotators: List of annotators
            weights: Custom weights for each annotator (if None, uses equal weights)
            name: Annotator name
            reliability_weighting: Whether to weight by historical success rate
        """
        super().__init__(name, annotators, AnnotationSource.MULTI)
        
        if weights and len(weights) != len(annotators):
            raise ValueError("Number of weights must match number of annotators")
        
        self.custom_weights = weights
        self.reliability_weighting = reliability_weighting
    
    def _combine_results(self, results: List[AnnotationResult], request: AnnotationRequest) -> AnnotationResult:
        """
        Combine results using weighted averaging.
        
        Args:
            results: Individual annotation results
            request: Original request
            
        Returns:
            Weighted combination result
        """
        successful_results = [(i, r) for i, r in enumerate(results) if r.success and r.label != "error"]
        
        if not successful_results:
            return AnnotationResult(
                label="error",
                confidence=0.0,
                source=self.source,
                success=False,
                error_message="No successful annotations for weighting"
            )
        
        # Calculate weights for each result
        weights = []
        labels = []
        confidences = []
        
        for i, result in successful_results:
            annotator = self.annotators[i]
            
            # Base weight
            if self.custom_weights:
                weight = self.custom_weights[i]
            else:
                weight = 1.0
            
            # Adjust by reliability if enabled
            if self.reliability_weighting:
                stats = annotator.get_stats()
                success_rate = stats.get('success_rate', 0.5)
                weight *= success_rate
            
            # Adjust by confidence
            weight *= result.confidence
            
            weights.append(weight)
            labels.append(result.label)
            confidences.append(result.confidence)
        
        # Find most weighted label
        label_weights = {}
        label_confidences = {}
        
        for label, weight, conf in zip(labels, weights, confidences):
            label_weights[label] = label_weights.get(label, 0) + weight
            if label not in label_confidences:
                label_confidences[label] = []
            label_confidences[label].append(conf)
        
        if not label_weights:
            return AnnotationResult(
                label="unknown",
                confidence=0.0,
                source=self.source,
                success=False,
                error_message="No weighted labels available"
            )
        
        # Select highest weighted label
        best_label = max(label_weights.keys(), key=lambda k: label_weights[k])
        
        # Calculate combined confidence (average of confidences for this label)
        combined_confidence = np.mean(label_confidences[best_label])
        
        # Calculate normalized weight (how much this label dominated)
        total_weight = sum(label_weights.values())
        dominance = label_weights[best_label] / total_weight if total_weight > 0 else 0
        
        return AnnotationResult(
            label=best_label,
            confidence=combined_confidence * dominance,  # Adjust confidence by dominance
            source=self.source,
            success=True,
            metadata={
                'label_weights': label_weights,
                'dominance': dominance,
                'contributing_annotators': len(successful_results)
            }
        )