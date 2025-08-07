"""
Gemini AI annotator implementation.
"""

import time
import os
import logging
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from PIL import Image
import google.generativeai as genai

from .base_annotator import BaseAnnotator, AnnotationResult, AnnotationRequest, AnnotationSource

logger = logging.getLogger(__name__)


class GeminiAnnotator(BaseAnnotator):
    """
    AI annotator using Google's Gemini Vision model.
    
    Provides automated object identification using Gemini's vision capabilities.
    Can be configured with different prompting strategies and confidence thresholds.
    """
    
    def __init__(self, 
                 name: str = "gemini",
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-2.5-flash",
                 confidence_threshold: float = 0.7,
                 max_retries: int = 3,
                 timeout_seconds: float = 30.0):
        """
        Initialize Gemini annotator.
        
        Args:
            name: Annotator name
            api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
            model_name: Gemini model to use
            confidence_threshold: Minimum confidence to return result
            max_retries: Maximum retry attempts for failed requests
            timeout_seconds: Request timeout
        """
        super().__init__(name, AnnotationSource.GEMINI)
        
        # Get API key
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            logger.warning("No Gemini API key provided. Annotator will be disabled.")
            self.enabled = False
            return
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        # Configure Gemini
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized Gemini annotator with model {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.enabled = False
    
    def annotate(self, request: AnnotationRequest) -> AnnotationResult:
        """
        Annotate an image using Gemini Vision.
        
        Args:
            request: Annotation request
            
        Returns:
            AnnotationResult with Gemini annotation
        """
        if not self.enabled:
            return AnnotationResult(
                label="error",
                confidence=0.0,
                source=self.source,
                success=False,
                error_message="Gemini annotator is disabled (no API key or initialization failed)"
            )
        
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                # Convert image for Gemini
                pil_image = self._prepare_image(request.image)
                
                # Create prompt based on context
                prompt = self._create_prompt(request)
                
                # Call Gemini
                response = self.model.generate_content([prompt, pil_image])
                
                # Parse response
                result = self._parse_response(response, request)
                result.processing_time = time.time() - start_time
                
                self._update_stats(result, result.processing_time)
                return result
                
            except Exception as e:
                logger.warning(f"Gemini annotation attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    # Final attempt failed
                    processing_time = time.time() - start_time
                    result = AnnotationResult(
                        label="error",
                        confidence=0.0,
                        source=self.source,
                        success=False,
                        error_message=f"Gemini failed after {self.max_retries} attempts: {str(e)}",
                        processing_time=processing_time
                    )
                    self._update_stats(result, processing_time)
                    return result
                
                # Wait before retry
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        # Should never reach here
        return AnnotationResult(
            label="error",
            confidence=0.0,
            source=self.source,
            success=False,
            error_message="Unexpected error in annotation loop"
        )
    
    def batch_annotate(self, requests: List[AnnotationRequest]) -> List[AnnotationResult]:
        """
        Batch annotate multiple images.
        
        For Gemini, we process sequentially with rate limiting.
        """
        results = []
        for i, request in enumerate(requests):
            result = self.annotate(request)
            results.append(result)
            
            # Rate limiting - avoid hitting API limits
            if i < len(requests) - 1:  # Don't wait after last request
                time.sleep(0.1)  # Small delay between requests
        
        return results
    
    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        """
        Prepare image for Gemini API.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            PIL Image (RGB format)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Resize if too large (Gemini has size limits)
        max_size = 2048
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to {pil_image.size} for Gemini")
        
        return pil_image
    
    def _create_prompt(self, request: AnnotationRequest) -> str:
        """
        Create appropriate prompt based on request context.
        
        Args:
            request: Annotation request
            
        Returns:
            Formatted prompt string
        """
        # Base prompt
        prompt = "Analyze this image and identify the main object. Respond with just the object name (one or two words maximum)."
        
        # Add context if available
        context_parts = []
        
        if request.yolo_detections:
            context_parts.append(f"YOLO detected: {', '.join(request.yolo_detections)}")
        
        if request.knn_prediction:
            context_parts.append(f"Previous AI thought it might be: {request.knn_prediction} (confidence: {request.knn_confidence:.2f})")
        
        if context_parts:
            prompt += f"\n\nContext: {' | '.join(context_parts)}"
            prompt += "\n\nIgnore the context if it seems wrong. Focus on what you actually see in the image."
        
        prompt += "\n\nRespond with ONLY the object name, nothing else."
        
        return prompt
    
    def _parse_response(self, response, request: AnnotationRequest) -> AnnotationResult:
        """
        Parse Gemini response into AnnotationResult.
        
        Args:
            response: Gemini API response
            request: Original request
            
        Returns:
            AnnotationResult
        """
        try:
            # Extract text from response
            label = response.text.strip().lower()
            
            # Clean up the response
            label = self._clean_label(label)
            
            if not label:
                return AnnotationResult(
                    label="unknown",
                    confidence=0.0,
                    source=self.source,
                    success=False,
                    error_message="Empty response from Gemini"
                )
            
            # Estimate confidence based on response characteristics
            confidence = self._estimate_confidence(response.text, request)
            
            # Create metadata
            metadata = {
                'raw_response': response.text,
                'model': self.model_name,
                'prompt_included_context': bool(request.yolo_detections or request.knn_prediction)
            }
            
            if hasattr(response, 'usage_metadata'):
                metadata['usage'] = response.usage_metadata
            
            return AnnotationResult(
                label=label,
                confidence=confidence,
                source=self.source,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            return AnnotationResult(
                label="error",
                confidence=0.0,
                source=self.source,
                success=False,
                error_message=f"Failed to parse Gemini response: {str(e)}",
                metadata={'raw_response': str(response)}
            )
    
    def _clean_label(self, label: str) -> str:
        """
        Clean and normalize the label from Gemini.
        
        Args:
            label: Raw label from Gemini
            
        Returns:
            Cleaned label
        """
        # Remove common unwanted phrases
        unwanted_phrases = [
            "the object is",
            "this is a",
            "this is an",
            "i see a",
            "i see an",
            "it appears to be",
            "it looks like",
            "the main object is"
        ]
        
        label_lower = label.lower()
        for phrase in unwanted_phrases:
            label_lower = label_lower.replace(phrase, "")
        
        # Clean punctuation and extra spaces
        label_clean = ''.join(c for c in label_lower if c.isalnum() or c.isspace())
        label_clean = ' '.join(label_clean.split())  # Remove extra whitespace
        
        # Limit to first two words
        words = label_clean.split()
        if len(words) > 2:
            label_clean = ' '.join(words[:2])
        
        return label_clean
    
    def _estimate_confidence(self, raw_response: str, request: AnnotationRequest) -> float:
        """
        Estimate confidence based on response characteristics.
        
        This is a heuristic since Gemini doesn't provide explicit confidence scores.
        
        Args:
            raw_response: Raw response text
            request: Original request
            
        Returns:
            Estimated confidence (0.0 to 1.0)
        """
        base_confidence = 0.8  # Default confidence for Gemini
        
        # Lower confidence for uncertain language
        uncertain_phrases = ["might be", "could be", "appears to", "seems like", "possibly", "maybe"]
        if any(phrase in raw_response.lower() for phrase in uncertain_phrases):
            base_confidence -= 0.2
        
        # Higher confidence for definitive language
        definitive_phrases = ["this is", "clearly", "definitely", "obviously"]
        if any(phrase in raw_response.lower() for phrase in definitive_phrases):
            base_confidence += 0.1
        
        # Lower confidence if response is very long (might be uncertain)
        if len(raw_response) > 50:
            base_confidence -= 0.1
        
        # Adjust based on context agreement
        if request.yolo_detections or request.knn_prediction:
            clean_response = self._clean_label(raw_response)
            if request.yolo_detections and clean_response in [d.lower() for d in request.yolo_detections]:
                base_confidence += 0.1
            if request.knn_prediction and clean_response == request.knn_prediction.lower():
                base_confidence += 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def is_available(self) -> bool:
        """Check if Gemini annotator is available."""
        return self.enabled and self.api_key is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini model."""
        return {
            'model_name': self.model_name,
            'api_key_configured': self.api_key is not None,
            'enabled': self.enabled,
            'confidence_threshold': self.confidence_threshold
        }