"""
Enhanced KNN Classifier with Online Learning capabilities.
Extends the base KNN classifier with interactive learning during inference.
"""

from src.knn_classifier import KNNObjectClassifier, Recognition
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class KNNOnlineClassifier(KNNObjectClassifier):
    """KNN classifier with online learning capabilities."""
    
    def __init__(self, 
                 n_neighbors: int = 3,
                 confidence_threshold: float = 0.6,
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 auto_save: bool = False,
                 batch_retrain_interval: int = 5,
                 max_samples_per_class: int = 100):
        """
        Initialize online learning KNN classifier.
        
        Args:
            n_neighbors: Number of neighbors for KNN
            confidence_threshold: Minimum confidence for "known" classification
            model_path: Path to save/load trained KNN model
            device: Device to run model on (cuda/cpu/auto)
            auto_save: Whether to auto-save after learning
            batch_retrain_interval: Retrain model every N samples (0 = always retrain)
            max_samples_per_class: Maximum samples to keep per class
        """
        super().__init__(n_neighbors, confidence_threshold, model_path, device, max_samples_per_class)
        self.auto_save = auto_save
        self.batch_retrain_interval = batch_retrain_interval
        self.samples_since_retrain = 0
        self.pending_retrain = False
        self.learning_history = []  # Track learning events
        
    def add_sample(self, image: np.ndarray, label: str, retrain: bool = True):
        """
        Add a training sample with optimized batch retraining.
        
        Args:
            image: Image as numpy array
            label: Object label
            retrain: Whether to consider retraining
        """
        # Use parent's thread-safe add_sample but control retraining
        self.samples_since_retrain += 1
        
        # Batch retraining optimization
        should_retrain = retrain and (
            self.batch_retrain_interval == 0 or  # Always retrain
            self.samples_since_retrain >= self.batch_retrain_interval  # Batch interval reached
        )
        
        # Call parent's add_sample with controlled retraining
        super().add_sample(image, label, retrain=should_retrain)
        
        if should_retrain:
            self.samples_since_retrain = 0
            self.pending_retrain = False
        else:
            self.pending_retrain = True
            
        # Track learning event
        self.learning_history.append({
            'label': label,
            'timestamp': np.datetime64('now'),
            'total_samples': len(self.X_train)
        })
        
        # Trim history if too long
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-500:]
            
        logger.info(f"Online learning: '{label}'. Total: {len(self.X_train)}, Pending retrain: {self.pending_retrain}")
    
    def retrain_model(self):
        """Force retrain the KNN model with current training data."""
        if self.pending_retrain:
            with self._lock:
                self._retrain_knn()
                self.pending_retrain = False
                self.samples_since_retrain = 0
                logger.info(f"Model retrained with {len(self.X_train)} samples")
    
    def predict_and_learn(self, 
                         image: np.ndarray, 
                         correct_label: Optional[str] = None,
                         confidence_threshold: Optional[float] = None,
                         force_learn: bool = False) -> Tuple[Recognition, bool]:
        """
        Online learning: Predict and optionally learn from correction.
        
        Args:
            image: Image to predict
            correct_label: If provided, adds this as training sample when needed
            confidence_threshold: Override default confidence threshold
            force_learn: Always add sample regardless of prediction
            
        Returns:
            Tuple of (Recognition result, learned) where learned indicates if model was updated
        """
        # Ensure model is up-to-date before prediction
        if self.pending_retrain:
            self.retrain_model()
        
        # Make prediction
        result = self.predict(image)
        learned = False
        
        threshold = confidence_threshold or self.confidence_threshold
        
        # Online learning: Learn from correction when needed
        if correct_label is not None:
            should_learn = force_learn or (
                not result.is_known or  # Unknown prediction
                result.label != correct_label or  # Wrong prediction
                result.confidence < threshold  # Low confidence
            )
            
            if should_learn:
                # Add sample with intelligent retraining
                self.add_sample(image, correct_label, retrain=True)
                learned = True
                
                logger.info(f"Online learning: '{correct_label}' (was: {result.label}, conf: {result.confidence:.2f})")
                
                # Auto-save if enabled
                if self.auto_save:
                    self.save_model()
                    logger.info("Model auto-saved after online learning")
        
        return result, learned
    
    def interactive_predict(self, 
                           image: np.ndarray,
                           get_user_input: Optional[callable] = None) -> Tuple[Recognition, bool]:
        """
        Interactive prediction with user feedback loop.
        
        Args:
            image: Image to predict
            get_user_input: Optional function to get user input (for GUI integration)
            
        Returns:
            Tuple of (Final recognition result, whether learning occurred)
        """
        result = self.predict(image)
        learned = False
        
        # If unknown or low confidence, get user feedback
        if not result.is_known or result.confidence < 0.7:
            if get_user_input:
                # Use provided input function (for GUI)
                user_label = get_user_input(result)
            else:
                # Default CLI input
                print(f"\nPrediction: {result.label} (confidence: {result.confidence:.2f})")
                print(f"Top 3: {sorted(result.all_scores.items(), key=lambda x: x[1], reverse=True)[:3]}")
                user_label = input("Correct label (or Enter to accept): ").strip()
            
            if user_label:
                # Learn from user correction
                _, learned = self.predict_and_learn(image, user_label, force_learn=True)
                
                # Re-predict to show improvement
                if learned:
                    result = self.predict(image)
                    print(f"Learned! New prediction: {result.label} (confidence: {result.confidence:.2f})")
        
        return result, learned
    
    def batch_predict_and_learn(self, 
                                images: list, 
                                labels: Optional[list] = None) -> list:
        """
        Batch prediction with optional learning.
        
        Args:
            images: List of images to predict
            labels: Optional list of correct labels for learning
            
        Returns:
            List of (Recognition, learned) tuples
        """
        results = []
        
        # Temporarily disable auto-retraining for batch operations
        original_interval = self.batch_retrain_interval
        if labels and len(images) > 10:
            self.batch_retrain_interval = max(10, len(images) // 10)
        
        for i, image in enumerate(images):
            label = labels[i] if labels and i < len(labels) else None
            result, learned = self.predict_and_learn(image, label)
            results.append((result, learned))
        
        # Ensure final retrain and restore settings
        if self.pending_retrain:
            self.retrain_model()
        self.batch_retrain_interval = original_interval
        
        # Save if auto-save enabled and learning occurred
        if self.auto_save and labels and any(r[1] for r in results):
            self.save_model()
            
        return results
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the online learning process."""
        stats = {
            "total_samples": len(self.X_train),
            "classes": self.get_known_classes(),
            "samples_per_class": self.get_sample_count(),
            "pending_retrain": self.pending_retrain,
            "samples_since_retrain": self.samples_since_retrain,
            "batch_interval": self.batch_retrain_interval,
            "auto_save": self.auto_save
        }
        return stats


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Create online learning classifier
    clf = KNNOnlineClassifier(
        n_neighbors=3,
        confidence_threshold=0.6,
        auto_save=True,
        batch_retrain_interval=5  # Retrain every 5 samples for efficiency
    )
    
    print("Online Learning KNN Classifier initialized")
    print(f"Learning stats: {clf.get_learning_stats()}")
    
    # Example: Load an image and learn interactively
    # img_path = "test_image.jpg"
    # img = cv2.imread(img_path)
    # 
    # # Method 1: Predict and learn if wrong
    # result, learned = clf.predict_and_learn(img, correct_label="apple")
    # 
    # # Method 2: Interactive learning with user feedback
    # result, learned = clf.interactive_predict(img)
    # 
    # # Method 3: Batch learning
    # images = [cv2.imread(f"img_{i}.jpg") for i in range(10)]
    # labels = ["apple", "banana", ...] 
    # results = clf.batch_predict_and_learn(images, labels)