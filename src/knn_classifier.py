"""
KNN Classifier for object recognition using ResNet18 embeddings.
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import os
from typing import Optional, Dict, List, Tuple, Any
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Recognition:
    """Result of KNN recognition."""
    label: str
    confidence: float
    all_scores: Dict[str, float]
    embedding: np.ndarray
    is_known: bool  # True if confidence > threshold


class KNNObjectClassifier:
    """KNN classifier for few-shot object recognition."""
    
    def __init__(self, 
                 n_neighbors: int = 3,  # Changed to 3 for more robust predictions
                 confidence_threshold: float = 0.6,
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 max_samples_per_class: int = 100,
                 embedding_dim: int = 512):
        """
        Initialize KNN classifier with ResNet18 feature extractor.
        
        Args:
            n_neighbors: Number of neighbors for KNN
            confidence_threshold: Minimum confidence for "known" classification
            model_path: Path to save/load trained KNN model
            device: Device to run model on (cuda/cpu/auto)
            max_samples_per_class: Maximum samples to keep per class
            embedding_dim: Dimension of feature embeddings
        """
        self.n_neighbors = n_neighbors
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path or "models/knn_classifier.pkl"
        self.max_samples_per_class = max_samples_per_class
        self.embedding_dim = embedding_dim
        
        # Setup device
        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize ResNet18 feature extractor
        self._setup_feature_extractor()
        
        # Initialize KNN
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        # Use numpy arrays from the start for efficiency
        self.X_train = np.empty((0, self.embedding_dim), dtype=np.float32)
        self.y_train = np.array([], dtype=object)
        self.trained = False
        
        # Thread safety
        import threading
        self._lock = threading.RLock()
        
        # Try to load existing model
        self.load_model()
        
    def _setup_feature_extractor(self):
        """Setup ResNet18 for feature extraction."""
        # Load pretrained ResNet18
        self.feature_extractor = resnet18(pretrained=True)
        # Remove the final classification layer
        self.feature_extractor = torch.nn.Sequential(
            *list(self.feature_extractor.children())[:-1]
        )
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature embedding from image using ResNet18.
        
        Args:
            image: Image as numpy array (BGR from OpenCV)
            
        Returns:
            Normalized feature embedding vector
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Convert to PIL Image
        image_pil = Image.fromarray(image_rgb)
        
        # Preprocess
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            embedding = self.feature_extractor(img_tensor).squeeze().cpu().numpy()
            
        # L2 normalize the embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.astype(np.float32)
    
    def add_sample(self, image: np.ndarray, label: str, retrain: bool = True):
        """
        Add a training sample to the classifier with memory management.
        
        Args:
            image: Image as numpy array
            label: Object label
            retrain: Whether to retrain the model immediately
        """
        with self._lock:
            embedding = self.extract_embedding(image)
            
            # Add new sample using numpy concatenation
            self.X_train = np.vstack([self.X_train, embedding.reshape(1, -1)])
            self.y_train = np.append(self.y_train, label)
            
            # Memory management: limit samples per class
            self._manage_memory()
            
            # Retrain KNN if requested and we have enough samples
            if retrain and len(self.X_train) >= self.n_neighbors:
                self._retrain_knn()
                
            logger.info(f"Added sample for '{label}'. Total samples: {len(self.X_train)}")
    
    def _manage_memory(self):
        """Manage memory by limiting samples per class."""
        unique_labels, counts = np.unique(self.y_train, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            if count > self.max_samples_per_class:
                # Keep only the most recent samples
                label_mask = self.y_train == label
                label_indices = np.where(label_mask)[0]
                
                # Keep the last max_samples_per_class samples
                keep_indices = label_indices[-self.max_samples_per_class:]
                remove_indices = label_indices[:-self.max_samples_per_class]
                
                # Create mask for samples to keep
                keep_mask = np.ones(len(self.y_train), dtype=bool)
                keep_mask[remove_indices] = False
                
                # Update arrays
                self.X_train = self.X_train[keep_mask]
                self.y_train = self.y_train[keep_mask]
                
                logger.debug(f"Pruned {len(remove_indices)} old samples for class '{label}'")
    
    def _retrain_knn(self):
        """Retrain the KNN model with current samples."""
        # Use min of n_neighbors and number of unique samples
        unique_labels = np.unique(self.y_train)
        actual_neighbors = min(self.n_neighbors, len(self.X_train), len(unique_labels))
        
        self.knn = KNeighborsClassifier(
            n_neighbors=actual_neighbors,
            metric='cosine',  # Use cosine similarity for normalized embeddings
            algorithm='brute'  # More stable for high-dimensional data
        )
        self.knn.fit(self.X_train, self.y_train)
        self.trained = True
            
    def add_samples_from_directory(self, directory: str):
        """
        Add training samples from a directory structure.
        
        Supports two structures:
        1. Subdirectories for classes:
            directory/
                class1/
                    image1.jpg
        2. Flat structure with class in filename:
            directory/
                class1_image1.jpg
                class2_image1.jpg
        """
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist")
            return
            
        # First try to load from subdirectories
        found_subdirs = False
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                found_subdirs = True
                class_name = item
                for img_file in os.listdir(item_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(item_path, img_file)
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                self.add_sample(img, class_name)
                                logger.info(f"Loaded {img_path} as class '{class_name}'")
                        except Exception as e:
                            logger.error(f"Error loading {img_path}: {e}")
                            
        # If no subdirectories, try flat structure
        if not found_subdirs:
            for img_file in os.listdir(directory):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(directory, img_file)
                    # Extract class name from filename (e.g., "apple1.png" -> "apple")
                    class_name = img_file.split('.')[0]  # Remove extension
                    class_name = ''.join([c for c in class_name if not c.isdigit()])  # Remove numbers
                    
                    if not class_name:
                        class_name = "unknown"
                        
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            self.add_sample(img, class_name)
                            logger.info(f"Loaded {img_path} as class '{class_name}'")
                    except Exception as e:
                        logger.error(f"Error loading {img_path}: {e}")
                        
    def predict(self, image: np.ndarray) -> Recognition:
        """
        Predict the class of an image with improved confidence calculation.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Recognition result with label, confidence, and scores
        """
        with self._lock:
            if not self.trained or len(self.X_train) == 0:
                return Recognition(
                    label="unknown",
                    confidence=0.0,
                    all_scores={},
                    embedding=np.array([]),
                    is_known=False
                )
                
            # Extract embedding
            embedding = self.extract_embedding(image)
            
            # Get k nearest neighbors and distances
            distances, indices = self.knn.kneighbors(
                embedding.reshape(1, -1), 
                n_neighbors=min(self.n_neighbors, len(self.X_train))
            )
            
            # Get labels of nearest neighbors
            neighbor_labels = self.y_train[indices[0]]
            neighbor_distances = distances[0]
            
            # Calculate weighted voting based on distance
            # Convert cosine distance to similarity (1 - distance)
            similarities = 1 - neighbor_distances
            
            # Calculate scores for each class
            unique_labels = np.unique(self.y_train)
            all_scores = {}
            
            for label in unique_labels:
                label_mask = neighbor_labels == label
                if np.any(label_mask):
                    # Weighted average of similarities for this class
                    all_scores[label] = np.sum(similarities[label_mask]) / np.sum(similarities)
                else:
                    all_scores[label] = 0.0
            
            # Get prediction and confidence
            if all_scores:
                pred_label = max(all_scores, key=all_scores.get)
                confidence = all_scores[pred_label]
                
                # Additional confidence adjustment based on distance to nearest neighbor
                # If nearest neighbor is very far, reduce confidence
                if neighbor_distances[0] > 0.5:  # Cosine distance > 0.5 means low similarity
                    confidence *= (1 - neighbor_distances[0])
            else:
                pred_label = "unknown"
                confidence = 0.0
            
            # Check if known based on adjusted confidence
            is_known = confidence >= self.confidence_threshold
            
            return Recognition(
                label=pred_label if is_known else "unknown",
                confidence=float(confidence),
                all_scores=all_scores,
                embedding=embedding,
                is_known=is_known
            )
        
    def get_known_classes(self) -> List[str]:
        """Get list of known class labels."""
        if self.trained:
            return list(self.knn.classes_)
        return []
        
    def get_sample_count(self) -> Dict[str, int]:
        """Get count of samples per class."""
        counts = {}
        for label in self.y_train:
            counts[label] = counts.get(label, 0) + 1
        return counts
        
    def save_model(self, path: Optional[str] = None):
        """Save the trained model to disk."""
        with self._lock:
            save_path = path or self.model_path
            
            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model data with numpy arrays
            model_data = {
                'X_train': self.X_train,
                'y_train': self.y_train,
                'n_neighbors': self.n_neighbors,
                'confidence_threshold': self.confidence_threshold,
                'max_samples_per_class': self.max_samples_per_class,
                'embedding_dim': self.embedding_dim
            }
            
            # Use numpy's save for better handling of arrays
            np.savez_compressed(save_path, **model_data)
                
            logger.info(f"Model saved to {save_path} ({len(self.X_train)} samples)")
        
    def load_model(self, path: Optional[str] = None) -> bool:
        """Load a trained model from disk."""
        with self._lock:
            load_path = path or self.model_path
            
            # Try both .npz and .pkl formats for compatibility
            if not os.path.exists(load_path):
                # Try .pkl for backward compatibility
                pkl_path = load_path.replace('.npz', '.pkl')
                if os.path.exists(pkl_path):
                    return self._load_pkl_model(pkl_path)
                logger.info(f"No saved model found at {load_path}")
                return False
                
            try:
                # Load numpy archive
                model_data = np.load(load_path, allow_pickle=True)
                
                self.X_train = model_data['X_train']
                self.y_train = model_data['y_train']
                self.n_neighbors = int(model_data.get('n_neighbors', self.n_neighbors))
                self.confidence_threshold = float(model_data.get('confidence_threshold', self.confidence_threshold))
                self.max_samples_per_class = int(model_data.get('max_samples_per_class', self.max_samples_per_class))
                
                # Ensure arrays are proper numpy arrays
                if not isinstance(self.X_train, np.ndarray):
                    self.X_train = np.array(self.X_train, dtype=np.float32)
                if not isinstance(self.y_train, np.ndarray):
                    self.y_train = np.array(self.y_train, dtype=object)
                
                # Retrain KNN
                if len(self.X_train) > 0:
                    self._retrain_knn()
                    
                logger.info(f"Model loaded from {load_path}. {len(self.X_train)} samples")
                return True
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
    
    def _load_pkl_model(self, path: str) -> bool:
        """Load legacy pickle model format."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Convert to numpy arrays
            self.X_train = np.array(model_data['X_train'], dtype=np.float32)
            self.y_train = np.array(model_data['y_train'], dtype=object)
            self.n_neighbors = model_data.get('n_neighbors', self.n_neighbors)
            self.confidence_threshold = model_data.get('confidence_threshold', self.confidence_threshold)
            
            if len(self.X_train) > 0:
                self._retrain_knn()
                
            logger.info(f"Legacy model loaded from {path}. {len(self.X_train)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error loading legacy model: {e}")
            return False
            
    def reset(self):
        """Reset the classifier, removing all training data."""
        self.X_train = []
        self.y_train = []
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.trained = False
        logger.info("Classifier reset")
        
    def update_confidence_threshold(self, threshold: float):
        """Update the confidence threshold for known/unknown classification."""
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated to {threshold}")


class AdaptiveKNNClassifier(KNNObjectClassifier):
    """
    Adaptive KNN that can learn from both successes and failures.
    Integrates with the feedback loop from Gemini API.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_history = []
        self.auto_save = True
        self.save_interval = 10  # Save after every N new samples
        
    def add_feedback_sample(self, image: np.ndarray, 
                           predicted_label: str,
                           correct_label: str,
                           source: str = "user"):
        """
        Add a sample based on feedback (correction).
        
        Args:
            image: The image that was misclassified
            predicted_label: What the model predicted
            correct_label: The correct label (from Gemini or user)
            source: Source of correction (user/gemini/manual)
        """
        # Add the corrected sample
        self.add_sample(image, correct_label)
        
        # Track feedback
        self.feedback_history.append({
            'predicted': predicted_label,
            'correct': correct_label,
            'source': source,
            'timestamp': np.datetime64('now')
        })
        
        # Auto-save if needed
        if self.auto_save and len(self.X_train) % self.save_interval == 0:
            self.save_model()
            
        logger.info(f"Learned from feedback: {predicted_label} -> {correct_label} (via {source})")
        
    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Get statistics about classifier performance."""
        if not self.feedback_history:
            return {}
            
        total = len(self.feedback_history)
        correct = sum(1 for f in self.feedback_history 
                     if f['predicted'] == f['correct'])
        
        return {
            'total_feedback': total,
            'correct_predictions': correct,
            'accuracy': correct / total if total > 0 else 0,
            'unique_corrections': len(set(f['correct'] for f in self.feedback_history))
        }