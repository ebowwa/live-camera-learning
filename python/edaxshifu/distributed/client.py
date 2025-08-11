"""
Client for distributed training with Modal.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import logging
import json
import base64
from datetime import datetime
from PIL import Image
import io
import os

from python.edaxshifu.knn_classifier import KNNObjectClassifier, Recognition
from .modal_config import ModalConfig

logger = logging.getLogger(__name__)

class DistributedTrainingClient:
    """Client for distributed training via Modal."""
    
    def __init__(self, 
                 config: Optional[ModalConfig] = None,
                 user_id: Optional[str] = None,
                 local_knn: Optional[KNNObjectClassifier] = None):
        """
        Initialize distributed training client.
        
        Args:
            config: Modal configuration
            user_id: Unique user identifier
            local_knn: Local KNN classifier instance
        """
        self.config = config or ModalConfig.from_env()
        self.user_id = user_id or self._generate_user_id()
        self.local_knn = local_knn
        
        # Track local state
        self.pending_annotations = []
        self.last_sync_time = None
        self.contribution_count = 0
        self.is_connected = False
        
        # Try to import Modal if available
        self.modal = None
        self.stub = None
        self._initialize_modal()
    
    def _generate_user_id(self) -> str:
        """Generate a unique user ID."""
        import uuid
        return f"user_{uuid.uuid4().hex[:8]}"
    
    def _initialize_modal(self):
        """Initialize Modal connection if available."""
        try:
            from . import MODAL_AVAILABLE
            if not MODAL_AVAILABLE:
                logger.info("Modal not available - running in local mode")
                return
                
            import modal
            self.modal = modal
            
            # Create stub for the app
            self.stub = modal.Stub(self.config.stub_name)
            self.is_connected = True
            logger.info(f"Connected to Modal app: {self.config.app_name}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Modal: {e}")
            logger.info("Continuing in local mode")
    
    def submit_annotation(self, 
                          image: np.ndarray, 
                          label: str,
                          confidence: float = 1.0,
                          bbox: Optional[List[int]] = None,
                          metadata: Optional[Dict] = None) -> bool:
        """
        Submit an annotation to the distributed system.
        
        Args:
            image: Image array
            label: Annotation label
            confidence: Annotation confidence
            bbox: Optional bounding box [x, y, w, h]
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            # Convert image to base64 for transmission
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            annotation = {
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
                "image": image_b64,
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "metadata": metadata or {}
            }
            
            if self.is_connected and self.stub:
                # Submit to Modal
                return self._submit_to_modal(annotation)
            else:
                # Queue locally for later sync
                self.pending_annotations.append(annotation)
                logger.info(f"Queued annotation locally ({len(self.pending_annotations)} pending)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to submit annotation: {e}")
            return False
    
    def _submit_to_modal(self, annotation: Dict) -> bool:
        """Submit annotation to Modal backend."""
        try:
            if not self.stub:
                return False
                
            # Call Modal function (will be defined in modal_functions.py)
            with self.stub.run():
                from .modal_functions import submit_annotation as modal_submit
                result = modal_submit.remote(annotation)
                
            self.contribution_count += 1
            logger.info(f"Submitted annotation to Modal (total: {self.contribution_count})")
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Modal submission failed: {e}")
            # Queue for retry
            self.pending_annotations.append(annotation)
            return False
    
    def sync_model(self, force: bool = False) -> bool:
        """
        Sync local model with distributed model.
        
        Args:
            force: Force sync even if recently synced
            
        Returns:
            Success status
        """
        try:
            if not self.is_connected or not self.stub:
                logger.info("Not connected to Modal - skipping sync")
                return False
            
            # Check if sync is needed
            if not force and self.last_sync_time:
                time_since_sync = (datetime.now() - self.last_sync_time).seconds
                if time_since_sync < self.config.aggregation_interval:
                    logger.info(f"Skipping sync - last sync {time_since_sync}s ago")
                    return False
            
            logger.info("Syncing with distributed model...")
            
            with self.stub.run():
                from .modal_functions import get_global_model
                model_data = get_global_model.remote(self.user_id)
            
            if model_data and self.local_knn:
                # Update local model
                self._update_local_model(model_data)
                self.last_sync_time = datetime.now()
                logger.info("Model sync completed successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Model sync failed: {e}")
            return False
    
    def _update_local_model(self, model_data: Dict):
        """Update local KNN model with distributed data."""
        try:
            if not self.local_knn:
                return
            
            # Decode model data
            X_train = np.frombuffer(base64.b64decode(model_data["X_train"]), dtype=np.float32)
            X_train = X_train.reshape(model_data["X_shape"])
            
            y_train = model_data["y_train"]
            class_names = model_data["class_names"]
            
            # Update local model
            self.local_knn.X_train = X_train
            self.local_knn.y_train = np.array(y_train)
            self.local_knn.class_names = class_names
            
            # Retrain KNN
            if len(X_train) > 0:
                self.local_knn.knn.fit(X_train, y_train)
                logger.info(f"Updated local model: {len(class_names)} classes, {len(X_train)} samples")
            
        except Exception as e:
            logger.error(f"Failed to update local model: {e}")
    
    def flush_pending_annotations(self) -> int:
        """
        Submit all pending annotations.
        
        Returns:
            Number of successfully submitted annotations
        """
        if not self.is_connected or not self.pending_annotations:
            return 0
        
        submitted = 0
        failed = []
        
        for annotation in self.pending_annotations:
            if self._submit_to_modal(annotation):
                submitted += 1
            else:
                failed.append(annotation)
        
        self.pending_annotations = failed
        logger.info(f"Flushed {submitted} annotations, {len(failed)} failed")
        return submitted
    
    def get_contribution_stats(self) -> Dict:
        """Get contribution statistics."""
        try:
            if self.is_connected and self.stub:
                with self.stub.run():
                    from .modal_functions import get_user_stats
                    stats = get_user_stats.remote(self.user_id)
                return stats
            else:
                # Return local stats
                return {
                    "user_id": self.user_id,
                    "contributions": self.contribution_count,
                    "pending": len(self.pending_annotations),
                    "last_sync": self.last_sync_time.isoformat() if self.last_sync_time else None,
                    "connected": self.is_connected
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def get_global_stats(self) -> Dict:
        """Get global training statistics."""
        try:
            if self.is_connected and self.stub:
                with self.stub.run():
                    from .modal_functions import get_global_stats
                    return get_global_stats.remote()
            return {}
        except Exception as e:
            logger.error(f"Failed to get global stats: {e}")
            return {}
    
    def enable_federation(self, enable: bool = True):
        """Enable/disable federated learning mode."""
        self.config.enable_federation = enable
        logger.info(f"Federated learning: {'enabled' if enable else 'disabled'}")
    
    def disconnect(self):
        """Disconnect from Modal and save pending work."""
        if self.pending_annotations:
            # Save pending annotations locally
            cache_file = f".pending_annotations_{self.user_id}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump(self.pending_annotations, f)
                logger.info(f"Saved {len(self.pending_annotations)} pending annotations")
            except Exception as e:
                logger.error(f"Failed to save pending annotations: {e}")
        
        self.is_connected = False
        logger.info("Disconnected from distributed training")
