"""
Modal serverless functions for distributed training.
This file defines the cloud functions that run on Modal.
"""

import os
import json
import base64
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

# These imports will work in Modal environment
try:
    import modal
    from modal import Image, Stub, Volume, Secret
    
    # Define the container image with required dependencies
    image = (
        Image.debian_slim()
        .pip_install(
            "numpy",
            "scikit-learn",
            "torch",
            "torchvision",
            "pillow",
            "opencv-python-headless",
        )
    )
    
    # Create the Modal stub
    stub = Stub("edaxshifu-distributed")
    
    # Create a volume for persistent storage
    volume = Volume.persisted("edaxshifu-models")
    
    MODAL_AVAILABLE = True
except ImportError:
    # For local development/testing
    MODAL_AVAILABLE = False
    stub = None
    volume = None
    image = None

# Global model storage (in production, use persistent storage)
GLOBAL_MODEL_PATH = "/models/global"
USER_MODELS_PATH = "/models/users"
ANNOTATIONS_PATH = "/annotations"

logger = logging.getLogger(__name__)

if MODAL_AVAILABLE:
    
    @stub.function(
        image=image,
        gpu="T4",
        timeout=300,
        volumes={"/models": volume},
    )
    def submit_annotation(annotation: Dict) -> Dict:
        """
        Process and store an annotation submission.
        
        Args:
            annotation: Annotation data with image, label, user_id, etc.
            
        Returns:
            Response with success status
        """
        try:
            import torch
            import torchvision.transforms as transforms
            from torchvision.models import resnet18
            from PIL import Image as PILImage
            import io
            
            # Decode image
            image_bytes = base64.b64decode(annotation["image"])
            image = PILImage.open(io.BytesIO(image_bytes))
            
            # Extract features using ResNet18
            model = resnet18(pretrained=True)
            model.eval()
            
            # Remove the final classification layer
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
            # Preprocess image
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
            ])
            
            input_tensor = preprocess(image).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = model(input_tensor).squeeze().numpy()
            
            # Store annotation with features
            annotation_id = f"{annotation['user_id']}_{datetime.now().timestamp()}"
            annotation_data = {
                "id": annotation_id,
                "user_id": annotation["user_id"],
                "timestamp": annotation["timestamp"],
                "label": annotation["label"],
                "confidence": annotation["confidence"],
                "features": base64.b64encode(features.tobytes()).decode(),
                "feature_shape": features.shape,
                "metadata": annotation.get("metadata", {})
            }
            
            # Save to persistent storage
            save_path = f"{ANNOTATIONS_PATH}/pending/{annotation_id}.json"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(annotation_data, f)
            
            # Trigger aggregation if enough annotations
            pending_count = len(os.listdir(f"{ANNOTATIONS_PATH}/pending"))
            if pending_count >= 10:  # Configurable threshold
                aggregate_models.spawn()
            
            return {"success": True, "annotation_id": annotation_id, "pending_count": pending_count}
            
        except Exception as e:
            logger.error(f"Failed to process annotation: {e}")
            return {"success": False, "error": str(e)}
    
    @stub.function(
        image=image,
        gpu="T4",
        timeout=600,
        volumes={"/models": volume},
        schedule=modal.Period(minutes=30),  # Run every 30 minutes
    )
    def aggregate_models():
        """
        Aggregate annotations and update global model.
        Runs periodically to combine contributions from all users.
        """
        try:
            from sklearn.neighbors import KNeighborsClassifier
            import shutil
            
            # Load pending annotations
            pending_dir = f"{ANNOTATIONS_PATH}/pending"
            if not os.path.exists(pending_dir):
                return {"success": False, "message": "No pending annotations"}
            
            annotation_files = os.listdir(pending_dir)
            if len(annotation_files) < 10:
                return {"success": False, "message": "Not enough annotations"}
            
            # Collect features and labels
            X_new = []
            y_new = []
            processed_ids = []
            
            for filename in annotation_files[:100]:  # Process up to 100 at a time
                filepath = os.path.join(pending_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        ann = json.load(f)
                    
                    # Decode features
                    features = np.frombuffer(
                        base64.b64decode(ann["features"]), 
                        dtype=np.float32
                    )
                    
                    X_new.append(features)
                    y_new.append(ann["label"])
                    processed_ids.append(ann["id"])
                    
                except Exception as e:
                    logger.error(f"Failed to process annotation {filename}: {e}")
                    continue
            
            if not X_new:
                return {"success": False, "message": "No valid annotations"}
            
            X_new = np.array(X_new)
            
            # Load existing global model
            global_model_file = f"{GLOBAL_MODEL_PATH}/model.npz"
            os.makedirs(GLOBAL_MODEL_PATH, exist_ok=True)
            
            if os.path.exists(global_model_file):
                # Load and update existing model
                data = np.load(global_model_file, allow_pickle=True)
                X_train = data["X_train"]
                y_train = data["y_train"].tolist()
                class_names = data["class_names"].tolist()
                
                # Append new data
                X_train = np.vstack([X_train, X_new])
                y_train.extend(y_new)
                
                # Update class names
                for label in y_new:
                    if label not in class_names:
                        class_names.append(label)
                
            else:
                # Create new model
                X_train = X_new
                y_train = y_new
                class_names = list(set(y_new))
            
            # Limit model size (configurable)
            max_samples = 10000
            if len(X_train) > max_samples:
                # Keep most recent samples
                X_train = X_train[-max_samples:]
                y_train = y_train[-max_samples:]
            
            # Save updated model
            np.savez(
                global_model_file,
                X_train=X_train,
                y_train=np.array(y_train),
                class_names=np.array(class_names),
                last_update=datetime.now().isoformat(),
                total_samples=len(X_train),
                version=datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            
            # Move processed annotations
            processed_dir = f"{ANNOTATIONS_PATH}/processed"
            os.makedirs(processed_dir, exist_ok=True)
            
            for ann_id in processed_ids:
                src = f"{pending_dir}/{ann_id}.json"
                dst = f"{processed_dir}/{ann_id}.json"
                if os.path.exists(src):
                    shutil.move(src, dst)
            
            logger.info(f"Aggregated {len(X_new)} new annotations. Total: {len(X_train)}")
            return {
                "success": True,
                "new_annotations": len(X_new),
                "total_samples": len(X_train),
                "classes": len(class_names)
            }
            
        except Exception as e:
            logger.error(f"Model aggregation failed: {e}")
            return {"success": False, "error": str(e)}
    
    @stub.function(
        image=image,
        volumes={"/models": volume},
        timeout=60,
    )
    def get_global_model(user_id: str) -> Optional[Dict]:
        """
        Get the current global model for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Model data or None
        """
        try:
            global_model_file = f"{GLOBAL_MODEL_PATH}/model.npz"
            
            if not os.path.exists(global_model_file):
                return None
            
            # Load model
            data = np.load(global_model_file, allow_pickle=True)
            
            # Encode for transmission
            model_data = {
                "X_train": base64.b64encode(data["X_train"].tobytes()).decode(),
                "X_shape": data["X_train"].shape,
                "y_train": data["y_train"].tolist(),
                "class_names": data["class_names"].tolist(),
                "last_update": str(data["last_update"]),
                "total_samples": int(data["total_samples"]),
                "version": str(data["version"])
            }
            
            # Log user sync
            sync_log = f"{USER_MODELS_PATH}/{user_id}/sync_log.json"
            os.makedirs(os.path.dirname(sync_log), exist_ok=True)
            
            sync_entry = {
                "timestamp": datetime.now().isoformat(),
                "version": model_data["version"],
                "samples": model_data["total_samples"]
            }
            
            # Append to log
            log_data = []
            if os.path.exists(sync_log):
                with open(sync_log, 'r') as f:
                    log_data = json.load(f)
            log_data.append(sync_entry)
            
            with open(sync_log, 'w') as f:
                json.dump(log_data[-100:], f)  # Keep last 100 entries
            
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to get global model: {e}")
            return None
    
    @stub.function(
        image=image,
        volumes={"/models": volume},
        timeout=30,
    )
    def get_user_stats(user_id: str) -> Dict:
        """
        Get statistics for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            User statistics
        """
        try:
            stats = {
                "user_id": user_id,
                "total_contributions": 0,
                "accepted_contributions": 0,
                "last_contribution": None,
                "sync_count": 0,
                "last_sync": None
            }
            
            # Count user annotations
            for folder in ["pending", "processed"]:
                folder_path = f"{ANNOTATIONS_PATH}/{folder}"
                if os.path.exists(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.startswith(user_id):
                            stats["total_contributions"] += 1
                            if folder == "processed":
                                stats["accepted_contributions"] += 1
            
            # Get sync history
            sync_log = f"{USER_MODELS_PATH}/{user_id}/sync_log.json"
            if os.path.exists(sync_log):
                with open(sync_log, 'r') as f:
                    log_data = json.load(f)
                    stats["sync_count"] = len(log_data)
                    if log_data:
                        stats["last_sync"] = log_data[-1]["timestamp"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return {"error": str(e)}
    
    @stub.function(
        image=image,
        volumes={"/models": volume},
        timeout=30,
    )
    def get_global_stats() -> Dict:
        """
        Get global training statistics.
        
        Returns:
            Global statistics
        """
        try:
            stats = {
                "total_users": 0,
                "total_annotations": 0,
                "pending_annotations": 0,
                "processed_annotations": 0,
                "model_version": None,
                "model_samples": 0,
                "model_classes": 0,
                "last_aggregation": None
            }
            
            # Count annotations
            for folder in ["pending", "processed"]:
                folder_path = f"{ANNOTATIONS_PATH}/{folder}"
                if os.path.exists(folder_path):
                    count = len(os.listdir(folder_path))
                    stats["total_annotations"] += count
                    if folder == "pending":
                        stats["pending_annotations"] = count
                    else:
                        stats["processed_annotations"] = count
            
            # Count unique users
            user_ids = set()
            user_dir = USER_MODELS_PATH
            if os.path.exists(user_dir):
                user_ids.update(os.listdir(user_dir))
            stats["total_users"] = len(user_ids)
            
            # Get model info
            global_model_file = f"{GLOBAL_MODEL_PATH}/model.npz"
            if os.path.exists(global_model_file):
                data = np.load(global_model_file, allow_pickle=True)
                stats["model_version"] = str(data["version"])
                stats["model_samples"] = int(data["total_samples"])
                stats["model_classes"] = len(data["class_names"])
                stats["last_aggregation"] = str(data["last_update"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get global stats: {e}")
            return {"error": str(e)}