"""
Modal configuration for distributed training.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModalConfig:
    """Configuration for Modal distributed training."""
    
    # Modal app settings
    app_name: str = "edaxshifu-distributed"
    stub_name: str = "edaxshifu"
    
    # Resource settings
    gpu: str = "T4"  # GPU type for model operations
    cpu: float = 2.0  # CPU cores
    memory: int = 8192  # Memory in MB
    timeout: int = 300  # Function timeout in seconds
    
    # Storage settings
    storage_backend: str = "modal"  # or "s3", "gcs"
    model_storage_path: str = "/models"
    annotation_storage_path: str = "/annotations"
    
    # Training settings
    batch_size: int = 32
    aggregation_interval: int = 1800  # 30 minutes in seconds
    min_annotations_for_update: int = 10
    consensus_threshold: float = 0.7  # For annotation validation
    
    # Network settings
    enable_federation: bool = False  # Federated learning mode
    enable_encryption: bool = True  # Encrypt model updates
    max_contributors: int = 1000
    
    # Authentication
    require_auth: bool = True
    api_key_env: str = "MODAL_API_KEY"
    
    @classmethod
    def from_env(cls) -> "ModalConfig":
        """Create config from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("MODAL_APP_NAME"):
            config.app_name = os.getenv("MODAL_APP_NAME")
        if os.getenv("MODAL_GPU"):
            config.gpu = os.getenv("MODAL_GPU")
        if os.getenv("MODAL_BATCH_SIZE"):
            config.batch_size = int(os.getenv("MODAL_BATCH_SIZE"))
        if os.getenv("MODAL_ENABLE_FEDERATION"):
            config.enable_federation = os.getenv("MODAL_ENABLE_FEDERATION").lower() == "true"
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "app_name": self.app_name,
            "gpu": self.gpu,
            "cpu": self.cpu,
            "memory": self.memory,
            "timeout": self.timeout,
            "storage_backend": self.storage_backend,
            "batch_size": self.batch_size,
            "aggregation_interval": self.aggregation_interval,
            "enable_federation": self.enable_federation,
            "enable_encryption": self.enable_encryption,
        }
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        valid = True
        
        if self.batch_size < 1:
            logger.error("Batch size must be at least 1")
            valid = False
            
        if self.consensus_threshold < 0 or self.consensus_threshold > 1:
            logger.error("Consensus threshold must be between 0 and 1")
            valid = False
            
        if self.aggregation_interval < 60:
            logger.error("Aggregation interval must be at least 60 seconds")
            valid = False
            
        return valid