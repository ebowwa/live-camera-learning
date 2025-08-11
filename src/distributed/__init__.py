"""
Distributed training module for EdaxShifu using Modal.
This module is optional and can be disabled if Modal is not available.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import Modal, but make it optional
MODAL_AVAILABLE = False
modal_app = None

try:
    import modal
    MODAL_AVAILABLE = True
    logger.info("Modal library detected - distributed training available")
except ImportError:
    logger.info("Modal not installed - distributed training disabled")
    logger.info("Install with: pip install modal")

def is_distributed_available() -> bool:
    """Check if distributed training is available."""
    return MODAL_AVAILABLE

__all__ = ['MODAL_AVAILABLE', 'is_distributed_available', 'modal_app']