"""
Bounding box utilities for object detection and cropping.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def draw_bounding_boxes(image: np.ndarray, bounding_boxes: List[Dict[str, Any]], 
                       colors: Optional[Dict[str, Tuple[int, int, int]]] = None) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Input image (BGR format)
        bounding_boxes: List of bounding box dictionaries with keys:
            - x, y, w, h: coordinates (normalized if 'normalized' key is True)
            - label: object label
            - confidence: detection confidence
            - normalized: whether coordinates are normalized (0-1)
        colors: Optional color mapping for labels
        
    Returns:
        Image with bounding boxes drawn
    """
    if not bounding_boxes:
        return image.copy()
    
    result_image = image.copy()
    height, width = image.shape[:2]
    
    # Default colors
    default_colors = {
        'default': (0, 255, 0),  # Green
        'person': (255, 0, 0),   # Blue
        'car': (0, 0, 255),      # Red
        'apple': (0, 255, 255),  # Yellow
        'phone': (255, 0, 255),  # Magenta
    }
    if colors:
        default_colors.update(colors)
    
    for bbox in bounding_boxes:
        try:
            # Extract coordinates
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            label = bbox.get('label', 'object')
            confidence = bbox.get('confidence', 0.0)
            normalized = bbox.get('normalized', False)
            
            # Convert normalized coordinates to pixel coordinates
            if normalized:
                x = int(x * width)
                y = int(y * height)
                w = int(w * width)
                h = int(h * height)
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            # Get color for this label
            color = default_colors.get(label.lower(), default_colors['default'])
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with background
            label_text = f"{label} ({confidence:.2f})"
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_w, label_h = label_size
            
            # Draw label background
            cv2.rectangle(result_image, (x, y - label_h - 10), (x + label_w + 10, y), color, -1)
            
            # Draw label text
            cv2.putText(result_image, label_text, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error drawing bounding box {bbox}: {e}")
            continue
    
    return result_image


def crop_object_from_bbox(image: np.ndarray, bbox: Dict[str, Any], 
                         padding: int = 10) -> Optional[np.ndarray]:
    """
    Crop an object from an image using bounding box coordinates.
    
    Args:
        image: Input image (BGR format)
        bbox: Bounding box dictionary with x, y, w, h coordinates
        padding: Additional padding around the object in pixels
        
    Returns:
        Cropped image or None if invalid bbox
    """
    try:
        height, width = image.shape[:2]
        
        # Extract coordinates
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        normalized = bbox.get('normalized', False)
        
        # Convert normalized coordinates to pixel coordinates
        if normalized:
            x = int(x * width)
            y = int(y * height)
            w = int(w * width)
            h = int(h * height)
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        # Ensure valid crop region
        if w <= 0 or h <= 0:
            return None
        
        # Crop the image
        cropped = image[y:y+h, x:x+w]
        
        return cropped if cropped.size > 0 else None
        
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Error cropping object from bbox {bbox}: {e}")
        return None


def crop_all_objects(image: np.ndarray, bounding_boxes: List[Dict[str, Any]], 
                    padding: int = 10) -> List[Dict[str, Any]]:
    """
    Crop all objects from an image using their bounding boxes.
    
    Args:
        image: Input image (BGR format)
        bounding_boxes: List of bounding box dictionaries
        padding: Additional padding around objects
        
    Returns:
        List of dictionaries with 'label', 'confidence', 'cropped_image' keys
    """
    results = []
    
    for bbox in bounding_boxes:
        cropped = crop_object_from_bbox(image, bbox, padding)
        if cropped is not None:
            results.append({
                'label': bbox.get('label', 'object'),
                'confidence': bbox.get('confidence', 0.0),
                'cropped_image': cropped,
                'original_bbox': bbox
            })
    
    return results


def normalize_bbox_coordinates(bbox: Dict[str, Any], image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Normalize bounding box coordinates to 0-1 range.
    
    Args:
        bbox: Bounding box dictionary
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Normalized bounding box dictionary
    """
    if bbox.get('normalized', False):
        return bbox.copy()
    
    normalized_bbox = bbox.copy()
    normalized_bbox.update({
        'x': bbox['x'] / image_width,
        'y': bbox['y'] / image_height,
        'w': bbox['w'] / image_width,
        'h': bbox['h'] / image_height,
        'normalized': True
    })
    
    return normalized_bbox


def denormalize_bbox_coordinates(bbox: Dict[str, Any], image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Convert normalized bounding box coordinates to pixel coordinates.
    
    Args:
        bbox: Normalized bounding box dictionary
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Pixel-coordinate bounding box dictionary
    """
    if not bbox.get('normalized', False):
        return bbox.copy()
    
    pixel_bbox = bbox.copy()
    pixel_bbox.update({
        'x': int(bbox['x'] * image_width),
        'y': int(bbox['y'] * image_height),
        'w': int(bbox['w'] * image_width),
        'h': int(bbox['h'] * image_height),
        'normalized': False
    })
    
    return pixel_bbox