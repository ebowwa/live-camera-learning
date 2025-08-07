#!/usr/bin/env python3
"""
Generate test data for the annotation interface.
Creates some failed recognition samples for testing.
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime

def create_test_images():
    """Create test images in the failed directory."""
    
    # Create directories
    failed_dir = "intelligent_captures/failed"
    os.makedirs(failed_dir, exist_ok=True)
    
    # Generate some test images
    for i in range(5):
        # Create a random colored image with text
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some text to make it identifiable
        text = f"Test Object {i+1}"
        cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"failed_{timestamp}_{i}.jpg"
        filepath = os.path.join(failed_dir, filename)
        cv2.imwrite(filepath, img)
        
        # Create metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "filepath": filepath,
            "yolo_detections": [
                {
                    "class_name": f"unknown_object_{i}",
                    "confidence": 0.3 + i * 0.05,
                    "bbox": [100, 100, 200, 200]
                }
            ],
            "recognition": {
                "label": "unknown",
                "confidence": 0.2,
                "is_known": False
            }
        }
        
        metadata_file = filepath.replace('.jpg', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Created test image: {filepath}")
    
    print(f"\nCreated 5 test images in {failed_dir}")
    print("You can now run the annotation interface to label these images.")

if __name__ == "__main__":
    create_test_images()