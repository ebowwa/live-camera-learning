#!/usr/bin/env python3
"""Simple test script for YOLO detection on a single image."""

import cv2
import numpy as np
from src.yolo_detector import YOLODetector

def test_yolo_on_image():
    # Initialize detector
    detector = YOLODetector("assets/yolo11n.onnx")
    
    # Create a test image (or load one)
    # For testing, let's create a simple colored rectangle
    test_image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), -1)
    cv2.putText(test_image, "Test Image", (200, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Try loading a real image if available
    try:
        test_image = cv2.imread("assets/images/apple1.png")
        if test_image is None:
            print("Using generated test image")
            test_image = np.ones((640, 640, 3), dtype=np.uint8) * 255
        else:
            print("Loaded apple1.png for testing")
    except:
        print("Using generated test image")
    
    # Run detection
    print("Running YOLO detection...")
    detections = detector.detect(test_image)
    
    # Print results
    if detections:
        print(f"Found {len(detections)} objects:")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")
            print(f"    BBox: {det['bbox']}")
    else:
        print("No objects detected")
    
    # Test hand detection
    print("\nTesting hand-with-object detection...")
    hand_detections = detector.detect_hands_with_objects(test_image)
    
    if hand_detections:
        print(f"Found {len(hand_detections)} handheld objects:")
        for det in hand_detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")
            print(f"    Holding gesture: {det.get('holding_gesture', False)}")
    else:
        print("No handheld objects detected")
    
    # Draw and save result
    result_image = detector.draw_detections(test_image, detections)
    cv2.imwrite("test_yolo_result.jpg", result_image)
    print("\nResult saved to test_yolo_result.jpg")

if __name__ == "__main__":
    test_yolo_on_image()