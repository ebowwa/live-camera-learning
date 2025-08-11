#!/usr/bin/env python3
"""
Test script for hand detection functionality.
Tests the HandDetector class without requiring a camera.
"""

import numpy as np
import cv2
import sys
from edaxshifu.hand_detector import HandDetector


def test_hand_detector_initialization():
    """Test that HandDetector initializes correctly."""
    print("Testing HandDetector initialization...")
    
    try:
        detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ HandDetector initialized successfully")
        detector.release()
        return True
    except Exception as e:
        print(f"✗ Failed to initialize HandDetector: {e}")
        return False


def test_empty_image_detection():
    """Test detection on an empty image."""
    print("\nTesting detection on empty image...")
    
    try:
        detector = HandDetector()
        
        # Create a blank image
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run detection
        detections = detector.detect(blank_image)
        
        print(f"✓ Detection completed. Found {len(detections)} hands (expected: 0)")
        
        detector.release()
        return len(detections) == 0
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        return False


def test_synthetic_hand_image():
    """Test detection on a synthetic image with hand-like features."""
    print("\nTesting synthetic hand image...")
    
    try:
        detector = HandDetector(static_image_mode=True)
        
        # Create a synthetic image with some features
        image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Add some colored regions that might look like skin
        cv2.circle(image, (320, 240), 50, (150, 130, 200), -1)  # Palm-like circle
        cv2.circle(image, (320, 180), 20, (150, 130, 200), -1)  # Finger-like circle
        cv2.circle(image, (280, 190), 20, (150, 130, 200), -1)  # Finger-like circle
        cv2.circle(image, (360, 190), 20, (150, 130, 200), -1)  # Finger-like circle
        
        # Run detection
        detections = detector.detect(image)
        
        print(f"✓ Detection completed. Found {len(detections)} hands")
        
        # Test drawing functions
        if len(detections) > 0:
            annotated = detector.draw_landmarks(
                image,
                detections[0],
                draw_connections=True,
                draw_bounding_box=True
            )
            print("✓ Drawing functions work correctly")
            
            # Test gesture detection
            gesture = detector.detect_gesture(detections[0])
            print(f"✓ Gesture detection works. Detected: {gesture}")
        
        detector.release()
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_hand_detection_methods():
    """Test various HandDetection methods."""
    print("\nTesting HandDetection methods...")
    
    try:
        from edaxshifu.hand_detector import HandDetection, HandLandmark
        
        # Create mock landmarks
        landmarks = [
            HandLandmark(x=0.5, y=0.5, z=0.0)
            for _ in range(21)  # Hand has 21 landmarks
        ]
        
        # Create a HandDetection object
        detection = HandDetection(
            landmarks=landmarks,
            handedness="Right",
            confidence=0.95
        )
        
        # Test methods
        fingertips = detection.get_fingertip_positions()
        print(f"✓ get_fingertip_positions returned {len(fingertips)} fingertips")
        
        bbox = detection.calculate_bounding_box(640, 480)
        print(f"✓ calculate_bounding_box returned: {bbox}")
        
        landmark = detection.get_landmark(8)  # Index finger tip
        if landmark:
            print("✓ get_landmark works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Method test failed: {e}")
        return False


def test_performance():
    """Test the performance of hand detection."""
    print("\nTesting performance...")
    
    import time
    
    try:
        detector = HandDetector()
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Warm up
        for _ in range(5):
            detector.detect(test_image)
        
        # Measure performance
        num_iterations = 20
        start_time = time.time()
        
        for _ in range(num_iterations):
            detector.detect(test_image)
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / num_iterations
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"✓ Average processing time: {avg_time*1000:.2f} ms")
        print(f"✓ Theoretical FPS: {fps:.1f}")
        
        detector.release()
        return True
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Hand Detection Test Suite")
    print("=" * 50)
    
    tests = [
        test_hand_detector_initialization,
        test_empty_image_detection,
        test_hand_detection_methods,
        test_synthetic_hand_image,
        test_performance
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
