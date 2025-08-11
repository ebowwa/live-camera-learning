#!/usr/bin/env python3
"""
Test script for the complete RTSP → Capture → Annotation pipeline
"""

import sys
import os
import time
import threading
from pathlib import Path

def test_rtsp_capture():
    """Test the intelligent capture system with RTSP."""
    print("=" * 60)
    print("RTSP CAPTURE TEST")
    print("=" * 60)
    
    # Import the intelligent capture system
    from edaxshifu.intelligent_capture import IntelligentCaptureSystem
    
    # Use webcam (0) for testing, but can be replaced with RTSP URL
    rtsp_url = "0"  # Change to RTSP URL like "rtsp://admin:admin@192.168.42.1:554/live"
    
    system = IntelligentCaptureSystem(
        rtsp_url=rtsp_url,
        yolo_model_path="python/assets/yolo11n.onnx",
        capture_dir="python/data/captures",
        confidence_threshold=0.5
    )
    
    # Load any existing training samples
    training_dir = "python/assets/images"
    if os.path.exists(training_dir):
        print(f"Loading training samples from {training_dir}")
        system.knn.add_samples_from_directory(training_dir)
        print(f"Loaded {len(system.knn.X_train)} training samples")
    
    print(f"\nStarting capture from: {rtsp_url}")
    print("Press 's' to manually trigger capture")
    print("Press ESC to stop\n")
    
    # Run for a limited time in test mode
    try:
        system.run(display=True)
    except KeyboardInterrupt:
        pass
    
    print(f"\nCapture Stats:")
    print(f"  Total captures: {system.stats['captures']}")
    print(f"  Successful recognitions: {system.stats['successful_recognitions']}")
    print(f"  Failed recognitions: {system.stats['failed_recognitions']}")
    
    return system.stats['failed_recognitions'] > 0


def test_annotation_interface():
    """Test the annotation interface."""
    print("\n" + "=" * 60)
    print("ANNOTATION INTERFACE TEST")
    print("=" * 60)
    
    from edaxshifu.annotation_interface import AnnotationInterface
    
    # Check if there are failed captures to annotate
    failed_dir = Path("python/data/captures/failed")
    if not failed_dir.exists():
        print("No failed directory found. Creating test data...")
        failed_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a test image if none exist
        import numpy as np
        import cv2
        import json
        
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_path = failed_dir / "test_capture.jpg"
        cv2.imwrite(str(test_path), test_img)
        
        # Create metadata
        meta_path = failed_dir / "test_capture.json"
        with open(meta_path, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "knn_prediction": "unknown",
                "confidence": 0.3,
                "detections": [{"class_name": "object", "confidence": 0.6}]
            }, f)
        print(f"Created test capture at {test_path}")
    
    # Count available annotations
    json_files = list(failed_dir.glob("*.json"))
    print(f"Found {len(json_files)} failed captures to annotate")
    
    if len(json_files) > 0:
        print("\nStarting annotation interface...")
        print("This will open in your browser at http://localhost:7860")
        print("Press Ctrl+C to stop\n")
        
        interface = AnnotationInterface(
            failed_dir="python/data/captures/failed",
            knn_classifier=None  # Will create new one if needed
        )
        
        # Run in a thread so we can stop it
        def run_interface():
            interface.launch()
        
        thread = threading.Thread(target=run_interface, daemon=True)
        thread.start()
        
        print("Interface launched! Check your browser.")
        print("Annotate some images, then press Enter to continue...")
        input()
    else:
        print("No failed captures to annotate.")


def test_full_pipeline():
    """Test the complete pipeline."""
    print("=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)
    print("\nThis test will:")
    print("1. Capture frames from RTSP/webcam")
    print("2. Run YOLO detection")
    print("3. Classify with KNN")
    print("4. Route failed recognitions for annotation")
    print("5. Update KNN with human feedback")
    print("\n" + "=" * 60)
    
    # Test capture system
    has_failures = test_rtsp_capture()
    
    if has_failures:
        print("\nFailed recognitions detected!")
        print("You can now annotate them...")
        time.sleep(2)
        
        # Test annotation interface
        test_annotation_interface()
    else:
        print("\nNo failed recognitions. All captures were successful!")
        print("Try capturing unknown objects to test the annotation flow.")
    
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RTSP Pipeline')
    parser.add_argument(
        '--component',
        choices=['capture', 'annotation', 'full'],
        default='full',
        help='Which component to test'
    )
    parser.add_argument(
        '--rtsp-url',
        default='0',
        help='RTSP URL or webcam index (default: 0 for webcam)'
    )
    
    args = parser.parse_args()
    
    if args.component == 'capture':
        test_rtsp_capture()
    elif args.component == 'annotation':
        test_annotation_interface()
    else:
        test_full_pipeline()


if __name__ == "__main__":
    main()
