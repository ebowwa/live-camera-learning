#!/usr/bin/env python3
"""
Example script demonstrating the HandDetector utility class.
This can be used with any video source: webcam, video file, or RTSP stream.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

from edaxshifu.hand_detector import HandDetector, HandDetection


def process_webcam():
    """Process webcam feed with hand detection."""
    print("Starting webcam hand detection...")
    print("Press 'q' to quit, 's' to save snapshot")
    
    # Initialize detector
    detector = HandDetector(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect hands
        detections = detector.detect(frame)
        
        # Process each detection
        for i, detection in enumerate(detections):
            # Draw landmarks
            frame = detector.draw_landmarks(
                frame, detection,
                draw_connections=True,
                draw_bounding_box=True
            )
            
            # Detect and display gesture
            gesture = detector.detect_gesture(detection)
            if gesture:
                cv2.putText(
                    frame,
                    f"Hand {i+1}: {gesture}",
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Check for pinch
            if detector.is_pinching(detection):
                cv2.putText(
                    frame,
                    "PINCHING!",
                    (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
        
        # Display frame
        cv2.imshow("Hand Detection", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("hand_snapshot.jpg", frame)
            print("Snapshot saved!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.release()


def process_image(image_path: str):
    """Process a single image for hand detection."""
    print(f"Processing image: {image_path}")
    
    # Initialize detector
    detector = HandDetector(
        max_num_hands=2,
        min_detection_confidence=0.5,
        static_image_mode=True  # Important for single images
    )
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Detect hands
    detections = detector.detect(image)
    
    print(f"Found {len(detections)} hand(s)")
    
    # Annotate image
    annotated = image.copy()
    for i, detection in enumerate(detections):
        # Draw landmarks
        annotated = detector.draw_landmarks(
            annotated, detection,
            draw_connections=True,
            draw_bounding_box=True
        )
        
        # Detect gesture
        gesture = detector.detect_gesture(detection)
        print(f"Hand {i+1}: {detection.handedness}, Gesture: {gesture}, "
              f"Confidence: {detection.confidence:.2f}")
        
        # Get fingertip positions
        fingertips = detection.get_fingertip_positions()
        for finger, landmark in fingertips.items():
            if landmark:
                x, y = landmark.to_pixel_coords(image.shape[1], image.shape[0])
                cv2.circle(annotated, (x, y), 8, (255, 255, 0), -1)
                cv2.putText(
                    annotated,
                    finger[:1].upper(),
                    (x-5, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1
                )
    
    # Save and display result
    output_path = "hand_detection_result.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"Result saved to {output_path}")
    
    # Display result
    cv2.imshow("Hand Detection Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    detector.release()


def benchmark_performance():
    """Benchmark the hand detector performance."""
    import time
    
    print("Benchmarking hand detector performance...")
    
    # Initialize detector
    detector = HandDetector(
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    
    # Create a dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(10):
        detector.detect(dummy_image)
    
    # Benchmark
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        detector.detect(dummy_image)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    fps = 1.0 / avg_time
    
    print(f"Average processing time: {avg_time*1000:.2f} ms")
    print(f"Theoretical FPS: {fps:.1f}")
    
    detector.release()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hand Detection Examples')
    parser.add_argument(
        '--mode',
        choices=['webcam', 'image', 'benchmark'],
        default='webcam',
        help='Mode of operation'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file (for image mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        process_webcam()
    elif args.mode == 'image':
        if not args.image:
            print("Error: --image path required for image mode")
            return
        process_image(args.image)
    elif args.mode == 'benchmark':
        benchmark_performance()


if __name__ == "__main__":
    main()
