#!/usr/bin/env python3

import argparse
import sys
import logging
from src.rtsp_stream import RTSPStream, RTSPViewer
from src.integrated_detector import IntegratedDetector, HandObjectTrigger

# Optional import for hand detection (requires mediapipe)
try:
    from src.hand_stream_viewer import HandStreamViewer, demo_gesture_handler
    HAND_DETECTION_AVAILABLE = True
except ImportError:
    HAND_DETECTION_AVAILABLE = False
    print("Note: Hand detection mode unavailable (mediapipe not compatible with Python 3.13)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_basic_stream(args):
    """Run basic RTSP streaming without detection."""
    stream = RTSPStream(args.url)
    viewer = RTSPViewer(stream, args.window_name)
    
    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\nStream interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_detection_mode(args):
    """Run RTSP with YOLO detection and auto-capture."""
    detector = IntegratedDetector(
        rtsp_url=args.url,
        yolo_model_path=args.model_path,
        capture_dir=args.capture_dir,
        conf_threshold=args.conf_threshold,
        capture_cooldown=args.capture_cooldown
    )
    
    # Set up capture callback if needed
    def on_capture(frame, detections):
        objects = ", ".join([d['class_name'] for d in detections])
        logger.info(f"Captured objects: {objects}")
    
    detector.set_capture_callback(on_capture)
    
    try:
        detector.run(display=not args.headless, window_name=args.window_name)
    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_hand_detection_mode(args):
    """Run RTSP with hand detection and gesture recognition."""
    if not HAND_DETECTION_AVAILABLE:
        print("Hand detection mode is not available. Please use Python 3.10-3.12 for mediapipe support.")
        print("You can still use --mode detect for YOLO-based detection.")
        sys.exit(1)
        
    stream = RTSPStream(args.url)
    viewer = HandStreamViewer(
        stream,
        window_name=args.window_name,
        max_hands=args.max_hands,
        detection_confidence=args.hand_confidence
    )
    
    # Set up gesture callback
    viewer.set_gesture_callback(demo_gesture_handler)
    
    # Optional: Set up custom hand callback
    def on_hands_detected(detections):
        if len(detections) > 0:
            logger.debug(f"Detected {len(detections)} hand(s)")
    
    viewer.set_hand_callback(on_hands_detected)
    
    try:
        viewer.run(show_stats=not args.no_stats)
    except KeyboardInterrupt:
        print("\nHand detection interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Intelligent RTSP Camera System')
    
    # Common arguments
    parser.add_argument(
        '--url',
        type=str,
        default='rtsp://admin:admin@192.168.42.1:554/live',
        help='RTSP stream URL (default: rtsp://admin:admin@192.168.42.1:554/live)'
    )
    parser.add_argument(
        '--window-name',
        type=str,
        default='Smart Camera',
        help='Window name for display (default: Smart Camera)'
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['stream', 'detect', 'hand'],
        default='stream',
        help='Operation mode: stream (basic), detect (YOLO), or hand (hand detection)'
    )
    
    # Detection mode arguments
    parser.add_argument(
        '--model-path',
        type=str,
        default='assets/yolo11n.onnx',
        help='Path to YOLO ONNX model (default: assets/yolo11n.onnx)'
    )
    parser.add_argument(
        '--capture-dir',
        type=str,
        default='captures',
        help='Directory for captured frames (default: captures)'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for detection (default: 0.5)'
    )
    parser.add_argument(
        '--capture-cooldown',
        type=float,
        default=2.0,
        help='Minimum seconds between captures (default: 2.0)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without display window'
    )
    
    # Hand detection mode arguments
    parser.add_argument(
        '--max-hands',
        type=int,
        default=2,
        help='Maximum number of hands to detect (default: 2)'
    )
    parser.add_argument(
        '--hand-confidence',
        type=float,
        default=0.5,
        help='Hand detection confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Disable statistics overlay in hand mode'
    )
    
    args = parser.parse_args()
    
    # Run appropriate mode
    if args.mode == 'detect':
        print(f"Starting detection mode with YOLO...")
        print(f"Model: {args.model_path}")
        print(f"Captures will be saved to: {args.capture_dir}")
        run_detection_mode(args)
    elif args.mode == 'hand':
        print(f"Starting hand detection mode...")
        print(f"Max hands: {args.max_hands}")
        print(f"Confidence threshold: {args.hand_confidence}")
        print("Gestures: open_palm, fist, peace, pointing, thumbs_up, pinch")
        print("Press 's' to save snapshot, 'r' to reset detector, ESC to quit")
        run_hand_detection_mode(args)
    else:
        print(f"Starting basic streaming mode...")
        run_basic_stream(args)


if __name__ == "__main__":
    main()