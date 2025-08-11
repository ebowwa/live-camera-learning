#!/usr/bin/env python3
"""
EdaxShifu - Intelligent Camera Learning System
Main entry point supporting both CLI and Web UI modes.
"""

import argparse
import sys
import os
import time
import logging
from src.rtsp_stream import RTSPStream, RTSPViewer
from src.integrated_detector import IntegratedDetector, HandObjectTrigger
from src.intelligent_capture import IntelligentCaptureSystem

# Import hand detection (should work with Python 3.11)
try:
    from examples.hand_stream_viewer import HandStreamViewer, demo_gesture_handler
    HAND_DETECTION_AVAILABLE = True
except ImportError as e:
    HAND_DETECTION_AVAILABLE = False
    print(f"Warning: Hand detection import failed: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_banner():
    """Display the EdaxShifu banner."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║                    🎯 EdaxShifu                          ║
║         Intelligent Camera Learning System                ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)


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


def run_intelligent_mode(args):
    """Run the full intelligent capture pipeline with KNN and annotation interface."""
    import threading
    from src.annotation_interface import create_annotation_app
    
    # Start annotation interface in background thread if not disabled
    if not args.no_annotation:
        def run_annotation():
            try:
                print(f"Starting annotation interface at http://localhost:{args.annotation_port}")
                app = create_annotation_app("models/knn_classifier.pkl")
                result = app.launch(
                    share=False,
                    port=args.annotation_port,
                    prevent_thread_lock=True
                )
                if result and hasattr(result, 'server_port'):
                    actual_port = result.server_port
                    print(f"🌐 Annotation interface available at http://localhost:{actual_port}")
                elif result and hasattr(result, 'local_url'):
                    print(f"🌐 Annotation interface available at {result.local_url}")
            except Exception as e:
                print(f"❌ Annotation interface error: {e}")
                print("Continuing without annotation interface...")
        
        annotation_thread = threading.Thread(target=run_annotation, daemon=True)
        annotation_thread.start()
        time.sleep(2)  # Let annotation interface start
    
    # Run capture system
    system = IntelligentCaptureSystem(
        rtsp_url=args.url,
        yolo_model_path=args.model_path,
        capture_dir=args.capture_dir,
        confidence_threshold=args.conf_threshold
    )
    
    # Load training samples if available
    training_dir = args.training_dir
    if training_dir and os.path.exists(training_dir):
        print(f"Loading training samples from {training_dir}")
        system.knn.add_samples_from_directory(training_dir)
    
    try:
        print("\n" + "="*60)
        print("🎯 EdaxShifu Intelligent Capture System")
        print("="*60)
        print(f"\n📹 Source: {args.url}")
        
        if not args.no_annotation:
            print(f"🌐 Annotation: http://localhost:{args.annotation_port}")
        
        print("\n🎮 Controls:")
        print("  's' - Manual capture")
        print("  'r' - Reset KNN classifier")
        print("  'i' - Show statistics")
        print("  'l' - Reload model (get new annotations)")
        print("  ESC - Exit")
        
        print("\n🔄 Live Learning Active:")
        print("  • Unknown objects → captures/failed/")
        print("  • Annotate in browser → Model updates")
        print("  • System learns in real-time")
        print("="*60 + "\n")
        
        system.run(display=not args.headless)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        print(f"Stats:")
        print(f"  Successful: {system.stats['successful_recognitions']}")
        print(f"  Failed: {system.stats['failed_recognitions']}")
        print(f"  Total: {system.stats['captures']}")
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


def run_web_interface(args):
    """Run the Gradio web interface."""
    from unified_interface import UnifiedEdaxShifu
    
    print("Starting Web Interface...")
    print(f"📹 Video source: {args.url}")
    print(f"🌐 Interface: http://localhost:{args.port}")
    print("="*60)
    print("\nFeatures available in the web interface:")
    print("• Live stream with YOLO detection")
    print("• Capture objects with one click")
    print("• Annotate unknown objects")
    print("• Teach new objects")
    print("• Real-time learning\n")
    
    try:
        app = UnifiedEdaxShifu(rtsp_url=args.url)
        interface = app.create_interface()
        
        # Try multiple ports if the requested one is in use
        for port_attempt in range(args.port, args.port + 10):
            try:
                print(f"Trying port {port_attempt}...")
                interface.launch(
                    server_port=port_attempt,
                    share=args.share,
                    server_name="0.0.0.0"
                )
                print(f"✅ Web interface running at http://localhost:{port_attempt}")
                break
            except OSError as e:
                if "address already in use" in str(e).lower() and port_attempt < args.port + 9:
                    continue
                else:
                    raise e
    except KeyboardInterrupt:
        print("\n\nShutting down web interface...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting web interface: {e}")
        sys.exit(1)


def main():
    """Main entry point with unified argument parsing."""
    parser = argparse.ArgumentParser(
        description='EdaxShifu - Intelligent Camera Learning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Launch web interface
  python main.py
  
  # Web interface with webcam
  python main.py --url 0
  
  # CLI mode with YOLO detection
  python main.py --cli --mode detect
  
  # CLI mode with intelligent capture
  python main.py --cli --mode intelligent
  
  # Hand detection mode (CLI only)
  python main.py --cli --mode hand
        """
    )
    
    # Interface selection
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Use command-line interface with OpenCV windows (default: web UI)'
    )
    
    # Common arguments
    parser.add_argument(
        '--url',
        type=str,
        default='rtsp://admin:admin@192.168.42.1:554/live',
        help='RTSP stream URL or webcam index (0 for webcam)'
    )
    
    # Web UI specific arguments
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port for web interface (default: 7860)'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public share link for web interface'
    )
    
    # CLI mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['stream', 'detect', 'hand', 'intelligent'],
        default='intelligent',
        help='CLI operation mode (default: intelligent)'
    )
    
    # Detection/Intelligence mode arguments
    parser.add_argument(
        '--model-path',
        type=str,
        default='assets/yolo11n.onnx',
        help='Path to YOLO ONNX model'
    )
    
    parser.add_argument(
        '--capture-dir',
        type=str,
        default='captures',
        help='Directory for captured frames'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for detection'
    )
    
    parser.add_argument(
        '--capture-cooldown',
        type=float,
        default=2.0,
        help='Minimum seconds between captures'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run CLI without display window'
    )
    
    # Hand detection mode arguments
    parser.add_argument(
        '--max-hands',
        type=int,
        default=2,
        help='Maximum number of hands to detect'
    )
    
    parser.add_argument(
        '--hand-confidence',
        type=float,
        default=0.5,
        help='Hand detection confidence threshold'
    )
    
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Disable statistics overlay in hand mode'
    )
    
    # Intelligent mode arguments
    parser.add_argument(
        '--training-dir',
        type=str,
        default='assets/images',
        help='Directory with training images for KNN'
    )
    
    parser.add_argument(
        '--no-annotation',
        action='store_true',
        help='Disable annotation interface in intelligent mode'
    )
    
    parser.add_argument(
        '--annotation-port',
        type=int,
        default=7860,
        help='Port for annotation interface'
    )
    
    args = parser.parse_args()
    
    # Convert URL for webcam if needed
    if args.url == '0':
        args.url = 0  # Convert string '0' to integer for OpenCV
    
    # Show banner
    print_banner()
    
    # Run appropriate interface
    if args.cli:
        # CLI mode with OpenCV windows
        print(f"Starting CLI mode: {args.mode}")
        print(f"📹 Source: {args.url}")
        
        if args.mode == 'detect':
            print(f"Model: {args.model_path}")
            print(f"Captures will be saved to: {args.capture_dir}")
            args.window_name = 'EdaxShifu Detection'
            run_detection_mode(args)
        elif args.mode == 'hand':
            print(f"Max hands: {args.max_hands}")
            print(f"Confidence threshold: {args.hand_confidence}")
            print("Gestures: open_palm, fist, peace, pointing, thumbs_up, pinch")
            print("Press 's' to save snapshot, 'r' to reset detector, ESC to quit")
            args.window_name = 'EdaxShifu Hand Detection'
            run_hand_detection_mode(args)
        elif args.mode == 'intelligent':
            print(f"YOLO Model: {args.model_path}")
            print(f"Training directory: {args.training_dir}")
            print(f"Captures will be saved to: {args.capture_dir}")
            run_intelligent_mode(args)
        else:  # stream mode
            print("Starting basic streaming mode...")
            args.window_name = 'EdaxShifu Stream'
            run_basic_stream(args)
    else:
        # Web UI mode (default)
        run_web_interface(args)


if __name__ == "__main__":
    main()
