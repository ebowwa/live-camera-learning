#!/usr/bin/env python3
"""
Run the complete EdaxShifu system with RTSP stream
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path

# Default RTSP URL for Seeed Studio reCamera
DEFAULT_RTSP = "rtsp://admin:admin@192.168.42.1:554/live"

def check_rtsp_url(url):
    """Check if RTSP URL is reachable."""
    import cv2
    
    print(f"Testing RTSP connection to: {url}")
    cap = cv2.VideoCapture(url)
    
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print("✓ RTSP stream is accessible")
            return True
        else:
            print("✗ RTSP stream opened but couldn't read frame")
            return False
    else:
        print("✗ Cannot connect to RTSP stream")
        return False


def setup_directories():
    """Ensure all required directories exist."""
    dirs = [
        "python/data/captures/successful",
        "python/data/captures/failed",
        "python/data/captures/dataset",
        "python/data/intelligent_captures",
        "python/models",
        "python/assets/images"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure ready")


def run_rtsp_system(rtsp_url, headless=False, skip_annotation=False):
    """Run the complete system with RTSP."""
    
    print("\n" + "=" * 60)
    print("EdaxShifu - Intelligent RTSP Camera System")
    print("=" * 60)
    print(f"\nRTSP URL: {rtsp_url}")
    print("\nInitializing...")
    
    # Setup directories
    setup_directories()
    
    # Check RTSP connection
    if rtsp_url != "0" and not rtsp_url.startswith("rtsp://"):
        print(f"\nWarning: URL doesn't look like RTSP: {rtsp_url}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    if rtsp_url != "0":
        if not check_rtsp_url(rtsp_url):
            print("\nFailed to connect to RTSP stream!")
            print("Possible issues:")
            print("1. Camera is not connected")
            print("2. Wrong IP address or credentials")
            print("3. Network connectivity issues")
            print(f"\nDefault URL for reCamera: {DEFAULT_RTSP}")
            
            response = input("\nUse webcam instead? (y/n): ")
            if response.lower() == 'y':
                rtsp_url = "0"
            else:
                return
    
    processes = []
    
    try:
        # Start annotation interface if not skipped
        if not skip_annotation:
            print("\n1. Starting annotation interface...")
            annotation_cmd = [sys.executable, "annotate.py"]
            annotation_proc = subprocess.Popen(
                annotation_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            processes.append(annotation_proc)
            print("   ✓ Annotation interface started at http://localhost:7860")
            time.sleep(2)
        
        # Start intelligent capture system
        print("\n2. Starting intelligent capture system...")
        capture_cmd = [
            sys.executable, "python/main.py",
            "--mode", "intelligent",
            "--url", rtsp_url,
            "--model-path", "python/assets/yolo11n.onnx",
            "--capture-dir", "python/data/captures",
            "--training-dir", "python/assets/images"
        ]
        
        if headless:
            capture_cmd.append("--headless")
        
        print(f"   Command: {' '.join(capture_cmd)}")
        capture_proc = subprocess.Popen(capture_cmd)
        processes.append(capture_proc)
        
        print("\n" + "=" * 60)
        print("System is running!")
        print("=" * 60)
        print("\nWorkflow:")
        print("1. RTSP stream → YOLO detection → Trigger")
        print("2. Capture → KNN classification")
        print("3. Success → Save to successful/")
        print("4. Failure → Save to failed/ → Annotation queue")
        print("5. Human labels → Update KNN → Continuous learning")
        
        print("\nControls:")
        print("- 's' : Manual capture")
        print("- 'r' : Reset KNN classifier")
        print("- 'i' : Show statistics")
        print("- ESC : Exit")
        
        if not skip_annotation:
            print(f"\nAnnotation UI: http://localhost:7860")
        
        print("\nPress Ctrl+C to stop all components")
        print("=" * 60)
        
        # Wait for capture process
        capture_proc.wait()
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        
        # Terminate all processes
        for proc in processes:
            proc.terminate()
            proc.wait()
        
        print("✓ All components stopped")
        
    except Exception as e:
        print(f"\nError: {e}")
        
        # Clean up processes
        for proc in processes:
            proc.terminate()


def main():
    parser = argparse.ArgumentParser(
        description='Run EdaxShifu with RTSP stream'
    )
    
    parser.add_argument(
        '--url',
        type=str,
        default=DEFAULT_RTSP,
        help=f'RTSP URL (default: {DEFAULT_RTSP})'
    )
    
    parser.add_argument(
        '--webcam',
        action='store_true',
        help='Use webcam instead of RTSP'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without display window'
    )
    
    parser.add_argument(
        '--skip-annotation',
        action='store_true',
        help='Skip starting annotation interface'
    )
    
    args = parser.parse_args()
    
    # Determine URL
    if args.webcam:
        rtsp_url = "0"
    else:
        rtsp_url = args.url
    
    run_rtsp_system(
        rtsp_url,
        headless=args.headless,
        skip_annotation=args.skip_annotation
    )


if __name__ == "__main__":
    main()
