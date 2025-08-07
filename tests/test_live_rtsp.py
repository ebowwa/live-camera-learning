#!/usr/bin/env python3
"""
Test the live RTSP to annotation flow
"""

import sys
import time
import cv2
from src.intelligent_capture import IntelligentCaptureSystem

def test_rtsp_flow():
    """Test RTSP capture creating real failed annotations."""
    print("=" * 60)
    print("TESTING RTSP TO ANNOTATION FLOW")
    print("=" * 60)
    
    # Use webcam for testing (replace with RTSP URL)
    rtsp_url = "0"  # or "rtsp://admin:admin@192.168.42.1:554/live"
    
    system = IntelligentCaptureSystem(
        rtsp_url=rtsp_url,
        yolo_model_path="assets/yolo11n.onnx",
        capture_dir="captures",
        confidence_threshold=0.5
    )
    
    print("\nSystem initialized!")
    print("This test will:")
    print("1. Capture from RTSP/webcam")
    print("2. Run YOLO detection")
    print("3. Attempt KNN classification")
    print("4. Save failed recognitions to captures/failed/")
    print("5. These can then be annotated via the Gradio interface")
    print("\nPress 's' to manually trigger a capture")
    print("Press ESC to exit\n")
    
    # Run for a short time to capture some samples
    start_time = time.time()
    timeout = 30  # Run for 30 seconds max
    
    try:
        while (time.time() - start_time) < timeout:
            results = system.process_frame()
            
            if results and results['display_frame'] is not None:
                cv2.imshow("RTSP Test", results['display_frame'])
                
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                print("Manual capture triggered!")
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        system.stream.release()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"\nStats:")
    print(f"  Frames processed: {system.stats['frames_processed']}")
    print(f"  Total captures: {system.stats['captures']}")
    print(f"  Successful recognitions: {system.stats['successful_recognitions']}")
    print(f"  Failed recognitions: {system.stats['failed_recognitions']}")
    
    if system.stats['failed_recognitions'] > 0:
        print(f"\n✅ {system.stats['failed_recognitions']} failed recognitions saved!")
        print("These are now available in the annotation interface.")
        print("\nTo annotate them, run in another terminal:")
        print("  uv run annotate.py")
    else:
        print("\n⚠️ No failed recognitions captured.")
        print("Try capturing unknown objects to test the annotation flow.")
    
    return system.stats['failed_recognitions'] > 0


if __name__ == "__main__":
    success = test_rtsp_flow()
    sys.exit(0 if success else 1)