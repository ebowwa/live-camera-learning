# rtsp_test.py
#
# Simple RTSP stream tester using OpenCV.
# Install dependencies with: pip install opencv-python

import cv2
import time

def test_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open RTSP stream: {rtsp_url}")
        return

    # Query stream properties
    fps_reported = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[INFO] Stream opened: {width:.0f}×{height:.0f} @ {fps_reported:.2f} FPS")

    frame_count = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — retrying in 1s")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            continue

        frame_count += 1
        cv2.imshow("RTSP Test Stream", frame)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    t_elapsed = time.time() - t_start
    avg_fps = frame_count / t_elapsed if t_elapsed > 0 else 0
    print(f"[INFO] Received {frame_count} frames in {t_elapsed:.2f}s → {avg_fps:.2f} FPS")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your camera's RTSP URL:
    rtsp_url = "rtsp://admin:admin@192.168.42.1:554/live"
    test_rtsp_stream(rtsp_url)
