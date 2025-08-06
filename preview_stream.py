import cv2
import subprocess
import threading
import time

# RTSP URL
rtsp_url = "rtsp://admin:admin@192.168.86.28:554/live"

# Audio recording function
def record_audio_from_rtsp(output_file="output.wav", duration=5):
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", rtsp_url,
        "-t", str(duration),
        "-ar", "16000",  # 16 kHz
        "-ac", "1",      # Mono
        "-f", "wav",
        output_file
    ]
    print("🎙️ Recording audio...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ Audio saved to", output_file)

# Start audio recording in background
audio_thread = threading.Thread(target=record_audio_from_rtsp)
audio_thread.start()

# Open video stream
cap = cv2.VideoCapture(rtsp_url)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to get frame.")
        break

    cv2.imshow("Camera", frame)

    # Stop after 5 seconds or if 'q' pressed
    if (time.time() - start_time > 5) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
audio_thread.join()
