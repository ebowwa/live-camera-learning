from flask import Flask, jsonify
import subprocess
import threading
import time
import cv2

app = Flask(__name__)

rtsp_url = "rtsp://admin:admin@192.168.86.28:554/live"

def save_single_snapshot(output_dir="snapshots"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(rtsp_url)
    time.sleep(0.5)  # small delay to stabilize stream

    ret, frame = cap.read()
    if ret and frame is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"snapshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Snapshot saved: {filename}")
    else:
        print("⚠️ Failed to capture snapshot.")
    cap.release()

def record_audio(output_file="output.wav", duration=2):
    cmd = [
        "ffmpeg", "-y",
        "-i", rtsp_url,
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def record_video(duration=0.5, save_dir="frames"):
    time.sleep(1)

    import os
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(rtsp_url)
    start = time.time()
    frame_count = 0

    while time.time() - start < duration:
        ret, frame = cap.read()
        if ret and frame is not None:
            filename = f"{save_dir}/frame_{frame_count:03d}.jpg"
            cv2.imwrite(filename, frame)
            frame_count += 1
        else:
            print("⚠️ Failed to grab frame")
            break

    cap.release()
    print(f"✅ Saved {frame_count} frames to {save_dir}")


@app.route("/trigger_record", methods=["POST"])
def trigger_record():
    print("🎯 Trigger received. Start recording...")

    audio_thread = threading.Thread(target=record_audio)
    video_thread = threading.Thread(target=save_single_snapshot)

    audio_thread.start()
    video_thread.start()

    audio_thread.join()
    video_thread.join()

    print("✅ Done.")
    return jsonify({"status": "ok", "message": "Recording complete"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=777)
