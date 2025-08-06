from flask import Flask, request, jsonify
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    # Optional: generate timestamped filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)
    print(f"📸 Image saved: {filepath}")
    return jsonify({"status": "ok", "filename": filename}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
