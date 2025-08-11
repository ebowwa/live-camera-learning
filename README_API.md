# EdaxShifu KNN API Server

REST API server that exposes the trained KNN model for object recognition inference over the local network.

## Features

- **REST API endpoints** for KNN model inference
- **Multiple input formats**: Base64 encoded images or file uploads
- **Network accessible**: Runs on `0.0.0.0` for local network access
- **Model management**: Reload model, update confidence threshold
- **Statistics**: Get model stats and known classes
- **Auto-generated docs**: FastAPI provides interactive API documentation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Start the API Server

```bash
python api_server.py
```

The server will start on `http://0.0.0.0:8000` by default.

### 3. Access API Documentation

Open `http://localhost:8000/docs` in your browser for interactive API documentation.

## API Endpoints

### Health Check
- **GET** `/` - Check server and model status

### Predictions
- **POST** `/predict` - Predict from base64 encoded image
- **POST** `/predict/upload` - Predict from uploaded image file

### Model Management
- **GET** `/model/stats` - Get model statistics
- **POST** `/model/reload` - Reload model from disk
- **PUT** `/model/confidence` - Update confidence threshold

## Usage Examples

### Using curl

#### Health Check
```bash
curl http://localhost:8000/
```

#### Upload Image for Prediction
```bash
curl -X POST "http://localhost:8000/predict/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/image.jpg"
```

#### Get Model Stats
```bash
curl http://localhost:8000/model/stats
```

#### Update Confidence Threshold
```bash
curl -X PUT "http://localhost:8000/model/confidence?threshold=0.7"
```

### Using Python

```python
import requests
import base64

# Health check
response = requests.get("http://localhost:8000/")
print(response.json())

# Predict from file upload
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict/upload", files=files)
    prediction = response.json()
    print(f"Predicted: {prediction['label']} (confidence: {prediction['confidence']:.2f})")

# Predict from base64
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()
    
response = requests.post("http://localhost:8000/predict", 
                        json={"image_base64": image_b64})
prediction = response.json()
print(f"Predicted: {prediction['label']} (confidence: {prediction['confidence']:.2f})")
```

### Using JavaScript/Node.js

```javascript
// Health check
fetch('http://localhost:8000/')
  .then(response => response.json())
  .then(data => console.log(data));

// Upload image for prediction
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict/upload', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(prediction => {
  console.log(`Predicted: ${prediction.label} (confidence: ${prediction.confidence})`);
});
```

## Command Line Options

```bash
python api_server.py --help
```

Options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--model-path`: Path to KNN model file (default: python/models/knn_classifier.npz)
- `--reload`: Enable auto-reload for development

## Network Access

The server runs on `0.0.0.0` by default, making it accessible from other devices on the same local network:

- **Local access**: `http://localhost:8000`
- **Network access**: `http://YOUR_IP_ADDRESS:8000`

To find your IP address:
```bash
# Linux/Mac
ip addr show | grep inet
# or
ifconfig | grep inet

# Windows
ipconfig
```

## Integration with Existing EdaxShifu

The API server works alongside the existing EdaxShifu system:

1. **Model sharing**: Uses the same `python/models/knn_classifier.npz` file
2. **Live updates**: Use `/model/reload` endpoint after training new objects
3. **Parallel operation**: Can run simultaneously with the Gradio interface

## Response Format

All prediction endpoints return:

```json
{
  "label": "apple",
  "confidence": 0.85,
  "is_known": true,
  "all_scores": {
    "apple": 0.85,
    "orange": 0.12,
    "banana": 0.03
  },
  "timestamp": "2025-08-10T23:30:00"
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid image, etc.)
- `503`: Service unavailable (model not loaded)
- `500`: Internal server error

## Security Notes

- This API is designed for **local network use only**
- No authentication is implemented (suitable for trusted local networks)
- CORS is enabled for all origins (local network convenience)
- For production use, consider adding authentication and HTTPS
