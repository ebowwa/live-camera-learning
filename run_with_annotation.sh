#!/bin/bash

# EdaxShifu - Run intelligent capture with annotation interface

echo "======================================"
echo "EdaxShifu - Intelligent Camera System"
echo "======================================"
echo ""

# Parse arguments
RTSP_URL="${1:-0}"  # Default to webcam if not specified

if [ "$RTSP_URL" == "0" ]; then
    echo "Using webcam for capture"
else
    echo "Using RTSP URL: $RTSP_URL"
fi

echo ""
echo "Starting components..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $CAPTURE_PID 2>/dev/null
    kill $ANNOTATION_PID 2>/dev/null
    exit 0
}

# Set up trap for cleanup
trap cleanup INT TERM

# Start the annotation interface in background
echo "1. Starting annotation interface..."
uv run python annotate.py &
ANNOTATION_PID=$!
echo "   Annotation interface PID: $ANNOTATION_PID"
echo "   Access at: http://localhost:7860"
echo ""

# Wait a moment for the interface to start
sleep 2

# Start the intelligent capture system
echo "2. Starting intelligent capture system..."
echo "   Controls:"
echo "   - 's' to manually capture"
echo "   - 'r' to reset KNN classifier"
echo "   - 'i' to show statistics"
echo "   - ESC to exit"
echo ""

if [ "$RTSP_URL" == "0" ]; then
    uv run python main.py --mode intelligent --url 0 &
else
    uv run python main.py --mode intelligent --url "$RTSP_URL" &
fi
CAPTURE_PID=$!
echo "   Capture system PID: $CAPTURE_PID"
echo ""

echo "======================================"
echo "System is running!"
echo ""
echo "- Capture window shows live RTSP stream"
echo "- Unknown objects go to annotation queue"
echo "- Label them at http://localhost:7860"
echo "- KNN updates automatically with feedback"
echo ""
echo "Press Ctrl+C to stop all components"
echo "======================================"

# Wait for processes
wait $CAPTURE_PID
wait $ANNOTATION_PID