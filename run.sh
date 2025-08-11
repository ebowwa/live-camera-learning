#!/bin/bash

# EdaxShifu - Complete System Launcher

echo "======================================"
echo "🎯 EdaxShifu - Intelligent Camera"
echo "======================================"

# Kill any existing processes
pkill -f "python.*main.py" 2>/dev/null
pkill -f "python.*annotate.py" 2>/dev/null
sleep 1

# Create necessary directories
mkdir -p python/data/captures/failed python/data/captures/successful python/data/captures/processed python/data/captures/dataset python/models python/assets/images

# Use first argument as URL, default to webcam
URL="${1:-0}"

echo ""
echo "📹 Video source: $URL"
echo "🌐 Annotation UI: http://localhost:7860"
echo ""

# Run the integrated system
echo "Starting complete system..."
uv run python python/main.py --mode intelligent --url "$URL"
