#!/usr/bin/env python3
"""
Test the live learning loop:
1. RTSP captures unknown objects
2. Gradio interface shows them immediately
3. Human annotations update KNN in real-time
4. RTSP system learns from annotations
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path

def check_live_system():
    """Check if the live learning system is working."""
    
    print("=" * 60)
    print("LIVE LEARNING SYSTEM TEST")
    print("=" * 60)
    
    # Check directories
    dirs = {
        "Failed": "python/data/captures/failed",
        "Processed": "python/data/captures/processed",
        "Dataset": "python/data/captures/dataset"
    }
    
    for name, path in dirs.items():
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
            print(f"✓ {name}: {count} images")
        else:
            print(f"✗ {name}: directory missing")
            os.makedirs(path, exist_ok=True)
    
    # Check model
    model_path = "python/models/knn_classifier.pkl"
    if os.path.exists(model_path):
        print(f"✓ Model exists: {model_path}")
        
        # Check model update time
        mod_time = os.path.getmtime(model_path)
        age = time.time() - mod_time
        if age < 60:
            print(f"  → Recently updated ({int(age)}s ago)")
        else:
            print(f"  → Last updated {int(age/60)} minutes ago")
    else:
        print("✗ No model found")
    
    print("\n" + "=" * 60)
    print("LIVE LEARNING WORKFLOW:")
    print("=" * 60)
    print("\n1. START CAPTURE SYSTEM:")
    print("   uv run main.py --mode intelligent --url 0")
    print("\n2. START ANNOTATION INTERFACE:")
    print("   uv run annotate.py")
    print("\n3. CAPTURE UNKNOWN OBJECTS:")
    print("   - Point camera at unknown objects")
    print("   - Press 's' to manually capture")
    print("   - Failed recognitions go to python/data/captures/failed/")
    print("\n4. ANNOTATE IN GRADIO:")
    print("   - Interface auto-refreshes every 5 seconds")
    print("   - Label unknown objects")
    print("   - Model updates immediately")
    print("\n5. LIVE LEARNING:")
    print("   - Capture system reloads model every 10 seconds")
    print("   - Press 'l' to force reload")
    print("   - New annotations improve recognition immediately")
    
    print("\n" + "=" * 60)
    print("MONITORING:")
    print("=" * 60)
    
    # Monitor for 30 seconds
    print("\nMonitoring directories for 30 seconds...")
    print("(Start the system now to see live updates)\n")
    
    start_time = time.time()
    last_counts = {name: 0 for name in dirs.keys()}
    
    try:
        while time.time() - start_time < 30:
            current_counts = {}
            changes = []
            
            for name, path in dirs.items():
                if os.path.exists(path):
                    count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                    current_counts[name] = count
                    
                    if count != last_counts[name]:
                        diff = count - last_counts[name]
                        if diff > 0:
                            changes.append(f"📥 {name}: +{diff} images")
                        else:
                            changes.append(f"📤 {name}: {diff} images")
                else:
                    current_counts[name] = 0
            
            if changes:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] " + " | ".join(changes))
                last_counts = current_counts
            
            # Check model updates
            if os.path.exists(model_path):
                mod_time = os.path.getmtime(model_path)
                if mod_time > start_time:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] 🔄 Model updated!")
                    start_time = mod_time  # Reset to avoid repeated messages
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        pass
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    # Final summary
    print("\nFinal counts:")
    for name, path in dirs.items():
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
            print(f"  {name}: {count} images")
    
    # Check if live learning worked
    failed_count = len([f for f in os.listdir(dirs["Failed"]) if f.endswith('.jpg')]) if os.path.exists(dirs["Failed"]) else 0
    processed_count = len([f for f in os.listdir(dirs["Processed"]) if f.endswith('.jpg')]) if os.path.exists(dirs["Processed"]) else 0
    
    if processed_count > 0:
        print(f"\n✅ Live learning successful! {processed_count} images annotated")
    elif failed_count > 0:
        print(f"\n⚠️ {failed_count} images waiting for annotation")
    else:
        print("\n💡 Start the system to begin live learning")


if __name__ == "__main__":
    check_live_system()
