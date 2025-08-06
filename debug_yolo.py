#!/usr/bin/env python3
"""Debug YOLO to see what's going wrong."""

import cv2
import numpy as np

def debug_yolo_model():
    # Load and inspect the YOLO model
    net = cv2.dnn.readNetFromONNX("assets/yolo11n.onnx")
    
    # Get layer names
    layer_names = net.getLayerNames()
    print(f"Total layers: {len(layer_names)}")
    
    # Get output layers
    output_layers = net.getUnconnectedOutLayersNames()
    print(f"Output layers: {output_layers}")
    
    # Create a test image
    test_img = np.ones((640, 640, 3), dtype=np.uint8) * 128
    
    # Prepare input
    blob = cv2.dnn.blobFromImage(test_img, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output
    outputs = net.forward(output_layers)
    
    print(f"\nNumber of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
        print(f"Output {i} dtype: {output.dtype}")
        print(f"Output {i} min/max: {output.min():.3f} / {output.max():.3f}")
        
        # Check the actual values
        if len(output.shape) == 3:
            print(f"First detection raw values: {output[0][0][:10]}")
        
    # Try to understand the format
    if len(outputs) == 1:
        output = outputs[0]
        print(f"\nAnalyzing single output:")
        print(f"Shape: {output.shape}")
        
        # Check if it's in format [1, 84, 8400] (YOLOv8/v11)
        if output.shape[1] == 84 or output.shape[1] == 85:
            print("Detected YOLOv8/v11 format (84/85 channels)")
            print("Format: [batch, 84/85, num_predictions]")
            print("Where 84 = 4 bbox + 80 classes")
            
            # Check actual predictions
            predictions = output[0].T  # Transpose to [num_predictions, 84]
            print(f"Number of predictions: {predictions.shape[0]}")
            
            # Look at first few predictions
            for i in range(min(3, predictions.shape[0])):
                pred = predictions[i]
                bbox = pred[:4]
                class_scores = pred[4:]
                max_class = np.argmax(class_scores)
                max_conf = class_scores[max_class]
                print(f"\nPrediction {i}:")
                print(f"  BBox (x,y,w,h): {bbox}")
                print(f"  Max class ID: {max_class}")
                print(f"  Max confidence: {max_conf:.3f}")

if __name__ == "__main__":
    debug_yolo_model()