#!/usr/bin/env python3
"""Test YOLO with a real image and debug preprocessing."""

import cv2
import numpy as np

def test_with_real_image():
    # Load the apple image
    img = cv2.imread("python/assets/images/apple1.png")
    if img is None:
        print("Could not load image")
        return
        
    print(f"Original image shape: {img.shape}")
    
    # Load model
    net = cv2.dnn.readNetFromONNX("python/assets/yolo11n.onnx")
    
    # Preprocess - this is critical!
    # YOLOv11 expects RGB input, normalized to [0,1]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
    print(f"Blob shape: {blob.shape}")
    print(f"Blob min/max: {blob.min():.3f} / {blob.max():.3f}")
    
    # Run inference
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    
    output = outputs[0]
    print(f"Output shape: {output.shape}")
    
    # Process predictions
    # YOLOv11 format: [1, 84, 8400] where 84 = 4 bbox + 80 classes
    predictions = output[0].T  # Shape: [8400, 84]
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Find high confidence detections
    detections = []
    for pred in predictions:
        # Extract bbox and scores
        x_center, y_center, width, height = pred[:4]
        class_scores = pred[4:]
        
        # Get best class
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence > 0.5:  # 50% threshold
            # The coordinates might be in YOLO format (relative to 640x640)
            # Or they might be in pixel coordinates already
            
            # Try interpreting as pixel coordinates for 640x640
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            # Scale to original image size
            x1 = int(x1 * w / 640)
            y1 = int(y1 * h / 640)
            x2 = int(x2 * w / 640)
            y2 = int(y2 * h / 640)
            
            detections.append({
                'bbox': [x1, y1, x2-x1, y2-y1],
                'class_id': class_id,
                'confidence': confidence,
                'raw_bbox': [x_center, y_center, width, height]
            })
    
    # COCO classes
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
               'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
               'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    print(f"\nFound {len(detections)} detections:")
    for det in detections:
        class_name = classes[det['class_id']] if det['class_id'] < len(classes) else 'unknown'
        print(f"  {class_name} (ID: {det['class_id']}): {det['confidence']:.3f}")
        print(f"    BBox: {det['bbox']}")
        print(f"    Raw: {det['raw_bbox']}")
    
    # Draw on image
    result = img.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        class_name = classes[det['class_id']] if det['class_id'] < len(classes) else 'unknown'
        
        # Draw box
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {det['confidence']:.2f}"
        cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite("debug_result.jpg", result)
    print("\nResult saved to debug_result.jpg")

if __name__ == "__main__":
    test_with_real_image()
