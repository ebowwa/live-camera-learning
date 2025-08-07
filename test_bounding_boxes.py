#!/usr/bin/env python3
"""
Test script for bounding box functionality in Gemini annotator.
"""

import os
import numpy as np
import cv2
from PIL import Image
from src.annotators import AnnotatorFactory, AnnotationRequest
from src.annotators.bbox_utils import draw_bounding_boxes, crop_object_from_bbox

def test_bounding_boxes():
    """Test bounding box detection and visualization."""
    
    print("=" * 60)
    print("🔍 BOUNDING BOX DETECTION TEST")
    print("=" * 60)
    
    # Check if API key is available
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("Set your API key: export GEMINI_API_KEY='your-key-here'")
        return
    
    # Initialize Gemini annotator
    try:
        gemini = AnnotatorFactory.create_gemini_annotator()
        if not gemini.is_available():
            print("❌ Gemini annotator not available")
            return
        
        print("✅ Gemini annotator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini: {e}")
        return
    
    # Create a test image with multiple objects
    print("\n📸 Creating test image with multiple objects...")
    
    # Create a test image (400x300 pixels)
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw some simple shapes to represent objects
    # Red rectangle (book)
    cv2.rectangle(test_image, (50, 50), (150, 120), (0, 0, 255), -1)
    cv2.putText(test_image, "BOOK", (75, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Blue circle (ball)
    cv2.circle(test_image, (250, 90), 40, (255, 0, 0), -1)
    cv2.putText(test_image, "BALL", (220, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Green rectangle (phone)
    cv2.rectangle(test_image, (300, 180), (360, 250), (0, 255, 0), -1)
    cv2.putText(test_image, "PHONE", (305, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Save test image
    cv2.imwrite("/tmp/test_objects.jpg", test_image)
    print("✅ Test image created: /tmp/test_objects.jpg")
    
    # Create annotation request
    request = AnnotationRequest(
        image=test_image,
        image_path="/tmp/test_objects.jpg",
        metadata={},
        yolo_detections=[],
        knn_prediction=None,
        knn_confidence=0.0,
        timestamp="test"
    )
    
    print("\n🤖 Getting AI annotation with bounding boxes...")
    
    try:
        # Get annotation from Gemini
        result = gemini.annotate(request)
        
        if result.success:
            print(f"✅ Primary label: {result.label}")
            print(f"✅ Confidence: {result.confidence:.2f}")
            
            if result.bounding_boxes:
                print(f"✅ Found {len(result.bounding_boxes)} bounding boxes:")
                
                for i, bbox in enumerate(result.bounding_boxes):
                    print(f"   {i+1}. {bbox.get('label', 'object')} "
                          f"(conf: {bbox.get('confidence', 0):.2f}) "
                          f"at [{bbox.get('x', 0):.3f}, {bbox.get('y', 0):.3f}, "
                          f"{bbox.get('w', 0):.3f}, {bbox.get('h', 0):.3f}]")
                
                # Draw bounding boxes on the image
                print("\n🎨 Drawing bounding boxes...")
                annotated_image = draw_bounding_boxes(test_image, result.bounding_boxes)
                
                # Save annotated image
                cv2.imwrite("/tmp/test_objects_annotated.jpg", annotated_image)
                print("✅ Annotated image saved: /tmp/test_objects_annotated.jpg")
                
                # Test cropping
                print("\n✂️ Testing object cropping...")
                for i, bbox in enumerate(result.bounding_boxes):
                    cropped = crop_object_from_bbox(test_image, bbox, padding=10)
                    if cropped is not None:
                        label = bbox.get('label', f'object_{i}')
                        crop_path = f"/tmp/cropped_{label}_{i}.jpg"
                        cv2.imwrite(crop_path, cropped)
                        print(f"✅ Cropped {label}: {crop_path} ({cropped.shape})")
                    else:
                        print(f"❌ Failed to crop object {i}")
                
            else:
                print("⚠️ No bounding boxes detected")
                print(f"Raw response: {result.metadata.get('raw_response', 'N/A')}")
        
        else:
            print(f"❌ Annotation failed: {result.error_message}")
    
    except Exception as e:
        print(f"❌ Error during annotation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🏁 TEST COMPLETE")
    print("=" * 60)
    print("\nCheck the following files:")
    print("• /tmp/test_objects.jpg - Original test image")
    print("• /tmp/test_objects_annotated.jpg - With bounding boxes")
    print("• /tmp/cropped_*.jpg - Individual cropped objects")

if __name__ == "__main__":
    test_bounding_boxes()