#!/usr/bin/env python3
"""
Test script to demonstrate annotator switching functionality.
"""

import os
import numpy as np
from src.annotators import AnnotatorFactory, AnnotationRequest

def test_annotator_switching():
    """Test different annotator modes."""
    
    # Create a dummy image and request
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    request = AnnotationRequest(
        image=dummy_image,
        image_path="test.jpg",
        metadata={},
        yolo_detections=["person", "object"],
        knn_prediction="unknown",
        knn_confidence=0.3,
        timestamp="2024-01-01"
    )
    
    # Test different annotator modes
    modes = [
        "human_only",
        "gemini_only", 
        "ai_first",
        "human_first",
        "consensus_ai_human",
        "weighted_ai_heavy",
        "weighted_human_heavy"
    ]
    
    print("=" * 60)
    print("ANNOTATOR SWITCHING TEST")
    print("=" * 60)
    
    # Check if Gemini is available
    gemini = AnnotatorFactory.create_gemini_annotator()
    print(f"\nGemini API available: {gemini.is_available()}")
    
    if not gemini.is_available():
        print("Note: Set GEMINI_API_KEY environment variable to enable AI modes")
    
    print("\nTesting annotator presets:")
    print("-" * 40)
    
    for mode in modes:
        try:
            annotator = AnnotatorFactory.create_preset(mode)
            print(f"✓ {mode:25} - Created successfully")
            
            # Show what type of annotator was created
            if hasattr(annotator, '__class__'):
                print(f"  Type: {annotator.__class__.__name__}")
                
                # For multi-annotators, show the chain
                if hasattr(annotator, 'annotators'):
                    chain = [a.name for a in annotator.annotators]
                    print(f"  Chain: {' -> '.join(chain)}")
                    
        except Exception as e:
            print(f"✗ {mode:25} - Failed: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    # Demonstrate how the Gradio interface would switch modes
    print("\nGradio Interface Mode Switching:")
    print("-" * 40)
    print("The dropdown in the interface allows switching between:")
    print("1. 👤 Human Only - Manual annotation only")
    print("2. 🤖 Gemini Only - AI annotation only (requires API key)")
    print("3. 🤖→👤 AI First - Try AI, fallback to human")
    print("4. 👤→🤖 Human First - Try human, fallback to AI")
    print("5. 🤝 Consensus - Both must agree")
    print("6. ⚖️ Weighted AI - 80% AI, 20% Human")
    print("7. ⚖️ Weighted Human - 80% Human, 20% AI")
    
    print("\nThe 'Auto-Annotate' button uses the selected mode to")
    print("automatically annotate failed detections.")

if __name__ == "__main__":
    test_annotator_switching()