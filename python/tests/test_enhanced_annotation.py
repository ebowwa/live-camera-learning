#!/usr/bin/env python3
"""
Test script for the enhanced annotation interface with AI+Human support.
"""

import os
import sys
import logging

from python.edaxshifu.annotation_interface import create_annotation_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🤖 Testing Enhanced EdaxShifu Annotation Interface")
    print("=" * 60)
    
    # Test with AI enabled
    print("Creating enhanced annotation app with AI support...")
    
    try:
        app = create_annotation_app(
            knn_model_path="models/knn_classifier.pkl",
            failed_dir="captures/failed", 
            dataset_dir="captures/dataset",
            use_ai_annotator=True,
            annotator_preset="ai_first"
        )
        
        print("✅ Enhanced annotation interface created successfully!")
        print("\nFeatures:")
        print("- 🤖 AI annotation suggestions (Gemini Vision)")
        print("- 👤 Human annotation interface") 
        print("- 📊 Enhanced statistics tracking")
        print("- 🔄 Dual annotation workflow")
        
        gemini_status = "✅ Available" if app.use_ai_annotator else "❌ Not available (check GEMINI_API_KEY)"
        print(f"\nAI Annotator Status: {gemini_status}")
        
        if app.use_ai_annotator and app.gemini_annotator:
            print(f"AI Model: {app.gemini_annotator.model_name}")
            print(f"AI Available: {app.gemini_annotator.is_available()}")
        
        print("\nLaunching interface on http://localhost:7860")
        print("To test AI features, set GEMINI_API_KEY environment variable")
        print("\nPress Ctrl+C to stop")
        
        # Launch the interface
        app.launch(share=False, port=7860)
        
    except KeyboardInterrupt:
        print("\n\n👋 Annotation interface stopped")
    except Exception as e:
        logger.error(f"Error launching interface: {e}")
        print(f"❌ Error: {e}")
        
        # Fallback to human-only mode
        print("\n🔄 Falling back to human-only annotation...")
        try:
            app = create_annotation_app(
                use_ai_annotator=False
            )
            print("✅ Human-only interface created")
            app.launch(share=False, port=7860)
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")

if __name__ == "__main__":
    main()
