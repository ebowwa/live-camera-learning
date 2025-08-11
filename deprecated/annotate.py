#!/usr/bin/env python3
"""
Launch the human annotation interface for failed recognitions.
"""

import argparse
import logging
from python.edaxshifu.annotation_interface import create_annotation_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Human Annotation Interface for EdaxShifu')
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/knn_classifier.pkl',
        help='Path to KNN model file (default: models/knn_classifier.pkl)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run the interface on (default: 7860)'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public shareable link'
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("EdaxShifu Human Annotation Interface")
    print("="*50)
    print(f"KNN Model: {args.model_path}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print("="*50)
    print("\nThe interface will help you:")
    print("1. Review images the AI couldn't recognize")
    print("2. Provide correct labels")
    print("3. Automatically train the KNN classifier")
    print("4. Move annotated images to the dataset")
    print("\nStarting interface...")
    
    # Create and launch the app
    app = create_annotation_app(args.model_path)
    app.launch(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
