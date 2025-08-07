#!/usr/bin/env python3
"""
Test online learning KNN classifier with webcam and RTSP streams.
Interactive learning: Press keys to teach the model in real-time.
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.knn_classifier_online import KNNOnlineClassifier
from typing import Optional


class OnlineLearningTester:
    """Test online learning with video streams."""
    
    def __init__(self, 
                 source: str = "0",
                 model_path: str = "models/online_test.pkl",
                 confidence_threshold: float = 0.6):
        """
        Initialize tester.
        
        Args:
            source: Video source - "0" for webcam, RTSP URL for stream
            model_path: Path to save/load model
            confidence_threshold: Confidence threshold for predictions
        """
        self.source = source
        self.model_path = model_path
        
        # Initialize online learning classifier
        self.classifier = KNNOnlineClassifier(
            n_neighbors=3,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            auto_save=True,  # Auto-save after learning
            batch_retrain_interval=5  # Retrain every 5 samples for efficiency
        )
        
        # Class labels for quick teaching (keys 1-9)
        self.quick_labels = {
            '1': 'object_1',
            '2': 'object_2', 
            '3': 'object_3',
            '4': 'object_4',
            '5': 'object_5',
            '6': 'face',
            '7': 'hand',
            '8': 'background',
            '9': 'other'
        }
        
        # Stats
        self.predictions_made = 0
        self.corrections_made = 0
        self.fps = 0
        
    def setup_video_capture(self) -> cv2.VideoCapture:
        """Setup video capture from webcam or RTSP."""
        if self.source == "0":
            # Webcam
            cap = cv2.VideoCapture(0)
            print("Using webcam (index 0)")
        elif self.source.startswith("rtsp://"):
            # RTSP stream
            cap = cv2.VideoCapture(self.source)
            print(f"Using RTSP stream: {self.source}")
        else:
            # Try as integer (other webcam index)
            try:
                cap = cv2.VideoCapture(int(self.source))
                print(f"Using webcam (index {self.source})")
            except:
                cap = cv2.VideoCapture(self.source)
                print(f"Using video file: {self.source}")
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.source}")
            
        # Set buffer size for RTSP to reduce latency
        if self.source.startswith("rtsp://"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        return cap
    
    def draw_ui(self, frame: np.ndarray, prediction: Optional[dict] = None):
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Title and FPS
        cv2.putText(frame, "Online Learning Classifier", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w-100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Prediction info
        if prediction:
            label = prediction.get('label', 'unknown')
            confidence = prediction.get('confidence', 0.0)
            is_known = prediction.get('is_known', False)
            
            # Color based on confidence
            if not is_known:
                color = (0, 0, 255)  # Red for unknown
            elif confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow for medium
            else:
                color = (0, 165, 255)  # Orange for low
            
            # Prediction text
            cv2.putText(frame, f"Prediction: {label}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Top 3 predictions (if available)
            if 'all_scores' in prediction and prediction['all_scores']:
                top_3 = sorted(prediction['all_scores'].items(), 
                              key=lambda x: x[1], reverse=True)[:3]
                for i, (cls, score) in enumerate(top_3):
                    cv2.putText(frame, f"{i+1}. {cls}: {score:.2%}", 
                               (w-200, 60 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Stats
        stats = self.classifier.get_learning_stats()
        total_samples = stats['total_samples']
        num_classes = len(stats['classes'])
        
        cv2.putText(frame, f"Samples: {total_samples} | Classes: {num_classes}", 
                   (10, h-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Predictions: {self.predictions_made} | Corrections: {self.corrections_made}", 
                   (10, h-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        instructions = "Keys: 1-9=Teach | L=Learn current | S=Stats | R=Reset | Q=Quit"
        cv2.putText(frame, instructions, (w//2 - 250, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def print_stats(self):
        """Print detailed statistics."""
        stats = self.classifier.get_learning_stats()
        print("\n" + "="*50)
        print("ONLINE LEARNING STATISTICS")
        print("="*50)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Number of classes: {len(stats['classes'])}")
        print(f"Classes: {', '.join(stats['classes']) if stats['classes'] else 'None'}")
        print(f"\nSamples per class:")
        for cls, count in stats['samples_per_class'].items():
            print(f"  {cls}: {count}")
        print(f"\nPending retrain: {stats['pending_retrain']}")
        print(f"Samples since retrain: {stats['samples_since_retrain']}")
        print(f"Batch interval: {stats['batch_interval']}")
        print(f"Auto-save: {stats['auto_save']}")
        print(f"\nSession stats:")
        print(f"  Predictions made: {self.predictions_made}")
        print(f"  Corrections made: {self.corrections_made}")
        if self.predictions_made > 0:
            accuracy = 1 - (self.corrections_made / self.predictions_made)
            print(f"  Session accuracy: {accuracy:.2%}")
        print("="*50 + "\n")
    
    def run(self):
        """Run interactive online learning test."""
        print("\n" + "="*50)
        print("ONLINE LEARNING CLASSIFIER TEST")
        print("="*50)
        print("\nControls:")
        print("  1-9: Teach model with quick labels")
        print("  L: Learn current frame (custom label)")
        print("  S: Show statistics")
        print("  R: Reset model")
        print("  Q: Quit")
        print("\nThe model will learn from your corrections in real-time!")
        print("="*50 + "\n")
        
        cap = self.setup_video_capture()
        
        # FPS calculation
        fps_frames = 0
        fps_start = time.time()
        
        # Current frame for learning
        current_frame = None
        last_prediction = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                current_frame = frame.copy()
                
                # Make prediction every frame
                try:
                    result = self.classifier.predict(current_frame)
                    self.predictions_made += 1
                    
                    last_prediction = {
                        'label': result.label,
                        'confidence': result.confidence,
                        'all_scores': result.all_scores,
                        'is_known': result.is_known
                    }
                except Exception as e:
                    print(f"Prediction error: {e}")
                    last_prediction = None
                
                # Calculate FPS
                fps_frames += 1
                if fps_frames >= 10:
                    fps_end = time.time()
                    self.fps = fps_frames / (fps_end - fps_start)
                    fps_frames = 0
                    fps_start = time.time()
                
                # Draw UI
                display_frame = self.draw_ui(frame, last_prediction)
                cv2.imshow("Online Learning Test", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                    
                elif key == ord('s'):
                    self.print_stats()
                    
                elif key == ord('r'):
                    # Reset model
                    self.classifier = KNNOnlineClassifier(
                        n_neighbors=3,
                        confidence_threshold=0.6,
                        model_path=self.model_path,
                        auto_save=True,
                        batch_retrain_interval=5
                    )
                    self.predictions_made = 0
                    self.corrections_made = 0
                    print("Model reset!")
                    
                elif key == ord('l'):
                    # Learn with custom label
                    print("\nEnter label for current frame:")
                    label = input("Label: ").strip()
                    if label:
                        _, learned = self.classifier.predict_and_learn(
                            current_frame, 
                            correct_label=label,
                            force_learn=True
                        )
                        if learned:
                            self.corrections_made += 1
                            print(f"Learned '{label}'!")
                            
                elif chr(key) in self.quick_labels:
                    # Quick teach with number keys
                    label = self.quick_labels[chr(key)]
                    _, learned = self.classifier.predict_and_learn(
                        current_frame,
                        correct_label=label,
                        force_learn=True
                    )
                    if learned:
                        self.corrections_made += 1
                        print(f"Learned '{label}'!")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Final stats
            print("\nFinal Statistics:")
            self.print_stats()
            
            # Save model
            if self.classifier.auto_save:
                print(f"Model saved to: {self.model_path}")


def main():
    parser = argparse.ArgumentParser(description="Test online learning with video streams")
    parser.add_argument(
        '--source',
        default='0',
        help='Video source: 0 for webcam, RTSP URL for stream, or video file path'
    )
    parser.add_argument(
        '--rtsp',
        help='RTSP stream URL (overrides --source)'
    )
    parser.add_argument(
        '--model',
        default='models/online_test.pkl',
        help='Path to save/load model'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Confidence threshold (0-1)'
    )
    
    args = parser.parse_args()
    
    # Use RTSP if provided
    source = args.rtsp if args.rtsp else args.source
    
    # Create models directory if needed
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    
    # Run tester
    tester = OnlineLearningTester(
        source=source,
        model_path=args.model,
        confidence_threshold=args.threshold
    )
    
    tester.run()


if __name__ == "__main__":
    main()