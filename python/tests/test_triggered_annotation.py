#!/usr/bin/env python3
"""
Test script for triggered annotation system with dynamic trigger selection.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import time

from edaxshifu.triggered_annotation import (
    TriggeredAnnotationSystem,
    ConfidenceTrigger, UnknownObjectTrigger, NoveltyTrigger
)
from edaxshifu.trigger_system import (
    KeyboardTrigger, MotionTrigger, TimerTrigger,
    CompositeTrigger
)


def main():
    parser = argparse.ArgumentParser(description="Test triggered annotation with flexible triggers")
    
    # Video source
    parser.add_argument('--source', default='0', 
                       help='Video source: 0 for webcam, RTSP URL, or video file')
    
    # Trigger selection
    parser.add_argument('--keyboard', action='store_true',
                       help='Enable keyboard trigger (press C to capture)')
    parser.add_argument('--keyboard-key', default='c',
                       help='Key for keyboard trigger')
    
    parser.add_argument('--motion', action='store_true',
                       help='Enable motion detection trigger')
    parser.add_argument('--motion-sensitivity', type=float, default=0.1,
                       help='Motion sensitivity (0-1, lower is more sensitive)')
    
    parser.add_argument('--confidence', action='store_true',
                       help='Enable low confidence trigger')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Confidence threshold (trigger when below)')
    
    parser.add_argument('--unknown', action='store_true',
                       help='Enable unknown object trigger')
    parser.add_argument('--unknown-frames', type=int, default=5,
                       help='Consecutive frames for unknown trigger')
    
    parser.add_argument('--timer', action='store_true',
                       help='Enable timer trigger')
    parser.add_argument('--timer-interval', type=float, default=10.0,
                       help='Timer interval in seconds')
    
    parser.add_argument('--novelty', action='store_true',
                       help='Enable novelty detection trigger')
    
    parser.add_argument('--composite', action='store_true',
                       help='Create composite trigger (motion AND unknown)')
    
    # General settings
    parser.add_argument('--cooldown', type=float, default=2.0,
                       help='Default cooldown between triggers')
    parser.add_argument('--model', default='models/triggered_test.pkl',
                       help='Model save path')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive annotation mode')
    
    args = parser.parse_args()
    
    # Create system
    print("\n" + "="*60)
    print("TRIGGERED ANNOTATION SYSTEM")
    print("="*60)
    
    system = TriggeredAnnotationSystem(model_path=args.model)
    
    # Configure triggers based on arguments
    triggers_added = []
    
    if args.keyboard:
        trigger = KeyboardTrigger(key=args.keyboard_key, name="keyboard")
        trigger.set_cooldown(0.5)  # Quick response for keyboard
        system.add_trigger(trigger)
        triggers_added.append(f"Keyboard (key: {args.keyboard_key})")
    
    if args.motion:
        trigger = MotionTrigger(sensitivity=args.motion_sensitivity, name="motion")
        trigger.set_cooldown(args.cooldown)
        system.add_trigger(trigger)
        triggers_added.append(f"Motion (sensitivity: {args.motion_sensitivity})")
    
    if args.confidence:
        trigger = ConfidenceTrigger(threshold=args.confidence_threshold, name="confidence")
        trigger.set_cooldown(args.cooldown)
        system.add_trigger(trigger)
        triggers_added.append(f"Low Confidence (< {args.confidence_threshold})")
    
    if args.unknown:
        trigger = UnknownObjectTrigger(consecutive_frames=args.unknown_frames, name="unknown")
        trigger.set_cooldown(args.cooldown * 2)  # Longer cooldown for unknown
        system.add_trigger(trigger)
        triggers_added.append(f"Unknown Object ({args.unknown_frames} frames)")
    
    if args.timer:
        trigger = TimerTrigger(interval_seconds=args.timer_interval, name="timer")
        system.add_trigger(trigger)
        triggers_added.append(f"Timer (every {args.timer_interval}s)")
    
    if args.novelty:
        trigger = NoveltyTrigger(difference_threshold=0.3, name="novelty")
        trigger.set_cooldown(args.cooldown)
        system.add_trigger(trigger)
        triggers_added.append("Novelty Detection")
    
    if args.composite:
        # Example: Trigger when both motion AND unknown object detected
        motion_t = MotionTrigger(sensitivity=0.2)
        unknown_t = UnknownObjectTrigger(consecutive_frames=3)
        composite = CompositeTrigger([motion_t, unknown_t], logic="AND", name="composite")
        composite.set_cooldown(args.cooldown * 3)
        system.add_trigger(composite)
        triggers_added.append("Composite (Motion + Unknown)")
    
    # If no triggers specified, add default set
    if not triggers_added:
        print("No triggers specified, using defaults...")
        system.setup_default_triggers()
        triggers_added = ["Keyboard (c)", "Motion", "Low Confidence", "Unknown Object", "Timer (disabled)"]
    
    print("\nActive Triggers:")
    for i, trigger in enumerate(triggers_added, 1):
        print(f"  {i}. {trigger}")
    
    print("\nControls:")
    print("  Q: Quit")
    print("  S: Show statistics")
    print("  E: Enable/disable triggers")
    print("  R: Reset model")
    if args.interactive:
        print("  When triggered: Enter label to teach")
    print("="*60 + "\n")
    
    # Setup video capture
    if args.source == '0':
        cap = cv2.VideoCapture(0)
    elif args.source.startswith('rtsp://'):
        cap = cv2.VideoCapture(args.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Failed to open video source: {args.source}")
        return
    
    # Start annotation processor
    system.start_annotation_processor()
    
    # Main loop
    last_trigger_time = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            result = system.process_frame(frame)
            
            # Draw UI
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Prediction info
            pred = result['prediction']
            color = (0, 255, 0) if pred['is_known'] else (0, 0, 255)
            cv2.putText(display_frame, f"Prediction: {pred['label']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, f"Confidence: {pred['confidence']:.2%}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Trigger status
            if result['triggers']:
                last_trigger_time = time.time()
                for event in result['triggers']:
                    print(f"TRIGGERED: {event.description}")
                    
                    # Interactive annotation if enabled
                    if args.interactive:
                        label = system.interactive_annotation(frame, event)
                        if label:
                            print(f"Annotated as: {label}")
            
            # Flash border if recently triggered
            if time.time() - last_trigger_time < 0.5:
                cv2.rectangle(display_frame, (0, 0), (w-1, h-1), (0, 255, 255), 5)
            
            # Statistics
            stats = system.get_statistics()
            cv2.putText(display_frame, f"Frames: {stats['frames_processed']}", (10, h-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Triggers: {stats['triggers_fired']}", (10, h-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Learned: {stats['auto_learned']}", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Triggered Annotation Test", display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Show statistics
                print("\n" + "="*50)
                print("STATISTICS")
                print("="*50)
                for k, v in stats.items():
                    if isinstance(v, dict):
                        print(f"\n{k}:")
                        for kk, vv in v.items():
                            print(f"  {kk}: {vv}")
                    else:
                        print(f"{k}: {v}")
                print("="*50 + "\n")
            elif key == ord('e'):
                # Toggle triggers
                print("\nTrigger Management:")
                triggers = system.list_triggers()
                for i, name in enumerate(triggers):
                    status = system.trigger_manager.triggers[name].enabled
                    print(f"  {i+1}. {name}: {'ENABLED' if status else 'DISABLED'}")
                
                choice = input("Enter trigger number to toggle (or Enter to skip): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(triggers):
                    trigger_name = triggers[int(choice)-1]
                    trigger = system.trigger_manager.triggers[trigger_name]
                    if trigger.enabled:
                        system.disable_trigger(trigger_name)
                    else:
                        system.enable_trigger(trigger_name)
            elif key == ord('r'):
                # Reset model
                system.classifier = system.classifier.__class__(
                    model_path=args.model,
                    auto_save=True
                )
                print("Model reset!")
                
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        # Cleanup
        system.stop_annotation_processor()
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        stats = system.get_statistics()
        print(f"Total frames: {stats['frames_processed']}")
        print(f"Total triggers: {stats['triggers_fired']}")
        print(f"Total annotations: {stats['annotations_created']}")
        print(f"Total learned: {stats['auto_learned']}")
        
        if stats['frames_processed'] > 0:
            trigger_rate = stats['triggers_fired'] / stats['frames_processed']
            print(f"Trigger rate: {trigger_rate:.2%}")
        
        print("="*50)


if __name__ == "__main__":
    main()
