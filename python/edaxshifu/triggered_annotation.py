#!/usr/bin/env python3
"""
Triggered Annotation System - Flexible trigger-based detection and annotation.
Uses the abstract trigger system to allow dynamic trigger selection.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import logging
from dataclasses import dataclass
import threading
import queue

# Import existing systems
from .trigger_system import (
    TriggerManager, TriggerEvent, TriggerType,
    KeyboardTrigger, MotionTrigger, TimerTrigger,
    ObjectDetectionTrigger, GestureTrigger, 
    AudioTrigger, CompositeTrigger, Trigger
)
from .knn_classifier_online import KNNOnlineClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceTrigger(Trigger):
    """Trigger when prediction confidence is below threshold."""
    
    def __init__(self, threshold: float = 0.5, name: str = "low_confidence"):
        super().__init__(name)
        self.threshold = threshold
        
    def check(self, frame: Optional[np.ndarray] = None, **kwargs) -> Optional[TriggerEvent]:
        """Check if confidence is below threshold."""
        prediction = kwargs.get('prediction')
        
        if prediction and prediction.get('confidence', 1.0) < self.threshold:
            event = TriggerEvent(
                trigger_type=TriggerType.OBJECT_DETECTION,
                timestamp=time.time(),
                data={
                    'confidence': prediction['confidence'],
                    'label': prediction.get('label', 'unknown'),
                    'threshold': self.threshold
                },
                confidence=1.0 - prediction['confidence'],  # Higher trigger confidence for lower prediction
                description=f"Low confidence: {prediction['confidence']:.2f} < {self.threshold}"
            )
            if self.fire(event):
                return event
        return None


class UnknownObjectTrigger(Trigger):
    """Trigger when unknown object is detected."""
    
    def __init__(self, consecutive_frames: int = 3, name: str = "unknown_object"):
        super().__init__(name)
        self.consecutive_frames = consecutive_frames
        self.unknown_buffer = []
        
    def check(self, frame: Optional[np.ndarray] = None, **kwargs) -> Optional[TriggerEvent]:
        """Check for unknown objects."""
        prediction = kwargs.get('prediction')
        
        if prediction:
            is_unknown = prediction.get('label') == 'unknown' or not prediction.get('is_known', True)
            
            # Add to buffer
            self.unknown_buffer.append(is_unknown)
            if len(self.unknown_buffer) > self.consecutive_frames:
                self.unknown_buffer.pop(0)
            
            # Check if we have enough consecutive unknown detections
            if len(self.unknown_buffer) == self.consecutive_frames and all(self.unknown_buffer):
                self.unknown_buffer = []  # Reset after trigger
                
                event = TriggerEvent(
                    trigger_type=TriggerType.OBJECT_DETECTION,
                    timestamp=time.time(),
                    data={'prediction': prediction},
                    confidence=1.0,
                    description="Unknown object detected consistently"
                )
                if self.fire(event):
                    return event
        return None


class NoveltyTrigger(Trigger):
    """Trigger when a novel/different object appears."""
    
    def __init__(self, difference_threshold: float = 0.3, name: str = "novelty"):
        super().__init__(name)
        self.difference_threshold = difference_threshold
        self.recent_labels = []
        self.buffer_size = 10
        
    def check(self, frame: Optional[np.ndarray] = None, **kwargs) -> Optional[TriggerEvent]:
        """Check for novel objects."""
        prediction = kwargs.get('prediction')
        
        if prediction and prediction.get('is_known', False):
            label = prediction.get('label')
            
            # Check if this is different from recent predictions
            if self.recent_labels and label not in self.recent_labels[-5:]:
                event = TriggerEvent(
                    trigger_type=TriggerType.OBJECT_DETECTION,
                    timestamp=time.time(),
                    data={
                        'new_label': label,
                        'recent_labels': list(set(self.recent_labels[-5:]))
                    },
                    confidence=prediction.get('confidence', 0.5),
                    description=f"Novel object detected: {label}"
                )
                
                # Update buffer
                self.recent_labels.append(label)
                if len(self.recent_labels) > self.buffer_size:
                    self.recent_labels.pop(0)
                
                if self.fire(event):
                    return event
            else:
                # Update buffer
                self.recent_labels.append(label)
                if len(self.recent_labels) > self.buffer_size:
                    self.recent_labels.pop(0)
                    
        return None


class TriggeredAnnotationSystem:
    """Main system for triggered detection and annotation."""
    
    def __init__(self, 
                 classifier: Optional[KNNOnlineClassifier] = None,
                 model_path: str = "models/triggered_annotations.pkl"):
        """
        Initialize triggered annotation system.
        
        Args:
            classifier: Optional pre-configured classifier
            model_path: Path to save/load model
        """
        # Initialize classifier
        self.classifier = classifier or KNNOnlineClassifier(
            n_neighbors=3,
            confidence_threshold=0.6,
            model_path=model_path,
            auto_save=True,
            batch_retrain_interval=5
        )
        
        # Initialize trigger manager
        self.trigger_manager = TriggerManager()
        
        # Annotation queue for async processing
        self.annotation_queue = queue.Queue()
        self.annotation_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'triggers_fired': 0,
            'annotations_created': 0,
            'auto_learned': 0
        }
        
        # Annotation callback
        self.annotation_callback = None
        
    def add_trigger(self, trigger: Trigger, callback: Optional[Callable] = None):
        """
        Add a trigger to the system.
        
        Args:
            trigger: Trigger instance to add
            callback: Optional specific callback for this trigger
        """
        # Set up trigger callback
        if callback:
            trigger.add_callback(callback)
        else:
            trigger.add_callback(self._on_trigger_fired)
        
        self.trigger_manager.add_trigger(trigger)
        logger.info(f"Added trigger: {trigger.name} (type: {trigger.__class__.__name__})")
    
    def setup_default_triggers(self):
        """Set up a default set of useful triggers."""
        # Keyboard trigger for manual capture
        keyboard_trigger = KeyboardTrigger(key='c')
        keyboard_trigger.set_cooldown(0.5)
        self.add_trigger(keyboard_trigger)
        
        # Motion trigger for activity
        motion_trigger = MotionTrigger(sensitivity=0.1)
        motion_trigger.set_cooldown(2.0)
        self.add_trigger(motion_trigger)
        
        # Low confidence trigger for uncertain predictions
        confidence_trigger = ConfidenceTrigger(threshold=0.5)
        confidence_trigger.set_cooldown(3.0)
        self.add_trigger(confidence_trigger)
        
        # Unknown object trigger
        unknown_trigger = UnknownObjectTrigger(consecutive_frames=5)
        unknown_trigger.set_cooldown(5.0)
        self.add_trigger(unknown_trigger)
        
        # Timer trigger for periodic capture
        timer_trigger = TimerTrigger(interval_seconds=30.0)
        timer_trigger.enabled = False  # Disabled by default
        self.add_trigger(timer_trigger)
        
        logger.info("Default triggers configured")
    
    def create_custom_trigger_set(self, config: Dict[str, Any]):
        """
        Create triggers from configuration.
        
        Args:
            config: Dictionary with trigger configurations
            
        Example:
            {
                'keyboard': {'key': 's', 'cooldown': 1.0},
                'motion': {'sensitivity': 0.2, 'enabled': True},
                'confidence': {'threshold': 0.4},
                'timer': {'interval': 10.0, 'enabled': False}
            }
        """
        for trigger_type, params in config.items():
            if trigger_type == 'keyboard':
                trigger = KeyboardTrigger(key=params.get('key', 's'))
            elif trigger_type == 'motion':
                trigger = MotionTrigger(sensitivity=params.get('sensitivity', 0.1))
            elif trigger_type == 'confidence':
                trigger = ConfidenceTrigger(threshold=params.get('threshold', 0.5))
            elif trigger_type == 'unknown':
                trigger = UnknownObjectTrigger(consecutive_frames=params.get('frames', 3))
            elif trigger_type == 'timer':
                trigger = TimerTrigger(interval_seconds=params.get('interval', 10.0))
            elif trigger_type == 'novelty':
                trigger = NoveltyTrigger(difference_threshold=params.get('threshold', 0.3))
            else:
                logger.warning(f"Unknown trigger type: {trigger_type}")
                continue
            
            # Configure trigger
            if 'cooldown' in params:
                trigger.set_cooldown(params['cooldown'])
            if 'enabled' in params:
                trigger.enabled = params['enabled']
            
            self.add_trigger(trigger)
    
    def _on_trigger_fired(self, event: TriggerEvent):
        """Handle trigger events."""
        self.stats['triggers_fired'] += 1
        logger.info(f"Trigger fired: {event.description}")
        
        # Add to annotation queue
        self.annotation_queue.put(event)
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a frame through prediction and triggers.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with prediction and trigger results
        """
        self.stats['frames_processed'] += 1
        
        # Make prediction
        result = self.classifier.predict(frame)
        
        prediction = {
            'label': result.label,
            'confidence': result.confidence,
            'all_scores': result.all_scores,
            'is_known': result.is_known
        }
        
        # Check all triggers
        trigger_events = self.trigger_manager.check_all(
            frame,
            prediction=prediction
        )
        
        return {
            'prediction': prediction,
            'triggers': trigger_events,
            'frame': frame
        }
    
    def start_annotation_processor(self):
        """Start background thread for processing annotations."""
        if not self.annotation_thread:
            self.running = True
            self.annotation_thread = threading.Thread(target=self._annotation_worker)
            self.annotation_thread.daemon = True
            self.annotation_thread.start()
            logger.info("Annotation processor started")
    
    def stop_annotation_processor(self):
        """Stop annotation processor."""
        self.running = False
        if self.annotation_thread:
            self.annotation_thread.join(timeout=2.0)
            self.annotation_thread = None
            logger.info("Annotation processor stopped")
    
    def _annotation_worker(self):
        """Worker thread for processing annotations."""
        while self.running:
            try:
                event = self.annotation_queue.get(timeout=1.0)
                self._process_annotation(event)
                self.annotation_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Annotation processing error: {e}")
    
    def _process_annotation(self, event: TriggerEvent):
        """Process an annotation event."""
        self.stats['annotations_created'] += 1
        
        # Call annotation callback if set
        if self.annotation_callback:
            self.annotation_callback(event)
        
        logger.info(f"Annotation processed: {event.description}")
    
    def interactive_annotation(self, frame: np.ndarray, event: TriggerEvent) -> Optional[str]:
        """
        Interactive annotation prompt.
        
        Args:
            frame: Frame that triggered annotation
            event: Trigger event
            
        Returns:
            Label if provided, None otherwise
        """
        print(f"\n{'='*50}")
        print(f"ANNOTATION TRIGGERED: {event.description}")
        print(f"Trigger type: {event.trigger_type.value}")
        print(f"Confidence: {event.confidence:.2f}")
        
        if 'prediction' in event.data:
            pred = event.data['prediction']
            print(f"Current prediction: {pred.get('label')} ({pred.get('confidence', 0):.2f})")
        
        print(f"{'='*50}")
        label = input("Enter label (or press Enter to skip): ").strip()
        
        if label:
            # Learn from annotation
            _, learned = self.classifier.predict_and_learn(frame, label, force_learn=True)
            if learned:
                self.stats['auto_learned'] += 1
                print(f"✓ Learned '{label}'")
            return label
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            **self.stats,
            'triggers': self.trigger_manager.get_status(),
            'classifier': self.classifier.get_learning_stats()
        }
    
    def list_triggers(self) -> List[str]:
        """List all configured triggers."""
        return list(self.trigger_manager.triggers.keys())
    
    def enable_trigger(self, name: str):
        """Enable a specific trigger."""
        self.trigger_manager.enable_trigger(name)
        logger.info(f"Enabled trigger: {name}")
    
    def disable_trigger(self, name: str):
        """Disable a specific trigger."""
        self.trigger_manager.disable_trigger(name)
        logger.info(f"Disabled trigger: {name}")
    
    def set_trigger_cooldown(self, name: str, cooldown: float):
        """Set trigger cooldown."""
        if name in self.trigger_manager.triggers:
            self.trigger_manager.triggers[name].set_cooldown(cooldown)
            logger.info(f"Set {name} cooldown to {cooldown}s")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Triggered Annotation System")
    parser.add_argument('--source', default='0', help='Video source')
    parser.add_argument('--triggers', nargs='+', 
                       choices=['keyboard', 'motion', 'confidence', 'unknown', 'timer', 'novelty'],
                       default=['keyboard', 'motion', 'confidence'],
                       help='Triggers to enable')
    args = parser.parse_args()
    
    # Create system
    system = TriggeredAnnotationSystem()
    
    # Configure triggers dynamically based on arguments
    trigger_config = {}
    for trigger in args.triggers:
        if trigger == 'keyboard':
            trigger_config['keyboard'] = {'key': 'c', 'cooldown': 0.5}
        elif trigger == 'motion':
            trigger_config['motion'] = {'sensitivity': 0.1, 'cooldown': 2.0}
        elif trigger == 'confidence':
            trigger_config['confidence'] = {'threshold': 0.5, 'cooldown': 3.0}
        elif trigger == 'unknown':
            trigger_config['unknown'] = {'frames': 5, 'cooldown': 5.0}
        elif trigger == 'timer':
            trigger_config['timer'] = {'interval': 10.0, 'enabled': True}
        elif trigger == 'novelty':
            trigger_config['novelty'] = {'threshold': 0.3, 'cooldown': 4.0}
    
    system.create_custom_trigger_set(trigger_config)
    
    print(f"Configured triggers: {system.list_triggers()}")
    print("System ready for triggered annotations!")
