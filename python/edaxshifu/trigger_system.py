"""
Flexible trigger system for capturing frames based on various inputs.
"""

import time
import threading
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of triggers available."""
    KEYBOARD = "keyboard"
    GESTURE = "gesture"
    AUDIO = "audio"
    OBJECT_DETECTION = "object_detection"
    TIMER = "timer"
    MANUAL = "manual"
    MOTION = "motion"


@dataclass
class TriggerEvent:
    """Event data when a trigger fires."""
    trigger_type: TriggerType
    timestamp: float
    data: Dict[str, Any]
    confidence: float = 1.0
    description: str = ""


class Trigger(ABC):
    """Abstract base class for all triggers."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.last_triggered = 0
        self.cooldown = 0  # seconds between triggers
        self.callbacks: List[Callable] = []
        
    @abstractmethod
    def check(self, frame: Optional[np.ndarray] = None, **kwargs) -> Optional[TriggerEvent]:
        """Check if trigger condition is met."""
        pass
    
    def can_trigger(self) -> bool:
        """Check if enough time has passed since last trigger."""
        return time.time() - self.last_triggered >= self.cooldown
    
    def fire(self, event: TriggerEvent):
        """Fire the trigger and notify callbacks."""
        if not self.enabled or not self.can_trigger():
            return False
            
        self.last_triggered = time.time()
        logger.info(f"Trigger '{self.name}' fired: {event.description}")
        
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error in trigger {self.name}: {e}")
                
        return True
    
    def add_callback(self, callback: Callable):
        """Add a callback to be called when trigger fires."""
        self.callbacks.append(callback)
        
    def set_cooldown(self, seconds: float):
        """Set minimum time between triggers."""
        self.cooldown = seconds


class KeyboardTrigger(Trigger):
    """Trigger on specific keyboard input."""
    
    def __init__(self, key: str = 's', name: str = "keyboard"):
        super().__init__(name)
        self.trigger_key = ord(key) if len(key) == 1 else key
        self.last_key = None
        
    def check(self, frame: Optional[np.ndarray] = None, **kwargs) -> Optional[TriggerEvent]:
        """Check for keyboard input (requires OpenCV window)."""
        key = cv2.waitKey(1) & 0xFF
        
        if key != 255 and key == self.trigger_key:  # 255 means no key pressed
            event = TriggerEvent(
                trigger_type=TriggerType.KEYBOARD,
                timestamp=time.time(),
                data={'key': chr(key) if key < 128 else key},
                description=f"Key '{chr(key)}' pressed"
            )
            if self.fire(event):
                return event
        return None


class GestureTrigger(Trigger):
    """Trigger on specific hand gestures."""
    
    def __init__(self, gesture_type: str = "thumbs_up", name: str = "gesture"):
        super().__init__(name)
        self.gesture_type = gesture_type
        self.gesture_buffer = []
        self.buffer_size = 3  # Need N consecutive detections
        
    def check(self, frame: Optional[np.ndarray] = None, **kwargs) -> Optional[TriggerEvent]:
        """Check for specific gestures in frame."""
        gestures = kwargs.get('gestures', [])
        
        # Look for target gesture
        detected = any(g.get('type') == self.gesture_type for g in gestures)
        
        # Add to buffer
        self.gesture_buffer.append(detected)
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)
            
        # Check if we have enough consecutive detections
        if len(self.gesture_buffer) == self.buffer_size and all(self.gesture_buffer):
            self.gesture_buffer = []  # Reset after trigger
            
            event = TriggerEvent(
                trigger_type=TriggerType.GESTURE,
                timestamp=time.time(),
                data={'gesture': self.gesture_type, 'gestures': gestures},
                confidence=0.9,
                description=f"Gesture '{self.gesture_type}' detected"
            )
            if self.fire(event):
                return event
        return None


class ObjectDetectionTrigger(Trigger):
    """Trigger when specific objects are detected."""
    
    def __init__(self, target_objects: List[str] = None, 
                 min_confidence: float = 0.6,
                 require_motion: bool = False,
                 name: str = "object"):
        super().__init__(name)
        self.target_objects = target_objects or ["person", "hand"]
        self.min_confidence = min_confidence
        self.require_motion = require_motion
        self.prev_detections = {}
        
    def check(self, frame: Optional[np.ndarray] = None, **kwargs) -> Optional[TriggerEvent]:
        """Check for target objects in detections."""
        detections = kwargs.get('detections', [])
        
        # Filter for target objects with sufficient confidence
        valid_detections = [
            d for d in detections
            if d.get('confidence', 0) >= self.min_confidence and
            (not self.target_objects or d.get('class_name') in self.target_objects)
        ]
        
        if not valid_detections:
            return None
            
        # Check for motion if required
        if self.require_motion:
            moving = self._check_motion(valid_detections)
            if not moving:
                return None
                
        # Create trigger event
        event = TriggerEvent(
            trigger_type=TriggerType.OBJECT_DETECTION,
            timestamp=time.time(),
            data={'detections': valid_detections},
            confidence=max(d['confidence'] for d in valid_detections),
            description=f"Detected: {', '.join(set(d['class_name'] for d in valid_detections))}"
        )
        
        if self.fire(event):
            return event
        return None
    
    def _check_motion(self, detections: List[Dict]) -> bool:
        """Check if objects are moving."""
        motion_detected = False
        
        for det in detections:
            obj_id = det.get('class_name', 'unknown')
            bbox = det.get('bbox', [0, 0, 0, 0])
            
            if obj_id in self.prev_detections:
                prev_bbox = self.prev_detections[obj_id]
                # Calculate movement
                movement = sum(abs(bbox[i] - prev_bbox[i]) for i in range(4))
                if movement > 20:  # Threshold for significant movement
                    motion_detected = True
                    
            self.prev_detections[obj_id] = bbox
            
        return motion_detected


class AudioTrigger(Trigger):
    """Trigger on audio events (clap, voice command, etc)."""
    
    def __init__(self, trigger_sound: str = "clap", 
                 volume_threshold: float = 0.3,
                 name: str = "audio"):
        super().__init__(name)
        self.trigger_sound = trigger_sound
        self.volume_threshold = volume_threshold
        # Note: Actual audio processing would require additional libraries
        
    def check(self, frame: Optional[np.ndarray] = None, **kwargs) -> Optional[TriggerEvent]:
        """Check for audio triggers."""
        audio_data = kwargs.get('audio_data')
        
        if audio_data:
            # Simplified audio detection
            volume = kwargs.get('volume', 0)
            
            if volume > self.volume_threshold:
                event = TriggerEvent(
                    trigger_type=TriggerType.AUDIO,
                    timestamp=time.time(),
                    data={'sound': self.trigger_sound, 'volume': volume},
                    confidence=min(volume / 1.0, 1.0),
                    description=f"Audio trigger: {self.trigger_sound}"
                )
                if self.fire(event):
                    return event
        return None


class TimerTrigger(Trigger):
    """Trigger at regular intervals."""
    
    def __init__(self, interval_seconds: float = 5.0, name: str = "timer"):
        super().__init__(name)
        self.interval = interval_seconds
        self.last_triggered = time.time()
        
    def check(self, frame: Optional[np.ndarray] = None, **kwargs) -> Optional[TriggerEvent]:
        """Check if interval has elapsed."""
        current_time = time.time()
        
        if current_time - self.last_triggered >= self.interval:
            event = TriggerEvent(
                trigger_type=TriggerType.TIMER,
                timestamp=current_time,
                data={'interval': self.interval},
                description=f"Timer interval {self.interval}s elapsed"
            )
            if self.fire(event):
                return event
        return None


class MotionTrigger(Trigger):
    """Trigger on significant motion in frame."""
    
    def __init__(self, sensitivity: float = 0.1, name: str = "motion"):
        super().__init__(name)
        self.sensitivity = sensitivity
        self.prev_frame = None
        self.motion_threshold = 1000  # Adjust based on testing
        
    def check(self, frame: Optional[np.ndarray] = None, **kwargs) -> Optional[TriggerEvent]:
        """Check for motion between frames."""
        if frame is None:
            return None
            
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return None
            
        # Calculate frame difference
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Calculate motion score
        motion_score = np.sum(thresh) / 255
        
        # Update previous frame
        self.prev_frame = gray
        
        if motion_score > self.motion_threshold:
            event = TriggerEvent(
                trigger_type=TriggerType.MOTION,
                timestamp=time.time(),
                data={'motion_score': motion_score},
                confidence=min(motion_score / (self.motion_threshold * 2), 1.0),
                description=f"Motion detected (score: {motion_score:.0f})"
            )
            if self.fire(event):
                return event
        return None


class CompositeTrigger(Trigger):
    """Combine multiple triggers with AND/OR logic."""
    
    def __init__(self, triggers: List[Trigger], 
                 logic: str = "OR",
                 name: str = "composite"):
        super().__init__(name)
        self.triggers = triggers
        self.logic = logic.upper()
        
    def check(self, frame: Optional[np.ndarray] = None, **kwargs) -> Optional[TriggerEvent]:
        """Check all sub-triggers."""
        events = []
        
        for trigger in self.triggers:
            if trigger.enabled:
                event = trigger.check(frame, **kwargs)
                if event:
                    events.append(event)
                    
        if self.logic == "OR" and events:
            # Any trigger fires
            composite_event = TriggerEvent(
                trigger_type=TriggerType.MANUAL,
                timestamp=time.time(),
                data={'events': events},
                confidence=max(e.confidence for e in events),
                description=f"Composite trigger ({len(events)} sub-triggers)"
            )
            if self.fire(composite_event):
                return composite_event
                
        elif self.logic == "AND" and len(events) == len([t for t in self.triggers if t.enabled]):
            # All enabled triggers fire
            composite_event = TriggerEvent(
                trigger_type=TriggerType.MANUAL,
                timestamp=time.time(),
                data={'events': events},
                confidence=min(e.confidence for e in events),
                description=f"All {len(events)} triggers fired"
            )
            if self.fire(composite_event):
                return composite_event
                
        return None


class TriggerManager:
    """Manages multiple triggers and their interactions."""
    
    def __init__(self):
        self.triggers: Dict[str, Trigger] = {}
        self.capture_callback: Optional[Callable] = None
        self.event_history: List[TriggerEvent] = []
        self.max_history = 100
        
    def add_trigger(self, trigger: Trigger):
        """Add a trigger to the manager."""
        self.triggers[trigger.name] = trigger
        logger.info(f"Added trigger: {trigger.name}")
        
    def remove_trigger(self, name: str):
        """Remove a trigger."""
        if name in self.triggers:
            del self.triggers[name]
            logger.info(f"Removed trigger: {name}")
            
    def set_capture_callback(self, callback: Callable):
        """Set callback for when any trigger fires."""
        self.capture_callback = callback
        
        # Add to all triggers
        for trigger in self.triggers.values():
            trigger.add_callback(self._on_trigger)
            
    def _on_trigger(self, event: TriggerEvent):
        """Internal callback when any trigger fires."""
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
            
        # Call capture callback
        if self.capture_callback:
            self.capture_callback(event)
            
    def check_all(self, frame: Optional[np.ndarray] = None, **kwargs) -> List[TriggerEvent]:
        """Check all enabled triggers."""
        events = []
        
        for trigger in self.triggers.values():
            if trigger.enabled:
                event = trigger.check(frame, **kwargs)
                if event:
                    events.append(event)
                    
        return events
    
    def enable_trigger(self, name: str):
        """Enable a specific trigger."""
        if name in self.triggers:
            self.triggers[name].enabled = True
            
    def disable_trigger(self, name: str):
        """Disable a specific trigger."""
        if name in self.triggers:
            self.triggers[name].enabled = False
            
    def get_status(self) -> Dict[str, Any]:
        """Get status of all triggers."""
        return {
            name: {
                'enabled': trigger.enabled,
                'last_triggered': trigger.last_triggered,
                'cooldown': trigger.cooldown
            }
            for name, trigger in self.triggers.items()
        }