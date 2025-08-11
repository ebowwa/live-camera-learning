"""
Live KNN model reloader using file system monitoring.
Watches for model file changes and triggers reloads automatically.
"""

import os
import time
import threading
from typing import Callable, Optional
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class ModelFileHandler(FileSystemEventHandler):
    """Handler for model file changes."""
    
    def __init__(self, model_path: str, reload_callback: Callable[[], None]):
        self.model_path = model_path
        self.reload_callback = reload_callback
        self.last_reload = 0
        self.cooldown = 2.0  # Prevent rapid reloads
        
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        # Check if it's our model file
        if os.path.abspath(event.src_path) == os.path.abspath(self.model_path):
            current_time = time.time()
            
            # Cooldown to prevent rapid successive reloads
            if current_time - self.last_reload > self.cooldown:
                logger.info(f"Model file changed: {event.src_path}, reloading...")
                try:
                    self.reload_callback()
                    self.last_reload = current_time
                    logger.info("✅ Model reloaded successfully")
                except Exception as e:
                    logger.error(f"❌ Model reload failed: {e}")


class LiveModelReloader:
    """
    Monitors KNN model file and automatically reloads when it changes.
    Enables live learning without system restart.
    """
    
    def __init__(self, model_path: str, reload_callback: Callable[[], None]):
        """
        Initialize live model reloader.
        
        Args:
            model_path: Path to the KNN model file to monitor
            reload_callback: Function to call when model needs reloading
        """
        self.model_path = model_path
        self.reload_callback = reload_callback
        self.observer = None
        self.handler = None
        self.running = False
        
    def start(self) -> bool:
        """Start monitoring the model file."""
        if self.running:
            logger.warning("Live model reloader already running")
            return False
            
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found yet: {self.model_path}, will watch for creation")
            
        try:
            # Create file system event handler
            self.handler = ModelFileHandler(self.model_path, self.reload_callback)
            
            # Create observer and watch the directory containing the model
            self.observer = Observer()
            model_dir = os.path.dirname(os.path.abspath(self.model_path))
            
            # Ensure the directory exists
            os.makedirs(model_dir, exist_ok=True)
            
            self.observer.schedule(self.handler, model_dir, recursive=False)
            self.observer.start()
            
            self.running = True
            logger.info(f"🔄 Live model reloader started - watching {self.model_path}")
            return True
            
        except Exception as e:
            logger.warning(f"File watcher failed to start: {e}")
            return False
    
    def stop(self):
        """Stop monitoring the model file."""
        if not self.running:
            return
            
        if self.observer:
            self.observer.stop()
            self.observer.join()
            
        self.running = False
        logger.info("🛑 Live model reloader stopped")
    
    def is_running(self) -> bool:
        """Check if the reloader is currently running."""
        return self.running


# Alternative approach: Polling-based reloader for simpler deployments
class PollingModelReloader:
    """
    Polls model file modification time and reloads when changed.
    Simpler alternative to file system watching.
    """
    
    def __init__(self, model_path: str, reload_callback: Callable[[], None], 
                 poll_interval: float = 2.0):
        """
        Initialize polling model reloader.
        
        Args:
            model_path: Path to the KNN model file to monitor
            reload_callback: Function to call when model needs reloading
            poll_interval: How often to check for changes (seconds)
        """
        self.model_path = model_path
        self.reload_callback = reload_callback
        self.poll_interval = poll_interval
        self.last_modified = 0
        self.polling_thread = None
        self.running = False
        
    def start(self) -> bool:
        """Start polling the model file."""
        if self.running:
            logger.warning("Polling model reloader already running")
            return False
            
        if not os.path.exists(self.model_path):
            logger.info(f"Model file not found yet: {self.model_path}, will watch for creation")
            self.last_modified = 0  # Will detect when file is first created
        else:
            # Get initial modification time
            self.last_modified = os.path.getmtime(self.model_path)
        
        # Start polling thread
        self.running = True
        self.polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.polling_thread.start()
        
        logger.info(f"📊 Polling model reloader started - checking {self.model_path} every {self.poll_interval}s")
        return True
    
    def stop(self):
        """Stop polling the model file."""
        self.running = False
        if self.polling_thread:
            self.polling_thread.join(timeout=5.0)
        logger.info("🛑 Polling model reloader stopped")
    
    def _poll_loop(self):
        """Main polling loop."""
        while self.running:
            try:
                if os.path.exists(self.model_path):
                    current_modified = os.path.getmtime(self.model_path)
                    
                    if current_modified > self.last_modified:
                        logger.info("Model file changed, reloading...")
                        try:
                            self.reload_callback()
                            self.last_modified = current_modified
                            logger.info("✅ Model reloaded successfully")
                        except Exception as e:
                            logger.error(f"❌ Model reload failed: {e}")
                            
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                
            time.sleep(self.poll_interval)
    
    def is_running(self) -> bool:
        """Check if the reloader is currently running."""
        return self.running