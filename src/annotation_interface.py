"""
Gradio-based human annotation interface for the intelligent capture system.
"""

import gradio as gr
import cv2
import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import threading
import queue
import logging
from PIL import Image
import numpy as np

from .knn_classifier import AdaptiveKNNClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnnotationTask:
    """Task for human annotation."""
    image_path: str
    image_array: np.ndarray
    metadata: Dict
    timestamp: str
    yolo_detections: List[str]
    knn_prediction: Optional[str] = None
    knn_confidence: Optional[float] = None


class AnnotationQueue:
    """Manages the queue of images waiting for annotation."""
    
    def __init__(self, failed_dir: str = "captures/failed", processed_dir: str = "captures/processed"):
        self.failed_dir = failed_dir
        self.processed_dir = processed_dir
        self.queue = queue.Queue()
        self.currently_annotating = None
        self.lock = threading.Lock()
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def scan_for_tasks(self) -> int:
        """Scan the failed directory for new annotation tasks."""
        if not os.path.exists(self.failed_dir):
            os.makedirs(self.failed_dir, exist_ok=True)
            return 0
            
        new_tasks = 0
        # Get all jpg files, sorted by modification time (newest first)
        files = sorted(
            [f for f in os.listdir(self.failed_dir) if f.endswith('.jpg')],
            key=lambda x: os.path.getmtime(os.path.join(self.failed_dir, x)),
            reverse=True
        )
        
        for filename in files:
            # Skip if this is the file currently being annotated
            if self.currently_annotating and filename == os.path.basename(self.currently_annotating):
                continue
                filepath = os.path.join(self.failed_dir, filename)
                metadata_file = filepath.replace('.jpg', '_metadata.json')
                
                # Load metadata if available
                metadata = {}
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                # Load image
                image = cv2.imread(filepath)
                if image is not None:
                    # Extract info from metadata
                    yolo_detections = []
                    if 'yolo_detections' in metadata:
                        yolo_detections = [d['class_name'] for d in metadata['yolo_detections']]
                    
                    task = AnnotationTask(
                        image_path=filepath,
                        image_array=image,
                        metadata=metadata,
                        timestamp=metadata.get('timestamp', 'unknown'),
                        yolo_detections=yolo_detections,
                        knn_prediction=metadata.get('knn_prediction', metadata.get('recognition', {}).get('label')),
                        knn_confidence=metadata.get('knn_confidence', metadata.get('recognition', {}).get('confidence'))
                    )
                    
                    # Check if not already in queue
                    if not self._is_in_queue(filepath):
                        self.queue.put(task)
                        new_tasks += 1
                    
        return new_tasks
    
    def _is_in_queue(self, filepath: str) -> bool:
        """Check if a file is already in the queue."""
        with self.lock:
            # Create temporary list to check
            temp_items = []
            in_queue = False
            
            # Empty and refill queue to check
            while not self.queue.empty():
                try:
                    item = self.queue.get_nowait()
                    temp_items.append(item)
                    if item.image_path == filepath:
                        in_queue = True
                except queue.Empty:
                    break
            
            # Put items back
            for item in temp_items:
                self.queue.put(item)
            
            return in_queue
    
    def get_next_task(self) -> Optional[AnnotationTask]:
        """Get the next annotation task from the queue."""
        try:
            task = self.queue.get_nowait()
            self.currently_annotating = task.image_path
            return task
        except queue.Empty:
            return None
    
    def mark_completed(self, task: AnnotationTask):
        """Mark a task as completed and move files."""
        # Move files to processed directory
        if os.path.exists(task.image_path):
            filename = os.path.basename(task.image_path)
            new_path = os.path.join(self.processed_dir, filename)
            shutil.move(task.image_path, new_path)
            
            # Move metadata too
            meta_file = task.image_path.replace('.jpg', '_metadata.json')
            if os.path.exists(meta_file):
                new_meta = os.path.join(self.processed_dir, filename.replace('.jpg', '_metadata.json'))
                shutil.move(meta_file, new_meta)
        
        self.currently_annotating = None
            
    def task_count(self) -> int:
        """Get number of pending tasks."""
        return self.queue.qsize()


class HumanAnnotationInterface:
    """Gradio interface for human annotation of failed recognitions."""
    
    def __init__(self, 
                 knn_classifier: Optional[AdaptiveKNNClassifier] = None,
                 failed_dir: str = "captures/failed",
                 dataset_dir: str = "captures/dataset"):
        """
        Initialize the annotation interface.
        
        Args:
            knn_classifier: KNN classifier to update with annotations
            failed_dir: Directory containing failed recognitions
            dataset_dir: Directory to save annotated samples
        """
        self.knn = knn_classifier
        self.failed_dir = failed_dir
        self.dataset_dir = dataset_dir
        self.annotation_queue = AnnotationQueue(failed_dir)
        
        # Statistics
        self.stats = {
            'total_annotated': 0,
            'unique_labels': set(),
            'session_start': datetime.now()
        }
        
        # Current task
        self.current_task: Optional[AnnotationTask] = None
        
        # Create interface
        self.interface = self._create_interface()
        
    def _create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        with gr.Blocks(title="EdaxShifu Annotation Interface", theme=gr.themes.Soft()) as interface:
            gr.Markdown(
                """
                # 🎯 EdaxShifu Human Annotation Interface
                
                Help the AI learn by labeling objects it couldn't recognize.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Image display
                    image_display = gr.Image(
                        label="Image to Annotate",
                        type="pil",
                        interactive=False
                    )
                    
                    # YOLO detections
                    yolo_info = gr.Textbox(
                        label="YOLO Detected",
                        interactive=False,
                        lines=1
                    )
                    
                    # KNN prediction
                    knn_info = gr.Textbox(
                        label="KNN Attempted",
                        interactive=False,
                        lines=1
                    )
                    
                with gr.Column(scale=1):
                    # Queue status
                    queue_status = gr.Textbox(
                        label="Queue Status",
                        value="0 images waiting",
                        interactive=False
                    )
                    
                    # Annotation input
                    gr.Markdown("### What is this object?")
                    
                    # Common labels for quick selection
                    common_labels = gr.Radio(
                        label="Quick Select",
                        choices=["person", "phone", "cup", "bottle", "book", "pen", "laptop", "other"],
                        value=None
                    )
                    
                    # Custom label input
                    custom_label = gr.Textbox(
                        label="Or type custom label",
                        placeholder="Enter object name...",
                        lines=1
                    )
                    
                    # Confidence in annotation
                    annotation_confidence = gr.Slider(
                        label="How confident are you?",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.1
                    )
                    
                    # Action buttons
                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary")
                        skip_btn = gr.Button("Skip", variant="secondary")
                        
                    # Statistics
                    gr.Markdown("### Session Statistics")
                    stats_display = gr.Textbox(
                        label="Stats",
                        value=self._format_stats(),
                        interactive=False,
                        lines=3
                    )
            
            # Status message
            status_msg = gr.Textbox(
                label="Status",
                value="Ready to annotate!",
                interactive=False
            )
            
            # Control buttons
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Queue")
                load_next_btn = gr.Button("➡️ Load Next")
                auto_refresh = gr.Checkbox(label="Auto-refresh (5s)", value=True)
                save_model_btn = gr.Button("💾 Save Model")
            
            # Event handlers
            def load_next_image():
                """Load the next image for annotation."""
                # First scan for new tasks
                new_tasks = self.annotation_queue.scan_for_tasks()
                if new_tasks > 0:
                    logger.info(f"Found {new_tasks} new images to annotate")
                
                # Get next task
                self.current_task = self.annotation_queue.get_next_task()
                
                if self.current_task:
                    # Convert image for display
                    image_rgb = cv2.cvtColor(self.current_task.image_array, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                    
                    # Format detections
                    yolo_text = ", ".join(self.current_task.yolo_detections) if self.current_task.yolo_detections else "None"
                    knn_text = f"{self.current_task.knn_prediction} ({self.current_task.knn_confidence:.2f})" if self.current_task.knn_prediction else "Failed"
                    
                    queue_text = f"{self.annotation_queue.task_count()} images waiting"
                    
                    return (
                        pil_image,
                        yolo_text,
                        knn_text,
                        queue_text,
                        "Image loaded. Please annotate.",
                        None,  # Clear radio selection
                        ""     # Clear custom input
                    )
                else:
                    return (
                        None,
                        "",
                        "",
                        "0 images waiting",
                        "No images to annotate. Click 'Refresh Queue' to check for new images.",
                        None,
                        ""
                    )
            
            def submit_annotation(common_label, custom_text, confidence):
                """Submit the annotation for the current image."""
                if not self.current_task:
                    return "No image loaded", self._format_stats()
                
                # Get the label (prefer custom if provided)
                label = custom_text.strip() if custom_text.strip() else common_label
                
                if not label:
                    return "Please provide a label", self._format_stats()
                
                # Save annotated image to dataset
                self._save_to_dataset(self.current_task, label)
                
                # Update KNN if available
                if self.knn:
                    self.knn.add_feedback_sample(
                        self.current_task.image_array,
                        predicted_label=self.current_task.knn_prediction or "unknown",
                        correct_label=label,
                        source="human"
                    )
                    logger.info(f"Updated KNN with label: {label}")
                
                # Update stats
                self.stats['total_annotated'] += 1
                self.stats['unique_labels'].add(label)
                
                # Mark task as completed and move files
                self.annotation_queue.mark_completed(self.current_task)
                self.current_task = None
                
                # Save the model after each annotation for live learning
                if self.knn:
                    self.knn.save_model()
                    logger.info("Model saved after annotation")
                
                return f"✅ Annotated as '{label}'", self._format_stats()
            
            def skip_image():
                """Skip the current image."""
                if self.current_task:
                    # Move to skipped folder
                    skipped_dir = os.path.join(os.path.dirname(self.failed_dir), "skipped")
                    os.makedirs(skipped_dir, exist_ok=True)
                    
                    filename = os.path.basename(self.current_task.image_path)
                    shutil.move(
                        self.current_task.image_path,
                        os.path.join(skipped_dir, filename)
                    )
                    
                    # Also move metadata if exists
                    metadata_file = self.current_task.image_path.replace('.jpg', '_metadata.json')
                    if os.path.exists(metadata_file):
                        shutil.move(
                            metadata_file,
                            os.path.join(skipped_dir, filename.replace('.jpg', '_metadata.json'))
                        )
                    
                    return "Image skipped", self._format_stats()
                return "No image to skip", self._format_stats()
            
            def save_model():
                """Save the KNN model."""
                if self.knn:
                    self.knn.save_model()
                    return "Model saved successfully!"
                return "No model to save"
            
            # Connect events
            load_next_btn.click(
                load_next_image,
                outputs=[image_display, yolo_info, knn_info, queue_status, status_msg, common_labels, custom_label]
            )
            
            def refresh_queue():
                """Manually refresh the queue."""
                new_tasks = self.annotation_queue.scan_for_tasks()
                queue_count = self.annotation_queue.task_count()
                
                # Auto-load if no current task and images waiting
                if not self.current_task and queue_count > 0:
                    return load_next_image()
                else:
                    return (
                        gr.update(),  # image_display
                        gr.update(),  # yolo_info
                        gr.update(),  # knn_info
                        f"{queue_count} images waiting",  # queue_status
                        f"Found {new_tasks} new images. {queue_count} total in queue.",  # status_msg
                        gr.update(),  # common_labels
                        gr.update()   # custom_label
                    )
            
            refresh_btn.click(
                refresh_queue,
                outputs=[image_display, yolo_info, knn_info, queue_status, status_msg, common_labels, custom_label]
            )
            
            submit_btn.click(
                submit_annotation,
                inputs=[common_labels, custom_label, annotation_confidence],
                outputs=[status_msg, stats_display]
            ).then(
                load_next_image,
                outputs=[image_display, yolo_info, knn_info, queue_status, status_msg, common_labels, custom_label]
            )
            
            skip_btn.click(
                skip_image,
                outputs=[status_msg, stats_display]
            ).then(
                load_next_image,
                outputs=[image_display, yolo_info, knn_info, queue_status, status_msg, common_labels, custom_label]
            )
            
            save_model_btn.click(
                save_model,
                outputs=status_msg
            )
            
            # Auto-refresh function
            def check_for_new_images():
                """Check for new images periodically."""
                new_count = self.annotation_queue.scan_for_tasks()
                queue_count = self.annotation_queue.task_count()
                
                # If no current task and images are waiting, load one
                if not self.current_task and queue_count > 0:
                    return load_next_image()
                else:
                    return (
                        gr.update(),  # image_display
                        gr.update(),  # yolo_info
                        gr.update(),  # knn_info
                        f"{queue_count} images waiting",  # queue_status
                        f"Queue checked. {new_count} new images found." if new_count > 0 else gr.update(),  # status_msg
                        gr.update(),  # common_labels
                        gr.update()   # custom_label
                    )
            
            # Load first image on startup and set up periodic refresh
            interface.load(
                load_next_image,
                outputs=[image_display, yolo_info, knn_info, queue_status, status_msg, common_labels, custom_label]
            )
            
            # Add a button click handler for auto-refresh
            def toggle_auto_refresh(checked):
                """Handle auto-refresh toggle."""
                if checked:
                    return "Auto-refresh enabled (checks every 5s)"
                else:
                    return "Auto-refresh disabled"
            
            auto_refresh.change(
                toggle_auto_refresh,
                inputs=[auto_refresh],
                outputs=[status_msg]
            )
            
        return interface
    
    def _save_to_dataset(self, task: AnnotationTask, label: str):
        """Save annotated image to the dataset."""
        # Create label directory
        label_dir = os.path.join(self.dataset_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"human_annotated_{timestamp}.jpg"
        filepath = os.path.join(label_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, task.image_array)
        
        # Save annotation metadata
        metadata = {
            'original_path': task.image_path,
            'label': label,
            'annotated_by': 'human',
            'timestamp': datetime.now().isoformat(),
            'original_metadata': task.metadata
        }
        
        metadata_file = filepath.replace('.jpg', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved annotated image to {filepath}")
        
    def _move_to_processed(self, task: AnnotationTask):
        """Move processed image to processed folder."""
        processed_dir = os.path.join(os.path.dirname(self.failed_dir), "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        filename = os.path.basename(task.image_path)
        shutil.move(
            task.image_path,
            os.path.join(processed_dir, filename)
        )
        
        # Also move metadata if exists
        metadata_file = task.image_path.replace('.jpg', '_metadata.json')
        if os.path.exists(metadata_file):
            shutil.move(
                metadata_file,
                os.path.join(processed_dir, filename.replace('.jpg', '_metadata.json'))
            )
            
    def _format_stats(self) -> str:
        """Format statistics for display."""
        duration = datetime.now() - self.stats['session_start']
        return (
            f"Annotated: {self.stats['total_annotated']}\n"
            f"Unique labels: {len(self.stats['unique_labels'])}\n"
            f"Session time: {str(duration).split('.')[0]}"
        )
    
    def launch(self, share: bool = False, port: int = 7860, prevent_thread_lock: bool = False):
        """Launch the Gradio interface."""
        logger.info(f"Launching annotation interface on port {port}")
        return self.interface.launch(
            share=share, 
            server_port=port, 
            server_name="0.0.0.0",
            prevent_thread_lock=prevent_thread_lock,
            quiet=True
        )


def create_annotation_app(knn_model_path: str = "models/knn_classifier.pkl", 
                         failed_dir: str = "captures/failed",
                         dataset_dir: str = "captures/dataset"):
    """Create and launch the annotation application."""
    # Load KNN classifier if available
    knn = None
    if os.path.exists(knn_model_path):
        knn = AdaptiveKNNClassifier(model_path=knn_model_path)
        knn.load_model()
        logger.info(f"Loaded KNN model with {len(knn.get_known_classes())} classes")
    else:
        logger.warning("No KNN model found, annotations won't update the classifier")
    
    # Create interface
    interface = HumanAnnotationInterface(
        knn_classifier=knn,
        failed_dir=failed_dir,
        dataset_dir=dataset_dir
    )
    
    return interface


if __name__ == "__main__":
    # Run standalone annotation interface
    app = create_annotation_app()
    app.launch(share=True)