"""
Gradio-based annotation interface with multi-annotator support.
Enhanced with AI+Human annotation capabilities.
"""

import gradio as gr
import cv2
import os
import json
import shutil
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import queue
import logging
from PIL import Image
import numpy as np

from .knn_classifier import AdaptiveKNNClassifier

# Import new annotator system
try:
    from .annotators import (
        AnnotatorFactory, BaseAnnotator, AnnotationRequest, AnnotationResult,
        create_dual_annotator
    )
    ANNOTATORS_AVAILABLE = True
except ImportError:
    ANNOTATORS_AVAILABLE = False
    logger.warning("New annotator system not available, using legacy human-only mode")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnnotationTask:
    """Task for multi-annotator annotation."""
    image_path: str
    image_array: np.ndarray
    metadata: Dict
    timestamp: str
    yolo_detections: List[str]
    knn_prediction: Optional[str] = None
    knn_confidence: Optional[float] = None
    ai_annotation: Optional[AnnotationResult] = None
    human_annotation: Optional[str] = None
    final_label: Optional[str] = None
    annotation_source: Optional[str] = None


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
    """Enhanced Gradio interface supporting multiple annotation methods."""
    
    def __init__(self, 
                 knn_classifier: Optional[AdaptiveKNNClassifier] = None,
                 failed_dir: str = "captures/failed",
                 dataset_dir: str = "captures/dataset",
                 use_ai_annotator: bool = True,
                 annotator_preset: str = "ai_first"):
        """
        Initialize the annotation interface.
        
        Args:
            knn_classifier: KNN classifier to update with annotations
            failed_dir: Directory containing failed recognitions
            dataset_dir: Directory to save annotated samples
            use_ai_annotator: Whether to enable AI annotation features
            annotator_preset: Preset for annotator configuration
        """
        self.knn = knn_classifier
        self.failed_dir = failed_dir
        self.dataset_dir = dataset_dir
        self.annotation_queue = AnnotationQueue(failed_dir)
        self.use_ai_annotator = use_ai_annotator and ANNOTATORS_AVAILABLE
        
        # Initialize AI annotators if available
        self.ai_annotator = None
        self.gemini_annotator = None
        
        if self.use_ai_annotator:
            try:
                # Create Gemini annotator for AI suggestions
                self.gemini_annotator = AnnotatorFactory.create_gemini_annotator()
                logger.info(f"Initialized AI annotator: {self.gemini_annotator.is_available()}")
            except Exception as e:
                logger.warning(f"Failed to initialize AI annotator: {e}")
                self.use_ai_annotator = False
        
        # Statistics
        self.stats = {
            'total_annotated': 0,
            'ai_annotations': 0,
            'human_annotations': 0,
            'consensus_annotations': 0,
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
            ai_status = "🤖 AI-Enhanced" if self.use_ai_annotator else "👤 Human-Only"
            gr.Markdown(
                f"""
                # 🎯 EdaxShifu Annotation Interface {ai_status}
                
                Help the AI learn by labeling objects it couldn't recognize.
                {f"✨ AI suggestions available via Gemini Vision" if self.use_ai_annotator else ""}
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
                    
                    # AI annotation display (only if AI is enabled)
                    if self.use_ai_annotator:
                        ai_annotation_display = gr.Textbox(
                            label="AI Suggestion (Gemini)",
                            interactive=False,
                            lines=2,
                            visible=True
                        )
                    else:
                        ai_annotation_display = gr.Textbox(visible=False)
                    
                with gr.Column(scale=1):
                    # Queue status
                    queue_status = gr.Textbox(
                        label="Queue Status",
                        value="0 images waiting",
                        interactive=False
                    )
                    
                    # AI suggestion button (only if AI is enabled)
                    if self.use_ai_annotator:
                        auto_ai_btn = gr.Button("🤖 Get AI Suggestion", variant="secondary")
                    else:
                        auto_ai_btn = gr.Button(visible=False)
                    
                    # Annotation input
                    gr.Markdown("### What is this object?")
                    
                    # Common labels for quick selection
                    common_labels = gr.Radio(
                        label="Quick Select",
                        choices=["person", "phone", "cup", "bottle", "book", "pen", "laptop", "cat", "dog", "other"],
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
                    
                    if self.use_ai_annotator:
                        return (
                            pil_image,
                            yolo_text,
                            knn_text,
                            "",  # ai_annotation_display (cleared)
                            queue_text,
                            "Image loaded. Use AI suggestion or annotate manually.",
                            None,  # Clear radio selection
                            ""     # Clear custom input
                        )
                    else:
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
                    if self.use_ai_annotator:
                        return (
                            None,
                            "",
                            "",
                            "",  # ai_annotation_display
                            "0 images waiting",
                            "No images to annotate. Click 'Refresh Queue' to check for new images.",
                            None,
                            ""
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
                self.stats['human_annotations'] += 1
                self.stats['unique_labels'].add(label)
                
                # Store annotation info
                if self.current_task:
                    self.current_task.human_annotation = label
                    self.current_task.final_label = label
                    self.current_task.annotation_source = "human"
                
                # Mark task as completed and move files
                self.annotation_queue.mark_completed(self.current_task)
                self.current_task = None
                
                # Save the model after each annotation for live learning
                if self.knn:
                    self.knn.save_model()
                    logger.info("Model saved after annotation")
                
                return f"✅ Annotated as '{label}'", self._format_stats()
            
            def get_ai_annotation():
                """Get AI annotation for current image."""
                if not self.current_task:
                    return "", "No image loaded"
                
                if not self.use_ai_annotator or not self.gemini_annotator:
                    return "AI annotator not available", "AI annotation unavailable"
                
                if not self.gemini_annotator.is_available():
                    return "AI annotator not available (check GEMINI_API_KEY)", "AI annotation failed"
                
                try:
                    # Create annotation request
                    if ANNOTATORS_AVAILABLE:
                        request = AnnotationRequest(
                            image=self.current_task.image_array,
                            image_path=self.current_task.image_path,
                            metadata=self.current_task.metadata,
                            yolo_detections=self.current_task.yolo_detections,
                            knn_prediction=self.current_task.knn_prediction,
                            knn_confidence=self.current_task.knn_confidence,
                            timestamp=self.current_task.timestamp
                        )
                        
                        # Get AI annotation
                        result = self.gemini_annotator.annotate(request)
                        self.current_task.ai_annotation = result
                        
                        if result.success:
                            ai_text = f"🤖 AI suggests: '{result.label}' (confidence: {result.confidence:.2f})"
                            if result.processing_time:
                                ai_text += f"\nProcessing time: {result.processing_time:.1f}s"
                            
                            status_text = f"AI annotation received: {result.label}"
                            return ai_text, status_text
                        else:
                            error_msg = result.error_message or "Unknown error"
                            return f"❌ AI annotation failed: {error_msg}", "AI annotation failed"
                    else:
                        return "Annotator system not available", "Import error"
                        
                except Exception as e:
                    logger.error(f"AI annotation error: {e}")
                    return f"❌ Error: {str(e)}", "AI annotation error"
            
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
            if self.use_ai_annotator:
                load_next_btn.click(
                    load_next_image,
                    outputs=[image_display, yolo_info, knn_info, ai_annotation_display, queue_status, status_msg, common_labels, custom_label]
                )
                
                # Connect AI annotation button
                auto_ai_btn.click(
                    get_ai_annotation,
                    outputs=[ai_annotation_display, status_msg]
                )
            else:
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
                    if self.use_ai_annotator:
                        return (
                            gr.update(),  # image_display
                            gr.update(),  # yolo_info
                            gr.update(),  # knn_info
                            gr.update(),  # ai_annotation_display
                            f"{queue_count} images waiting",  # queue_status
                            f"Found {new_tasks} new images. {queue_count} total in queue.",  # status_msg
                            gr.update(),  # common_labels
                            gr.update()   # custom_label
                        )
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
            
            if self.use_ai_annotator:
                refresh_btn.click(
                    refresh_queue,
                    outputs=[image_display, yolo_info, knn_info, ai_annotation_display, queue_status, status_msg, common_labels, custom_label]
                )
                
                submit_btn.click(
                    submit_annotation,
                    inputs=[common_labels, custom_label, annotation_confidence],
                    outputs=[status_msg, stats_display]
                ).then(
                    load_next_image,
                    outputs=[image_display, yolo_info, knn_info, ai_annotation_display, queue_status, status_msg, common_labels, custom_label]
                )
                
                skip_btn.click(
                    skip_image,
                    outputs=[status_msg, stats_display]
                ).then(
                    load_next_image,
                    outputs=[image_display, yolo_info, knn_info, ai_annotation_display, queue_status, status_msg, common_labels, custom_label]
                )
            else:
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
            if self.use_ai_annotator:
                interface.load(
                    load_next_image,
                    outputs=[image_display, yolo_info, knn_info, ai_annotation_display, queue_status, status_msg, common_labels, custom_label]
                )
            else:
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
        stats_text = f"Total: {self.stats['total_annotated']}\n"
        
        if self.use_ai_annotator:
            stats_text += (
                f"AI: {self.stats['ai_annotations']} | "
                f"Human: {self.stats['human_annotations']} | "
                f"Consensus: {self.stats['consensus_annotations']}\n"
            )
        
        stats_text += (
            f"Unique labels: {len(self.stats['unique_labels'])}\n"
            f"Session: {str(duration).split('.')[0]}"
        )
        
        return stats_text
    
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
                         dataset_dir: str = "captures/dataset",
                         use_ai_annotator: bool = True,
                         annotator_preset: str = "ai_first"):
    """Create and launch the enhanced annotation application."""
    # Load KNN classifier if available
    knn = None
    if os.path.exists(knn_model_path):
        knn = AdaptiveKNNClassifier(model_path=knn_model_path)
        knn.load_model()
        logger.info(f"Loaded KNN model with {len(knn.get_known_classes())} classes")
    else:
        logger.warning("No KNN model found, annotations won't update the classifier")
    
    # Create enhanced interface
    interface = HumanAnnotationInterface(
        knn_classifier=knn,
        failed_dir=failed_dir,
        dataset_dir=dataset_dir,
        use_ai_annotator=use_ai_annotator,
        annotator_preset=annotator_preset
    )
    
    return interface


if __name__ == "__main__":
    # Run standalone annotation interface
    app = create_annotation_app()
    app.launch(share=True)