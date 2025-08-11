"""
Enhanced Gradio-based annotation interface with multi-annotator support.
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
from .annotators import (
    AnnotatorFactory, BaseAnnotator, AnnotationRequest, AnnotationResult,
    create_dual_annotator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedAnnotationTask:
    """Enhanced task for multi-annotator annotation."""
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


class EnhancedAnnotationInterface:
    """Enhanced Gradio interface supporting multiple annotation methods."""
    
    def __init__(self, 
                 knn_classifier: Optional[AdaptiveKNNClassifier] = None,
                 failed_dir: str = "captures/failed",
                 dataset_dir: str = "captures/dataset",
                 annotator_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced annotation interface.
        
        Args:
            knn_classifier: KNN classifier to update with annotations
            failed_dir: Directory containing failed recognitions
            dataset_dir: Directory to save annotated samples
            annotator_config: Configuration for annotators
        """
        self.knn = knn_classifier
        self.failed_dir = failed_dir
        self.dataset_dir = dataset_dir
        
        # Initialize annotators
        self._setup_annotators(annotator_config)
        
        # Statistics
        self.stats = {
            'total_annotated': 0,
            'ai_annotations': 0,
            'human_annotations': 0,
            'consensus_annotations': 0,
            'unique_labels': set(),
            'session_start': datetime.now()
        }
        
        # Current task and queue
        self.current_task: Optional[EnhancedAnnotationTask] = None
        self.annotation_queue = queue.Queue()
        
        # Create interface
        self.interface = self._create_interface()
    
    def _setup_annotators(self, config: Optional[Dict[str, Any]]):
        """Setup the annotator system."""
        if not config:
            # Default configuration: AI-first with human fallback
            config = {
                'type': 'ai_first',
                'ai_config': {'confidence_threshold': 0.7},
                'human_config': {'interactive_mode': False}
            }
        
        annotator_type = config.get('type', 'ai_first')
        
        if annotator_type == 'ai_first':
            self.annotator = create_dual_annotator(
                ai_config=config.get('ai_config', {}),
                human_config=config.get('human_config', {}),
                combination_strategy='fallback'
            )
        elif annotator_type == 'consensus':
            self.annotator = create_dual_annotator(
                ai_config=config.get('ai_config', {}),
                human_config=config.get('human_config', {}),
                combination_strategy='consensus'
            )
        else:
            # Use factory to create annotator
            self.annotator = AnnotatorFactory.create_from_config(config)
        
        # Also create individual annotators for comparison
        self.gemini_annotator = AnnotatorFactory.create_gemini_annotator()
        self.human_annotator = AnnotatorFactory.create_human_annotator()
        
        logger.info(f"Initialized annotator: {self.annotator}")
    
    def _create_interface(self) -> gr.Blocks:
        """Create the enhanced Gradio interface."""
        
        with gr.Blocks(title="EdaxShifu Enhanced Annotation Interface", theme=gr.themes.Soft()) as interface:
            gr.Markdown(
                """
                # 🤖 EdaxShifu Enhanced Annotation Interface
                
                Multi-annotator system combining AI and human intelligence.
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
                    
                    # Detection info
                    with gr.Row():
                        yolo_info = gr.Textbox(
                            label="YOLO Detected",
                            interactive=False,
                            lines=1
                        )
                        knn_info = gr.Textbox(
                            label="KNN Prediction",
                            interactive=False,
                            lines=1
                        )
                    
                    # AI Annotation display
                    ai_annotation_display = gr.Textbox(
                        label="AI Annotation (Gemini)",
                        interactive=False,
                        lines=2
                    )
                    
                with gr.Column(scale=1):
                    # System status
                    annotator_status = gr.Textbox(
                        label="Annotator Status",
                        value=f"Using: {self.annotator.name}",
                        interactive=False
                    )
                    
                    queue_status = gr.Textbox(
                        label="Queue Status",
                        value="0 images waiting",
                        interactive=False
                    )
                    
                    # Annotation controls
                    gr.Markdown("### Annotation Options")
                    
                    # Auto-annotate with AI
                    auto_ai_btn = gr.Button("🤖 Get AI Suggestion", variant="secondary")
                    
                    # Manual annotation
                    gr.Markdown("#### Manual Annotation")
                    common_labels = gr.Radio(
                        label="Quick Select",
                        choices=["person", "phone", "cup", "bottle", "book", "pen", "laptop", "cat", "dog", "other"],
                        value=None
                    )
                    
                    custom_label = gr.Textbox(
                        label="Custom Label",
                        placeholder="Enter object name...",
                        lines=1
                    )
                    
                    annotation_confidence = gr.Slider(
                        label="Confidence",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.1
                    )
                    
                    # Action buttons
                    with gr.Row():
                        submit_human_btn = gr.Button("👤 Submit Human", variant="primary")
                        skip_btn = gr.Button("⏭️ Skip", variant="secondary")
                    
                    # Advanced options
                    with gr.Accordion("Advanced Options", open=False):
                        use_consensus = gr.Checkbox(label="Use AI+Human Consensus", value=False)
                        force_human = gr.Checkbox(label="Always Require Human Input", value=False)
                        ai_threshold = gr.Slider(
                            label="AI Confidence Threshold", 
                            minimum=0.0, 
                            maximum=1.0, 
                            value=0.7,
                            step=0.1
                        )
                    
                    # Statistics
                    gr.Markdown("### Session Statistics")
                    stats_display = gr.Textbox(
                        label="Stats",
                        value=self._format_stats(),
                        interactive=False,
                        lines=4
                    )
            
            # Status and controls
            status_msg = gr.Textbox(
                label="Status",
                value="Ready to annotate!",
                interactive=False
            )
            
            with gr.Row():
                load_next_btn = gr.Button("➡️ Load Next Image")
                refresh_btn = gr.Button("🔄 Refresh Queue")
                save_model_btn = gr.Button("💾 Save Model")
                auto_refresh = gr.Checkbox(label="Auto-refresh", value=True)
            
            # Event handlers
            def load_next_image():
                """Load the next image for annotation."""
                self.current_task = self._get_next_task()
                
                if self.current_task:
                    # Convert image for display
                    image_rgb = cv2.cvtColor(self.current_task.image_array, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                    
                    # Format info
                    yolo_text = ", ".join(self.current_task.yolo_detections) if self.current_task.yolo_detections else "None"
                    knn_text = f"{self.current_task.knn_prediction} ({self.current_task.knn_confidence:.2f})" if self.current_task.knn_prediction else "No prediction"
                    
                    return (
                        pil_image,  # image_display
                        yolo_text,  # yolo_info
                        knn_text,   # knn_info
                        "",         # ai_annotation_display (cleared)
                        "Image loaded. Use AI suggestion or annotate manually.",  # status_msg
                        None,       # common_labels (cleared)
                        ""          # custom_label (cleared)
                    )
                else:
                    return (
                        None, "", "", "", 
                        "No images to annotate. Click 'Refresh Queue'.",
                        None, ""
                    )
            
            def get_ai_annotation():
                """Get AI annotation for current image."""
                if not self.current_task:
                    return "", "No image loaded"
                
                if not self.gemini_annotator.is_available():
                    return "AI annotator not available (check GEMINI_API_KEY)", "AI annotation failed"
                
                try:
                    # Create annotation request
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
                        
                        # Auto-accept if confidence is high enough
                        return ai_text, status_text
                    else:
                        error_msg = result.error_message or "Unknown error"
                        return f"❌ AI annotation failed: {error_msg}", "AI annotation failed"
                        
                except Exception as e:
                    logger.error(f"AI annotation error: {e}")
                    return f"❌ Error: {str(e)}", "AI annotation error"
            
            def submit_human_annotation(common_label, custom_text, confidence, use_consensus_flag):
                """Submit human annotation."""
                if not self.current_task:
                    return "No image loaded", self._format_stats()
                
                # Get the label (prefer custom if provided)
                human_label = custom_text.strip() if custom_text.strip() else common_label
                
                if not human_label:
                    return "Please provide a label", self._format_stats()
                
                # Store human annotation
                self.current_task.human_annotation = human_label
                
                # Determine final label based on strategy
                if use_consensus_flag and self.current_task.ai_annotation and self.current_task.ai_annotation.success:
                    # Consensus mode
                    ai_label = self.current_task.ai_annotation.label
                    if ai_label.lower() == human_label.lower():
                        final_label = human_label
                        source = "consensus"
                        self.stats['consensus_annotations'] += 1
                    else:
                        # Disagreement - prefer human
                        final_label = human_label
                        source = "human_override"
                        self.stats['human_annotations'] += 1
                else:
                    # Human annotation
                    final_label = human_label
                    source = "human"
                    self.stats['human_annotations'] += 1
                
                self.current_task.final_label = final_label
                self.current_task.annotation_source = source
                
                # Save and update
                self._save_annotation(self.current_task, final_label, source)
                self._update_knn(self.current_task, final_label)
                
                # Update stats
                self.stats['total_annotated'] += 1
                self.stats['unique_labels'].add(final_label)
                
                return f"✅ Annotated as '{final_label}' (source: {source})", self._format_stats()
            
            def skip_image():
                """Skip current image."""
                if self.current_task:
                    self._move_to_skipped(self.current_task)
                    return "Image skipped", self._format_stats()
                return "No image to skip", self._format_stats()
            
            def save_model():
                """Save the model."""
                if self.knn:
                    self.knn.save_model()
                    return "Model saved!"
                return "No model to save"
            
            # Connect events
            load_next_btn.click(
                load_next_image,
                outputs=[image_display, yolo_info, knn_info, ai_annotation_display, status_msg, common_labels, custom_label]
            )
            
            auto_ai_btn.click(
                get_ai_annotation,
                outputs=[ai_annotation_display, status_msg]
            )
            
            submit_human_btn.click(
                submit_human_annotation,
                inputs=[common_labels, custom_label, annotation_confidence, use_consensus],
                outputs=[status_msg, stats_display]
            ).then(
                load_next_image,
                outputs=[image_display, yolo_info, knn_info, ai_annotation_display, status_msg, common_labels, custom_label]
            )
            
            skip_btn.click(
                skip_image,
                outputs=[status_msg, stats_display]
            ).then(
                load_next_image,
                outputs=[image_display, yolo_info, knn_info, ai_annotation_display, status_msg, common_labels, custom_label]
            )
            
            save_model_btn.click(
                save_model,
                outputs=status_msg
            )
            
            refresh_btn.click(
                lambda: ("Queue refreshed", self._format_stats()),
                outputs=[status_msg, stats_display]
            )
            
            # Load first image on startup
            interface.load(
                load_next_image,
                outputs=[image_display, yolo_info, knn_info, ai_annotation_display, status_msg, common_labels, custom_label]
            )
            
        return interface
    
    def _get_next_task(self) -> Optional[EnhancedAnnotationTask]:
        """Get next annotation task from failed directory."""
        if not os.path.exists(self.failed_dir):
            return None
        
        files = [f for f in os.listdir(self.failed_dir) if f.endswith('.jpg')]
        if not files:
            return None
        
        # Get newest file
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.failed_dir, x)), reverse=True)
        filepath = os.path.join(self.failed_dir, files[0])
        
        # Load metadata
        metadata_file = filepath.replace('.jpg', '_metadata.json')
        metadata = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Load image
        image = cv2.imread(filepath)
        if image is None:
            return None
        
        # Extract info
        yolo_detections = []
        if 'yolo_detections' in metadata:
            yolo_detections = [d['class_name'] for d in metadata['yolo_detections']]
        
        return EnhancedAnnotationTask(
            image_path=filepath,
            image_array=image,
            metadata=metadata,
            timestamp=metadata.get('timestamp', str(time.time())),
            yolo_detections=yolo_detections,
            knn_prediction=metadata.get('knn_prediction'),
            knn_confidence=metadata.get('knn_confidence')
        )
    
    def _save_annotation(self, task: EnhancedAnnotationTask, label: str, source: str):
        """Save annotated image and metadata."""
        # Create label directory
        label_dir = os.path.join(self.dataset_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source}_annotated_{timestamp}.jpg"
        filepath = os.path.join(label_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, task.image_array)
        
        # Save metadata
        annotation_metadata = {
            'original_path': task.image_path,
            'final_label': label,
            'annotation_source': source,
            'timestamp': datetime.now().isoformat(),
            'yolo_detections': task.yolo_detections,
            'knn_prediction': task.knn_prediction,
            'knn_confidence': task.knn_confidence,
            'original_metadata': task.metadata
        }
        
        # Add AI annotation info if available
        if task.ai_annotation:
            annotation_metadata['ai_annotation'] = task.ai_annotation.to_dict()
        
        if task.human_annotation:
            annotation_metadata['human_annotation'] = task.human_annotation
        
        metadata_file = filepath.replace('.jpg', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(annotation_metadata, f, indent=2)
        
        # Move original files to processed
        processed_dir = os.path.join(os.path.dirname(self.failed_dir), "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        original_filename = os.path.basename(task.image_path)
        shutil.move(task.image_path, os.path.join(processed_dir, original_filename))
        
        # Move original metadata
        original_metadata = task.image_path.replace('.jpg', '_metadata.json')
        if os.path.exists(original_metadata):
            shutil.move(original_metadata, os.path.join(processed_dir, original_filename.replace('.jpg', '_metadata.json')))
        
        logger.info(f"Saved annotation: {filepath}")
    
    def _update_knn(self, task: EnhancedAnnotationTask, label: str):
        """Update KNN classifier with annotation."""
        if self.knn:
            self.knn.add_feedback_sample(
                task.image_array,
                predicted_label=task.knn_prediction or "unknown",
                correct_label=label,
                source=task.annotation_source or "human"
            )
            self.knn.save_model()
    
    def _move_to_skipped(self, task: EnhancedAnnotationTask):
        """Move task to skipped directory."""
        skipped_dir = os.path.join(os.path.dirname(self.failed_dir), "skipped")
        os.makedirs(skipped_dir, exist_ok=True)
        
        filename = os.path.basename(task.image_path)
        shutil.move(task.image_path, os.path.join(skipped_dir, filename))
        
        # Move metadata too
        metadata_file = task.image_path.replace('.jpg', '_metadata.json')
        if os.path.exists(metadata_file):
            shutil.move(metadata_file, os.path.join(skipped_dir, filename.replace('.jpg', '_metadata.json')))
    
    def _format_stats(self) -> str:
        """Format statistics for display."""
        duration = datetime.now() - self.stats['session_start']
        return (
            f"Total annotated: {self.stats['total_annotated']}\n"
            f"AI annotations: {self.stats['ai_annotations']}\n"
            f"Human annotations: {self.stats['human_annotations']}\n"
            f"Consensus: {self.stats['consensus_annotations']}\n"
            f"Unique labels: {len(self.stats['unique_labels'])}\n"
            f"Session time: {str(duration).split('.')[0]}"
        )
    
    def launch(self, share: bool = False, port: int = 7860):
        """Launch the interface."""
        logger.info(f"Launching enhanced annotation interface on port {port}")
        return self.interface.launch(
            share=share,
            server_port=port,
            server_name="0.0.0.0",
            prevent_thread_lock=False,
            quiet=True
        )


def create_enhanced_annotation_app(knn_model_path: str = "models/knn_classifier.pkl",
                                 failed_dir: str = "captures/failed", 
                                 dataset_dir: str = "captures/dataset",
                                 annotator_preset: str = "ai_first"):
    """Create enhanced annotation application."""
    # Load KNN classifier if available
    knn = None
    if os.path.exists(knn_model_path):
        knn = AdaptiveKNNClassifier(model_path=knn_model_path)
        knn.load_model()
        logger.info(f"Loaded KNN model with {len(knn.get_known_classes())} classes")
    
    # Create annotator configuration
    annotator_config = AnnotatorFactory.create_preset(annotator_preset)
    
    # Create interface
    interface = EnhancedAnnotationInterface(
        knn_classifier=knn,
        failed_dir=failed_dir,
        dataset_dir=dataset_dir,
        annotator_config={'type': annotator_preset}
    )
    
    return interface