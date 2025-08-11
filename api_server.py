#!/usr/bin/env python3
"""
EdaxShifu KNN API Server
Exposes the trained KNN model for inference via REST API endpoints.
"""

import os
import sys
import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import base64
import io

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image

from python.edaxshifu.knn_classifier import AdaptiveKNNClassifier, Recognition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    image_base64: str
    confidence_threshold: Optional[float] = None

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    is_known: bool
    all_scores: Dict[str, float]
    timestamp: str

class ModelStatsResponse(BaseModel):
    known_classes: List[str]
    total_samples: int
    sample_counts: Dict[str, int]
    confidence_threshold: float
    model_trained: bool

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    known_classes_count: int
    total_samples: int

class KNNAPIServer:
    """FastAPI server for KNN model inference."""
    
    def __init__(self, model_path: str = "models/knn_classifier.npz"):
        self.model_path = model_path
        self.knn_classifier = None
        self.app = FastAPI(
            title="EdaxShifu KNN API",
            description="REST API for KNN object recognition inference",
            version="1.0.0"
        )
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        self._load_model()
    
    def _load_model(self):
        """Load the trained KNN model."""
        try:
            self.knn_classifier = AdaptiveKNNClassifier(
                model_path=self.model_path,
                confidence_threshold=0.6
            )
            
            # Try to load existing model
            if os.path.exists(self.model_path):
                success = self.knn_classifier.load_model()
                if success:
                    logger.info(f"Loaded KNN model from {self.model_path}")
                    logger.info(f"Known classes: {self.knn_classifier.get_known_classes()}")
                else:
                    logger.warning("Failed to load existing model, starting fresh")
            else:
                logger.info("No existing model found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error initializing KNN classifier: {e}")
            raise
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            if not self.knn_classifier:
                return HealthResponse(
                    status="error",
                    model_loaded=False,
                    known_classes_count=0,
                    total_samples=0
                )
            
            return HealthResponse(
                status="healthy",
                model_loaded=self.knn_classifier.trained,
                known_classes_count=len(self.knn_classifier.get_known_classes()),
                total_samples=len(self.knn_classifier.X_train) if self.knn_classifier.X_train is not None else 0
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict_image(request: PredictionRequest):
            """Predict object class from base64 encoded image."""
            if not self.knn_classifier or not self.knn_classifier.trained:
                raise HTTPException(
                    status_code=503,
                    detail="KNN model not loaded or not trained"
                )
            
            try:
                image_data = base64.b64decode(request.image_base64)
                image = Image.open(io.BytesIO(image_data))
                
                img_array = np.array(image)
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                else:  # RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                if request.confidence_threshold is not None:
                    original_threshold = self.knn_classifier.confidence_threshold
                    self.knn_classifier.confidence_threshold = request.confidence_threshold
                
                recognition = self.knn_classifier.predict(img_array)
                
                if request.confidence_threshold is not None:
                    self.knn_classifier.confidence_threshold = original_threshold
                
                return PredictionResponse(
                    label=recognition.label,
                    confidence=recognition.confidence,
                    is_known=recognition.is_known,
                    all_scores=recognition.all_scores,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing image: {str(e)}"
                )
        
        @self.app.post("/predict/upload", response_model=PredictionResponse)
        async def predict_upload(
            file: UploadFile = File(...),
            confidence_threshold: Optional[float] = Form(None)
        ):
            """Predict object class from uploaded image file."""
            if not self.knn_classifier or not self.knn_classifier.trained:
                raise HTTPException(
                    status_code=503,
                    detail="KNN model not loaded or not trained"
                )
            
            try:
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                
                img_array = np.array(image)
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                else:  # RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                if confidence_threshold is not None:
                    original_threshold = self.knn_classifier.confidence_threshold
                    self.knn_classifier.confidence_threshold = confidence_threshold
                
                recognition = self.knn_classifier.predict(img_array)
                
                if confidence_threshold is not None:
                    self.knn_classifier.confidence_threshold = original_threshold
                
                return PredictionResponse(
                    label=recognition.label,
                    confidence=recognition.confidence,
                    is_known=recognition.is_known,
                    all_scores=recognition.all_scores,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Upload prediction error: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing uploaded file: {str(e)}"
                )
        
        @self.app.get("/model/stats", response_model=ModelStatsResponse)
        async def get_model_stats():
            """Get model statistics and information."""
            if not self.knn_classifier:
                raise HTTPException(
                    status_code=503,
                    detail="KNN model not loaded"
                )
            
            return ModelStatsResponse(
                known_classes=self.knn_classifier.get_known_classes(),
                total_samples=len(self.knn_classifier.X_train) if self.knn_classifier.X_train is not None else 0,
                sample_counts=self.knn_classifier.get_sample_count(),
                confidence_threshold=self.knn_classifier.confidence_threshold,
                model_trained=self.knn_classifier.trained
            )
        
        @self.app.post("/model/reload")
        async def reload_model():
            """Reload the model from disk (useful after new training)."""
            try:
                if self.knn_classifier:
                    success = self.knn_classifier.load_model()
                    if success:
                        return {
                            "status": "success",
                            "message": "Model reloaded successfully",
                            "known_classes": self.knn_classifier.get_known_classes(),
                            "total_samples": len(self.knn_classifier.X_train) if self.knn_classifier.X_train is not None else 0
                        }
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail="Failed to reload model"
                        )
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="KNN classifier not initialized"
                    )
            except Exception as e:
                logger.error(f"Model reload error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error reloading model: {str(e)}"
                )
        
        @self.app.put("/model/confidence")
        async def update_confidence_threshold(threshold: float):
            """Update the confidence threshold for predictions."""
            if not self.knn_classifier:
                raise HTTPException(
                    status_code=503,
                    detail="KNN model not loaded"
                )
            
            if not 0.0 <= threshold <= 1.0:
                raise HTTPException(
                    status_code=400,
                    detail="Confidence threshold must be between 0.0 and 1.0"
                )
            
            self.knn_classifier.update_confidence_threshold(threshold)
            return {
                "status": "success",
                "message": f"Confidence threshold updated to {threshold}",
                "new_threshold": threshold
            }

def create_app(model_path: str = "models/knn_classifier.npz") -> FastAPI:
    """Create and configure the FastAPI app."""
    server = KNNAPIServer(model_path)
    return server.app

def main():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="EdaxShifu KNN API Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 for network access)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/knn_classifier.npz",
        help="Path to KNN model file"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    print("🎯 EdaxShifu KNN API Server")
    print("=" * 50)
    print(f"📡 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🧠 Model: {args.model_path}")
    print(f"🌐 API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 50)
    
    app = create_app(args.model_path)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
