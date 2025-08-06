import cv2
import numpy as np
import time
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RTSPStream:
    def __init__(self, rtsp_url: str):
        # Handle webcam index (e.g., "0" for webcam)
        try:
            self.rtsp_url = int(rtsp_url)
        except (ValueError, TypeError):
            self.rtsp_url = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps_reported = 0.0
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self.start_time = None
        
    def connect(self) -> bool:
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            logger.error(f"Unable to open RTSP stream: {self.rtsp_url}")
            return False
            
        self.fps_reported = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Stream opened: {self.width}×{self.height} @ {self.fps_reported:.2f} FPS")
        return True
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap:
            return False, None
        return self.cap.read()
        
    def reconnect(self) -> bool:
        if self.cap:
            self.cap.release()
        time.sleep(1)
        return self.connect()
        
    def get_stats(self) -> dict:
        if self.start_time:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        else:
            avg_fps = 0
            
        return {
            'frame_count': self.frame_count,
            'avg_fps': avg_fps,
            'reported_fps': self.fps_reported,
            'resolution': f"{self.width}×{self.height}"
        }
        
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            

class RTSPViewer:
    def __init__(self, stream: RTSPStream, window_name: str = "RTSP Stream"):
        self.stream = stream
        self.window_name = window_name
        
    def run(self):
        if not self.stream.connect():
            return
            
        self.stream.start_time = time.time()
        
        try:
            while True:
                ret, frame = self.stream.read_frame()
                
                if not ret:
                    logger.warning("Frame grab failed — attempting reconnection")
                    if not self.stream.reconnect():
                        break
                    continue
                    
                self.stream.frame_count += 1
                cv2.imshow(self.window_name, frame)
                
                if cv2.waitKey(1) == 27:  # ESC key
                    break
                    
        finally:
            stats = self.stream.get_stats()
            logger.info(f"Session stats: {stats['frame_count']} frames, {stats['avg_fps']:.2f} avg FPS")
            self.stream.release()
            cv2.destroyAllWindows()