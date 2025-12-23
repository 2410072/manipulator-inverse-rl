import cv2
import numpy as np
from pathlib import Path
import threading

class GlobalVideoRecorder:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GlobalVideoRecorder, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.is_recording = False
        self.video_path = None
        self.writer = None
        self.frame_size = None
        self.fps = 30
        self._initialized = True
        self.frame_counter = 0

    def start(self, path, fps=30):
        self.stop() # Ensure previous session is closed
        self.video_path = Path(path)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.is_recording = True
        self.frame_counter = 0
        print(f"GlobalVideoRecorder: Started recording to {self.video_path}")

    def capture_frame(self, frame_rgb):
        if not self.is_recording:
            return

        if frame_rgb is None:
            return

        # Convert RGB (from gym) to BGR (for cv2)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Initialize writer on first frame
        if self.writer is None:
            h, w = frame_bgr.shape[:2]
            
            # Select codec based on extension
            ext = self.video_path.suffix.lower()
            if ext == '.webm':
                # VP8 is widely supported in Electron/VSCode
                fourcc = cv2.VideoWriter_fourcc(*'vp80')
            elif ext == '.mp4':
                # mp4v is supported by OpenCV but NOT by all Electron apps (VSCode might fail)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                 fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Default fallback

            self.writer = cv2.VideoWriter(str(self.video_path), fourcc, float(self.fps), (w, h))
            if not self.writer.isOpened():
                print(f"GlobalVideoRecorder: Failed to open writer for {self.video_path}")
                self.stop()
                return
            else:
                print(f"GlobalVideoRecorder: Writer initialized successfully.")

        self.writer.write(frame_bgr)
        self.frame_counter += 1

    def stop(self):
        if self.writer:
            self.writer.release()
            print(f"GlobalVideoRecorder: Stopped. Saved {self.frame_counter} frames to {self.video_path}")
            self.writer = None
        self.is_recording = False
        self.video_path = None

global_recorder = GlobalVideoRecorder()
