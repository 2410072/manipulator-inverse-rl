from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
import mss
from mss.exception import ScreenShotError

Region = Optional[Tuple[int, int, int, int]]  # (left, top, width, height) or None


@dataclass
class RecordConfig:
    video_path: str | Path
    screenshot_path: str | Path   # 1録画につき1枚だけ保存
    fps: int = 30
    region: Region = None         # None: primary monitor full
    monitor_index: int = 1        # mss.monitors[1] is usually primary
    codec: str = "mp4v"           # Default: mp4v for better Linux compatibility (avc1 often fails to write frames)
    screenshot_ext: str = ".png"  # ".png" or ".jpg"
    screenshot_quality: int = 95  # jpeg quality


class ScreenRecordOneShot:
    """
    start〜stop の間だけ録画し、stop直前にスクリーンショットを1枚だけ保存する。
    """

    def __init__(self, cfg: RecordConfig):
        self.cfg = cfg
        self.video_path = Path(cfg.video_path)
        self.screenshot_path = Path(cfg.screenshot_path)

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._sct: Optional[Any] = None
        self._mon: Optional[Dict[str, int]] = None
        self._started = False
        self._error_disabled = False  # New flag to track if recording is disabled due to error

    def __enter__(self) -> "ScreenRecordOneShot":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _resolve_monitor(self) -> Dict[str, int]:
        assert self._sct is not None
        monitors = self._sct.monitors
        if len(monitors) <= self.cfg.monitor_index:
            raise RuntimeError(
                f"monitor_index={self.cfg.monitor_index} is invalid. "
                f"Available monitors length={len(monitors)}"
            )
        base = monitors[self.cfg.monitor_index]
        if self.cfg.region is None:
            return {
                "left": int(base["left"]),
                "top": int(base["top"]),
                "width": int(base["width"]),
                "height": int(base["height"]),
            }
        left, top, width, height = self.cfg.region
        return {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}

    def _grab_bgr(self, sct) -> Optional[np.ndarray]:
        assert self._mon is not None
        try:
            img = sct.grab(self._mon)           # BGRA
            frame = np.asarray(img, dtype=np.uint8)   # (H,W,4)
            return frame[:, :, :3]                    # BGR
        except ScreenShotError as e:
            print(f"ScreenShotError in _grab_bgr: {e}")
            return None

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Recorder already started.")
        self._started = True

        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        self._stop_evt.clear()
        self._stop_evt.clear()
        try:
            self._sct = mss.mss()
            self._mon = self._resolve_monitor()

            w, h = int(self._mon["width"]), int(self._mon["height"])
            
            # Try configured codec first, then fallback to mp4v if avc1 fails
            codecs_to_try = [self.cfg.codec]
            if self.cfg.codec == 'avc1':
                codecs_to_try.append('mp4v')
            
            writer_opened = False
            for codec in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore
                    self._writer = cv2.VideoWriter(str(self.video_path), fourcc, float(max(1, self.cfg.fps)), (w, h))
                    if self._writer.isOpened():
                        writer_opened = True
                        print(f"Successfully started video recording with codec: {codec}")
                        break
                    else:
                        print(f"Failed to open VideoWriter with codec: {codec}")
                except Exception as e:
                     print(f"Error initializing codec {codec}: {e}")

            if not writer_opened:
                raise RuntimeError(
                    f"Failed to open VideoWriter for {self.video_path}. "
                    f"Tried codecs: {codecs_to_try}. Ensure codecs are available."
                )

            self._thread = threading.Thread(target=self._run, name="ScreenRecordThread", daemon=True)
            self._thread.start()
        except (ScreenShotError, RuntimeError) as e:
            print(f"Warning: Screen recording disabled due to error: {e}")
            self._error_disabled = True
            if self._sct:
                self._sct.close()
                self._sct = None
            if self._writer:
                self._writer.release()
                self._writer = None

    def stop(self) -> None:
        if not self._started:
            return

        # 停止直前に「この録画のスクショを1枚」だけ保存
        try:
            # Try to save screenshot even if error occurred, as a last resort
            if self._sct is not None and self._mon is not None:
                frame = self._grab_bgr(self._sct)
                if frame is not None:
                    self._save_screenshot(frame)
                else:
                    print(f"Warning: Failed to grab screenshot frame for {self.screenshot_path.name}")
            else:
                 print(f"Skipping screenshot: sct or mon is None (started={self._started})")

        except Exception as e:
            print(f"Error saving screenshot {self.screenshot_path}: {e}")
        finally:
            self._stop_evt.set()
            if self._thread:
                self._thread.join(timeout=5)

            if self._writer is not None:
                self._writer.release()
                self._writer = None

            if self._sct is not None:
                self._sct.close()
                self._sct = None

            self._thread = None
            self._mon = None
            self._started = False

    def _run(self) -> None:
        assert self._writer is not None
        fps = max(1, int(self.cfg.fps))
        interval = 1.0 / fps
        next_t = time.perf_counter()

        # mss on Linux needs thread-local instance
        with mss.mss() as sct:
            while not self._stop_evt.is_set():
                if self._error_disabled:
                    break

                now = time.perf_counter()
                if now < next_t:
                    time.sleep(min(0.005, next_t - now))
                    continue
                next_t += interval

                frame = self._grab_bgr(sct)
                if frame is None:
                    print("Warning: Screen capture failed in thread. Stopping recording.")
                    self._error_disabled = True
                    break
                    
                self._writer.write(frame)

    def _save_screenshot(self, frame_bgr: np.ndarray) -> None:
        ext = self.cfg.screenshot_ext.lower()
        path = self.screenshot_path
        if path.suffix.lower() != ext:
            path = path.with_suffix(ext)

        result = False
        if ext in (".jpg", ".jpeg"):
            result = cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.screenshot_quality)])
        else:
            result = cv2.imwrite(str(path), frame_bgr)
            
        if result:
            print(f"Saved screenshot to {path}")
        else:
            print(f"Failed to save screenshot using cv2.imwrite to {path}")


def record_during(func, *, cfg: RecordConfig, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    with ScreenRecordOneShot(cfg):
        return func(*args, **kwargs)
