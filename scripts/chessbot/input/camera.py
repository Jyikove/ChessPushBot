"""One camera reader, shared read-only frames over a timestamped latest-value API."""
from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    sequence: int
    time: float
    bgr: np.ndarray


class Camera(threading.Thread):
    def __init__(self, device, fps):
        super().__init__(name="camera", daemon=True)
        self.device, self.fps = device, fps
        self.stopping = threading.Event()
        self.condition = threading.Condition()
        self.frame = None
        self.error = None

    def latest(self, after=-1, timeout=0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.error is not None or self.stopping.is_set()
                or (self.frame is not None and self.frame.sequence > after), timeout)
            if self.error:
                raise RuntimeError(f"camera: {self.error}") from self.error
            if self.frame is not None and self.frame.sequence > after:
                return self.frame
            return None

    def close(self):
        with self.condition:
            self.stopping.set()
            self.condition.notify_all()
        self.join(timeout=1)

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise OSError(f"cannot open {self.device}")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            seq = 0
            while not self.stopping.is_set():
                ok, bgr = cap.read()
                at = time.monotonic()
                if not ok or bgr is None:
                    raise OSError(f"no frame from {self.device}")
                if bgr.shape != (480, 640, 3):
                    raise ValueError(f"expected native 640x480 BGR, got {bgr.shape}")
                bgr.setflags(write=False)
                with self.condition:
                    self.frame = CameraFrame(seq, at, bgr)
                    self.condition.notify_all()
                seq += 1
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            if cap is not None:
                cap.release()
