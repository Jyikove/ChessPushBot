"""Write timestamped control frames to an MP4 without blocking the control loop."""
import json
import math
from queue import Empty, Full, Queue
import threading

import cv2

from .preview import FRAME_SIZE, render


class Recorder:
    def __init__(self, output, config):
        self.output = output
        self.fps = float(config["fps"])
        self.codec = config["codec"]
        capacity = config["queue_frames"]
        if (not math.isfinite(self.fps) or self.fps <= 0 or len(self.codec) != 4
                or type(capacity) is not int or capacity < 1):
            raise ValueError("invalid recording settings")
        self.queue = Queue(maxsize=capacity)
        self.done = threading.Event()
        self.error = None
        self.dropped = 0
        self.frames_written = 0
        self.first = None
        self.worker = threading.Thread(target=self.save, name="chess-video", daemon=True)

    def __enter__(self):
        # Validate encoder access before enabling motion.
        self.writer = cv2.VideoWriter(str(self.output / "recording.mp4"),
                                      cv2.VideoWriter_fourcc(*self.codec), self.fps, FRAME_SIZE)
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError("cannot open recording encoder")
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.worker.join()
        self.writer.release()
        (self.output / "video_stats.json").write_text(json.dumps(dict(
            fps=self.fps, frames_written=self.frames_written, dropped_queue=self.dropped,
            error=None if self.error is None else str(self.error)), indent=2) + "\n")
        if self.error:
            raise RuntimeError(f"recording failed: {self.error}") from self.error

    def add(self, at, bgr, obs, state):
        if self.error:
            raise RuntimeError(f"recording failed: {self.error}") from self.error
        try:
            self.queue.put_nowait((at, bgr, obs, state))
        except Full:
            self.dropped += 1

    def save(self):
        last_frame = None
        try:
            with (self.output / "video_frames.jsonl").open("x", buffering=1) as log:
                while not self.done.is_set() or not self.queue.empty():
                    try:
                        at, bgr, obs, state = self.queue.get(timeout=.05)
                    except Empty:
                        continue
                    if self.first is None:
                        self.first = at
                    index = round((at - self.first) * self.fps)
                    if index < self.frames_written:
                        continue
                    while self.frames_written < index and last_frame is not None:
                        self.writer.write(last_frame)
                        self.frames_written += 1
                    last_frame = render(bgr, obs, state)
                    self.writer.write(last_frame)
                    self.frames_written += 1
                    log.write(json.dumps(dict(frame=index, camera_time=at, state=state)) + "\n")
        except Exception as error:
            self.error = error
