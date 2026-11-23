"""Read the existing 100 Hz IMU stream; publish heading and recent acceleration."""
from collections import deque
from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from ..modeling.imu_heading import Heading
from .imu_protocol import Parser


@dataclass(frozen=True)
class ImuFrame:
    time: float
    yaw_rad: float
    gyro_z_rad_s: float
    magnetic_yaw_rad: float | None
    acceleration: tuple
    horizontal_rms: float | None


class Imu(threading.Thread):
    def __init__(self, device, baud, body_from_imu):
        super().__init__(name="imu", daemon=True)
        self.device, self.baud = device, baud
        self.rotation = np.asarray(body_from_imu, dtype=float)
        if (self.rotation.shape != (3, 3)
                or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.rotation), 1.)):
            raise ValueError("R_body_imu must be a proper rotation")
        self.heading = Heading()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.recent = deque(maxlen=512)
        self.last = None
        self.error = None

    def close(self):
        self.stopping.set()
        self.join(timeout=1)

    def latest(self, after=-math.inf):
        with self.lock:
            if self.error:
                raise RuntimeError(f"IMU: {self.error}") from self.error
            if self.last is None:
                return None
            recent = [(t, x, y) for t, x, y in self.recent if t > after]
            acceleration = tuple((t, x) for t, x, _ in recent)
            rms = math.sqrt(sum(x*x+y*y for _, x, y in recent)/len(recent)) if recent else None
            return ImuFrame(*self.last, acceleration, rms)

    def run(self):
        try:
            import serial

            parser = Parser()
            previous = None
            start = None
            stationary = []
            baseline = None
            with serial.Serial(self.device, self.baud, timeout=.03, exclusive=True) as port:
                while not self.stopping.is_set():
                    data = port.read(min(256, max(1, port.in_waiting)))
                    now = time.monotonic()
                    rows = [row for row in parser.feed(data) if "gyro_rad_s" in row]
                    if not rows:
                        continue
                    rates = [(self.rotation @ row["gyro_rad_s"])[2] for row in rows]
                    fields = [self.rotation @ row["mag_uT"] for row in rows]
                    accel = [9.80665 * (self.rotation @ row["accel_g"])[:2] for row in rows]
                    self.heading.update(list(zip(rates, fields)), now)
                    if baseline is None:
                        if ((previous is not None and now-previous > .15)
                                or max(map(abs, rates)) > .10):
                            start, stationary = None, []
                        else:
                            start = now if start is None else start
                            stationary.extend(accel)
                            if now-start >= 1. and len(stationary) >= 30:
                                baseline = np.mean(stationary, axis=0)
                        previous = now
                        continue
                    # Host receive times are distributed within a batch, as in the keyboard estimator.
                    dt = (now-previous)/len(rows) if now-previous <= .15 else 0.
                    _, heading, _, _ = self.heading.snapshot()
                    mx, my, _ = fields[-1]
                    magnetic = math.atan2(-my, mx) if math.hypot(mx, my) > 1e-6 else None
                    with self.lock:
                        for index, raw in enumerate(accel):
                            x, y = raw - baseline
                            self.recent.append((now-(len(rows)-1-index)*dt, float(x), float(y)))
                        self.last = (now, heading[0], heading[2], magnetic)
                    previous = now
        except Exception as error:
            with self.lock:
                self.error = error
