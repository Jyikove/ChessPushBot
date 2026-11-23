"""The Pi keyboard script's yaw estimator, separated from hardware access."""
from collections import deque
import math
from threading import Lock
import time

HEADING_STALE = 0.3
HEADING_DISPLAY_PERIOD = 0.1
HEADING_RATE_WINDOW = 1.0
MAG_SNAP_ZONE = math.radians(0.25)
MAG_REARM_ZONE = math.radians(1)
MAG_REARM_TIME = 0.2

def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Heading:
    def __init__(self):
        self.lock = Lock()
        self.angle = 0.0
        self.previous_rate = 0.0
        self.magnetic_zero = None
        self.magnetic_armed = False
        self.magnetic_outside_since = None
        self.latest = None
        self.magnetic_field = None
        self.received = deque()

    def update(self, samples, received_at):
        samples = [(rate, field) for rate, field in samples if math.isfinite(rate)]
        if not samples:
            return
        with self.lock:
            step = 0.0
            if self.latest is not None:
                elapsed = received_at - self.latest[1]
                if 0 < elapsed <= HEADING_STALE:
                    # Frames delivered in one read share a host timestamp.
                    step = elapsed / len(samples)
                elif elapsed > HEADING_STALE:
                    self.magnetic_outside_since = None
            for index, (rate, field) in enumerate(samples):
                delta = (self.previous_rate + rate) * 0.5 * step
                self.previous_rate = rate
                sample_at = received_at - (len(samples) - 1 - index) * step
                self.advance(delta, field, sample_at)
            self.latest = self.angle, received_at, self.previous_rate
            self.magnetic_field = (*samples[-1][1], received_at)
            self.received.extend([received_at] * len(samples))
            self.expire_samples(received_at)

    def advance(self, delta, field, sample_at):
        self.angle = wrap_angle(self.angle + delta)
        x, y, _ = field
        horizontal = math.hypot(x, y)
        if not math.isfinite(horizontal) or horizontal < 1e-6:
            self.magnetic_outside_since = None
            return
        magnetic_yaw = math.atan2(-y, x)
        if self.magnetic_zero is None:
            self.magnetic_zero = magnetic_yaw
        relative = wrap_angle(magnetic_yaw - self.magnetic_zero)
        if self.magnetic_armed:
            if abs(relative) <= MAG_SNAP_ZONE:
                self.angle = relative
                self.magnetic_armed = False
                self.magnetic_outside_since = None
        elif abs(relative) > MAG_REARM_ZONE:
            if self.magnetic_outside_since is None:
                self.magnetic_outside_since = sample_at
            elif sample_at - self.magnetic_outside_since >= MAG_REARM_TIME:
                self.magnetic_armed = True
        else:
            self.magnetic_outside_since = None

    def expire_samples(self, now):
        cutoff = now - HEADING_RATE_WINDOW
        while self.received and self.received[0] <= cutoff:
            self.received.popleft()

    def snapshot(self, clock=time.monotonic):
        with self.lock:
            now = clock()
            self.expire_samples(now)
            return now, self.latest, len(self.received) / HEADING_RATE_WINDOW, self.magnetic_field
