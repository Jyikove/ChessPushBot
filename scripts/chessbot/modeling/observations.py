"""Image-relative observations passed from perception to control."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Piece:
    box: tuple
    confidence: float
    distance_m: float | None = None
    body_xy_m: tuple | None = None
    display_box: tuple | None = None

    @property
    def x(self):
        return (self.box[0] + self.box[2]) / 2

    @property
    def bottom(self):
        return self.box[3]


@dataclass(frozen=True)
class Zone:
    entry_px: tuple
    lateral_m: float
    forward_m: float
    left_m: float
    right_m: float
    pixels: int
    line_px: tuple | None = None


@dataclass(frozen=True)
class StartZone:
    marker_px: tuple
    forward_m: float
    footprint_m: tuple


@dataclass(frozen=True)
class FenceLine:
    lateral_m: float
    endpoints_px: tuple
    support: int
    span_m: float


@dataclass(frozen=True)
class Observation:
    time: float
    yaw: float
    yaw_rate: float
    pieces: tuple
    zone: Zone | None
    imu: tuple  # (sample time s, longitudinal acceleration m/s²)
    start_zone: StartZone | None = None
    fence: FenceLine | None = None
    horizontal_accel_rms: float | None = None
    image_motion_px: float | None = None
