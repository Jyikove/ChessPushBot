"""Sensor freshness, reactive policy and integer PWM. No device access."""
from dataclasses import dataclass
import math

from .execution.drive import STOP
from .modeling.observations import Observation
from .planning.policy import Reactive


@dataclass(frozen=True)
class SensorFrame:
    camera_time: float
    imu_time: float
    bgr: object
    yaw: float
    gyro_z: float
    acceleration: tuple
    horizontal_rms: float | None


class Runtime:
    def __init__(self, cfg, vision, drive, max_camera_age, max_imu_age, one_cycle, test_stepper=None):
        self.policy = Reactive(cfg)
        self.vision, self.drive = vision, drive
        self.max_camera_age, self.max_imu_age = max_camera_age, max_imu_age
        self.one_cycle = one_cycle
        self.test_stepper = test_stepper
        self.preview_image = None
        self.last_camera_time = self.last_time = -math.inf
        self.stop_reason = None

    def stop(self, reason):
        self.stop_reason = self.stop_reason or reason
        return (0., 0., 0.), STOP, None

    def tick(self, now, frame, skip=False):
        if self.stop_reason is not None:
            return self.stop(self.stop_reason)
        valid = (math.isfinite(now) and now > self.last_time
                 and 0 <= now-frame.camera_time <= self.max_camera_age
                 and frame.camera_time > self.last_camera_time
                 and 0 <= now-frame.imu_time <= self.max_imu_age
                 and all(math.isfinite(v) for v in (frame.yaw, frame.gyro_z))
                 and (frame.horizontal_rms is None or
                      math.isfinite(frame.horizontal_rms) and frame.horizontal_rms >= 0)
                 and all(math.isfinite(a) and self.last_time <= t <= now
                         for t, a in frame.acceleration))
        if not valid:
            self.policy.fail(now if math.isfinite(now) else 0., "invalid_or_stale_sensor")
            return self.stop("invalid_or_stale_sensor")
        self.last_camera_time, self.last_time = frame.camera_time, now
        pieces, zone, start, fence, motion, self.preview_image = self.vision.sense(
            frame.bgr, timestamp=frame.camera_time)
        obs = Observation(now, frame.yaw, frame.gyro_z, pieces, zone,
                          frame.acceleration, start, fence, frame.horizontal_rms, motion)
        velocity = (self.policy.step(obs) if self.test_stepper is None
                    else self.test_stepper.step(self.policy, obs, skip))
        if self.one_cycle and self.policy.cycle_completed:
            self.stop("one_cycle_complete")
            return (0., 0., 0.), STOP, obs
        if self.policy.state == "SF":
            self.stop("policy_fault")
            return (0., 0., 0.), STOP, obs
        return velocity, self.drive.command(velocity), obs
