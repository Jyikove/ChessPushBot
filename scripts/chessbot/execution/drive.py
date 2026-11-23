"""Map desired body motion to signed Pi-order PWM without opening any device."""
import json
import math

STOP = (0, 0, 0, 0)
PI_WHEEL_ORDER = ("FL", "RL", "FR", "RR")


class Drive:
    def __init__(self, profile_path, full_scale_m_s, yaw_arm_m, use_trims=True):
        profile = json.loads(profile_path.read_text())
        self.pwm = profile["pwm"]
        self.gains = profile["wheel_gains"]
        self.trims = profile["trims"] if use_trims else dict.fromkeys("wasd", (0., 0.))
        self.full_scale_m_s = full_scale_m_s
        self.yaw_arm_m = yaw_arm_m
        if (profile["version"] != 1 or set(profile["confirmed"]) != set("wasd")
                or type(self.pwm) is not int or not 1 <= self.pwm <= 100
                or len(self.gains) != 4
                or any(not math.isfinite(gain) or gain <= 0 for gain in self.gains)
                or set(self.trims) != set("wasd")
                or any(len(pair) != 2 or any(not math.isfinite(v) or abs(v) > .35 for v in pair)
                       for pair in self.trims.values())
                or any(not math.isfinite(v) or v <= 0 for v in (full_scale_m_s, yaw_arm_m))):
            raise ValueError("invalid drive or wheel calibration profile")

    def command(self, velocity):
        """Input is forward m/s, left m/s, counterclockwise rad/s."""
        vx, vy, wz = velocity
        if not all(math.isfinite(v) for v in velocity):
            raise ValueError("non-finite velocity")
        f = vx / self.full_scale_m_s
        r = -vy / self.full_scale_m_s
        c = -self.yaw_arm_m * wz / self.full_scale_m_s
        # Blend directional trims in proportion to the base translation axes.
        # Select both from the uncorrected axes, including on diagonals.
        cross_f, yaw_f = self.trims["w" if f >= 0 else "s"]
        cross_r, yaw_r = self.trims["d" if r >= 0 else "a"]
        forward = f + abs(r) * cross_r
        right = r + abs(f) * cross_f
        clockwise = c + abs(f) * yaw_f + abs(r) * yaw_r
        raw = (forward + right + clockwise, forward - right + clockwise,
               forward - right - clockwise, forward + right - clockwise)
        compensated = tuple(v * gain for v, gain in zip(raw, self.gains, strict=True))
        scale = self.pwm / max(1., max(map(abs, compensated)))
        return tuple(round(v * scale) for v in compensated)
