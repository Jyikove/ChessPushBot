"""Sweep from behind the pile, checking the crossing before each entry."""
import math

import cv2
import numpy as np

from core import WHEEL, wrap


class Policy:
    def __init__(self, setup):
        self.field = np.asarray(setup["field_size_m"], dtype=float)
        self.reverse = setup["side"] == "B"
        self.origin = self._local(setup["starting_xy_m"])
        goal = np.sort(self._local(np.reshape(setup["goal_xyxy_m"], (2, 2))), axis=0)
        self.goal_x = goal[:, 0].mean()
        self.goal_y = goal[:, 1].mean()
        self.goal_lane = goal[:, 1] + [.11, -.11]
        self.center_x, self.center_y = self.field / 2
        camera = setup["camera"]
        self.intrinsics = np.asarray(camera["K"])
        self.distortion = np.asarray(camera["opencv_distortion"])
        self.optical_to_car = np.asarray(camera["R_ground_optical"])
        self.camera_origin = np.asarray(camera["camera_base_position_m"])
        self.inverse = np.linalg.pinv(WHEEL)
        self.reset_id = None

    def _local(self, xy):
        xy = np.asarray(xy, dtype=float)
        return self.field - xy if self.reverse else xy

    def _move_to(self, phase, xy, heading, time):
        self.phase = phase
        self.target = np.array(xy)
        self.heading = heading
        self.since = time

    def _approach(self, time):
        self.deep = False
        approach_offset, entry_offset = (
            ((.38, -.43), (.48, -.17)) if self.sweep == 0
            else ((.10, -.36), (.17, -.07))
        )
        center = np.array([self.center_x, self.center_y])
        approach = center + approach_offset
        self.rollout_y = approach[1]
        self.entry = center + entry_offset
        self._move_to("approach", approach, 0., time)

    def _look(self, time):
        self.last_frame = None
        self.frames = 0
        self.blocked_frames = 0
        self._move_to("look", self.pos, math.radians(112), time)

    def _blocked(self, rgb, rotation):
        # The amber push plate is distinct from the red and white glyphs.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (8, 100, 40), (31, 255, 255))
        _, _, boxes, _ = cv2.connectedComponentsWithStats(mask)
        for x, y, width, height, area in boxes[1:]:
            if width < 18 or area < 140:
                continue
            pixel = np.array([[[x + .5 * width, y + height - 2]]])
            u, v = cv2.undistortPoints(pixel, self.intrinsics, self.distortion)[0, 0]
            ray = self.optical_to_car @ np.array([u, v, 1.])
            if ray[2] >= -.05:
                continue
            # Project the lower edge of the plate, which sits close to the floor.
            plate = self.camera_origin + (.01 - self.camera_origin[2]) / ray[2] * ray
            px, py = self.pos + rotation @ plate[:2]
            if abs(px - self.entry[0]) < .33 and self.pos[1] - .08 < py < self.entry[1] + .29:
                return True
        return False

    def _leave_entry(self, yaw, time):
        # Clear the pile laterally before turning back out to the travel lane.
        self._move_to("clear", (self.pos[0], self.rollout_y - .10), yaw, time)

    def _detour(self, time):
        self.deep = True
        self.entry = np.array([self.center_x + .64, self.goal_y - .06 + .08 * self.sweep])
        self._move_to("detour", (self.entry[0], self.rollout_y - .10), 0., time)

    def _deliver(self, time):
        self._move_to("deliver", (self.goal_x, self.goal_y - .04 + .08 * self.sweep),
                      math.pi, time)

    def _retreat(self, time):
        self._move_to("retreat", (self.goal_x + .34, self.pos[1]), math.pi, time)

    def _next_sweep(self, time):
        if self.sweep == 0:
            self.sweep = 1
            self._move_to("outside", (self.goal_x + .57, self.center_y - .44), 0., time)
        else:
            self.phase = "done"

    def act(self, obs):
        encoder = np.asarray(obs["encoder_m"])
        yaw = wrap(obs["yaw_from_start_rad"])
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        time = obs["time_s"]
        gyro = obs["gyro_z_rad_s"]
        if obs["reset_id"] != self.reset_id:
            self.reset_id = obs["reset_id"]
            self.pos = self.origin.copy()
            self.previous = encoder
            self.sweep = 0
            self._approach(time)
        self.pos += rotation @ (self.inverse @ (encoder - self.previous))[:2]
        self.previous = encoder

        error = wrap(self.heading - yaw)
        aligned = abs(error) < .15
        reached = np.linalg.norm(self.target - self.pos) < .034 and aligned
        elapsed = time - self.since
        if self.phase in ("approach", "detour") and reached:
            self._look(time)
        elif self.phase == "look":
            stamp = obs["image_time_s"]
            if aligned and abs(gyro) < .45 and obs["rgb"] is not None and (
                stamp is not None and self.since <= stamp and time - stamp <= .11
                and stamp != self.last_frame
            ):
                self.last_frame = stamp
                self.frames += 1
                self.blocked_frames += self._blocked(obs["rgb"], rotation)
            if self.blocked_frames >= 2:
                if self.deep:
                    self._leave_entry(yaw, time)
                else:
                    self._detour(time)
            elif aligned and (self.frames >= 3 or elapsed >= 2.8):
                self._move_to("enter", self.entry, math.pi, time)
            elif elapsed >= 3.3:
                self._leave_entry(yaw, time)
        elif self.phase == "enter":
            if reached or (elapsed >= 2.8 and aligned and self.goal_lane[0] < self.pos[1] < self.goal_lane[1]):
                self._deliver(time)
            elif elapsed >= 3.3:
                self._leave_entry(yaw, time)
        elif self.phase == "clear" and (reached or elapsed >= 2.5):
            if self.deep:
                self._next_sweep(time)
            else:
                self._detour(time)
        elif self.phase == "deliver" and (self.pos[0] <= self.goal_x + .034 or elapsed >= 12.):
            self._retreat(time)
        elif self.phase == "retreat" and (reached or elapsed >= 3.):
            self._next_sweep(time)
        elif self.phase == "outside" and reached:
            self._approach(time)
        elif self.phase in ("approach", "detour", "outside") and elapsed >= 12.:
            self._next_sweep(time)
        if self.phase == "done":
            return {"motor_pwm": [0, 0, 0, 0]}

        error = wrap(self.heading - yaw)
        p = 2. * (self.target - self.pos)
        speed, yaw_limit = .23, 1.4
        if self.phase == "look" or (self.phase == "enter" and abs(error) > .40):
            p[:] = 0.
        elif self.phase == "deliver":
            # Keep the push face square and bleed off cross-track error gradually.
            p[1] = np.clip(p[1], -.023, .023)
            yaw_limit = .50
        p *= min(1., speed / max(speed, np.linalg.norm(p)))
        w = np.clip(3. * error - .18 * gyro, -yaw_limit, yaw_limit)
        raw = (WHEEL @ np.r_[rotation.T @ p, .11 * w]) * (237. / .32)
        pwm = np.where(abs(raw) >= 3., raw + 18 * np.sign(raw), 0.)
        return {"motor_pwm": np.rint(np.clip(pwm, -255, 255)).astype(int).tolist()}
