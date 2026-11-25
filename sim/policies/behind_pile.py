"""Sweep from behind the pile, checking the crossing before each entry."""
import math

import cv2
import numpy as np

from core import WHEEL, wrap

MATCH_SECONDS = 120.
APPROACH_SPEED = .20
RETURN_SPEED = .17
# Wing (+1 enters from low Y), X past midfield, and edge of the goal to sweep.
SWEEP_REGIONS = ((-1, .56, .13), (1, .27, -.13),
                 (1, .59, -.03), (-1, .28, .04))


class Policy:
    def __init__(self, setup):
        self.field = np.asarray(setup["field_size_m"], dtype=float)
        self.reverse = setup["side"] == "B"
        self.origin = self._local(setup["starting_xy_m"])
        goal = np.sort(self._local(np.reshape(setup["goal_xyxy_m"], (2, 2))), axis=0)
        self.goal_x = goal[:, 0].mean()
        self.goal_y = goal[:, 1].mean()
        self.goal_front = goal[1, 0] - .0025
        self.goal_sides = goal[:, 1] + [.0025, -.0025]
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
        if self.sweep >= 2:
            self._move_to("approach", self.approach, 0., time)
            return
        self.wing = 1
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
        self._observe("look", self.wing * math.radians(112), time)

    def _observe(self, phase, heading, time):
        self.last_frame = None
        self.frames = 0
        self.blocked_frames = 0
        self._move_to(phase, self.pos, heading, time)

    def _floor_point(self, x, y, z=0.):
        pixel = np.array([[[float(x), float(y)]]])
        u, v = cv2.undistortPoints(pixel, self.intrinsics, self.distortion)[0, 0]
        ray = self.optical_to_car @ np.array([u, v, 1.])
        if ray[2] < -.05:
            return self.camera_origin + (z - self.camera_origin[2]) / ray[2] * ray

    def _edge_points(self, edge, rotation):
        rows, cols = np.nonzero(edge)
        if len(rows) < 12:
            return np.empty((0, 2))
        pixels = np.column_stack((cols, rows)).astype(float)
        uv = cv2.undistortPoints(pixels, self.intrinsics, self.distortion).reshape(-1, 2)
        rays = np.column_stack((uv, np.ones(len(uv)))) @ self.optical_to_car.T
        rays = rays[rays[:, 2] < -.055]
        points = self.camera_origin - self.camera_origin[2] * rays / rays[:, 2, None]
        return points[:, :2] @ rotation.T

    @staticmethod
    def _edge_coordinate(points, axis, span):
        if len(points) < 12:
            return None
        values = points[:, axis]
        bins = np.rint(values / .012).astype(int)
        labels, counts = np.unique(bins, return_counts=True)
        peak = labels[counts.argmax()] * .012
        match = np.abs(values - peak) < .018
        if match.sum() >= 12 and np.ptp(points[match, 1 - axis]) >= span:
            return float(np.median(values[match]))

    def _correct_at_goal(self, obs, rotation):
        frame = obs["rgb"]
        stamp = obs["image_time_s"]
        if (frame is None or stamp is None or stamp == self.goal_frame or
                obs["time_s"] - stamp > .11 or abs(obs["gyro_z_rad_s"]) > .30):
            return
        self.goal_frame = stamp
        hue, saturation, value = cv2.split(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))
        if self.reverse:
            tint = (hue >= 103) & (hue <= 142)
        else:
            tint = (hue <= 12) | (hue >= 169)
        mask = tint & (saturation >= 33) & (saturation <= 130) & (value >= 85)
        floor = (saturation <= 38) & (value >= 95) & ~mask
        # The near edge of the printed rectangle has goal color above it.
        # Require floor beyond the outline so a row of pieces cannot vote as an edge.
        front_edge = np.zeros_like(mask)
        front_edge[:-5] = mask[:-5] & ~mask[1:-4] & floor[5:]
        front = self._edge_points(front_edge, rotation)
        front = front[(front[:, 0] < -.12) & (front[:, 0] > -1.25)]
        x = self._edge_coordinate(front, 0, .13)
        if x is None or abs(self.goal_front - x - self.pos[0]) > .8:
            return
        corrected_x = self.goal_front - x
        # Side edges vote independently and can be partly hidden by pieces.
        left = np.zeros_like(mask)
        right = np.zeros_like(mask)
        left[:, 5:] = mask[:, 5:] & ~mask[:, 4:-1] & floor[:, :-5]
        right[:, :-5] = mask[:, :-5] & ~mask[:, 1:-4] & floor[:, 5:]
        lateral = []
        for edge, line in zip((left, right), self.goal_sides):
            points = self._edge_points(edge, rotation)
            points = points[(points[:, 0] < x + .02) & (points[:, 0] > x - .41)]
            y = self._edge_coordinate(points, 1, .12)
            if y is not None and abs(line - y - self.pos[1]) < .48:
                lateral.append(line - y)
        if lateral and np.ptp(lateral) <= .06:
            self.pos[:] = corrected_x, np.mean(lateral)
        elif abs(corrected_x - self.pos[0]) <= .20:
            self.pos[0] = corrected_x

    def _end_wall_distance(self, rgb, rotation):
        # Track the visible bottom of the end wall even when disks hide the goal outline.
        cols = np.arange(52, rgb.shape[1] - 51, 6)
        wall = np.median(rgb[88:145, 155:485], axis=(0, 1))
        similar = np.max(np.abs(rgb[140:, cols] - wall), axis=2) <= 12
        visible = similar[:-4] | similar[1:-3] | similar[2:-2] | similar[3:-1] | similar[4:]
        rows = np.argmax(~visible, axis=0) + 140
        supported = (rows > 190) & (rows < 459)
        edge = np.zeros(rgb.shape[:2], dtype=bool)
        edge[rows[supported], cols[supported]] = True
        points = self._edge_points(edge, rotation)
        if len(points) < 16:
            return None
        # A disk occluding the seam makes the wall look farther away.
        distances = -points[:, 0]
        near = float(np.quantile(distances, .2))
        if .15 < near < 1.16 and np.count_nonzero(np.abs(distances - near) < .035) >= 12:
            return near

    def _blocked(self, rgb, rotation):
        # The amber push plate is distinct from the red and white glyphs.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (8, 100, 40), (31, 255, 255))
        _, _, boxes, _ = cv2.connectedComponentsWithStats(mask)
        for x, y, width, height, area in boxes[1:]:
            if width < 18 or area < 140:
                continue
            # Project the lower edge of the plate, which sits close to the floor.
            plate = self._floor_point(x + .5 * width, y + height - 2, .01)
            if plate is None:
                continue
            px, py = self.pos + rotation @ plate[:2]
            if (abs(px - self.entry[0]) < .33 and
                    -.08 < self.wing * (py - self.pos[1]) <
                    self.wing * (self.entry[1] - self.pos[1]) + .29):
                return True
        return False

    def _push_has_target(self, rgb, rotation):
        # A compact dark silhouette of disk width ahead of the push face.
        # Check both colors so an occluding opponent piece cannot hide a batch.
        end_wall = self._end_wall_distance(rgb, rotation)
        # Exclude the disks already past our goal line when the wall is visible.
        reach = 1.35 if end_wall is None else min(1.35, end_wall - self.goal_front + .012)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = cv2.inRange(gray, 0, 80)
        _, _, boxes, _ = cv2.connectedComponentsWithStats(mask)
        for x, y, width, height, area in boxes[1:]:
            if width < 11 or height < 5 or area < .28 * width * height:
                continue
            piece = self._floor_point(x + .5 * width, y + height - 1)
            if piece is None or not (.15 < piece[0] < reach and abs(piece[1]) < .20):
                continue
            width_m = width * (piece[0] - self.camera_origin[0]) / self.intrinsics[0, 0]
            if .027 < width_m < .19:
                return True
        return False

    def _fresh_frame(self, obs, aligned):
        stamp = obs["image_time_s"]
        if (aligned and abs(obs["gyro_z_rad_s"]) < .45 and obs["rgb"] is not None and
                stamp is not None and self.since <= stamp and obs["time_s"] - stamp <= .11
                and stamp != self.last_frame):
            self.last_frame = stamp
            self.frames += 1
            return obs["rgb"]

    def _leave_entry(self, yaw, time, retry):
        # Clear the pile laterally before turning back out to the travel lane.
        self.retry_entry = retry
        self._move_to("clear", (self.pos[0], self.rollout_y - self.wing * .10), yaw, time)

    def _detour(self, time):
        self.deep = True
        lane = self.goal_y - .06 + .08 * self.sweep if self.sweep < 2 else self.delivery_lane
        self.entry = np.array([max(self.center_x + .64, self.entry[0] + .12), lane])
        self._move_to("detour", (self.entry[0], self.rollout_y - self.wing * .10), 0., time)

    def _deliver(self, time):
        lane = self.goal_y - .04 + .08 * self.sweep if self.sweep < 2 else self.delivery_lane
        self.wall_frame = None
        self.wall_distance = None
        self._move_to("deliver", (self.goal_x if self.sweep < 2 else self.goal_front - .052, lane),
                      math.pi, time)

    def _retreat(self, time, yaw):
        clear_x = self.goal_x + .34 if self.sweep < 2 else self.goal_front + .27
        heading = math.pi if self.sweep < 2 else yaw
        self._move_to("retreat", (clear_x, self.pos[1]), heading, time)

    def _next_sweep(self, time):
        self.sweep += 1
        if self.sweep == 1 and MATCH_SECONDS - time >= 24.:
            self._move_to("outside", (self.goal_x + .57, self.center_y - .44), 0., time)
            return
        self.sweep = max(self.sweep, 2)
        self._plan_sweep(time)

    def _return_seconds(self, x):
        return max(0., x - self.goal_x) / RETURN_SPEED + 1.5

    def _plan_sweep(self, time):
        wing, depth, lane = SWEEP_REGIONS[self.region]
        self.region = (self.region + 1) % len(SWEEP_REGIONS)
        staging_x = self.goal_x + .57
        available = MATCH_SECONDS - time
        near_wing = self.wing
        candidates = ((wing, self.center_x + depth, lane),
                      (wing, self.center_x - .40, lane),
                      (near_wing, self.goal_x + .62, -near_wing * .13),
                      (-near_wing, self.goal_x + .62, near_wing * .13))
        for wing, entry_x, lane in candidates:
            approach_x = entry_x - .05
            rollout_y = self.center_y - wing * .45
            entry = np.array([entry_x, self.goal_y + lane])
            if wing == self.wing and np.linalg.norm(entry - self.entry) < .18:
                continue
            outside = np.array([min(staging_x, approach_x), rollout_y])
            crossing = np.linalg.norm(self.pos - outside)
            if self.pos[0] > staging_x + .25 and wing != self.wing:
                # Change wing on our side of the board, clear of the pile.
                reposition = (staging_x, self.rollout_y - self.wing * .10)
                crossing = np.linalg.norm(self.pos - reposition) + np.linalg.norm(outside - reposition)
            else:
                reposition = None
            # Reorienting out of the previous sweep takes time even on a short leg.
            approach_time = max(2.2, crossing / APPROACH_SPEED)
            approach_time += max(0., approach_x - outside[0]) / APPROACH_SPEED
            entry_seconds = 4.5 if entry_x <= self.goal_x + .62 else 6.
            budget = approach_time + self._return_seconds(entry_x) + entry_seconds
            if available >= budget:
                self.wing = wing
                self.rollout_y = rollout_y
                self.delivery_lane = self.goal_y + lane
                self.entry = entry
                self.approach = (approach_x, rollout_y)
                self.outside = outside
                self.short = entry_x < self.center_x
                if reposition is None:
                    self._move_to("outside", outside, 0., time)
                else:
                    self._move_to("reposition", reposition, 0., time)
                return
        self.phase = "done"

    def act(self, obs):
        encoder = np.asarray(obs["encoder_m"])
        yaw = wrap(obs["yaw_from_start_rad"])
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        time = obs["time_s"]
        gyro = obs["gyro_z_rad_s"]
        if obs["reset_id"] != self.reset_id:
            first_tick = self.reset_id is None
            self.reset_id = obs["reset_id"]
            self.pos = self.origin.copy()
            self.previous = encoder
            self.goal_frame = None
            if first_tick:
                self.sweep = 0
                self.region = 0
                self._approach(time)
            if not first_tick or MATCH_SECONDS - time < 24.:
                self.sweep = max(self.sweep + 1, 2)
                self.wing = 1
                self.rollout_y = self.center_y - .45
                self._plan_sweep(time)
        travel = rotation @ (self.inverse @ (encoder - self.previous))[:2]
        self.pos += travel
        self.previous = encoder
        if (self.sweep >= 2 and self.phase in ("check", "deliver", "retreat") and
                abs(wrap(math.pi - yaw)) < .13):
            self._correct_at_goal(obs, rotation)

        error = wrap(self.heading - yaw)
        guarded_return = self.sweep >= 2 and self.phase in ("deliver", "retreat")
        goal_distance = self.pos[0]
        if guarded_return:
            if self.wall_distance is not None:
                self.wall_distance += travel[0]
            stamp = obs["image_time_s"]
            if (obs["rgb"] is not None and stamp is not None and stamp != self.wall_frame and
                    time - stamp <= .11 and abs(error) < .20 and abs(gyro) < .30):
                self.wall_frame = stamp
                distance = self._end_wall_distance(obs["rgb"], rotation)
                if distance is not None and distance < .76:
                    self.wall_distance = distance
            if self.wall_distance is not None:
                goal_distance = self.wall_distance
            # Keep the departure waypoint planner in the same coordinate frame.
            if self.phase == "retreat" and self.wall_distance is not None:
                self.pos[0] = goal_distance

        aligned = abs(error) < .15
        reached = np.linalg.norm(self.target - self.pos) < .034 and aligned
        elapsed = time - self.since
        remaining = MATCH_SECONDS - time
        can_detour = remaining >= self._return_seconds(self.center_x + .75) + 9.
        in_goal_lane = self.goal_lane[0] < self.pos[1] < self.goal_lane[1]
        if remaining <= 0.:
            self.phase = "done"
        elif (self.phase in ("outside", "reposition", "approach", "detour", "look") and
              remaining < self._return_seconds(self.entry[0]) + 3.):
            self._next_sweep(time)
        elif self.phase in ("approach", "detour") and reached:
            self._look(time)
        elif self.phase == "look":
            frame = self._fresh_frame(obs, aligned)
            if frame is not None:
                self.blocked_frames += self._blocked(frame, rotation)
            if self.blocked_frames >= 2:
                if not self.deep and can_detour and not (self.sweep >= 2 and self.short):
                    self._detour(time)
                else:
                    self._leave_entry(yaw, time, retry=False)
            elif aligned and (self.frames >= 3 or elapsed >= 2.8):
                self._move_to("enter", self.entry, math.pi, time)
            elif elapsed >= 3.3:
                self._leave_entry(yaw, time, retry=not self.deep and can_detour)
        elif self.phase == "enter":
            return_due = remaining <= self._return_seconds(self.pos[0]) + .7
            if reached and self.sweep >= 2 and not return_due:
                self._observe("check", math.pi, time)
            elif reached or ((elapsed >= 2.8 or return_due) and aligned and in_goal_lane):
                self._deliver(time)
            elif elapsed >= 3.3 or return_due:
                self._leave_entry(yaw, time, retry=not self.deep and can_detour)
        elif self.phase == "check":
            frame = self._fresh_frame(obs, aligned)
            due = remaining <= self._return_seconds(self.pos[0])
            if ((frame is not None and self._push_has_target(frame, rotation)) or
                    (self.frames == 0 and (due or elapsed >= .8))):
                self._deliver(time)
            elif self.frames >= 2 or due or elapsed >= .8:
                self._leave_entry(yaw, time, retry=False)
        elif self.phase == "clear" and (reached or elapsed >= 2.5):
            if self.retry_entry and can_detour:
                self._detour(time)
            else:
                self._next_sweep(time)
        elif self.phase == "deliver" and (goal_distance <= self.target[0] + .034 or elapsed >= 12. or
                (guarded_return and goal_distance < .58 and abs(error) > .25)):
            self._retreat(time, yaw)
        elif self.phase == "retreat" and (reached or elapsed >= (3.6 if guarded_return else 3.) or
                (guarded_return and goal_distance >= self.target[0] - .034)):
            if self.sweep == 1 and aligned:
                self._correct_at_goal(obs, rotation)
            self._next_sweep(time)
        elif self.phase == "outside" and reached:
            self._approach(time)
        elif self.phase == "reposition" and reached:
            self._move_to("outside", self.outside, 0., time)
        elif self.phase in ("approach", "detour", "outside", "reposition") and elapsed >= 12.:
            self._next_sweep(time)
        if self.phase == "done":
            return {"motor_pwm": [0, 0, 0, 0]}

        error = wrap(self.heading - yaw)
        p = 2. * (self.target - self.pos)
        speed, yaw_limit = .23, 1.4
        if self.phase in ("look", "check") or (self.phase == "enter" and abs(error) > .40):
            p[:] = 0.
        elif self.phase == "deliver":
            # Keep the push face square and bleed off cross-track error gradually.
            p[1] = np.clip(p[1], -.023, .023)
            yaw_limit = .50
            if guarded_return:
                p[0] = -max(0., 2. * (goal_distance - self.target[0]))
                if goal_distance < .62:
                    p[1] = 0.
                    speed, yaw_limit = .13, .30
        elif self.phase == "retreat" and self.sweep >= 2:
            back = max(0., 2. * (self.target[0] - goal_distance))
            p = -back * np.array([math.cos(self.heading), math.sin(self.heading)])
            yaw_limit = .30
        p *= min(1., speed / max(speed, np.linalg.norm(p)))
        w = np.clip(3. * error - .18 * gyro, -yaw_limit, yaw_limit)
        raw = (WHEEL @ np.r_[rotation.T @ p, .11 * w]) * (237. / .32)
        pwm = np.where(abs(raw) >= 3., raw + 18 * np.sign(raw), 0.)
        return {"motor_pwm": np.rint(np.clip(pwm, -255, 255)).astype(int).tolist()}
