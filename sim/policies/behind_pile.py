"""Sweep from behind the pile, checking the crossing before each entry."""
import math

import cv2
import numpy as np

from core import WHEEL, wrap
from piece_values import PieceValues

MATCH_SECONDS = 120.
APPROACH_SPEED = .20
RETURN_SPEED = .17
OPENING_LANE_STEP = .04
# Wing (+1 enters from low Y), X past midfield, and edge of the goal to sweep.
SWEEP_REGIONS = ((-1, .56, .13), (1, .27, -.13),
                 (1, .59, -.03), (-1, .28, .04))
TRAFFIC_PHASES = ("outside", "reposition", "approach", "detour", "enter", "deliver")
RAID_TRAVEL = ("raid_clear", "raid_lane", "raid_transit", "raid_align",
               "raid_push", "raid_clearance", "raid_exit", "raid_rescan_shift")
RECOVER_TRAVEL = ("recover_flank", "recover_transit", "recover_stage", "recover_align", "recover_push")
WATCHED_TRAVEL = TRAFFIC_PHASES + RAID_TRAVEL + RECOVER_TRAVEL


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
        self.home_goal = goal
        self.enemy_goal = np.sort(self.field - goal, axis=0)
        self.center_x, self.center_y = self.field / 2
        camera = setup["camera"]
        self.intrinsics = np.asarray(camera["K"])
        self.distortion = np.asarray(camera["opencv_distortion"])
        self.optical_to_car = np.asarray(camera["R_ground_optical"])
        self.camera_origin = np.asarray(camera["camera_base_position_m"])
        self.inverse = np.linalg.pinv(WHEEL)
        self.reset_id = None
        self.exit_reason = None
        self.failed_sweeps = 0
        self.last_delivery = 0.
        self.raid_entries = []
        self.raid_denied = 0
        self.raid_push_offset = 0.
        self.raid_carry_observation = None
        self.raid_followup = False
        self.raid_short_target = None
        self.raid_rescan_shifted = False
        self.raid_rescan_advanced = False
        self.raid_entry_rechecked = False
        self.opening_lane_step = 0.
        self.recovery_tried = False
        self.rank_model = None
        self.rank_stamp = None
        self.rank_observations = []
        self.rank_raw = []
        self.opponent = None
        self.home_bank = (0., 0.)
        self.home_loose = (0., [])
        self.enemy_bank = (0., [])

    def _local(self, xy):
        xy = np.asarray(xy, dtype=float)
        return self.field - xy if self.reverse else xy

    def _move_to(self, phase, xy, heading, time):
        self.phase = phase
        self.target = np.array(xy)
        self.heading = heading
        self.since = time
        self.traffic_stamp = None
        self.traffic_gap = math.inf
        self.traffic_hits = 0
        self.motion_frame = None
        self.motion_quiet = 0.
        self.raid_edge_samples = []
        self.raid_line_progress = -math.inf
        self.drive_command = (0., 0.)

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
        self.opening_lane_step = 0.
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

    def _end_wall_distance(self, rgb, rotation, axis=0, sign=-1):
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
        distances = sign * points[:, axis]
        near = float(np.quantile(distances, .2))
        if .15 < near < 1.16 and np.count_nonzero(np.abs(distances - near) < .035) >= 12:
            return near

    def _plates(self, rgb):
        # The amber push plate is distinct from the red and white glyphs.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (8, 100, 40), (31, 255, 255))
        _, _, boxes, _ = cv2.connectedComponentsWithStats(mask)
        for x, y, width, height, area in boxes[1:]:
            if width < 18 or area < 140:
                continue
            # Project the lower edge of the plate, which sits close to the floor.
            plate = self._floor_point(x + .5 * width, y + height - 2, .01)
            if plate is not None:
                yield plate[:2]

    def _blocked(self, rgb, rotation):
        for plate in self._plates(rgb):
            px, py = self.pos + rotation @ plate
            if (abs(px - self.entry[0]) < .33 and
                    -.08 < self.wing * (py - self.pos[1]) <
                    self.wing * (self.entry[1] - self.pos[1]) + .29):
                return True
        return False

    def _traffic_ahead(self, obs, direction):
        stamp = obs["image_time_s"]
        if obs["rgb"] is None or stamp is None or obs["time_s"] - stamp > .23:
            self.traffic_hits = 0
            self.traffic_gap = math.inf
        elif stamp != self.traffic_stamp:
            previous = self.traffic_stamp
            self.traffic_stamp = stamp
            cross = np.array([-direction[1], direction[0]])
            gap = math.inf
            for plate in self._plates(obs["rgb"]):
                # The front-mounted push face also sweeps sideways on this drive.
                if not .14 < plate[0] < .65:
                    continue
                delta = plate - [.13, 0.]
                ahead = delta @ direction
                if -.04 < ahead < .32 and abs(delta @ cross) < .24:
                    gap = min(gap, ahead)
            self.traffic_gap = gap
            if gap < .225:
                self.traffic_hits = (self.traffic_hits + 1 if previous is not None and
                                     stamp - previous <= .32 else 1)
            else:
                self.traffic_hits = 0
        return self.traffic_gap, self.traffic_hits >= 2

    def _remember_opponent(self, plate, rotation, time):
        position = self.pos + rotation @ plate
        velocity = np.zeros(2)
        if self.opponent is not None:
            last, previous, old_velocity = self.opponent
            age = time - last
            if .15 <= age <= .45:
                pace = (position - previous) / age
                if np.linalg.norm(pace) < .55:
                    velocity = .55 * old_velocity + .45 * pace
        self.opponent = time, position, velocity

    def _entry_risk(self, time, wing, y):
        if self.opponent is None or time - self.opponent[0] > 3.:
            return .18
        _, p, v = self.opponent
        # Approaching a wing increases its risk. A visibly departing opponent
        # gives that entrance a short-lived advantage.
        if p[0] < self.enemy_goal[0, 0] - .65:
            return .06
        predicted_y = p[1] + 1.2 * v[1]
        if wing * (v[1]) > .06 and abs(predicted_y - y) > abs(p[1] - y):
            return .07
        if abs(predicted_y - y) < .45:
            return .65
        return .15

    def _utility(self, time, seconds, own=0., deny=0., loss=0., risk=0.):
        after = MATCH_SECONDS - time - seconds
        if after < 0. or risk >= 1.:
            return -math.inf
        # Denial near the horn is harder to undo. All terms are in points
        # until the time discount, so a long route needs a larger payoff.
        last_chance = max(0., 1. - after / 48.)
        retained = .60 + .40 * last_chance ** 2
        return ((own + retained * deny) * (1. - risk) - loss) / (1. + seconds / 14.)

    def _home_exposure(self, time, seconds):
        seen, points = self.home_bank
        if time - seen > 24. or points <= 0.:
            return 0.
        risk = .05
        if self.opponent is not None and time - self.opponent[0] <= 5.:
            _, p, v = self.opponent
            if p[0] < 1.35 and (v[0] < -.035 or p[0] < .65):
                risk = .32
        return points * risk * min(1., seconds / 20.)

    def _ranked_pieces(self, rgb, stamp, rotation):
        if stamp == self.rank_stamp:
            return self.rank_observations
        if self.rank_model is None:
            self.rank_model = PieceValues()
        previous = self.rank_raw if self.rank_stamp is not None and stamp - self.rank_stamp <= .45 else []
        raw, confirmed = [], []
        for box, team, score in self.rank_model.observe(rgb):
            x0, _, x1, y1 = box
            point = self._floor_point((x0 + x1) / 2, y1 - 1.)
            if point is None or not .19 < point[0] < 1.35:
                continue
            position = self.pos + rotation @ (point[:2] + [.028, 0.])
            raw.append((position, team, score))
            agree = score is not None and any(
                side == team and value == score and np.linalg.norm(at - position) < .07
                for at, side, value in previous
            )
            confirmed.append((position, team, score if agree else None))
        self.rank_stamp, self.rank_raw, self.rank_observations = stamp, raw, confirmed
        return confirmed

    def _valued_disks(self, rgb, stamp, rotation, own):
        red = not self.reverse if own else self.reverse
        team = "A" if red else "B"
        observed = [(p, value) for p, side, value in self._ranked_pieces(rgb, stamp, rotation)
                    if side == team]
        for p in self._colored_disks(rgb, rotation, red):
            if all(np.linalg.norm(p - previous) > .052 for previous, _ in observed):
                observed.append((p, None))
        return observed

    @staticmethod
    def _points(offers):
        return sum(1 if value is None else value for _, value in offers)

    def _no_progress(self, obs, yaw):
        stamp = obs["image_time_s"]
        if obs["rgb"] is None or stamp is None or obs["time_s"] - stamp > .23:
            self.motion_frame = None
            self.motion_quiet = 0.
            self.raid_edge_samples.clear()
            self.raid_line_progress = -math.inf
            return False
        previous = self.motion_frame
        if previous is not None and stamp == previous[0]:
            return False
        rgb = cv2.resize(obs["rgb"], (160, 120), interpolation=cv2.INTER_AREA)
        gray = np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[35:114, 8:152])
        self.motion_frame = stamp, yaw, gray
        if previous is None or stamp - previous[0] > .32:
            self.motion_quiet = 0.
            return False
        drive, turn = self.drive_command
        yaw_changed = abs(wrap(yaw - previous[1])) >= .018
        quiet = turn >= .42 and drive < .07 and not yaw_changed
        if drive >= .10 and not yaw_changed:
            mask = None
            if self.phase in ("deliver", "raid_push", "recover_push"):
                # Disks already carried with the car are poor motion landmarks.
                mask = np.full(previous[2].shape, 255, dtype=np.uint8)
                mask[38:, 24:-24] = 0
            corners = cv2.goodFeaturesToTrack(previous[2], 32, .025, 6, mask=mask)
            if corners is not None and len(corners) >= 6:
                tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous[2], gray, corners, None)
                if tracked is not None and status is not None:
                    valid = status.ravel() == 1
                    if valid.sum() >= 6:
                        motion = np.linalg.norm((tracked - corners).reshape(-1, 2)[valid], axis=1)
                        quiet = float(np.quantile(motion, .75)) < .60
        self.motion_quiet = self.motion_quiet + stamp - previous[0] if quiet else 0.
        if self.phase == "raid_push" and MATCH_SECONDS - stamp <= 20.:
            if self._raid_line_advancing(obs["rgb"], stamp, yaw, quiet):
                return False
        return self.motion_quiet >= 1.2 - 1e-6

    def _give_way(self, direction, yaw, time):
        # Reverse the approach before turning or picking a different region.
        if self.phase.startswith("raid_"):
            if self.phase == "raid_rescan_shift":
                self.exit_retried = True
                self.exit_reason = "raid_followup_shift_blocked"
                phase = "raid_exit_yield"
            elif self.phase in ("raid_clearance", "raid_exit"):
                phase = "raid_exit_yield"
            else:
                self.exit_reason = "raid_blocked"
                phase = "raid_yield"
        elif self.phase.startswith("recover_"):
            self.exit_reason = "recovery_blocked"
            self.recover_resume = self.phase == "recover_transit" and not self.recover_retried
            phase = "recover_yield"
        else:
            self.failed_sweeps += 1
            self.exit_reason = "delivery_blocked" if self.phase == "deliver" else "transit_blocked"
            phase = "yield"
        self._move_to(phase, self.pos - .27 * direction, yaw, time)

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
        if not retry:
            self.failed_sweeps += 1
        self.retry_entry = retry
        self._move_to("clear", (self.pos[0], self.rollout_y - self.wing * .10), yaw, time)

    def _detour(self, time):
        self.deep = True
        lane = self.goal_y - .06 + .08 * self.sweep if self.sweep < 2 else self.delivery_lane
        self.entry = np.array([max(self.center_x + .64, self.entry[0] + .12), lane])
        self._move_to("detour", (self.entry[0], self.rollout_y - self.wing * .10), 0., time)

    def _opening_lane_step(self, frame, stamp, rotation):
        # Sample the edge of a confirmed high-value cluster without crossing
        # the pile to center the first push on its deepest piece.
        for point, value in self._valued_disks(frame, stamp, rotation, own=True):
            if (value == 10 and self.entry[1] + .08 < point[1] < self.goal_lane[1] and
                    self.center_x - .20 < point[0] < self.entry[0] - .21):
                return OPENING_LANE_STEP
        return 0.

    def _deliver(self, time):
        lane = self.goal_y - .04 + .08 * self.sweep if self.sweep < 2 else self.delivery_lane
        if self.sweep == 0:
            lane += self.opening_lane_step
        self.wall_frame = None
        self.wall_distance = None
        self.exit_reason = None
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
        if self.sweep >= 3:
            seen, offers = self.enemy_bank
            if time - seen < 18. and any(value == 10 for _, value in offers):
                if self._plan_raid(time, MATCH_SECONDS - time, minimum=max(0., self._known_recovery_value(time))):
                    return
            if self._plan_recovery(time):
                return
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
                gain = (1.5 if entry_x < self.center_x else 3.) / (1 + self.failed_sweeps)
                if self.recovery_tried and available <= 36.:
                    gain *= .55
                threat = self._home_exposure(time, budget) * .35
                own = self._utility(time, budget, own=gain, loss=threat, risk=.15)
                if self.sweep >= 3 and (self.failed_sweeps or self.recovery_tried):
                    if self._plan_raid(time, max_cost=available, minimum=max(0., own)):
                        return
                    if own <= 0.:
                        break
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
                    self._move_to("reposition", reposition, math.pi, time)
                return
        if (self.failed_sweeps or self.recovery_tried) and self._plan_raid(time, max_cost=available, minimum=0.):
            return
        self.phase = "done"

    def _plan_raid(self, time, max_cost, minimum=0.):
        goal = self.enemy_goal
        scout_x = goal[:, 0].mean() - .04
        clear_x = goal[0, 0] - .26
        candidates = []
        for wing, goal_y in zip((1, -1), goal[:, 1]):
            if wing in self.raid_entries:
                continue
            side_y = goal_y - wing * .28
            # Switch wings outside the mouth of the goal.
            clear = self.pos[0] > clear_x and abs(self.pos[1] - side_y) > .20
            lane_x = min(self.pos[0], clear_x) if clear else self.pos[0]
            travel = abs(self.pos[0] - lane_x) + abs(self.pos[1] - side_y) + abs(scout_x - lane_x)
            cost = travel / APPROACH_SPEED + (np.ptp(goal[:, 1]) + .37) / .16 + 3.5
            if cost > max_cost:
                continue
            stamp, pieces = self.enemy_bank
            denial = self._points(pieces) if time - stamp < 18. and pieces else 2.2
            risk = self._entry_risk(time, wing, side_y)
            value = self._utility(time, cost, deny=denial, loss=self._home_exposure(time, cost), risk=risk)
            if value > minimum:
                candidates.append((-value, cost, wing, side_y, scout_x, lane_x, clear))
        if not candidates:
            return False
        _, cost, wing, side_y, scout_x, lane_x, clear = min(candidates)
        self.raid_entries.append(wing)
        self.raid_short_target = None
        self.raid_wing = wing
        self.raid_scout = np.array([scout_x, side_y])
        self.raid_lane = np.array([lane_x, side_y])
        self.raid_deadline = min(MATCH_SECONDS - .2, time + cost + 3.)
        self.exit_reason = None
        if clear:
            self._move_to("raid_clear", (lane_x, self.pos[1]), wing * math.pi / 2, time)
        else:
            self._move_to("raid_lane", self.raid_lane, 0., time)
        return True

    def _colored_disks(self, rgb, rotation, red_side):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, _, boxes, _ = cv2.connectedComponentsWithStats(cv2.inRange(gray, 0, 82))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        for x, y, w, h, area in boxes[1:]:
            if w < 13 or h < 6 or area < .25 * w * h:
                continue
            point = self._floor_point(x + .5 * w, y + h - 1)
            if point is None or not .20 < point[0] < 1.25:
                continue
            width = w * (point[0] - self.camera_origin[0]) / self.intrinsics[0, 0]
            if not .031 < width < .15:
                continue
            glyph = hsv[y:y + max(3, h // 2), x + w // 5:x + 4 * w // 5]
            hue, sat, value = cv2.split(glyph)
            red = ((hue < 10) | (hue > 164)) & (sat > 120) & (value > 65)
            pale = (sat < 50) & (value > 143)
            matches = red.mean() > .035 if red_side else red.mean() < .025 and pale.mean() > .07
            if matches:
                yield self.pos + rotation @ (point[:2] + [.026, 0.])

    def _raid_targets(self, rgb, stamp, rotation):
        return [(p, value) for p, value in self._valued_disks(rgb, stamp, rotation, own=False)
                if np.all(p > self.enemy_goal[0] + .025) and np.all(p < self.enemy_goal[1] - .025)]

    def _raid_carries_residual(self, obs, rotation, aligned):
        stamp = obs["image_time_s"]
        if (not aligned or obs["rgb"] is None or stamp is None or
                obs["time_s"] - stamp > .11 or np.linalg.norm(self.target - self.pos) > .17):
            return False
        previous = self.raid_carry_observation
        seen = []
        confirmed_high = False
        for point, value in self._valued_disks(obs["rgb"], stamp, rotation, own=False):
            local = rotation.T @ (point - self.pos)
            front, lateral = local
            if .19 < front < .39 and abs(lateral) < .13:
                seen.append(local)
                confirmed_high |= value is not None and value >= 5
        if previous is None or stamp > previous[0]:
            supported = (previous is not None and .15 <= stamp - previous[0] <= .32 and
                         any(np.linalg.norm(at - new) < .045 for at in previous[1] for new in seen))
            self.raid_carry_observation = stamp, seen, supported
        else:
            supported = previous[2]
        # At close range the push face can hide the glyph. Match consecutive
        # detections in the car frame while retaining the original rank uncertainty.
        return confirmed_high or (self.raid_denied >= 5 and supported)

    def _raid_choice(self, time, targets):
        elapsed = MATCH_SECONDS - time
        if not targets:
            return None
        end = self.enemy_goal[1 if self.raid_wing > 0 else 0, 1] + .09 * self.raid_wing
        wing_risk = self._entry_risk(time, self.raid_wing, self.pos[1])
        options = []
        for p, _ in targets:
            lane = float(np.clip(p[0], self.enemy_goal[0, 0] + .10, self.field[0] - .15))
            path = abs(lane - self.pos[0]) / .15 + abs(end - self.pos[1]) / .16 + 1.
            if path > elapsed:
                continue
            swept = [(q, v) for q, v in targets if abs(q[0] - lane) < .125]
            denied = self._points(swept)
            utility = self._utility(time, path, deny=denied,
                                    loss=self._home_exposure(time, path), risk=wing_risk)
            options.append((utility, denied, -path, lane))
        return max(options) if options else None

    def _raid_direction(self, remaining, lane, targets):
        goal = self.enemy_goal
        goal_center_x = goal[:, 0].mean()
        # Larger batches already have a straight sweep and clearance route.
        if (len(self.raid_entries) != 1 or self.raid_denied >= 10 or
                lane < goal_center_x + .02 or remaining < 22. or
                not any(value is not None and value >= 3 for _, value in targets)):
            return 0.
        end_y = goal[1 if self.raid_wing > 0 else 0, 1] + self.raid_wing * .09
        delta_y = end_y - self.pos[1]
        if abs(delta_y) < .50:
            return 0.

        def expected_denial(offset):
            total = 0.
            for (x, y), value in targets:
                progress = np.clip((y - self.pos[1]) / delta_y, 0., 1.)
                miss = abs(x - (lane - offset * progress))
                capture = np.clip((.175 - miss) / .085, 0., 1.)
                finish_x = x - .85 * offset * (1. - progress)
                # A disk displaced toward midfield is harder to return with a side sweep.
                retention = .55 + .35 * np.clip((goal_center_x - finish_x) / .20, 0., 1.)
                total += (value or 1) * capture * retention
            return total

        offset = .14
        return offset if expected_denial(offset) > expected_denial(0.) + .20 else 0.

    def _plan_recovery(self, time, verified=False):
        if self.recovery_tried or time < 53. or (not verified and not self.failed_sweeps and self.pos[0] > .85):
            return False
        if self.pos[0] > 2.27 and not verified:
            return False
        y = self.pos[1]
        if verified:
            y = max(self.home_loose[1], key=lambda offer: offer[1] or 1)[0][1]
        wing = 1 if y >= self.goal_y else -1
        outer = self.field[1] - .20 if wing > 0 else .20
        scout_x = self.home_goal[1, 0] + .24
        approach = abs(self.pos[0] - scout_x) + abs(outer - self.pos[1])
        # Include the scan, turn, and short push. Reserve time to leave the goal.
        if MATCH_SECONDS - time < approach / .18 + 12.:
            return False
        self.recovery_tried = True
        self.recover_wing = wing
        self.recover_outer = outer
        self.recover_scout_x = scout_x
        self.recover_line = self.home_goal[1 if wing > 0 else 0, 1]
        self.recover_retried = False
        self.exit_reason = None
        self._move_to("recover_flank", (self.pos[0], outer), wing * math.pi / 2, time)
        return True

    def _finish_recovery(self, time):
        self.home_loose = (time, [])
        self._plan_sweep(time)

    def _known_recovery_value(self, time):
        seen, pieces = self.home_loose
        if self.recovery_tried or time - seen > 18. or not pieces:
            return 0.
        _, value = max(pieces, key=lambda offer: offer[1] or 1)
        path = max(0., self.pos[0] - self.goal_front) / APPROACH_SPEED + 10.
        return self._utility(time, path, own=value or 1, risk=.22)

    def _recovery_step(self, obs, rotation, reached, aligned, blocked, direction, yaw):
        time = obs["time_s"]
        elapsed = time - self.since
        wing = self.recover_wing
        heading = -wing * math.pi / 2
        end_y = self.recover_line + wing * .012
        if blocked:
            self._give_way(direction, yaw, time)
        elif self.phase == "recover_yield":
            if reached or elapsed >= 2.:
                if self.recover_resume and MATCH_SECONDS - time >= 31.:
                    self.recover_retried = True
                    self._move_to("recover_wait", self.pos, math.pi, time)
                else:
                    self._finish_recovery(time)
        elif self.phase == "recover_wait" and elapsed >= 2.5:
            self._move_to("recover_transit", (self.recover_scout_x, self.recover_outer), math.pi, time)
        elif self.phase == "recover_flank" and (reached or elapsed >= 4.5):
            self._move_to("recover_transit", (self.recover_scout_x, self.recover_outer), math.pi, time)
        elif self.phase == "recover_transit" and reached:
            self._observe("recover_scan", wrap(math.pi + wing * .42), time)
        elif self.phase == "recover_scan":
            frame = self._fresh_frame(obs, aligned)
            if frame is not None:
                targets = [(p, value) for p, value in self._valued_disks(
                               frame, obs["image_time_s"], rotation, own=True)
                           if .06 < p[0] < self.home_goal[1, 0] - .026 and
                           .015 < wing * (p[1] - self.recover_line) < .34]
                if targets and self.frames >= 2:
                    # Compare score and travel while leaving room behind the disk.
                    (x, y), _ = max(targets, key=lambda t: self._utility(
                        time, 5. + max(0., wing * (t[0][1] - self.recover_line)) / .14,
                        own=t[1] or 1, risk=.12))
                    self.recover_lane_x = float(np.clip(x + .025, .16, .30))
                    self._move_to("recover_stage", (self.pos[0], y + wing * .23), heading, time)
                    return
            if self.frames >= 4 or elapsed >= 3.:
                self._finish_recovery(time)
        elif self.phase == "recover_stage" and reached:
            self._move_to("recover_align", (self.recover_lane_x, self.pos[1]), heading, time)
        elif self.phase == "recover_align" and reached:
            if wing * (self.pos[1] - end_y) > .10:
                self._move_to("recover_push", (self.pos[0], end_y), heading, time)
            else:
                self._move_to("recover_backout", (self.pos[0], self.recover_line + wing * .32), yaw, time)
        elif self.phase == "recover_push" and (reached or elapsed >= 4.5):
            self._move_to("recover_backout", (self.pos[0], self.recover_line + wing * .32), yaw, time)
        elif self.phase == "recover_backout" and (reached or elapsed >= 2.8):
            self._finish_recovery(time)
        elif elapsed >= 10.:
            self._finish_recovery(time)

    def _raid_goal_edge(self, rgb, rotation, exit_edge=False):
        hue, saturation, value = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
        tint = ((hue <= 12) | (hue >= 169)) if self.reverse else (hue >= 103) & (hue <= 142)
        mask = tint & (saturation >= 33) & (saturation <= 130) & (value >= 85)
        floor = (saturation <= 38) & (value >= 95) & ~mask
        edge = np.zeros_like(mask)
        if exit_edge:
            edge[5:] = mask[5:] & ~mask[1:-4] & floor[:-5]
        else:
            edge[:-5] = mask[:-5] & ~mask[1:-4] & floor[5:]
        points = self._edge_points(edge, rotation)
        ahead = self.raid_wing * points[:, 1]
        points = points[(ahead > .10) & (ahead < .68)]
        return self._edge_coordinate(points, 1, .10 if exit_edge else .12)

    def _raid_line_advancing(self, rgb, stamp, yaw, quiet):
        # A carried batch can fill the flow mask. The painted line stays fixed
        # in the world and can corroborate progress independently of wheel slip.
        if not quiet or abs(wrap(self.heading - yaw)) >= .15:
            self.raid_edge_samples.clear()
            self.raid_line_progress = -math.inf
            return False
        c, s = math.cos(yaw), math.sin(yaw)
        offset = self._raid_goal_edge(rgb, np.array([[c, -s], [s, c]]), exit_edge=True)
        self.raid_edge_samples = [(t, p) for t, p in self.raid_edge_samples if stamp - t <= 1.05]
        if offset is not None:
            self.raid_edge_samples.append((stamp, offset * self.raid_wing))
        if offset is not None and len(self.raid_edge_samples) >= 4:
            times, distances = np.asarray(self.raid_edge_samples).T
            speed = -np.diff(distances) / np.diff(times)
            if np.count_nonzero((speed > .06) & (speed < .22)) > len(speed) / 2:
                self.raid_line_progress = stamp
        # Bridge one camera frame when the approaching line is occluded.
        return stamp - self.raid_line_progress <= .25

    def _raid_goal_side(self, rgb, rotation):
        offset = self._raid_goal_edge(rgb, rotation)
        if offset is not None:
            line = self.enemy_goal[0 if self.raid_wing > 0 else 1, 1]
            if abs(line - offset - self.pos[1]) < .38:
                self.pos[1] = line - offset

    def _short_raid_choice(self, time, targets):
        goal = self.enemy_goal
        options = []
        for point, value in targets:
            if value is None or value < 5:
                continue
            wing = 1 if point[1] > goal[:, 1].mean() else -1
            line = goal[1 if wing > 0 else 0, 1]
            if abs(point[1] - line) > .19:
                continue
            stage_y = point[1] - wing * .26
            lane = float(np.clip(point[0], goal[0, 0] + .08, self.field[0] - .17))
            travel = abs(stage_y - self.pos[1]) + abs(lane - self.pos[0])
            seconds = travel / .19 + abs(line + wing * .09 - stage_y) / .15 + 6.
            utility = self._utility(time, seconds, deny=value, risk=.17)
            options.append((utility, -seconds, lane, stage_y, wing, tuple(point)))
        return max(options, default=None)

    def _start_short_raid(self, time, choice):
        _, neg_seconds, lane, stage_y, wing, target = choice
        self.raid_short_target = np.array(target)
        self.raid_wing = wing
        self.raid_denied = 0
        self.raid_push_offset = 0.
        self.raid_deadline = min(MATCH_SECONDS - 4.2, time - neg_seconds - 3.)
        self.raid_scout = np.array([lane, stage_y])
        self.exit_reason = None
        self._move_to("raid_lane", (self.pos[0], stage_y), 0., time)

    def _recent_opponent_gap(self, time, target, origin=None):
        if self.opponent is None or time - self.opponent[0] > 2.:
            return math.inf
        _, point, velocity = self.opponent
        origin = self.pos if origin is None else origin
        obstacle = point + .6 * velocity - origin
        step = target - origin
        along = np.clip(obstacle @ step / max(.01, step @ step), 0., 1.)
        return np.linalg.norm(obstacle - along * step)

    def _try_rescan_vantage(self, time, vantage):
        step = vantage - self.pos
        if np.linalg.norm(step) < .20:
            return False
        look_seconds = np.linalg.norm(step) / .18 + .7
        entry_seconds = (self.enemy_goal[0, 0] + .08 - vantage[0]) / .19
        minimum_sweep = entry_seconds + .375 / .15 + 6.
        if (time + look_seconds > self.raid_rescan_until or
                MATCH_SECONDS - time < look_seconds + minimum_sweep or
                self._recent_opponent_gap(time, vantage) < .42):
            return False
        delta = self.enemy_goal.mean(axis=0) - vantage
        self._move_to("raid_rescan_shift", vantage, math.atan2(delta[1], delta[0]), time)
        return True

    def _finish_rescan(self, time, reason):
        self.exit_reason = reason
        center = self.enemy_goal.mean(axis=0)
        if not self.raid_rescan_shifted and not self.raid_rescan_advanced:
            wing = 1 if self.pos[1] <= center[1] else -1
            across = np.array([self.pos[0] - .08, center[1] + wing * .16])
            if self._try_rescan_vantage(time, across):
                self.raid_rescan_shifted = True
                return
        if not self.raid_rescan_advanced:
            # A closer view can resolve disks hidden behind the lip, while
            # staying outside the mouth until a profitable push is confirmed.
            line = self.enemy_goal[0, 0] - .33
            forward = np.array([line, self.pos[1]])
            if line > self.pos[0] and self._try_rescan_vantage(time, forward):
                self.raid_rescan_advanced = True
                return
        self.phase = "done"

    def _raid_rescan_step(self, obs, rotation, aligned):
        time = obs["time_s"]
        if time >= self.raid_rescan_until:
            self.exit_reason = "raid_followup_search_timeout"
            self.phase = "done"
            return
        frame = self._fresh_frame(obs, aligned)
        if frame is not None:
            occupied = any(np.linalg.norm(plate) < 1.08 and
                           (self.pos + rotation @ plate)[0] > self.enemy_goal[0, 0] - .24
                           for plate in self._plates(frame))
            targets = self._raid_targets(frame, obs["image_time_s"], rotation)
            short = self._short_raid_choice(time, targets)
            if short and short[0] > 0.:
                _, _, lane, stage_y, _, _ = short
                turn = np.array([self.pos[0], stage_y])
                stage = np.array([lane, stage_y])
                approach_gap = min(self._recent_opponent_gap(time, turn),
                                   self._recent_opponent_gap(time, stage, origin=turn))
                if not occupied or .43 <= approach_gap < math.inf:
                    recovery = self._known_recovery_value(time)
                    if recovery > short[0] and self._plan_recovery(time, verified=True):
                        return
                    self._start_short_raid(time, short)
                    return
            if occupied:
                # Briefly keep the defender and the scoring area in view.
                # A departing car can uncover a useful target in the next frames.
                if self.frames < 3 and time - self.since < 4.1 and MATCH_SECONDS - time >= 13.:
                    return
                self._finish_rescan(time, "raid_followup_occupied")
                return
            recovery = self._known_recovery_value(time)
            if recovery > 0. and self._plan_recovery(time, verified=True):
                return
            if any(value is not None and value >= 5 for _, value in targets):
                self.enemy_bank = time, targets
                if self._plan_raid(time, MATCH_SECONDS - time - 2., minimum=0.):
                    return
        if self.frames >= 3 or time - self.since >= 4.1:
            self._finish_rescan(time, "raid_followup_no_target")

    def _recheck_raid_entry(self, time, heading):
        exit_y = self.enemy_goal[1 if self.raid_wing > 0 else 0, 1] + self.raid_wing * .09
        # Back far enough to bring the near painted edge into the floor view.
        vantage = self.pos - np.array([0., self.raid_wing * .13])
        seconds = abs(exit_y - vantage[1]) / .16 + 6.0
        if (self.raid_entry_rechecked or len(self.raid_entries) < 2 or
                MATCH_SECONDS - time < seconds or self._recent_opponent_gap(time, vantage) < .42):
            return False
        self.raid_entry_rechecked = True
        self.raid_deadline = MATCH_SECONDS - 4.5
        center_x = self.enemy_goal[:, 0].mean()
        turn = -.23 if self.pos[0] < center_x else .23
        self._move_to("raid_scan_back", vantage, heading + self.raid_wing * turn, time)
        return True

    def _short_raid_scan(self, obs, rotation, aligned):
        time = obs["time_s"]
        frame = self._fresh_frame(obs, aligned)
        if frame is not None:
            gap, blocked = self._traffic_ahead(obs, np.array([1., 0.]))
            if blocked:
                self._give_way(rotation[:, 0], self.heading, time)
                return
            nearby = self._raid_targets(frame, obs["image_time_s"], rotation)
            if (self.frames >= 2 and not math.isfinite(gap) and any(
                    np.linalg.norm(point - self.raid_short_target) <= .11 for point, _ in nearby)):
                self._move_to("raid_align", self.pos, self.heading, time)
                return
        if self.frames >= 3 or time - self.since >= 3.3:
            reason = "raid_short_blocked" if math.isfinite(self.traffic_gap) else "raid_short_unconfirmed"
            self._raid_exit(time, reason)

    def _end_raid(self, time, reason):
        self.exit_reason = reason
        self.failed_sweeps = 0
        if reason == "raid_complete":
            seen, pieces = self.enemy_bank
            self.enemy_bank = seen, [(p, v) for p, v in pieces if abs(p[0] - self.raid_pushed_lane) > .15]
        # After a sweep, an unobserved short pickup at home is not worth
        # crossing the field again unless there is time for a full collection.
        home_budget = max(0., self.pos[0] - self.goal_front) / APPROACH_SPEED
        home_budget += self._return_seconds(self.center_x + .18) + 5.5
        if ((reason in ("raid_complete", "raid_no_target") or reason.startswith("raid_short_"))
                and MATCH_SECONDS - time < home_budget):
            if (reason == "raid_complete" and not self.raid_followup and len(self.raid_entries) == 1 and
                    MATCH_SECONDS - time >= 15. and self.pos[0] <= self.enemy_goal[0, 0] - .40):
                self.raid_followup = True
                self.raid_rescan_until = time + 9.
                delta = self.enemy_goal.mean(axis=0) - self.pos
                self._observe("raid_rescan", math.atan2(delta[1], delta[0]), time)
            else:
                self.phase = "done"
            return
        self._plan_sweep(time)

    def _raid_exit(self, time, reason, retry=False):
        self.exit_reason = reason
        goal = self.enemy_goal
        if reason == "raid_no_target":
            flank = -self.raid_wing
        else:
            flank = self.raid_wing
        line = goal[1 if flank > 0 else 0, 1]
        clearance = .09
        if self.phase == "raid_clearance":
            # Leave along the deeper path so the exit does not crowd the scoring line.
            clearance = max(clearance, flank * (self.pos[1] - line))
        target = np.array([goal[0, 0] - .60, line + flank * (.24 if retry else clearance)])
        if (reason == "raid_complete" and not retry and self.raid_short_target is None and
                len(self.raid_entries) == 1 and MATCH_SECONDS - time >= 20.):
            # Clear the mouth on the inside of the receiving flank. This
            # leaves the outer corridor open to a returning defender.
            inside = np.array([target[0], line - flank * .17])
            if self._recent_opponent_gap(time, inside) >= .42:
                target = inside
        if self.raid_short_target is not None and reason == "raid_complete" and not retry:
            # Leave outward from the short sweep. The defender may approach
            # from inside the mouth while we turn toward midfield.
            target = np.array([goal[0, 0] - .46, line + flank * .27])
        if retry and self.opponent is not None and time - self.opponent[0] <= 3.:
            _, p, v = self.opponent
            obstacle = p + .7 * v
            alternatives = (target, np.array([target[0], line - flank * .16]))
            def separation(point):
                step = point - self.pos
                along = np.clip((obstacle - self.pos) @ step / max(.01, step @ step), 0., 1.)
                return np.linalg.norm(obstacle - self.pos - along * step)
            target = max(alternatives, key=separation)
        delta = target - self.pos
        self.exit_retried = retry
        # Clear the scoring mouth before stopping, with the camera facing the escape corridor.
        self._move_to("raid_exit", target, math.atan2(delta[1], delta[0]), time)

    def _raid_step(self, obs, rotation, reached, aligned, blocked, direction, yaw):
        time = obs["time_s"]
        elapsed = time - self.since
        heading = self.raid_wing * math.pi / 2
        if blocked:
            self._give_way(direction, yaw, time)
        elif self.phase == "raid_backout":
            if reached:
                self._end_raid(time, "raid_complete")
            elif self._no_progress(obs, yaw) or elapsed >= 5.2:
                self.exit_reason = "raid_backout_incomplete"
                self.phase = "done"
        elif self.phase == "raid_scan_back":
            if reached:
                self._observe("raid_scan", self.heading, time)
            elif elapsed >= 1.7 or self._no_progress(obs, yaw):
                self._end_raid(time, "raid_no_target")
        elif self.phase == "raid_rescan":
            self._raid_rescan_step(obs, rotation, aligned)
        elif self.phase == "raid_rescan_shift":
            if reached:
                self._observe("raid_rescan", self.heading, time)
            elif elapsed >= 3.5 or time >= self.raid_rescan_until:
                self.phase = "done"
                self.exit_reason = "raid_followup_shift_timeout"
        elif self.phase == "raid_exit_yield" and (reached or elapsed >= 2.):
            if not self.exit_retried and MATCH_SECONDS - time >= 5.:
                self._raid_exit(time, self.exit_reason, retry=True)
            else:
                self.exit_reason = "raid_exit_blocked"
                self.phase = "done"
        elif self.phase == "raid_yield" and (reached or elapsed >= 2.):
            if self.raid_short_target is not None:
                self._raid_exit(time, "raid_short_blocked")
                return
            recovery = self._known_recovery_value(time)
            if not self._plan_raid(time, MATCH_SECONDS - time, minimum=max(0., recovery)):
                if recovery > 0. and self._plan_recovery(time, verified=True):
                    return
                self._end_raid(time, "raid_blocked")
        elif self.phase not in ("raid_yield", "raid_clearance", "raid_exit", "raid_exit_yield") and (
                time >= self.raid_deadline or elapsed >= 13.):
            if self.raid_short_target is not None:
                self._raid_exit(time, "raid_short_timeout")
            elif self.phase == "raid_push":
                self._give_way(rotation[:, 0], yaw, time)
                self.exit_reason = "raid_timeout"
            else:
                self._end_raid(time, "raid_timeout")
        elif self.phase == "raid_clear" and reached:
            self._move_to("raid_lane", self.raid_lane, 0., time)
        elif self.phase == "raid_lane" and reached:
            self._move_to("raid_transit", self.raid_scout, 0., time)
        elif self.phase == "raid_transit" and (reached or self.pos[0] >= self.raid_scout[0] + .03):
            self.raid_target_seen = False
            self._observe("raid_scan", heading, time)
        elif self.phase == "raid_scan":
            if self.raid_short_target is not None:
                self._short_raid_scan(obs, rotation, aligned)
                return
            frame = self._fresh_frame(obs, aligned)
            if frame is not None:
                self._raid_goal_side(frame, rotation)
                _, obstructed = self._traffic_ahead(obs, np.array([1., 0.]))
                if obstructed:
                    self._give_way(rotation[:, 0], yaw, time)
                    return
                targets = self._raid_targets(frame, obs["image_time_s"], rotation)
                self.raid_target_seen |= bool(targets)
                if targets:
                    self.enemy_bank = time, targets
                if (targets and self.frames >= 2 and not math.isfinite(self.traffic_gap) and
                        (self.raid_entry_rechecked or self.raid_deadline - time >= 7.)):
                    raid = self._raid_choice(time, targets)
                    home = self._known_recovery_value(time)
                    if home > max(0., raid[0] if raid else 0.) and self._plan_recovery(time, verified=True):
                        return
                    if (raid and raid[0] > 0. and
                            (not self.raid_entry_rechecked or MATCH_SECONDS - time >= -raid[2] + 4.5) and
                            (not self.raid_followup or any(
                            v is not None and v >= 5 and abs(p[0] - raid[3]) < .125
                            for p, v in targets))):
                        self.raid_denied = raid[1]
                        self.raid_push_offset = self._raid_direction(MATCH_SECONDS - time, raid[3], targets)
                        self._move_to("raid_align", (raid[3], self.pos[1]), heading, time)
                        return
            if self.frames >= 4 or elapsed >= (1.7 if self.raid_entry_rechecked else 4.):
                if not math.isfinite(self.traffic_gap) and self._recheck_raid_entry(time, heading):
                    return
                if (self.recovery_tried and self.frames >= 4 and not self.raid_target_seen and
                        not math.isfinite(self.traffic_gap) and MATCH_SECONDS - time < 29.):
                    self._raid_exit(time, "raid_no_target")
                elif not self._plan_raid(time, MATCH_SECONDS - time):
                    self._end_raid(time, "raid_no_target")
        elif self.phase == "raid_align" and reached:
            self.raid_carry_observation = None
            exit_y = self.enemy_goal[1 if self.raid_wing > 0 else 0, 1] + self.raid_wing * .09
            self.raid_pushed_lane = self.pos[0]
            end = (self.pos[0] - self.raid_push_offset, exit_y)
            delta = end - self.pos
            push_heading = math.atan2(delta[1], delta[0])
            self._move_to("raid_push", end, push_heading, time)
        elif self.phase == "raid_push":
            if reached and self.raid_push_offset > 0.:
                # Retrace the swept strip instead of crossing the receiving flank.
                forward = np.array([math.cos(self.heading), math.sin(self.heading)])
                self._move_to("raid_backout", self.pos - .80 * forward, self.heading, time)
                return
            remaining = MATCH_SECONDS - time
            can_clear = (self.raid_short_target is None and 9. <= remaining <= 22. and
                         not math.isfinite(self.traffic_gap))
            diagonal = (can_clear and remaining >= 11. and self.raid_denied < 10 and
                        self._raid_carries_residual(obs, rotation, aligned))
            if reached and can_clear and (self.raid_denied >= 10 or diagonal):
                self.exit_reason = "raid_complete"
                self.exit_retried = False
                if diagonal:
                    # Move the visible disk away from the goal mouth as well as past its side line.
                    self.raid_clearance_limit = 2.8
                    self._move_to("raid_clearance", self.pos + [-.19, self.raid_wing * .30],
                                  heading + self.raid_wing * .45, time)
                else:
                    self.raid_clearance_limit = 1.2
                    self._move_to("raid_clearance", self.target + [0., self.raid_wing * .16], heading, time)
            elif reached:
                self._raid_exit(time, "raid_complete")
            elif elapsed >= 8.:
                self._give_way(rotation[:, 0], yaw, time)
                self.exit_reason = "raid_timeout"
        elif self.phase == "raid_clearance" and (reached or elapsed >= self.raid_clearance_limit):
            self._raid_exit(time, "raid_complete")
        elif self.phase == "raid_exit" and (reached or elapsed >= 6.):
            self._end_raid(time, self.exit_reason or "raid_complete")

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
            self.exit_reason = None
            if not first_tick:
                self.failed_sweeps += 1
                self.opponent = None
                self.rank_stamp = None
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
        fresh = obs["rgb"] is not None and obs["image_time_s"] == time
        if (fresh and abs(gyro) < .30 and abs(wrap(self.heading - yaw)) < .13):
            if self.phase == "recover_flank":
                wing = self.recover_wing
                distance = self._end_wall_distance(obs["rgb"], rotation, axis=1, sign=wing)
                if distance is not None:
                    y = (self.field[1] if wing > 0 else 0.) - wing * distance
                    if abs(y - self.pos[1]) < .30:
                        self.pos[1] = y
            elif self.phase == "recover_transit":
                distance = self._end_wall_distance(obs["rgb"], rotation)
                if distance is not None and abs(distance - self.pos[0]) < .55:
                    self.pos[0] = distance
        if (self.phase == "raid_transit" and obs["rgb"] is not None and
                obs["image_time_s"] == time and abs(yaw) < .13 and abs(gyro) < .30):
            distance = self._end_wall_distance(obs["rgb"], -rotation)
            if distance is not None and abs(self.field[0] - distance - self.pos[0]) < .60:
                self.pos[0] = self.field[0] - distance
        if (self.sweep >= 2 and self.phase in ("check", "deliver", "retreat") and
                abs(wrap(math.pi - yaw)) < .13):
            self._correct_at_goal(obs, rotation)

        if fresh and self.sweep >= 1:
            landmarks = [p for p in self._plates(obs["rgb"]) if .17 < p[0] < .95]
            if landmarks:
                self._remember_opponent(min(landmarks, key=np.linalg.norm), rotation, time)
            if (self.phase in ("deliver", "retreat") and abs(wrap(math.pi - yaw)) < .14 and
                    .54 < self.pos[0] < 1.07):
                own = self._valued_disks(obs["rgb"], time, rotation, own=True)
                bank = [(p, v) for p, v in own if np.all(p > self.home_goal[0] + .032) and
                        np.all(p < self.home_goal[1] - .032)]
                if bank:
                    self.home_bank = time, self._points(bank)
                loose = [(p, v) for p, v in own if .06 < p[0] < self.goal_front - .02 and
                         0. < abs(p[1] - self.goal_y) - np.ptp(self.home_goal[:, 1]) / 2 < .32]
                if loose:
                    self.home_loose = time, loose

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
        gap, blocked = math.inf, False
        traffic_direction = self.target - self.pos
        if self.phase == "deliver":
            traffic_direction = np.array([-1., 0.])
        distance = np.linalg.norm(traffic_direction)
        if distance > .025:
            traffic_direction /= distance
        else:
            traffic_direction = rotation[:, 0]
        watched = self.sweep >= 2 and self.phase in WATCHED_TRAVEL
        if (watched and distance > .025 and
                (self.phase != "enter" or abs(error) < .40)):
            gap, blocked = self._traffic_ahead(obs, rotation.T @ traffic_direction)
        if remaining <= 0.:
            self.phase = "done"
        elif watched and not blocked and self._no_progress(obs, yaw):
            self._give_way(traffic_direction, yaw, time)
            self.exit_reason = "visual_stall"
        elif self.phase.startswith("raid_"):
            self._raid_step(obs, rotation, reached, aligned, blocked, traffic_direction, yaw)
        elif self.phase.startswith("recover_"):
            self._recovery_step(obs, rotation, reached, aligned, blocked, traffic_direction, yaw)
        elif blocked:
            self._give_way(traffic_direction, yaw, time)
        elif self.phase == "yield" and (reached or elapsed >= 2.):
            self._next_sweep(time)
        elif (self.phase in ("outside", "reposition", "approach", "detour", "look") and
              remaining < self._return_seconds(self.entry[0]) + 3.):
            self._next_sweep(time)
        elif self.phase in ("approach", "detour") and reached:
            self._look(time)
        elif self.phase == "look":
            frame = self._fresh_frame(obs, aligned)
            if frame is not None:
                self.blocked_frames += self._blocked(frame, rotation)
                if self.sweep == 0 and not self.deep:
                    self.opening_lane_step = (0. if self.blocked_frames else
                                              self._opening_lane_step(frame, obs["image_time_s"], rotation))
            if self.blocked_frames >= 2:
                if not self.deep and can_detour and not (self.sweep >= 2 and self.short):
                    self._detour(time)
                else:
                    self._leave_entry(yaw, time, retry=False)
            elif aligned and (self.frames >= 3 or elapsed >= 2.8):
                if self.sweep == 0:
                    self.entry[1] = min(self.goal_lane[1], self.entry[1] + self.opening_lane_step)
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
        elif self.phase == "deliver":
            if goal_distance <= self.target[0] + .034:
                self.exit_reason = "goal_edge"
                self.failed_sweeps = 0
                self.last_delivery = time
            elif guarded_return and goal_distance < .58 and abs(error) > .25:
                self.exit_reason = "goal_alignment"
            elif elapsed >= 12.:
                self.exit_reason = "delivery_timeout"
            if self.exit_reason is not None:
                if self.exit_reason != "goal_edge":
                    self.failed_sweeps += 1
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
            self.failed_sweeps += 1
            self._next_sweep(time)
        if self.phase == "done":
            return {"motor_pwm": [0, 0, 0, 0]}

        error = wrap(self.heading - yaw)
        p = 2. * (self.target - self.pos)
        speed, yaw_limit = .23, 1.4
        if self.phase in ("look", "check", "raid_scan", "raid_rescan", "recover_scan", "recover_wait") or (
                self.phase in ("enter", "reposition", "recover_flank", "recover_transit", "recover_stage", "raid_exit")
                and abs(error) > .40):
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
        elif self.phase in ("yield", "raid_yield", "recover_yield", "raid_exit_yield"):
            speed, yaw_limit = .18, .30
        elif self.phase in ("raid_push", "raid_clearance"):
            speed, yaw_limit = .16, .30
        elif self.phase == "raid_backout":
            speed, yaw_limit = .18, .30
        elif self.phase == "raid_scan_back":
            speed = .16
        elif self.phase == "raid_exit":
            speed, yaw_limit = .19, 1.
        elif self.phase in ("recover_push", "recover_backout"):
            speed, yaw_limit = .14, .30
        elif self.phase == "recover_align":
            speed = .15
        if gap < .32 and self.phase in WATCHED_TRAVEL:
            speed = min(speed, .09)
        p *= min(1., speed / max(speed, np.linalg.norm(p)))
        w = np.clip(3. * error - .18 * gyro, -yaw_limit, yaw_limit)
        self.drive_command = float(np.linalg.norm(p)), abs(float(w))
        raw = (WHEEL @ np.r_[rotation.T @ p, .11 * w]) * (237. / .32)
        pwm = np.where(abs(raw) >= 3., raw + 18 * np.sign(raw), 0.)
        return {"motor_pwm": np.rint(np.clip(pwm, -255, 255)).astype(int).tolist()}
