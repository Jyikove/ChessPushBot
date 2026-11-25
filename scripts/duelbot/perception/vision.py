import cv2
import numpy as np

from .piece_values import PieceValues


class Vision:
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

    def _rail_targets(self, rgb, time, rotation):
        lo, hi = self.enemy_goal
        # Rail-side projections can extend slightly behind the visible end wall.
        return [(p, v) for p, v in self._valued_disks(rgb, time, rotation, own=False)
                if hi[0] - .135 < p[0] < hi[0] + .045 and lo[1] + .006 < p[1] < hi[1] - .006]

    def _rail_guard(self, rgb):
        return any(.16 < forward < 1.23 and abs(lateral) < .48
                   for forward, lateral in self._plates(rgb))
