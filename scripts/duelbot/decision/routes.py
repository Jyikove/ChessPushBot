import math

import numpy as np

from ..settings import APPROACH_SPEED, MATCH_SECONDS


class RouteSelection:
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

    @staticmethod
    def _points(offers):
        return sum(1 if value is None else value for _, value in offers)

    def _raid_choice(self, time, targets):
        elapsed = MATCH_SECONDS - time
        if not targets:
            return None
        end = self.enemy_goal[1 if self.raid_wing > 0 else 0, 1] + .09 * self.raid_wing
        wing_risk = self._entry_risk(time, self.raid_wing, self.pos[1])
        front = self.enemy_goal[0, 0] + .10
        rear = self.field[0] - .15
        # Score whole blade passes at several depths, including between disks.
        lanes = [front, (front + rear) / 2, rear]
        lanes.extend(float(np.clip(p[0], front, rear)) for p, _ in targets)
        options = []
        for lane in lanes:
            swept = [(q, v) for q, v in targets if abs(q[0] - lane) < .125]
            if not swept:
                continue
            turn = (lane, self.pos[1])
            if min(self._recent_opponent_gap(time, turn),
                   self._recent_opponent_gap(time, (lane, end), origin=np.asarray(turn))) < .52:
                continue
            path = abs(lane - self.pos[0]) / .15 + abs(end - self.pos[1]) / .16 + 1.
            if path > elapsed:
                continue
            denied = self._points(swept)
            # The center of the plate carries more reliably than its tips.
            weighted = sum(self._points([(q, v)]) * (1. - .45 * abs(q[0] - lane) / .125)
                           for q, v in swept)
            utility = self._utility(time, path, deny=weighted,
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

    def _known_recovery_value(self, time):
        seen, pieces = self.home_loose
        if self.recovery_tried or time - seen > 18. or not pieces:
            return 0.
        _, value = max(pieces, key=lambda offer: offer[1] or 1)
        path = max(0., self.pos[0] - self.goal_front) / APPROACH_SPEED + 10.
        return self._utility(time, path, own=value or 1, risk=.22)

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

    def _recent_opponent_gap(self, time, target, origin=None):
        if self.opponent is None or time - self.opponent[0] > 2.:
            return math.inf
        _, point, velocity = self.opponent
        origin = self.pos if origin is None else origin
        obstacle = point + .6 * velocity - origin
        step = target - origin
        along = np.clip(obstacle @ step / max(.01, step @ step), 0., 1.)
        return np.linalg.norm(obstacle - along * step)

    def _rail_choice(self, time, targets, heading):
        if not targets:
            return None
        lane = self.field[0] - .115
        wing = self.raid_wing
        turn = np.array([lane, self.pos[1]])
        finish = np.array([lane, self.enemy_goal[1 if wing > 0 else 0, 1] + wing * .17])
        if min(self._recent_opponent_gap(time, turn),
               self._recent_opponent_gap(time, finish, origin=turn)) < .52:
            return None
        seconds = abs(lane - self.pos[0]) / .15 + abs(finish[1] - self.pos[1]) / .16 + 2.
        value = self._utility(time, seconds + 3., deny=self._points(targets),
                              risk=self._entry_risk(time, wing, self.pos[1]))
        if value > 0. and time + seconds < self.raid_deadline:
            return value, lane, heading - wing * .11
        return None
