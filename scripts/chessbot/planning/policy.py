"""Image-relative reactive policy. Inputs contain no scene/world position."""
from collections import deque
import math

import numpy as np

from ..decision.targets import choose_outbound, choose_return, return_pickup_travel_s

SCORE_ENTRY_DISTANCE_M = .30


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Impact:
    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = deque()
        self.filtered = 0.
        self.previous = None
        self.negative_since = None
        self.longest = 0.
        self.velocity_delta = 0.

    def feed(self, t, value):
        previous = self.previous
        self.previous = t
        if previous is None or t <= previous:
            return False
        dt = t - previous
        self.filtered += (-math.expm1(-dt / .025)) * (value - self.filtered)
        self.samples.append((t, self.filtered * dt))
        while self.samples and self.samples[0][0] < t - self.cfg["drop_window_s"]:
            self.samples.popleft()
        if self.filtered <= -self.cfg["negative_m_s2"]:
            if self.negative_since is None:
                self.negative_since = previous
        else:
            self.negative_since = None
        duration = 0. if self.negative_since is None else t - self.negative_since
        self.longest = max(self.longest, duration)
        self.velocity_delta = sum(value for _, value in self.samples)
        return (duration >= self.cfg["negative_duration_s"]
                and self.velocity_delta <= -self.cfg["velocity_drop_m_s"])


class Reactive:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = "INITIALIZE"
        self.enter = 0.
        self.zero = None
        self.opening_heading = math.radians(cfg["opening_search_heading_deg"])
        self.captured_count = 0
        self.captured_total = 0
        self.round = 1
        self.target = None
        self.last_target = None
        self.last_seen = None
        self.first_missing = None
        self.processed = []
        self.next_id = 0
        self.events = []
        self.heading_stable = None
        self.push_heading = -math.pi
        self.impact = Impact(cfg["collision"])
        self.last_cmd = (0., 0., 0.)
        self.capture_armed = None
        self.cluster_samples = deque()
        self.cycle_completed = 0
        self.start_hint_until = -1.
        self.return_pickup_deadline = 0.
        self.fence_until = -1.
        self.fence_direction = 0.
        self.stationary_since = None
        self.restarting = False
        self.escape_command = (0., 0., 0.)

    def evidence(self, o):
        return dict(yaw_deg=math.degrees(wrap(o.yaw - self.zero)),
                    detections=len(o.pieces), target=self.target,
                    piece_box=self.last_target.box if self.last_target else None,
                    target_distance_m=self.last_target.distance_m if self.last_target else None,
                    last_seen_s=self.last_seen, first_missing_s=self.first_missing,
                    capture_armed_s=self.capture_armed,
                    captured_count=self.captured_count, round=self.round,
                    goal=None if o.zone is None else vars(o.zone),
                    start_zone=None if o.start_zone is None else vars(o.start_zone))

    def event(self, kind, o, reason, **extra):
        self.events.append(dict(t=o.time, kind=kind, state=self.state, reason=reason,
                                **self.evidence(o), **extra))

    def transition(self, state, o, reason):
        self.event("transition", o, reason, previous=self.state, next=state,
                   dwell_s=o.time - self.enter)
        self.state, self.enter = state, o.time
        self.heading_stable = None

    def fail(self, time, message):
        self.events.append(dict(t=time, kind="transition", previous=self.state, next="SF",
                                reason=message, dwell_s=time-self.enter))
        self.state, self.enter = "SF", time

    def match(self, o):
        if self.last_target is None or not o.pieces:
            return None
        last = self.last_target
        matches = sorted(o.pieces, key=lambda p: math.hypot(p.x-last.x, p.bottom-last.bottom))
        best = matches[0]
        width = last.box[2]-last.box[0]
        return best if (abs(best.x-last.x) < max(.09, width) and
                        abs(best.bottom-last.bottom) < max(.075, width*1.1)) else None

    def clear_target(self):
        self.target = self.last_target = self.last_seen = self.first_missing = self.capture_armed = None
        self.cluster_samples.clear()

    def arm_capture(self, t):
        if self.capture_armed is None:
            self.capture_armed = t
        self.first_missing = None
        self.cluster_samples.clear()

    def visible_cluster(self, o):
        cfg = self.cfg["visible_cluster_capture"]
        samples = self.cluster_samples
        if samples and o.time-samples[-1][0] > cfg["max_gap_s"]:
            samples.clear()
        if (self.last_cmd[0] <= .06 or o.image_motion_px is None
                or o.image_motion_px < cfg["min_image_motion_px"]):
            return None
        points = [p.body_xy_m for p in o.pieces if p.body_xy_m is not None
                  and 0. < p.body_xy_m[0] <= cfg["cluster_forward_m"]
                  and abs(p.body_xy_m[1]) <= cfg["cluster_half_width_m"]]
        contacts = [p[0] for p in points if p[0] <= cfg["contact_forward_m"]
                    and abs(p[1]) <= cfg["contact_half_width_m"]
                    and any(math.dist(p, q) >= cfg["distinct_center_m"] for q in points)]
        if not contacts:
            return None
        samples.append((o.time, min(contacts)))
        while len(samples) > 1 and o.time-samples[1][0] >= cfg["hold_s"]-1e-9:
            samples.popleft()
        span = max(x for _, x in samples)-min(x for _, x in samples)
        held = o.time-samples[0][0]
        if held >= cfg["hold_s"]-1e-9 and span <= cfg["max_forward_span_m"]:
            return dict(cluster_hold_s=held, cluster_forward_span_m=span,
                        cluster_frame_count=len(samples))
        return None

    def finish_capture(self, o, returning, home_w, reason, declared, **evidence):
        if declared:
            self.captured_count += 1
            self.captured_total += 1
        self.processed.append((self.target, o.time, self.last_target.x, self.last_target.bottom))
        self.event("capture_judgment", o, reason, declared=declared,
                   collection_phase="return" if returning else "outbound", **evidence)
        if returning:
            self.transition("FIND_SCORE_ZONE", o,
                            "return_pickup_confirmed" if declared else "return_verify_timeout")
            command = self.return_pickup_command(None, home_w)
        else:
            self.transition("TURN_HOME", o, "batch_capacity" if declared else "outbound_verify_timeout")
            command = (0., 0., 0.)
        self.clear_target()
        return command

    def return_pickup_command(self, piece, heading_command):
        cfg = self.cfg["return_pickup"]
        lateral = piece.body_xy_m[1] if piece is not None and piece.body_xy_m is not None else 0.
        limit = cfg["lateral_speed_m_s"]
        correction = float(np.clip(1.4*lateral, -limit, limit))
        return cfg["west_speed_m_s"], correction, heading_command

    def select_target(self, o, piece, reason):
        previous = self.target
        self.target = self.next_id
        self.next_id += 1
        self.last_target, self.last_seen = piece, o.time
        self.capture_armed = self.first_missing = None
        self.event("target_selected", o, reason, previous_target=previous)

    def should_switch_target(self, candidate, current):
        if candidate is None or candidate is current:
            return False
        if current is None:
            return True
        if candidate.distance_m is None:
            return False
        return (current.distance_m is None or
                candidate.distance_m > current.distance_m+self.cfg["far_target_switch_margin_m"])

    def turn(self, o, yaw, target):
        error = wrap(target - yaw)
        tolerance = math.radians(self.cfg["yaw_tolerance_deg"])
        if abs(error) <= tolerance and abs(o.yaw_rate) < .22:
            if self.heading_stable is None:
                self.heading_stable = o.time
            return 0., o.time-self.heading_stable >= self.cfg["yaw_stable_s"]
        self.heading_stable = None
        limit = self.cfg["pivot_max_yaw_rate_rad_s"]
        return self.heading_command(error, o.yaw_rate, limit), False

    @staticmethod
    def heading_command(error, rate, limit=2.4):
        if abs(error) < .035:
            return 0.
        return float(np.clip(2.8*error - .22*rate, -limit, limit))

    def step(self, o, *, allow_stall=True):
        if allow_stall and self.stalled(o):
            cfg = self.cfg["stall"]
            self.event("stall", o, "commanded_motion_but_stationary",
                       stationary_s=o.time-self.stationary_since,
                       horizontal_accel_rms=o.horizontal_accel_rms,
                       gyro_rad_s=o.yaw_rate,
                       abandoned_command=self.last_cmd)
            speed = cfg["escape_speed_m_s"]
            vx, vy, _ = self.last_cmd
            planar_speed = math.hypot(vx, vy)
            self.escape_command = ((-speed*vx/planar_speed, -speed*vy/planar_speed, 0.)
                                   if planar_speed > .06 else (-speed, 0., 0.))
            self.restarting = True
            self.captured_count = 0
            self.clear_target()
            self.processed.clear()
            self.start_hint_until = self.fence_until = -1.
            self.stationary_since = None
            self.transition("STALL_ESCAPE", o, "round_aborted")
            command = self.escape_command
        else:
            command = self._step(o)
        self.last_cmd = command
        return command

    def stalled(self, o):
        cfg = self.cfg["stall"]
        vx, vy, wz = self.last_cmd
        stationary = (
            self.state not in ("INITIALIZE", "STALL_ESCAPE", "SF")
            and all(math.isfinite(v) for v in (o.time, o.yaw, o.yaw_rate))
            and (math.hypot(vx, vy) > .06 or abs(wz) > .12)
            and o.horizontal_accel_rms is not None
            and o.horizontal_accel_rms <= cfg["max_horizontal_accel_rms_m_s2"]
            and abs(o.yaw_rate) <= cfg["max_yaw_rate_rad_s"])
        if not stationary:
            self.stationary_since = None
            return False
        if self.stationary_since is None:
            self.stationary_since = o.time
        return o.time-self.stationary_since >= cfg["stationary_s"]-1e-9

    def fence_guidance(self, o):
        if o.zone is not None:
            self.fence_until = -1.
            return None
        cfg = self.cfg["fence"]
        if o.fence is not None and abs(o.fence.lateral_m) <= cfg["side_trigger_m"]:
            direction = -math.copysign(1., o.fence.lateral_m)
            if o.time > self.fence_until or direction != self.fence_direction:
                self.event("fence_guidance", o, "score_zone_hidden_near_side_wall",
                           fence=vars(o.fence), inward_body_y=direction)
            self.fence_direction = direction
            self.fence_until = o.time+cfg["hold_s"]
            self.start_hint_until = -1.
        if o.time <= self.fence_until:
            return cfg["forward_speed_m_s"], self.fence_direction*cfg["inward_speed_m_s"]
        return None

    def _step(self, o):
        if self.state == "SF":
            return 0., 0., 0.
        if not all(math.isfinite(v) for v in (o.time, o.yaw, o.yaw_rate)):
            self.fail(o.time, "nonfinite_sensor")
            return 0., 0., 0.
        if self.state == "INITIALIZE":
            self.zero = o.yaw
            self.transition("TURN_CENTER", o, "opening_turn_to_center")
        yaw = wrap(o.yaw - self.zero)
        search_heading = self.opening_heading if self.round == 1 else 0.
        search_w = self.heading_command(wrap(search_heading-yaw), o.yaw_rate)
        home_w = self.heading_command(wrap(-math.pi-yaw), o.yaw_rate)
        entry_priority = (o.zone is not None
                          and o.zone.forward_m <= self.cfg["return_pickup"]["score_entry_priority_m"])
        if self.state == "TURN_CENTER":
            w, settled = self.turn(o, yaw, self.opening_heading)
            if settled:
                self.transition("SEARCH_PIECE", o, "opening_heading_stable")
            return 0., 0., w
        if self.state == "SEARCH_PIECE":
            piece = choose_outbound(o, self.processed)
            if piece is not None:
                self.select_target(o, piece, "farthest_visible_piece")
                self.transition("APPROACH_PIECE", o, "target_visible")
            return (.15 if self.state == "SEARCH_PIECE" else 0.), 0., search_w
        if self.state == "APPROACH_PIECE":
            piece = self.match(o)
            candidate = choose_outbound(o, self.processed)
            if self.should_switch_target(candidate, piece):
                reason = "farther_piece_appeared" if piece is not None else "lost_target_reselected"
                self.select_target(o, candidate, reason)
                piece = candidate
            if piece is None:
                if o.time-self.last_seen >= self.cfg["target_lost_s"]:
                    self.event("capture_judgment", o, "lost_before_capture_region", declared=False)
                    self.target = self.last_target = None
                    self.transition("SEARCH_PIECE", o, "target_lost_before_region")
                return .08, 0., search_w
            self.last_seen, self.last_target = o.time, piece
            x_error = self.cfg["piece_aim_x_fraction"]-piece.x
            if (piece.bottom >= self.cfg["capture_bottom_fraction"]
                    and abs(x_error) < self.cfg["capture_horizontal_fraction"]):
                self.arm_capture(o.time)
                self.transition("VERIFY_CAPTURE", o, "piece_in_lower_capture_region")
            speed = float(np.clip(.30-.18*max(0., piece.bottom-.32)
                                  -.27*abs(x_error), .09, .30))
            return speed, float(np.clip(.64*x_error, -.26, .26)), search_w
        if self.state in ("RETURN_COLLECT_PIECE", "RETURN_VERIFY_CAPTURE") and entry_priority:
            self.processed.append((self.target, o.time, self.last_target.x, self.last_target.bottom))
            self.transition("FIND_SCORE_ZONE", o, "score_entry_priority")
            self.clear_target()
            return .08, 0., home_w
        if self.state in ("VERIFY_CAPTURE", "RETURN_VERIFY_CAPTURE"):
            returning = self.state == "RETURN_VERIFY_CAPTURE"
            heading_command = home_w if returning else search_w
            cluster = self.visible_cluster(o)
            if cluster is not None:
                return self.finish_capture(o, returning, home_w, "visible_cluster_moving_with_pusher",
                                           True, **cluster)
            piece = self.match(o)
            expired = o.time-self.capture_armed >= self.cfg["capture_verify_timeout_s"]
            if piece is not None:
                self.last_seen, self.last_target = o.time, piece
                if expired:
                    return self.finish_capture(o, returning, home_w, "capture_unconfirmed_timeout", False)
                if self.first_missing is not None:
                    self.event("capture_judgment", o, "piece_reappeared", declared=False)
                    self.first_missing = None
                    approach = "RETURN_COLLECT_PIECE" if returning else "APPROACH_PIECE"
                    self.transition(approach, o, "track_reacquired")
                if returning and piece.body_xy_m is not None:
                    return .14, float(np.clip(.9*piece.body_xy_m[1], -.12, .12)), home_w
                error = self.cfg["piece_aim_x_fraction"] - piece.x
                return .14, float(np.clip(.45*error, -.20, .20)), heading_command
            if self.first_missing is None:
                self.first_missing = o.time
            missing_s = o.time - self.first_missing
            if missing_s >= self.cfg["capture_disappearance_s"]-1e-9:
                return self.finish_capture(o, returning, home_w, "disappeared_after_lower_region",
                                           True, missing_s=missing_s)
            return .13, 0., heading_command
        if self.state == "TURN_HOME":
            error = wrap(-math.pi - yaw)
            tolerance = math.radians(self.cfg["turn_home_tolerance_deg"])
            if abs(error) <= tolerance + 1e-9:
                self.transition("FIND_SCORE_ZONE", o, "home_heading_in_tolerance")
                return 0., 0., 0.
            w = self.heading_command(error, o.yaw_rate, self.cfg["pivot_max_yaw_rate_rad_s"])
            return 0., 0., w
        if self.state in ("FIND_SCORE_ZONE", "RETURN_COLLECT_PIECE", "RETURN_VERIFY_CAPTURE",
                          "ALIGN_SCORE_ZONE"):
            guidance = self.fence_guidance(o)
            if guidance is not None:
                if self.state != "FIND_SCORE_ZONE":
                    self.transition("FIND_SCORE_ZONE", o, "side_wall_priority")
                    self.clear_target()
                return *guidance, home_w
        if self.state == "FIND_SCORE_ZONE":
            if not entry_priority:
                piece = choose_return(o, self.processed, self.cfg["return_pickup"])
                if piece is not None:
                    self.select_target(o, piece, "return_piece_ahead")
                    self.return_pickup_deadline = (o.time+self.cfg["return_pickup"]["timeout_s"]
                                                   +return_pickup_travel_s(piece, self.cfg["return_pickup"]))
                    self.transition("RETURN_COLLECT_PIECE", o, "align_on_east_side")
                    return self.return_pickup_command(piece, home_w)
            if o.zone:
                self.start_hint_until = -1.
                self.event("zone_recognition", o, "blue_entry_detected")
                self.transition("ALIGN_SCORE_ZONE", o, "blue_entry_detected")
                return .22, 0., home_w
            if o.start_zone:
                if o.time > self.start_hint_until:
                    self.event("start_zone_hint", o, "search_right_of_start_zone")
                self.start_hint_until = o.time+self.cfg["start_zone_hint_s"]
            if o.time <= self.start_hint_until:
                return .08, -self.cfg["start_zone_right_m_s"], home_w
            return .22, 0., home_w
        if self.state == "RETURN_COLLECT_PIECE":
            cfg = self.cfg["return_pickup"]
            piece = self.match(o)
            lost = piece is None and o.time-self.last_seen >= self.cfg["target_lost_s"]
            if lost or o.time >= self.return_pickup_deadline:
                self.processed.append((self.target, o.time, self.last_target.x, self.last_target.bottom))
                self.transition("FIND_SCORE_ZONE", o, "return_pickup_abandoned")
                self.clear_target()
                return .08, 0., home_w
            if piece is None or piece.body_xy_m is None:
                return self.return_pickup_command(None, home_w)
            self.last_seen, self.last_target = o.time, piece
            lateral = piece.body_xy_m[1]
            if abs(lateral) <= cfg["alignment_m"] and piece.bottom >= self.cfg["capture_bottom_fraction"]:
                self.arm_capture(o.time)
                self.transition("RETURN_VERIFY_CAPTURE", o, "return_piece_in_lower_capture_region")
            return self.return_pickup_command(piece, home_w)
        if self.state == "ALIGN_SCORE_ZONE":
            if not o.zone:
                self.transition("FIND_SCORE_ZONE", o, "entry_no_longer_visible")
                return .17, 0., home_w
            lateral = o.zone.lateral_m
            heading_offset = float(np.clip(
                .22*math.atan2(lateral, max(.25, o.zone.forward_m)), -.12, .12))
            desired = wrap(-math.pi + heading_offset)
            w = self.heading_command(wrap(desired-yaw), o.yaw_rate)
            aligned = (abs(lateral) <= self.cfg["entry_lateral_tolerance_m"]
                       and abs(wrap(-math.pi-yaw)) < math.radians(6))
            if aligned and o.zone.forward_m <= SCORE_ENTRY_DISTANCE_M:
                self.push_heading = -math.pi
                self.impact = Impact(self.cfg["collision"])
                self.transition("PUSH_PIECES", o, "entry_corridor_aligned")
                return .19, 0., home_w
            # Align laterally before advancing out of sight of the entrance line.
            advance = .16 if o.zone.forward_m > .44 or aligned else 0.
            return advance, float(np.clip(1.1*lateral, -.26, .26)), w
        if self.state == "PUSH_PIECES":
            for t, accel in o.imu:
                hit = self.impact.feed(t, accel)
                if (hit and self.last_cmd[0] > 0 and
                        t-self.enter >= self.cfg["collision"]["armed_after_command_s"]):
                    self.event("collision", o, "longitudinal_deceleration_and_velocity_drop",
                               sample_t=t, filtered_m_s2=self.impact.filtered,
                               delta_velocity_m_s=self.impact.velocity_delta,
                               negative_duration_s=t-self.impact.negative_since)
                    self.transition("RETREAT", o, "forward_obstruction_confirmed")
                    return 0., 0., home_w
            return .19, 0., self.heading_command(wrap(self.push_heading-yaw), o.yaw_rate)
        if self.state == "RETREAT":
            if o.zone:
                # Blue front boundary and in-zone pieces must clear the pusher sweep.
                distance = o.zone.forward_m
                nearby_scored = [p for p in o.pieces if p.bottom < o.zone.entry_px[1]+.035
                                 and abs(p.x-o.zone.entry_px[0]) < .20]
                if (distance >= self.cfg["retreat_clearance_m"] and
                        all(p.bottom < .78 for p in nearby_scored)):
                    self.transition("TURN_SEARCH", o, "visual_pusher_clearance")
                    return 0., 0., home_w
            if o.time-self.enter >= self.cfg["retreat_timeout_s"]:
                self.transition("TURN_SEARCH", o, "retreat_timeout")
                return 0., 0., home_w
            return -.22, 0., home_w
        if self.state == "STALL_ESCAPE":
            if o.time-self.enter >= self.cfg["stall"]["escape_s"]:
                self.transition("TURN_SEARCH", o, "escape_manoeuvre_finished")
                return 0., 0., 0.
            return self.escape_command
        if self.state == "TURN_SEARCH":
            w, settled = self.turn(o, yaw, 0.)
            if settled:
                reason = "stall_new_round" if self.restarting else "search_heading_recovered"
                if not self.restarting:
                    self.cycle_completed += 1
                self.restarting = False
                self.round += 1
                self.captured_count = 0
                self.transition("SEARCH_PIECE", o, reason)
            return 0., 0., w
        raise RuntimeError(self.state)
