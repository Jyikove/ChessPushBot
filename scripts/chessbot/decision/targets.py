"""Select a visible piece for the outbound or return leg."""
import math


def eligible_pieces(obs, processed):
    return [piece for piece in obs.pieces if not any(
        obs.time-t < 2.0 and abs(piece.x-x) < .11 and piece.bottom > y-.07
        for _, t, x, y in processed)]


def choose_outbound(obs, processed):
    eligible = eligible_pieces(obs, processed)
    return max(eligible, default=None, key=lambda piece: (
        piece.distance_m if piece.distance_m is not None else -math.inf,
        piece.bottom, piece.confidence))


def choose_return(obs, processed, cfg):
    # Pass over candidates beyond the visible score-zone entrance.
    upper = obs.zone.forward_m-.05 if obs.zone is not None else math.inf
    candidates = [piece for piece in eligible_pieces(obs, processed)
                  if piece.body_xy_m is not None and cfg["minimum_forward_m"] <= piece.body_xy_m[0] < upper]
    return min(candidates, default=None, key=lambda piece: return_pickup_travel_s(piece, cfg))


def return_pickup_travel_s(piece, cfg):
    forward, lateral = piece.body_xy_m
    return max((forward-cfg["minimum_forward_m"])/cfg["west_speed_m_s"],
               abs(lateral)/cfg["lateral_speed_m_s"])
