"""Confirm current detections and smooth their on-screen box coordinates."""
from dataclasses import dataclass, replace
import math

import numpy as np


@dataclass(frozen=True)
class Track:
    raw: tuple
    display: tuple
    hits: int


class BoxTracking:
    def __init__(self, *, min_hits, match_iou, current_weight, max_gap_s):
        if (type(min_hits) is not int or min_hits < 1
                or not 0 < match_iou <= 1 or not 0 < current_weight <= 1
                or not math.isfinite(max_gap_s) or max_gap_s <= 0):
            raise ValueError("invalid detector tracking settings")
        self.min_hits = min_hits
        self.match_iou = match_iou
        self.weight = current_weight
        self.max_gap_s = max_gap_s
        self.previous = []
        self.timestamp = None

    def matches(self, pieces):
        if not pieces or not self.previous:
            return {}
        old = np.asarray([track.raw for track in self.previous])
        new = np.asarray([piece.box for piece in pieces])
        lo = np.maximum(old[:, None, :2], new[None, :, :2])
        hi = np.minimum(old[:, None, 2:], new[None, :, 2:])
        intersection = np.maximum(hi - lo, 0.).prod(axis=2)
        union = ((old[:, 2:] - old[:, :2]).prod(axis=1)[:, None]
                 + (new[:, 2:] - new[:, :2]).prod(axis=1)[None, :] - intersection)
        iou = intersection / np.maximum(union, 1e-9)
        rows, cols = np.nonzero(iou >= self.match_iou)
        used, matched = set(), {}
        for order in np.argsort(-iou[rows, cols], kind="stable"):
            row, col = rows[order], cols[order]
            if row not in used and col not in matched:
                used.add(row)
                matched[col] = self.previous[row]
        return matched

    def update(self, pieces, timestamp=None):
        if (timestamp is not None and self.timestamp is not None
                and not 0 < timestamp - self.timestamp <= self.max_gap_s):
            self.previous = []
        self.timestamp = timestamp
        matches = self.matches(pieces)
        tracks, confirmed = [], []
        for index, piece in enumerate(pieces):
            old = matches.get(index)
            if old is None:
                current = Track(piece.box, piece.box, 1)
            else:
                display = tuple(self.weight * new + (1 - self.weight) * prev
                                for new, prev in zip(piece.box, old.display, strict=True))
                current = Track(piece.box, display, min(old.hits + 1, self.min_hits))
            tracks.append(current)
            if current.hits >= self.min_hits:
                confirmed.append(replace(piece, display_box=current.display))
        # Only detections seen on this frame are eligible for the next one.
        self.previous = tracks
        return tuple(confirmed)
