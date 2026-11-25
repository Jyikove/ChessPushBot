"""Competition scoring from terminal geometry and floor contact."""
import numpy as np
from field_spec import FIELD, FIELD_WIDTH_M, bounds
from piece_spec import HALF_HEIGHT_M, PIECES, RADIUS_M

# The field drawing centers its strokes on the rectangle edges, with a
# separate 4 mm image border at the two field ends.
_strokes = FIELD["line_widths_mm"]
_goal_line_m = _strokes["goal_outline"] / 2000
_outer_line_m = _strokes["outer_border"] / 1000
GOAL_INTERIORS = np.asarray([
    (max(x0+_goal_line_m, _outer_line_m), y0+_goal_line_m,
     min(x1-_goal_line_m, FIELD_WIDTH_M-_outer_line_m), y1-_goal_line_m)
    for x0,y0,x1,y1 in (bounds("goal_A"),bounds("goal_B"))
])
SIDES = ("A", "B")
GOAL_SIDE = np.asarray([SIDES.index(p["team"]) for p in PIECES])
VALUES = np.asarray([p["points"] for p in PIECES])


def qualify(positions, axes, floor_contact):
    """A tilted finite cylinder's support on XY, with strict inner boundaries."""
    limits = GOAL_INTERIORS[GOAL_SIDE]
    xy_axis = np.abs(axes[:, :2])
    support = HALF_HEIGHT_M*xy_axis + RADIUS_M*np.sqrt(np.maximum(0.,1.-xy_axis**2))
    xy = positions[:, :2]
    return (floor_contact & np.all(xy-support > limits[:, :2]+1e-9,axis=1)
            & np.all(xy+support < limits[:, 2:]-1e-9,axis=1))


def score(positions, axes, floor_contact):
    valid = qualify(positions,axes,floor_contact)
    return {side: {"points":int(VALUES[valid & (GOAL_SIDE==i)].sum()),
                   "pieces":[p.copy() for p,ok in zip(PIECES,valid,strict=True) if ok and p["team"]==side]}
            for i,side in enumerate(SIDES)}
