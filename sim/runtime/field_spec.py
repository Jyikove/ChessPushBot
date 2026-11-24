"""Competition field geometry in metres, from assets/field/field.json."""
import json
from pathlib import Path

FIELD = json.loads((Path(__file__).resolve().parents[2] / "assets/field/field.json").read_text())
FIELD_WIDTH_M, FIELD_DEPTH_M = FIELD["field_size_m"]
WALL_HEIGHT_M = FIELD["wall_height_mm"] / 1000
WALL_THICKNESS_M = FIELD["wall_thickness_mm"] / 1000


def bounds(name):
    region = next(region for region in FIELD["regions"] if region["name"] == name)
    x, y, w, h = region["xywh_mm"]
    return x/1000, y/1000, (x+w)/1000, (y+h)/1000


def polygon(name):
    x0, y0, x1, y1 = bounds(name)
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def center(name):
    x0, y0, x1, y1 = bounds(name)
    return (x0+x1)/2, (y0+y1)/2


GOALS = (polygon("goal_A"), polygon("goal_B"))
GOAL_A_X0, GOAL_A_Y0, GOAL_A_X1, GOAL_A_Y1 = bounds("goal_A")
