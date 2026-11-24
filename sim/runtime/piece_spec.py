"""Competition geometry and identity. Mass remains a simulation assumption."""
import json
from pathlib import Path

SPEC = json.loads(
    (Path(__file__).resolve().parents[2] / "assets/chess/chess.json").read_text(encoding="utf-8")
)
RADIUS_M = SPEC["diameter_mm"] * .0005
HALF_HEIGHT_M = SPEC["height_mm"] * .0005
SPAWN_Z_M = HALF_HEIGHT_M + .0001
MASS_KG = .02
PAIR_GAP_M = .001
RANDOM_PAIR_GAP_M = .008
WALL_GAP_M = .001
ROBOT_GAP_M = .002


def make_identities():
    teams = (
        [(piece, n) for piece in SPEC["types"] if piece["team"] == team
         for n in range(1, piece["count"]+1)]
        for team in ("A", "B")
    )
    return tuple({
        "instance_id": i,
        "source_piece_id": f'{piece["id"]}_{n}',
        "class": "red" if piece["team"] == "A" else "white",
        "side": piece["side"],
        "team": piece["team"],
        "character": piece["character"],
        "points": piece["points"],
    } for i, (piece, n) in enumerate((q for pair in zip(*teams, strict=True) for q in pair), 1))


PIECES = make_identities()


def inertia(mass=MASS_KG):
    ix = mass * (3*RADIUS_M**2+(2*HALF_HEIGHT_M)**2) / 12
    return ix, ix, .5*mass*RADIUS_M**2


def identity(index):
    return dict(PIECES[index])
