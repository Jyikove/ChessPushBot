"""JSON-lines subprocess boundary for local strategy code."""
import base64
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from core import validate_action


def state_label(policy):
    phase = getattr(policy, "phase", None)
    if not isinstance(phase, str):
        steps,stage = getattr(policy, "steps", ()),getattr(policy, "stage", None)
        if not isinstance(steps, (list,tuple)) or not isinstance(stage, int):
            return "UNAVAILABLE"
        phase = getattr(steps[stage], "phase", "UNAVAILABLE") if 0<=stage<len(steps) else "DONE"
    if phase in ("scoop","deliver") and getattr(policy,"yield_since",None) is not None:
        phase += " / YIELD"
    return str(phase).upper()[:50]


def main(path):
    path = Path(path).resolve()
    sys.path.insert(0,str(path.parent))
    spec = importlib.util.spec_from_file_location("active_strategy", path)
    module = importlib.util.module_from_spec(spec)
    with redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
        policy = module.Policy(json.loads(sys.stdin.readline()))
    print('{"ready":true}',flush=True)
    image = None
    image_time = None
    for line in sys.stdin:
        observation = json.loads(line)
        encoded = observation.pop("camera_png")
        if encoded is not None:
            bgr = cv2.imdecode(np.frombuffer(base64.b64decode(encoded),np.uint8),cv2.IMREAD_COLOR)
            image = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            image.setflags(write=False)
            image_time = observation["time_s"]
        observation.update(rgb=image,image_time_s=image_time)
        with redirect_stdout(sys.stderr):
            action = policy.act(observation)
        validate_action(action)
        print(json.dumps(dict(action=action,state=state_label(policy))),flush=True)


if __name__=="__main__":
    main(sys.argv[1])
