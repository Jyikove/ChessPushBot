"""Manual state navigation for the no-motor inspection session."""
from .decision.targets import choose_outbound, choose_return
from .planning.policy import Impact


NEXT_STATE = {
    "INITIALIZE": "TURN_CENTER",
    "TURN_CENTER": "SEARCH_PIECE",
    "SEARCH_PIECE": "APPROACH_PIECE",
    "APPROACH_PIECE": "VERIFY_CAPTURE",
    "VERIFY_CAPTURE": "TURN_HOME",
    "TURN_HOME": "FIND_SCORE_ZONE",
    "FIND_SCORE_ZONE": "ALIGN_SCORE_ZONE",
    "RETURN_COLLECT_PIECE": "RETURN_VERIFY_CAPTURE",
    "RETURN_VERIFY_CAPTURE": "FIND_SCORE_ZONE",
    "ALIGN_SCORE_ZONE": "PUSH_PIECES",
    "PUSH_PIECES": "RETREAT",
    "RETREAT": "TURN_SEARCH",
    "TURN_SEARCH": "SEARCH_PIECE",
}


class TestStepper:
    def __init__(self):
        self.wait_for_zone = False

    def skip(self, p, obs):
        state = NEXT_STATE.get(p.state)
        if state is None:
            return
        if p.zero is None:
            p.zero = obs.yaw
        if state == "SEARCH_PIECE" and p.state == "TURN_SEARCH":
            p.round += 1
            p.captured_count = 0
        if state in ("SEARCH_PIECE", "TURN_HOME", "FIND_SCORE_ZONE"):
            p.clear_target()
            p.start_hint_until = p.fence_until = -1.
        if state in ("VERIFY_CAPTURE", "RETURN_VERIFY_CAPTURE") and p.last_target:
            p.arm_capture(obs.time)
        if state == "PUSH_PIECES":
            p.impact = Impact(p.cfg["collision"])
        self.wait_for_zone = state == "ALIGN_SCORE_ZONE"
        p.transition(state, obs, "test_manual_skip")

    def step(self, p, obs, skip):
        zero = (0., 0., 0.)
        if skip:
            self.skip(p, obs)
            p.last_cmd = zero
            return zero
        if p.state in ("APPROACH_PIECE", "VERIFY_CAPTURE", "RETURN_VERIFY_CAPTURE") and p.last_target is None:
            target = (choose_return(obs, p.processed, p.cfg["return_pickup"])
                      if p.state == "RETURN_VERIFY_CAPTURE" else choose_outbound(obs, p.processed))
            if target is None:
                return zero
            p.select_target(obs, target, "test_visible_piece")
            if p.state.endswith("VERIFY_CAPTURE"):
                p.arm_capture(obs.time)
        if self.wait_for_zone and p.state == "ALIGN_SCORE_ZONE" and obs.zone is None:
            return zero
        self.wait_for_zone = False
        return p.step(obs, allow_stall=False)
