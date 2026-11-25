MATCH_SECONDS = 120.
APPROACH_SPEED = .20
RETURN_SPEED = .17
OPENING_LANE_STEP = .04
# Wing (+1 enters from low Y), X past midfield, and edge of the goal to sweep.
SWEEP_REGIONS = ((-1, .56, .13), (1, .27, -.13),
                 (1, .59, -.03), (-1, .28, .04))
TRAFFIC_PHASES = ("outside", "reposition", "approach", "detour", "enter", "deliver")
RAID_TRAVEL = ("raid_clear", "raid_lane", "raid_transit", "raid_align",
               "raid_push", "raid_clearance", "raid_exit", "raid_rescan_shift")
RECOVER_TRAVEL = ("recover_flank", "recover_transit", "recover_stage", "recover_align", "recover_push")
WATCHED_TRAVEL = TRAFFIC_PHASES + RAID_TRAVEL + RECOVER_TRAVEL + ("late_return",)
