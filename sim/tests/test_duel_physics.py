import json
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

SIM=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SIM/"runtime"))
from duel_physics import DuelPhysics


class DualCarTest(unittest.TestCase):
    def setUp(self):
        camera=json.loads((SIM/"resources/camera/camera_profile.json").read_text())
        self.env=DuelPhysics(camera,77,"baseline",render=False)
        self.addCleanup(self.env.close)

    def test_wheel_inputs_advance_both_cars_on_same_clock(self):
        env=self.env
        starts=env.d.xpos[list(env.cars)].copy()
        for _ in range(3):
            env.controls([{"motor_pwm":[120]*4},{"motor_pwm":[150]*4}])
        xy=env.d.xpos[list(env.cars)]
        self.assertGreater(xy[0,0],starts[0,0])
        self.assertLess(xy[1,0],starts[1,0])
        self.assertAlmostEqual(env.time,.3)
        self.assertEqual(env.sensor_packet(0)["time_s"],env.sensor_packet(1)["time_s"])
        self.assertFalse({"pose","opponent","score"} & env.sensor_packet(1).keys())

    def test_referee_reset_preserves_pieces_and_match_time(self):
        env=self.env
        for address in env.qpos:
            env.d.qpos[address+1]+=.3
        before=env.d.qpos.copy()
        piece_addresses=env.m.jnt_qposadr[env.m.body_jntadr[env.pieces]]
        env._reset_cars()
        self.assertEqual(env.time,0.)
        self.assertEqual(env.epoch,1)
        for address,start in zip(env.qpos,env.spawn,strict=True):
            np.testing.assert_array_equal(env.d.qpos[address:address+7],start)
        for address in piece_addresses:
            np.testing.assert_array_equal(env.d.qpos[address:address+7],before[address:address+7])

    def test_opposing_drives_register_physical_contact(self):
        env=self.env
        for address,x in zip(env.qpos,(1.2,1.8),strict=True):
            env.d.qpos[address:address+2]=[x,.35]
        mujoco.mj_forward(env.m,env.d)
        for _ in range(15):
            env.controls([{"motor_pwm":[160]*4}]*2)
        self.assertGreater(len(env.collisions),0)
        self.assertGreater(env.contact_steps,0)

    def test_placed_piece_counts_with_actual_floor_contact(self):
        env=self.env
        joint_id=env.m.body_jntadr[env.pieces[0]]
        start=env.m.jnt_qposadr[joint_id]
        env.d.qpos[start:start+2]=[.2,1.]
        mujoco.mj_forward(env.m,env.d)
        env.controls([{"motor_pwm":[0]*4}]*2)
        self.assertEqual(env.scoreboard()["A"]["points"],10)
