"""Two independent wheel drives in one world. Private truth stays in the runner."""
from collections import deque
import math

import mujoco
import numpy as np

from core import validate_action, wrap
from field_spec import center
from match_referee import score
from physics import Physics, wheel_force
from piece_spec import SPAWN_Z_M
from referee import sample_central_layout

HZ = 240
CONTROL_STEPS = 24
CAMERA_STEPS = 48
MAT = mujoco.mjtObj
STARTS = {"A": (*center("start_A"), .001), "B": (*center("start_B"), .001)}
HEADINGS = {"A": 0., "B": math.pi}


class DuelPhysics(Physics):
    def __init__(self, camera, seed, appearance, render=True):
        points = sample_central_layout(seed)
        super().__init__({"camera":camera},np.column_stack((points,np.full(32,SPAWN_Z_M))),
                         car=STARTS["A"],opponent=(STARTS["B"],HEADINGS["B"]),
                         render=render,appearance=appearance)
        self.cars = (self.car,mujoco.mj_name2id(self.m,MAT.mjOBJ_BODY,"car_B"))
        joints = self.m.body_jntadr[list(self.cars)]
        self.qpos = self.m.jnt_qposadr[joints]
        self.qvel = self.m.jnt_dofadr[joints]
        self.spawn = tuple(self.d.qpos[a:a+7].copy() for a in self.qpos)
        self.piece_dof = self.m.jnt_dofadr[self.m.body_jntadr[self.pieces]]
        self.speed = np.zeros((2,4))
        self.pwm = np.zeros((2,4))
        self.encoder = np.zeros((2,4))
        self.pending = deque()
        self.epoch = 0
        self.collisions = []
        self.resets = []
        self.contact_steps = 0
        self.lock_steps = 0
        self.contact = False
        self.min_distance = math.inf
        self.layout = points.tolist()
        self.floor = mujoco.mj_name2id(self.m,MAT.mjOBJ_GEOM,"floor")
        self.piece_index = {int(body):i for i,body in enumerate(self.pieces)}
        self.floor_contact = np.zeros(len(self.pieces),dtype=bool)

    def sensor_packet(self, slot, image=None):
        body = self.cars[slot]
        rotation = self.d.xmat[body].reshape(3,3)
        yaw = math.atan2(rotation[1,0],rotation[0,0])
        return dict(time_s=self.time,dt_s=CONTROL_STEPS/HZ,reset_id=self.epoch,
                    yaw_from_start_rad=wrap(yaw-slot*math.pi),
                    gyro_z_rad_s=float(self.d.qvel[self.qvel[slot]+5]),
                    encoder_m=self.encoder[slot].tolist(),camera_png=image)

    def controls(self, actions):
        validated = np.asarray([validate_action(a) for a in actions],dtype=float)
        self.pending.append((self.steps+6,validated))
        for _ in range(CONTROL_STEPS):
            self._step()
        if self.lock_steps >= 3*HZ and self.time < 120.:
            self._reset_cars()

    def _step(self, *, settling=False):
        if self.pending and self.pending[0][0] <= self.steps:
            _,self.pwm = self.pending.popleft()
        self.d.xfrc_applied[:]=0
        for i,body in enumerate(self.cars):
            start=self.qvel[i]
            force,torque=wheel_force(self.pwm[i],self.speed[i],self.wheel_deadband,
                                      self.wheel_max_speed,self.d.xmat[body].reshape(3,3),
                                      self.d.qvel[start:start+6])
            self.d.xfrc_applied[body,:3]=force
            self.d.xfrc_applied[body,5]=torque
        mujoco.mj_step(self.m,self.d)
        if not np.isfinite(self.d.qpos).all() or any(w.number for w in self.d.warning):
            raise RuntimeError("MuJoCo numerical warning")
        if not settling:
            self.steps+=1
        self.encoder+=self.speed/HZ
        self.floor_contact[:]=False
        touching=False
        for contact in self.d.contact:
            g1,g2=int(contact.geom1),int(contact.geom2)
            b1,b2=int(self.m.geom_bodyid[g1]),int(self.m.geom_bodyid[g2])
            if g1==self.floor and b2 in self.piece_index:
                self.floor_contact[self.piece_index[b2]]=True
            elif g2==self.floor and b1 in self.piece_index:
                self.floor_contact[self.piece_index[b1]]=True
            touching |= (b1,b2)==self.cars or (b2,b1)==self.cars
        if settling:
            return
        self.min_distance=min(self.min_distance,float(np.linalg.norm(
            self.d.xpos[self.cars[0],:2]-self.d.xpos[self.cars[1],:2])))
        if touching:
            if not self.contact:
                self.collisions.append({"start_s":self.time,"duration_s":0.})
            self.contact_steps+=1
            self.collisions[-1]["duration_s"]+=1/HZ
        self.contact=touching
        slow=all(np.linalg.norm(self.d.qvel[j:j+2])<.02 and abs(self.d.qvel[j+5])<.15 for j in self.qvel)
        trying=all(np.max(np.abs(pwm))>25 for pwm in self.pwm)
        self.lock_steps=self.lock_steps+1 if touching and slow and trying else 0

    def _reset_cars(self):
        # A referee reset keeps the live board and clock, and creates a fresh
        # sensor origin for both policies. Skip occupied starts for manual review.
        positions=self.d.xpos[self.pieces,:2]
        if any(np.any(np.linalg.norm(positions-origin[:2],axis=1)<.23) for origin in self.spawn):
            if not self.resets or self.resets[-1]["status"]!="start_occupied":
                self.resets.append(dict(time_s=self.time,status="start_occupied"))
            return
        for i in range(2):
            a,b=self.qpos[i],self.qvel[i]
            self.d.qpos[a:a+7]=self.spawn[i]
            self.d.qvel[b:b+6]=0.
        self.pending.clear()
        self.pwm[:]=self.encoder[:]=self.speed[:]=0.
        self.d.qacc_warmstart[:]=0.
        mujoco.mj_forward(self.m,self.d)
        self.epoch+=1
        self.contact=False
        self.lock_steps=0
        self.resets.append(dict(time_s=self.time,status="reset"))

    def scoreboard(self):
        return score(self.d.xpos[self.pieces],self.d.xmat[self.pieces].reshape(-1,3,3)[:,:,2],self.floor_contact)

    def settle(self):
        self.pending.clear()
        self.pwm[:]=0.
        indices=np.asarray([*self.qvel,*self.piece_dof])[:,None]+np.arange(6)
        quiet=0
        for i in range(5*HZ):
            self._step(settling=True)
            velocities=self.d.qvel[indices]
            still=np.max(np.linalg.norm(velocities[:,:3],axis=1))<.005 and np.max(
                np.linalg.norm(velocities[:,3:],axis=1))<.15
            quiet=quiet+1 if still else 0
            if quiet>=HZ//4:
                return (i+1)/HZ, True
        return 5.,False
