"""Two deterministic sparring openings that only read onboard heading and encoders."""
import math
import numpy as np
from core import WHEEL,wrap


class RoutePolicy:
    def __init__(self, setup, behind):
        self.origin=np.asarray(setup["starting_xy_m"],dtype=float)
        self.start_yaw=setup["starting_yaw_rad"]
        self.reverse=setup["side"]=="B"
        self.field=np.asarray(setup["field_size_m"])
        # Route is expressed from A's end, with B using a 180 degree field rotation.
        self.route=([(1.88,.57,0.),(1.98,.83,math.pi),(.20,.96,math.pi)]
                    if behind else [(1.58,.61,0.),(1.62,.82,math.pi),(.20,.91,math.pi)])
        self.route.extend([(.54,.91,math.pi),(.77,.56,0.),(1.60,.64,0.),
                           (1.67,.93,math.pi),(.20,1.04,math.pi)])
        self.inverse=np.linalg.pinv(WHEEL)
        self.previous=None
        self.reset_id=None
        self.pos=None
        self.stage=0
        self.since=0.

    def act(self, obs):
        encoder=np.asarray(obs["encoder_m"])
        yaw=wrap(obs["yaw_from_start_rad"]+self.start_yaw)
        t=obs["time_s"]
        if obs["reset_id"]!=self.reset_id:
            self.reset_id=obs["reset_id"]
            self.pos=self.origin.copy()
            self.previous=encoder
            self.stage,self.since=0,t
        c,s=np.cos(yaw),np.sin(yaw)
        rotation=np.array([[c,-s],[s,c]])
        self.pos+=rotation@(self.inverse@(encoder-self.previous))[:2]
        self.previous=encoder
        if self.stage>=len(self.route):
            return {"motor_pwm":[0,0,0,0]}
        x,y,nominal=self.route[self.stage]
        target=np.array([x,y])
        if self.reverse:
            target=self.field-target
        desired=wrap(nominal+self.start_yaw)
        err=wrap(desired-yaw)
        delta=target-self.pos
        distance=np.linalg.norm(delta)
        if distance<.034 and abs(err)<.15 or t-self.since>15.:
            self.stage+=1
            self.since=t
        # During a bulk delivery, settle yaw before applying the forward push.
        p=rotation.T@delta*2.0
        if abs(err)>.40 and self.stage in (2,3,7):
            p*=0.
        if np.linalg.norm(p)>.23:
            p*=.23/np.linalg.norm(p)
        w=np.clip(3.*err-.18*obs["gyro_z_rad_s"],-1.4,1.4)
        raw=(WHEEL@np.r_[p,.11*w])*(237/.32)
        pwm=np.where(abs(raw)>=3.,raw+18*np.sign(raw),0)
        return {"motor_pwm":np.rint(np.clip(pwm,-255,255)).astype(int).tolist()}
