"""Public contract, timestamps, wheel commands; no simulator dependencies."""
import math
import numpy as np

def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi

def validate_obs(o):
    if set(o)!={'time_s','dt_s','rgb','image_time_s','pose'}: raise ValueError('observation keys')
    a=o['rgb']; p=o['pose']
    if not isinstance(a,np.ndarray) or a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<=0: raise ValueError('RGB uint8 H W 3')
    if set(p)!={'x_m','y_m','yaw_rad','time_s'}: raise ValueError('pose keys')
    if not np.isfinite([o['time_s'],o['dt_s'],o['image_time_s'],*p.values()]).all(): raise ValueError('finite times and pose')
    if o['dt_s']<=0 or o['time_s']<0 or not 0<=o['image_time_s']<=o['time_s']+1e-9 or not 0<=p['time_s']<=o['time_s']+1e-9: raise ValueError('time range')

def validate_action(a):
    if not isinstance(a,dict) or set(a)!={'motor_pwm'}: raise ValueError('action keys')
    u=a['motor_pwm']
    if not isinstance(u,list) or len(u)!=4 or any(type(x)!=int or abs(x)>255 for x in u): raise ValueError('four integer PWM -255..255')
    return u

class PoseHistory:
    def __init__(self): self.samples=[]
    def add(self,p):
        t=p['time_s']; a=np.array([p['x_m'],p['y_m'],p['yaw_rad']],float)
        if self.samples and t<self.samples[-1][0]: raise ValueError('pose time decreased')
        if self.samples and t==self.samples[-1][0]:
            if not np.allclose(a,self.samples[-1][1],atol=1e-10,rtol=0): raise ValueError('conflicting duplicate pose')
            return
        self.samples.append((t,a)); self.samples=self.samples[-100:]
    def at(self,t):
        for stamp,p in self.samples:
            if abs(stamp-t)<1e-8:return p.copy()
        for (t0,p0),(t1,p1) in zip(self.samples,self.samples[1:]):
            if t0<t<t1:
                f=(t-t0)/(t1-t0);p=p0+f*(p1-p0);p[2]=wrap(p0[2]+f*wrap(p1[2]-p0[2]));return p
        return None

# logical 0 FL,1 FR,2 RL,3 RR; normalized forward, left, CCW wheel traction
WHEEL=np.array([[1,-1,-1],[1,1,1],[1,1,-1],[1,-1,1]],float)
class Mixer:
    def __init__(self):self.last=np.zeros(4)
    def command(self,vx,vy,wz,dt):
        u=WHEEL@np.array([vx,vy,.11*wz])/.32*255
        u=np.where(abs(u)>2,np.sign(u)*(abs(u)+28),0)
        u*=min(1,160/max(1,abs(u).max()))
        u=np.clip(u,self.last-800*dt,self.last+800*dt)
        self.last=np.rint(u);return {'motor_pwm':[int(x) for x in self.last]}

class MockAdapter:
    """Offline only. Caller explicitly supplies logical→unverified driver map."""
    def __init__(self,index_map):
        if len(set(index_map))!=4:raise ValueError('distinct map required')
        self.index_map=list(index_map);self.calls=[]
    def apply(self,a):
        self.calls.extend(zip(self.index_map,validate_action(a)))
