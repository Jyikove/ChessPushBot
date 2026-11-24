"""Private evaluator. Never imported by policy worker. Exact cylinder support for convex polygons."""
import numpy as np
from field_spec import FIELD_WIDTH_M, FIELD_DEPTH_M, bounds
from piece_spec import HALF_HEIGHT_M, RADIUS_M, RANDOM_PAIR_GAP_M, SPAWN_Z_M, WALL_GAP_M

def polygon_planes(poly):
    p=np.asarray(poly,float)
    if p.shape[0]<3 or p.shape[1]!=2 or not np.isfinite(p).all(): raise ValueError('polygon')
    e=np.roll(p,-1,axis=0)-p
    if np.any(np.cross(e,np.roll(e,-1,axis=0))<-1e-12): raise ValueError('provide CCW convex polygon; decompose concave zones explicitly')
    n=np.column_stack([e[:,1],-e[:,0]]); n/=np.linalg.norm(n,axis=1)[:,None]
    return n,np.sum(n*p,axis=1)

def contains(pos,R,poly,r=RADIUS_M,h=HALF_HEIGHT_M):
    n,b=polygon_planes(poly);n3=np.column_stack([n,np.zeros(len(n))]);axis=np.asarray(R)[:,2]
    axial=n3@axis; support=h*np.abs(axial)+r*np.sqrt(np.maximum(0,1-axial**2))
    return bool(np.all(n@np.asarray(pos)[:2]+support<=b+1e-12))

def intersects_upright(xy,poly,r=RADIUS_M):
    # Separation of circle from a convex polygon, with vertex distance handling.
    n,b=polygon_planes(poly);p=np.asarray(poly);xy=np.asarray(xy)
    if np.all(n@xy<=b):return True
    for a,c in zip(p,np.roll(p,-1,axis=0)):
        t=np.clip(np.dot(xy-a,c-a)/np.dot(c-a,c-a),0,1)
        if np.linalg.norm(xy-(a+t*(c-a)))<=r+1e-12:return True
    return False

class Referee:
    def __init__(self,goals,n=32,*,center_only=False,time_limit=120.): self.center_only=center_only;self.goals=goals;self.planes=[polygon_planes(g) for g in goals];self.n=n;self.time_limit=float(time_limit);self.reset()
    def reset(self):
        self.since=[None]*self.n;self.active=set();self.t=0.;self.windows=[];self.window_start=0.;self.window_count=0;self.done=False;self.reason=None;self.events=[]
    def update(self,t,poses):
        if self.done or t<self.t-1e-9 or len(poses)!=self.n:raise ValueError('referee time/count/end')
        # Runner calls on each physics step and never beyond timeout.
        if t>self.time_limit+1e-8:raise ValueError('must stop at configured time limit')
        previous=set(self.active)
        positions=np.asarray([p for p,R in poses]);axes=np.asarray([R[:,2] for p,R in poses]);inside_all=np.zeros(self.n,bool)
        for normals,bounds in self.planes:
            if self.center_only:
                inside_all|=np.all(positions[:,:2]@normals.T<=bounds+1e-12,axis=1)
                continue
            n3=np.column_stack([normals,np.zeros(len(normals))]);axial=axes@n3.T
            support=HALF_HEIGHT_M*np.abs(axial)+RADIUS_M*np.sqrt(np.maximum(0,1-axial**2))
            inside_all|=np.all(positions[:,:2]@normals.T+support<=bounds+1e-12,axis=1)
        for i,inside in enumerate(inside_all):
            if not inside:self.since[i]=None;self.active.discard(i)
            else:
                if self.since[i] is None:self.since[i]=float(t)
                if self.center_only or t-self.since[i]>=.5-1e-10:self.active.add(i)
        self.t=float(t)
        if self.active!=previous:self.events.append({'t':self.t,'enter':sorted(self.active-previous),'leave':sorted(previous-self.active)})
        if self.t-self.window_start>=10-1e-8:self._window()
        if len(self.active)==self.n:self.done=True;self.reason='all_in'
        elif t>=self.time_limit-1e-8:self.done=True;self.reason='timeout'
    def _window(self):
        self.windows.append({'start_s':self.window_start,'end_s':self.t,'span_s':self.t-self.window_start,'net':len(self.active)-self.window_count,'partial':self.t-self.window_start<10-1e-8});self.window_start=self.t;self.window_count=len(self.active)
    def finish(self,error=None):
        if error:self.done=True;self.reason='abnormal:'+error
        if not self.done:raise ValueError('cannot voluntarily end match')
        if self.t>self.window_start+1e-9:self._window()
        return {'N':len(self.active),'T_s':self.t,'rate_per_10s':None if error or self.t<=0 else 10*len(self.active)/self.t,'termination':self.reason,'windows':self.windows,'events':self.events}

def sample_layout(seed,goals,n=32,car=(1.5,1.0)):
    rng=np.random.default_rng(seed);pts=[]
    for _ in range(100000):
        edge=RADIUS_M+WALL_GAP_M
        p=rng.uniform([edge,edge],[FIELD_WIDTH_M-edge,FIELD_DEPTH_M-edge])
        if any(intersects_upright(p,g) for g in goals):continue
        if np.linalg.norm(p-np.asarray(car))<.27:continue
        if any(np.linalg.norm(p-q[:2])<2*RADIUS_M+RANDOM_PAIR_GAP_M for q in pts):continue
        pts.append([*p,SPAWN_Z_M])
        if len(pts)==n:return np.array(pts)
    raise RuntimeError('layout rejection exhausted')


def sample_central_layout(seed, n=32, gap=.003):
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bounds("pieces")
    points = []
    for _ in range(25000):
        p = rng.uniform([x0+RADIUS_M, y0+RADIUS_M], [x1-RADIUS_M, y1-RADIUS_M])
        if any(np.linalg.norm(p-q) < 2*RADIUS_M+gap for q in points):
            continue
        points.append(p)
        if len(points) == n:
            return np.array(points)
    raise RuntimeError("central field layout rejection exhausted")
