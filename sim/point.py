"""Paired deterministic decision experiment. Run with the project's mechanical-model venv."""
from pathlib import Path
import json, math, subprocess,tempfile,itertools,statistics as st
import numpy as np
from math import sin,cos,atan2,pi,dist
import sys
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base/'runtime'))
gx,gy,gh=0.,.42,1.15
cam=json.loads((base/'resources/camera/camera_profile.json').read_text());R=np.array(cam['R_ground_optical']); lens=cam['camera_base_position_m'];k1,k2=cam['opencv_distortion'][:2];f=cam['focal_px'];grid=np.array([(x/10+.05,y/10+.05) for x in range(30) for y in range(20)])
V=.35; O=pi/2; modes=['full','passive','active'];names=['全图','被动','主动']
def angle(a):return (a+pi)%(2*pi)-pi
def view(pts,p,h):
    d=pts-p;c,s=cos(h),sin(h);a,b=d.T;local=np.column_stack((c*a+s*b,-s*a+c*b,np.full(len(a),.015)))-lens
    X,Y,Z=(local@R).T;z=np.maximum(.01,Z);x=X/z;y=Y/z;r=x*x+y*y;q=1+k1*r+k2*r*r
    return (Z>0)&(np.hypot(a,b)<=1.5)&(abs(f*x*q)<=319.5)&(abs(f*y*q)<=239.5)
def plan(proc,p,known,load,prefix,ref,turn_args=()):
    pts=sorted(known);v=[len(pts),4,1.15,gx,gy,gy+gh,len(load),prefix,ref,*turn_args,*itertools.chain.from_iterable([p,*pts])];proc.stdin.write(' '.join(map(str,v))+'\n');proc.stdin.flush();_,_,ref,*idx=map(float,proc.stdout.readline().split());return [pts[int(i)-1] for i in idx],ref

def run(proc,sample,mode,turn_cost=False,slots=False):
    truth=np.array(sample['points']); allpts=set(map(tuple,truth));seen=allpts.copy() if mode=='full' else set();taken=set();done=set();load=set();cov=np.zeros(len(grid),bool);p=tuple(sample['start']);h=t=distance=empty=search=origin=0.;tick=.5;ref=-1;frame=0;trace=[];scores=[];inspected=False
    counts=[0]*8; deliveries=[]
    if slots:
        from goal_slots import plan as slot_plan,CENTERS
    turning=0.
    def cost():return distance+(V*turning if turn_cost else 0.)
    def observe():
        if mode!='full':
            seen.update(map(tuple,truth[view(truth,p,h)]));cov[:]|=view(grid,p,h)
    def move(duration,kind,rate=0,searching=False,interrupt=False):
        nonlocal p,h,t,distance,empty,search,frame,turning
        end=min(120,t+duration);old=len(seen);start=p;begin=t
        while t<end-1e-10:
            nxt=min(end,(frame+1)/5);dt=nxt-t;t=nxt
            if kind=='turn':h=angle(h+rate*dt)
            if kind=='drive':p=(p[0]+cos(h)*V*dt,p[1]+sin(h)*V*dt);distance+=V*dt;empty+=V*dt*(not load)
            if abs(t-(frame+1)/5)<1e-8:
                frame+=1;observe()
                if interrupt and len(seen)>old:break
        if kind=='turn':turning+=t-begin
        if searching:search+=t-begin
        if t>begin:trace.append([begin,t,*start,*p,kind,searching,h]);
    def turn(a,searching=False,interrupt=False):
        delta=angle(a-h);move(abs(delta)/O,'turn',math.copysign(O,delta),searching,interrupt)
    observe(); guard=0
    while t<120-1e-8 and len(done)<32:
        guard+=1;assert guard<10000
        known=seen-taken
        if distance>=tick-1e-8:tick+=.5
        if not load and (not known or (mode=='active' and not inspected and (len(known)<4 or min(dist(p,v) for v in known)>.65))):
            # Compare uniform coverage from this pose; no board/piece truth in search policy.
            cand=[(np.count_nonzero(view(grid,p,a)&~cov)/(1+abs(angle(a-h))/O),a) for a in np.arange(24)*pi/12];gain,a=max(cand);inspected=True
            if gain>=5 and abs(angle(a-h))>.02:
                turn(a,True,not known);continue
            if not known:
                # Move toward nearest uncovered grid cell (continue until detection or next planning tick).
                options=grid[~cov] if (~cov).any() else grid
                delta=options-np.array(p);ds=np.linalg.norm(delta,axis=1);mask=ds>.18
                if not mask.any():options=grid;ds=np.linalg.norm(grid-p,axis=1);mask=ds>.7
                dst=options[np.argmin(np.where(mask,ds,np.inf))];turn(atan2(dst[1]-p[1],dst[0]-p[0]),True,True)
                if seen-taken:continue
                move(min(dist(p,dst),tick-distance)/V,'drive',searching=True,interrupt=True);continue
        if slots:
            pts=sorted(known)
            slot,ref,indices=slot_plan(proc,counts,[p,*pts],h,len(load),cost()-origin,ref)
            seq=[pts[i] for i in indices];goal=(float(gx),CENTERS[slot])
        else:
            seq,ref=plan(proc,p,known,load,cost()-origin,ref,(h,V/O) if turn_cost else ());last=seq[-1] if seq else p;goal=(float(gx),float(np.clip(last[1],gy,gy+gh)))
        for idx,dst in enumerate(seq+[goal]):
            length=dist(p,dst)
            if length>1e-9:turn(atan2(dst[1]-p[1],dst[0]-p[0]));move(min(length,max(0,tick-distance))/V,'drive')
            if dist(p,dst)>1e-7:break
            p=dst
            if idx<len(seq):load.add(dst);taken.add(dst)
            else:
                before=t;move(.5,'deposit');
                if t-before>=.5-1e-8:
                    if slots:
                        assert counts[slot]<4 and load
                        counts[slot]+=len(load);deliveries.append([t,slot,len(load),counts[slot]])
                    done.update(load);scores.append([t,len(done)]);load.clear();origin=cost();ref=-1;inspected=False
            if distance>=tick-1e-8 or t>=120-1e-8:break
    assert done<=taken<=seen<=allpts and len(done)+len(seen-done)+len(allpts-seen)==32 and abs(sum(v[1]-v[0] for v in trace)-t)<1e-6
    if slots:assert sum(counts)==len(done)
    return dict(counts=counts,deliveries=deliveries,score=len(done),seconds=t,search=search,empty=empty,unseen=len(allpts-seen),pending=len(seen-done),distance=distance,trace=trace,scores=scores,seen=sorted(seen),done=sorted(done))


if __name__=='__main__':
    out=base.parent.parent/'决策层点模型实验/point-comparison';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        exe=Path(tmp)/'solver'
        subprocess.run(['c++','-O3','-std=c++17',base/'balance_solver.cpp','-o',exe],check=True)
        with subprocess.Popen([exe,'turns'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as proc:
            for seed in range(18,28):
                rng=np.random.default_rng(seed);pts=[[.305,.21]]
                while len(pts)<33:
                    q=rng.uniform([.16,.08],[2.84,1.92]).tolist()
                    if all(dist(q,x)>.12 for x in pts):pts.append(q)
                board=dict(seed=seed,start=pts[0],points=pts[1:])
                rows.append(dict(**board,runs={m:[run(proc,board,m,True,slots) for slots in (False,True)] for m in modes}))
            proc.stdin.close()
    (out/'结果.json').write_text(json.dumps(rows,separators=(',',':')))
    import point_report
