from pathlib import Path
import os,sys,json,subprocess,tempfile,math,itertools
os.environ.setdefault('MUJOCO_GL','glfw');sim=Path(__file__).resolve().parent;sys.path.insert(0,str(sim/'runtime'));import numpy as np,cv2,mujoco
from goal_slots import CENTERS, plan
from physics import Physics
from referee import Referee
from camera import Camera, prepare,postprocess
from core import wrap,WHEEL
def make_board(seed):
 rng=np.random.default_rng(seed)
 start=[.305,.21]
 points=[start]
 while len(points)<33:
  q=rng.uniform([.16,.08],[2.84,1.92]).tolist()
  if all(math.dist(q,other)>.12 for other in points):points.append(q)
 return dict(seed=seed,start=start,points=points[1:])
board=make_board(int(os.environ.get('BOARD_SEED','77')))
reports=sim.parent.parent/'三维仿真/报告'/f"sim-seed-{board['seed']}"
out=sim.parent.parent/'三维仿真/视频'/f"sim-seed-{board['seed']}";c=dict(camera=json.loads((sim/'resources/camera/camera_profile.json').read_text()),goals=[[[0,.42],[.11,.42],[.11,1.57],[0,1.57]]]);c['goals']=c['goals'][:1];grid=np.array([(x/10+.05,y/10+.05) for x in range(30) for y in range(20)])
# Enclose tilted chess cylinders and every pusher collision primitive by spheres.
cad=json.loads((sim.parent/'assets/robot/simulation.json').read_text())
Rplate=max([math.dist(q,(0,0,0)) for part in cad['convex_parts'] for q in part['vertices']]+[math.hypot(*cap['position'][:2])+cap['radius']+abs(cap['position'][2])+cap['half_height'] for cap in cad['tip_caps']]);Rpiece=math.hypot(.0235,.0075);ZONE=.11;SWEEP=Rplate+2*Rpiece; SAFE=SWEEP+.035;GATE=ZONE+SAFE+.075
# Even if only the rim touches the goal, the opposite rim can project a full diameter outside.
def clearance(p):return math.hypot(max(0,p[0]-ZONE,-p[0]),max(0,.42-p[1],p[1]-1.57))-SWEEP
# Wall-normal extent of the working blade, including its rounded tips.
blade=np.array([q[:2] for a in cad['convex_parts'] if a['part']=='blade' for q in a['vertices']]);tips=cad['tip_caps']; STOPPING=.018; WALL_BUFFER=.022; LINE_MARGIN=.003
class PushEnd:
 def begin(self,t,p):self.start=t;self.window=[(t,p[0])];self.deadline=t+min(5.,(max(0,p[0]-.195)/.085)+1.);self.reason=None
 def check(self,t,p,h):
  normal=np.array([math.cos(h),-math.sin(h)]);xs=p[0]+blade@normal;self.wall=min(min(xs),min(p[0]+np.dot(cap['position'][:2],normal)-cap['radius'] for cap in tips));self.back=max(xs);self.window.append((t,p[0]));self.window=[v for v in self.window if t-v[0]<=.81]
  if self.wall<=WALL_BUFFER+STOPPING:self.reason='WALL BUFFER'
  elif self.back<=ZONE-LINE_MARGIN:self.reason='BLADE AT LINE'
  elif abs(wrap(h-math.pi))>.18:self.reason='HEADING DRIFT'
  elif t-self.start>=1 and t-self.window[0][0]>=.7 and self.window[0][1]-p[0]<.012:self.reason='NO PROGRESS'
  elif t>=self.deadline:self.reason='TIME LIMIT'
  return self.reason
class Brain:
 def __init__(self,solver):
  self.slot_views=[];self.recounts=[];self.cargo=set();self.review={};self.review_log=[];self.credited=False
  self.counts=[0]*8;self.slot=None;self.s=solver;self.cam=Camera(c,sim/'resources/detector');self.db={};self.seq=[];self.goal=None;self.load=0;self.prefix=0.;self.ref=-1;self.odo=0.;self.tick=.5;self.prev=None;self.hold=None;self.cover=np.zeros(len(grid),bool);self.plans=0;self.boxes=[];self.scan=None;self.phase=None;self.dock_y=None;self.release=None;self.end=PushEnd();self.end.reason=None
 def perceive(self,rgb,p,h,t):
  mat,tf=prepare(rgb[:,:,::-1],(416,320));raw,_=self.cam.detector.forward(mat);self.boxes=postprocess(raw,tf);used=set();cc=c['camera'];rot=np.array([[math.cos(h),-math.sin(h)],[math.sin(h),math.cos(h)]]);lens=p+rot@cc['camera_base_position_m'][:2];pose=np.r_[p,h];self.cover|=self.visible(p,h)
  slot_centers=[]
  for b in sorted(self.boxes,key=lambda b:-b['confidence']):
   x,y,u,v=b['box'];
   if min(x,y)<1 or u>638 or v>478:continue
   q=self.cam.project([[(x+u)/2,y]],pose)[0]
   if q is None:continue
   q=q-.0235*(q-lens)/max(1e-9,np.linalg.norm(q-lens));
   if q[0]<=ZONE and .42<=q[1]<=1.57:
    # Temporary count in a single image, never a goal-interior world-map track.
    if self.phase=='BACK' and q[0]>=0 and abs(q[1]-CENTERS[self.slot])<=(1.57-.42)/16 and all(np.linalg.norm(q-z)>.028 for z in slot_centers):slot_centers.append(q)
    continue
   if not(.023<q[0]<2.98 and .023<q[1]<1.977 and np.linalg.norm(q-p)<=1.5):continue
   if self.phase=='BACK':
    # Use only fresh retreat frames. Defer any cargo/accounting change until braking ends.
    self.observe_retreat(q,t,used)
    continue
   near=[(np.linalg.norm(q-v['p']),k) for k,v in self.db.items() if k not in used and (not v['taken'] or t-v['t']<2)]
   d,k=min(near,default=(99,-1));
   if d>.085:k=len(self.db);self.db[k]=dict(p=q,hits=0,t=t,taken=False)
   v=self.db[k];v.update(p=.65*v['p']+.35*q,hits=v['hits']+1,t=t);used.add(k)
  if self.phase=='BACK' and p[0]>=ZONE+SAFE+.025:
   # Only recount a fully framed slot: project its four corners into the actual image.
   half=(1.57-.42)/16;yy=CENTERS[self.slot]
   corners=np.array([[x,y,.015] for x in (0,ZONE) for y in (yy-half,yy+half)])
   corners[:,:2]=(corners[:,:2]-p)@rot
   xyz=(corners-cc['camera_base_position_m'])@np.array(cc['R_ground_optical'])
   uv,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),np.asarray(cc['K']),np.asarray(cc['opencv_distortion']))
   uv=uv.reshape(-1,2)
   if np.all(xyz[:,2]>0) and np.all((uv>[12,12])&(uv<[627,467])):self.slot_views.append((t,len(slot_centers)))
 def observe_retreat(self,q,t,used):
  nearby=sorted((np.linalg.norm(q-v['p']),k) for k,v in self.review.items() if k not in used)
  if nearby and nearby[0][0]<.035:
   _,k=nearby[0];v=self.review[k]
   if t-v['t']>.31:v['hits']=0
   v.update(p=q.copy(),hits=v['hits']+1,t=t)
  else:
   k=len(self.review);self.review[k]=dict(p=q.copy(),hits=1,t=t)
  used.add(k)
 def finish_review(self,t):
  observations=[v['p'] for v in self.review.values() if v['hits']>=2 and t-v['t']<.61]
  # Match against the pre-review map. A crowded match cannot establish cargo identity.
  snapshot={k:v['p'].copy() for k,v in self.db.items()}
  assigned=set()
  for q in observations:
   near=sorted((np.linalg.norm(q-p),k) for k,p in snapshot.items() if np.linalg.norm(q-p)<.065)
   unique=(len(near)==1 and near[0][0]<.035 and sum(np.linalg.norm(z-snapshot[near[0][1]])<.065 for z in observations)==1)
   debit=False
   if unique:
    k=near[0][1];v=self.db[k]
    outside=math.hypot(max(0.,q[0]-ZONE),max(0.,.42-q[1],q[1]-1.57))
    if k in self.cargo and self.credited and outside>.015 and self.hold-self.db[k]['t']<1.0:
     self.counts[self.slot]-=1;debit=True
   else:
    # Reuse a nearby available track if possible; never debit an ambiguous observation.
    free=[k for _,k in near if k not in assigned and not self.db[k]['taken']]
    k=free[0] if free else len(self.db)
   assigned.add(k)
   self.db[k]=dict(p=q.copy(),hits=2,t=t,taken=False,pending=True)
   self.review_log.append(dict(t=t,xy=q.tolist(),track=k,slot=self.slot,debit=debit))
  counts=[n for ts,n in self.slot_views]
  supported=[min(counts[i:i+3]) for i in range(len(counts)-2)]
  if self.credited and supported and max(supported)>0:
   # Visual estimate, not proof of absence; full framing does not rule out occlusion.
   before=self.counts[self.slot];self.counts[self.slot]=max(supported)
   self.recounts.append(dict(t=t,slot=self.slot+1,before=before,observed=max(supported),views=counts))
  self.slot_views.clear();self.review.clear();self.cargo.clear()
 def visible(self,p,h):
  d=grid-p;co,si=math.cos(h),math.sin(h);a,b=d.T;cc=c['camera'];q=(np.column_stack((co*a+si*b,-si*a+co*b,np.full(len(a),.015)))-cc['camera_base_position_m'])@np.array(cc['R_ground_optical']);z=q[:,2];x,y=(q[:,:2]/np.maximum(.01,z[:,None])).T;r=x*x+y*y;k,l=cc['opencv_distortion'][:2];v=1+k*r+l*r*r;return (z>0)&(abs(x*v*cc['focal_px'])<319.5)&(abs(y*v*cc['focal_px'])<239.5)&(np.linalg.norm(d,axis=1)<=1.5)
 def candidates(self,t):return [k for k,v in self.db.items() if v['hits']>=2 and not v['taken'] and t-v['t']<40 and clearance(v['p'])>.04]
 def act(self,o):
  v=o['pose'];p=np.array([v['x_m'],v['y_m']]);h=v['yaw_rad'];t=o['time_s'];self.pos=p;self.yaw=h;self.target=None
  self.speed=self.rate=0.
  if self.prev is not None:
   q,a=self.prev;ds=np.linalg.norm(p-q);self.speed=ds/o["dt_s"];self.rate=math.degrees(wrap(h-a))/o["dt_s"];self.odo+=ds;self.prefix+=ds+.35/(math.pi/2)*abs(wrap(h-a))
  self.prev=p,h
  if abs(t-o['image_time_s'])<1e-6:self.perceive(o['rgb'],p,h,t)
  ids=self.candidates(t);expired=self.odo>=self.tick
  if expired:self.tick+=.5
  if self.phase=='HOLD':
   if t-self.hold<.5:return self.cmd('STOP / '+self.end.reason,0,0)
   self.credited=self.end.back<=ZONE-LINE_MARGIN
   if self.credited:self.counts[self.slot]+=self.load
   self.phase='BACK';self.release=None
  if self.phase=='BACK':
   if p[0]<ZONE+SAFE+.225:self.release=None;return self.cmd('CLEAR GOAL / REVERSE',-.18,0)
   if self.release is None:self.release=t
   if t-self.release<.4:return self.cmd('CLEAR GOAL / BRAKE',0,0)
   self.finish_review(t)
   ids=self.candidates(t)
   self.phase=None;self.goal=None;self.load=0;self.prefix=0.;self.ref=-1;self.hold=None
  if self.phase=='RESTAGE':
   # Already outside the protected zone. Move to a different staging point, preserving cargo.
   if t-self.restart<.4:return self.cmd('REALIGN / BRAKE',0,0)
   self.phase='STAGE'
  if self.phase=='IN':
   if self.end.check(t,p,h):self.phase='HOLD';self.hold=t;return self.cmd('STOP / '+self.end.reason,0,0)
   return self.cmd('PUSH TO GOAL / STRAIGHT',.08 if self.end.back<ZONE+.055 else .155,0)
  if (ids or self.load) and (self.goal is None or expired) and self.phase is None:
   ids=sorted(ids,key=lambda k:tuple(self.db[k]['p']))[:60];coords=[p,*[self.db[k]['p'] for k in ids]]
   decision=plan(self.s,self.counts,coords,h,self.load,self.prefix,self.ref)
   if decision is None:return self.cmd('ALL GOAL SLOTS FULL',0,0)
   self.slot,self.ref,route=decision;self.seq=[ids[j] for j in route];self.goal=np.array([0.,CENTERS[self.slot]]);self.plans+=1;self.scan=None
  if self.goal is None:
   if self.scan is None or abs(wrap(self.scan-h))<.08:
    vals=[(np.count_nonzero(self.visible(p,a)&~self.cover)/(1+abs(wrap(a-h))),a) for a in np.arange(24)*math.pi/12];gain,a=max(vals);self.scan=a if gain>=5 else None
   if self.scan is not None:return self.cmd('SCAN / ROTATE',0,np.clip(3*wrap(self.scan-h),-1.55,1.55))
   candidates=grid[(~self.cover)&(grid[:,0]>.18)&(grid[:,0]<2.82)&(grid[:,1]>.18)&(grid[:,1]<1.82)];dst=min(candidates,key=lambda q:np.linalg.norm(q-p)) if len(candidates) else np.array([1.5,1.]);return self.follow(p,h,dst,'SEARCH')
  if self.seq:
   k=self.seq[0];dst=self.db[k]['p'];
   if np.linalg.norm(p-dst)<.095:
    self.seq.pop(0);self.db[k]['taken']=True;self.cargo.add(k);self.load+=1;return self.cmd('PICKUP / ESTIMATE',0,0)
   return self.follow(p,h,dst,'COLLECT')
  # Score line belongs to the pieces. Keep the vehicle base reachable inside the wall.
  if self.phase is None:self.phase='STAGE';self.dock_y=np.clip(self.goal[1],.48,1.51)
  gate=np.array([GATE,self.dock_y]);self.target=gate
  if self.phase=='STAGE' and np.linalg.norm(p-gate)>.035:return self.follow(p,h,gate,'GOAL APPROACH')
  err=wrap(math.pi-h)
  if self.phase!='ALIGN':
   self.phase='ALIGN';self.align_start=t;self.improved=t;self.best_err=abs(err);self.aligned=None
  if abs(err)<self.best_err-.015:self.best_err=abs(err);self.improved=t
  # Project the push axis to the goal line. Allow at most 25 mm of sideways drift
  # and keep the predicted piece centre one radius plus 10 mm from either end.
  dx=max(0.,p[0]-ZONE);side=dx*math.tan(err) if abs(err)<.18 else math.inf
  feasible=abs(err)<.16 and abs(side)<=.025 and abs(p[1]+side-CENTERS[self.slot])<=(1.57-.42)/16-Rpiece-.01
  if not feasible:
   self.aligned=None
   if t-self.improved>1.5 or t-self.align_start>6.:
    self.phase='RESTAGE';self.restart=t;self.dock_y=CENTERS[self.slot]+(.015 if p[1]<CENTERS[self.slot] else -.015);return self.cmd('REALIGN / BRAKE',0,0)
   return self.cmd('GOAL ALIGN / TURN',0,math.copysign(np.clip(2.6*abs(err),.42,1.4),err))
  if self.aligned is None:self.aligned=t
  if t-self.aligned<.4:return self.cmd('GOAL ALIGN / BRAKE',0,0)
  self.aligned=None;self.phase='IN';self.end.begin(t,p);return self.cmd('PUSH TO GOAL / STRAIGHT',.155,0)
 def follow(self,p,h,dst,label):
  self.target=dst;d=dst-p;a=wrap(math.atan2(d[1],d[0])-h);return self.cmd(label+' / TURN',0,np.clip(3*a,-1.55,1.55)) if abs(a)>.12 else self.cmd(label+' / DRIVE',np.clip(2*np.linalg.norm(d),.075,.30),0)
 def cmd(self,state,v,w):
  # Final command gate applies to every caller, including scans and new plans.
  if w and clearance(self.pos)<.035:
   v=w=0;state='GOAL GUARD / STOP'
  self.state=state;u=WHEEL@np.array([v,0,.11*w]);return dict(motor_pwm=np.rint(np.where(abs(u)>1e-6,np.sign(u)*(18+237*abs(u)/.32),0)).clip(-255,255).astype(int).tolist())
def text(im,s,xy,size=.55,color=(215,229,233)):
 cv2.putText(im,s,xy,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
def compose(o,top,b,ref,a):
 im=np.full((900,1280,3),(30,33,36),np.uint8);rgb=o['rgb'][:,:,::-1].copy();t=o['time_s'];p=b.pos;h=b.yaw
 for box in b.boxes:
  x,y,u,v=map(round,box['box']);cv2.rectangle(rgb,(x,y),(u,v),(36,240,130),2);cv2.putText(rgb,f"{box['confidence']:.2f}",(x,max(y-3,14)),0,.43,(26,155,50),1,cv2.LINE_AA)
 def panel(pic,x):im[30:450,x:x+640]=cv2.copyMakeBorder(cv2.resize(pic,(560,420)),0,0,40,40,cv2.BORDER_CONSTANT,value=(30,33,36))
 panel(rgb,0);panel(top[:,:,::-1],640);text(im,'01   ONBOARD CAMERA   /   YOLO',(18,21));text(im,'02   OVERHEAD CAMERA   /   PHYSICS',(658,21));text(im,'03   WORLD MODEL   /   VISUAL ESTIMATES',(18,478));text(im,f"04   STATE  /  SEED {board['seed']}  /  CENTER SCORING",(658,478));x0,y0=38,855;z=176
 def xy(v):return (int(x0+v[0]*z),int(y0-v[1]*z))
 cv2.rectangle(im,xy([0,2]),xy([3,0]),(53,58,61),-1)
 for q in grid[b.cover]:cv2.rectangle(im,xy(q+[-.049,.049]),xy(q+[.049,-.049]),(73,81,82),-1)
 cv2.line(im,xy([0,.42]),xy([0,1.57]),(222,181,60),4);path=[p,*[b.db[k]['p'] for k in b.seq]]
 for j,y in enumerate(CENTERS):
  col=(40,75,220) if b.counts[j]>=4 else (105,175,100)
  if b.goal is not None and j==b.slot:col=(60,210,255)
  cv2.line(im,xy([0,y-.065]),xy([0,y+.065]),col,6)
  text(im,str(b.counts[j]),xy([-.13,y-.025]),.43,col)
 if b.goal is not None:path.append(b.goal)
 if len(path)>1:cv2.polylines(im,[np.array(list(map(xy,path)))],False,(82,187,250),2)
 ids=b.candidates(t)
 for v in b.db.values():
  if v.get('pending') and not v['taken']:cv2.circle(im,xy(v['p']),6,(55,180,255),1)
 for v in b.db.values():
  if not v["taken"] and v["hits"]>=2 and t-v["t"]<40 and clearance(v["p"])<=.04:cv2.circle(im,xy(v["p"]),4,(55,180,255),1)
 for k in ids:cv2.circle(im,xy(b.db[k]['p']),4,(130,228,160),-1)
 cv2.arrowedLine(im,xy(p),xy(p+.17*np.array([math.cos(h),math.sin(h)])),(241,194,92),3,tipLength=.35);text(im,'Green: selectable   |   Rings: mapped near goal   |   Amber line: plan',(18,883),.47)
 rows=[('SIM CLOCK',f'{t:05.1f} / 120.0 s'),('STATE',b.state),('BASE XY',f'{p[0]:.3f}, {p[1]:.3f} m'),('YAW',f'{math.degrees(h):+.1f} deg'),('TARGET XY','--' if b.target is None else f'{b.target[0]:.2f}, {b.target[1]:.2f} m'),('CAM / MAP',f'{len(b.boxes)} boxes  /  {len(ids)} tracks'),('TRIP',f'{b.load} estimated on plate  /  {len(b.seq)} queued'),('PLANNER',f'{b.plans} solves  /  {b.odo:.2f} m traveled'),('WHEEL PWM','   '.join(map(str,a['motor_pwm']))),('REFEREE ONLY',f'{len(ref.active)} / 32 in home goal')]
 rows.extend([('SPEED / YAW RATE',f'{b.speed:.3f} m/s   {b.rate:+.1f} deg/s'),('GOAL SLOT','--' if b.goal is None else f'{b.slot+1} / 8  (estimated: {b.counts[b.slot]})')])
 for i,(k,v) in enumerate(rows):yy=508+i*32;text(im,k,(658,yy),.48,(154,174,180));text(im,v,(847,yy),.62,(142,224,177) if i in (1,9) else (237,237,231))
 return im

def main():
 out.mkdir(parents=True,exist_ok=True);reports.mkdir(parents=True,exist_ok=True);vid=out/'四视图.mp4';log=[];audit=[];prefix=int(os.environ.get('TEST_SECONDS','120'));positions=np.array([[*p,.0076] for p in board['points']]);env=Physics(c,positions,car=(*board['start'],.001));ref=Referee(c['goals'],32,center_only=True)
 # Overview framing only: fill the pane with the complete 3 x 2 m field.
 camid=mujoco.mj_name2id(env.m,mujoco.mjtObj.mjOBJ_CAMERA,'overview');env.m.cam_fovy[camid]=44.
 # Evaluate at all 240 Hz physics steps, including motor delay / rotation after braking.
 original=ref.update;breakdown={};checks=dict(turn_steps=0,unsafe_turn_steps=0,min_turn_clearance=999.,protected_turn_contacts=0);steps_seen=0
 def audited(t,poses):
  nonlocal steps_seen
  original(t,poses);new=env.judge_contact_events[steps_seen:];steps_seen+=len(new)
  if abs(env.d.qvel[5])>.08 or any(a['motor_pwm'][j]!=a['motor_pwm'][0] for j in range(1,4)):
   m=breakdown.setdefault(b.state,dict(samples=0,contacts=0,min_clear=999.));m['samples']+=1;cl=clearance(env.d.xpos[env.car]);m['min_clear']=min(m['min_clear'],cl);checks['turn_steps']+=1;checks['min_turn_clearance']=min(checks['min_turn_clearance'],cl);checks['unsafe_turn_steps']+=int(cl<=0)
   for ev in new:
    q=poses[ev['piece_id']][0];dist=math.hypot(max(0,q[0]-ZONE,-q[0]),max(0,.42-q[1],q[1]-1.57));touch=int(ev['role']=='plate' and dist<=Rpiece);checks['protected_turn_contacts']+=touch;m['contacts']+=touch
 ref.update=audited
 with tempfile.TemporaryDirectory() as tmp:
  exe=Path(tmp)/'solver';subprocess.run(['c++','-O3','-std=c++17',sim/'balance_solver.cpp','-o',exe],check=True)
  with subprocess.Popen([exe,'turns'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as solver,subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x900','-r','5','-i','-','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(vid)],stdin=subprocess.PIPE) as enc:
   b=Brain(solver)
   try:
    for k in range(prefix*10+1):
     mujoco.mj_kinematics(env.m,env.d);mujoco.mj_camlight(env.m,env.d);o=env.obs();a=b.act(o)
     if k%2==0 or ref.done:
      im=compose(o,env.render(True),b,ref,a);enc.stdin.write(im.tobytes());log.append(dict(t=env.time,pose=o['pose'],state=b.state,pwm=a['motor_pwm'],detections=len(b.boxes),tracks=len(b.candidates(env.time)),route=b.seq.copy(),pending_map={k:v['p'].tolist() for k,v in b.db.items() if v.get('pending') and not v['taken']},estimated_load=b.load,goal_slot=b.slot,slot_counts=b.counts.copy(),score=len(ref.active),odo=b.odo,plans=b.plans));
      if k==20:cv2.imwrite(str(out/'画面预览.jpg'),im)
     if k%150==0:print('sim',env.time,b.state,len(ref.active),flush=True)
     if ref.done or k==prefix*10:break
     n=len(env.judge_contact_events);is_turn=a['motor_pwm'][0]!=a['motor_pwm'][1];x0=len(ref.events);env.step(a,ref)
     # Contact coordinates + full-step poses checked separately below on every rotation step.
     audit.append(dict(t=o['time_s'],state=b.state,phase=b.phase,turn=is_turn,clear=clearance(b.pos),pwm=a['motor_pwm'],finish=b.end.reason,blade_back=getattr(b.end,'back',None),wall=getattr(b.end,'wall',None)))
   finally:enc.stdin.close();solver.stdin.close();env.close()
   if enc.wait():raise RuntimeError('video encoding failed')
 (reports/'运行记录.json').write_text(json.dumps(dict(seed=board['seed'],board=board,retreat_reviews=b.review_log,visual_recounts=b.recounts,scoring='center_inside_immediate',checks=checks,breakdown=breakdown,audit=audit,plate_radius=Rplate,safe_x=ZONE+SAFE+.025,result=ref.finish() if ref.done else None,frames=log),ensure_ascii=False));print('DONE',log[-1])
if __name__=='__main__':main()
