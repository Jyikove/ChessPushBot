"""Private MuJoCo runner: finite four-wheel PWM force/slip model; no velocity targets."""
import sys,math,xml.etree.ElementTree as E
from pathlib import Path
import numpy as np,mujoco,cv2
ROOT=Path(__file__).resolve().parents[1]
from sim003_scene import scene_xml
from camera_alignment_adapter import load_profile,apply_scene,warp
from core import WHEEL,validate_action
class Physics:
 def __init__(self,c,positions,car=(1.5,1.,.001),yaw=0,render=True):
  self.c=c;self.car_start=car;self.positions=positions
  xml=E.fromstring(scene_xml(ROOT.parent/'assets/robot',position=car,pieces=positions,yaw=yaw))
  # Static fixed camera, gimbal posture represented explicitly by published extrinsics.
  for name in ['pan','tilt']:
   joint=xml.find('.//joint[@name="'+name+'"]');parent=xml.find('.//body[@name="'+('pan_body' if name=='pan' else 'tilt_body')+'"]');parent.remove(joint)
  xml.remove(xml.find('actuator'))
  profile,self.map,rf=load_profile(ROOT/'resources/camera')
  profile.update(c['camera'])
  tilt=xml.find('.//body[@name="tilt_body"]');E.SubElement(tilt,'camera',name='onboard')
  apply_scene(xml,profile,rf)
  world=xml.find('worldbody');E.SubElement(world,'camera',name='overview',pos='1.5 1 3',xyaxes='1 0 0 0 1 0',fovy='60')
  self.m=mujoco.MjModel.from_xml_string(E.tostring(xml,encoding='unicode'));self.d=mujoco.MjData(self.m)
  for _ in range(240):mujoco.mj_step(self.m,self.d)
  self.d.time=0;self.d.qvel[:]=0;self.d.qacc_warmstart[:]=0;mujoco.mj_forward(self.m,self.d)
  self.car=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,'car');self.pieces=[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,f'piece_{i:02}') for i in range(len(positions))]
  self.renderer=mujoco.Renderer(self.m,height=960,width=1280) if render else None;self.opt=mujoco.MjvOption();self.opt.geomgroup[:]=[1,1,0,0,0,0]
  self.wheel_speed=np.zeros(4);self.delay=[];self.u=np.zeros(4);self.max_force=0;self.steps=0;self.contacts=0;self.rgb=None;self.image_time=0;self.last_control_contacts=set();self.judge_contact_events=[];self.robot_piece_contacts=0;self.first_robot_piece_contact=None
 def pose(self):
  p=self.d.xpos[self.car];r=self.d.xmat[self.car].reshape(3,3)
  return dict(x_m=float(p[0]),y_m=float(p[1]),yaw_rad=math.atan2(r[1,0],r[0,0]),time_s=self.time)
 @property
 def time(self):return self.steps/240
 def truth(self):return [(self.d.xpos[i].copy(),self.d.xmat[i].reshape(3,3).copy()) for i in self.pieces]
 def render(self,overview=False):
  self.renderer.update_scene(self.d,camera='overview' if overview else 'onboard',scene_option=self.opt)
  a=self.renderer.render().copy()
  return cv2.resize(a,(640,480)) if overview else warp(a,self.map)
 def obs(self):
  if self.rgb is None or self.steps%48==0:self.rgb=self.render();self.image_time=self.time
  return dict(time_s=self.time,dt_s=.1,rgb=self.rgb.copy(),image_time_s=self.image_time,pose=self.pose())
 def step(self,a,ref):
  self.last_control_contacts=set();u=np.array(validate_action(a));self.delay.append((self.steps+6,u))
  for _ in range(24):
   while self.delay and self.delay[0][0]<=self.steps:_,self.u=self.delay.pop(0)
   effective=np.sign(self.u)*np.maximum(0,abs(self.u)-18)/(255-18)*.32
   self.wheel_speed+=(effective-self.wheel_speed)*(1-math.exp(-1/240/.09))
   r=self.d.xmat[self.car].reshape(3,3);bodyv=r.T@self.d.qvel[:3];omega=self.d.qvel[5]
   slip=self.wheel_speed-WHEEL@np.array([bodyv[0],bodyv[1],.11*omega])
   traction=np.clip(8*slip,-.65,.65)
   wrench=WHEEL.T@traction;fw=r@np.array([wrench[0],wrench[1],0])
   self.d.xfrc_applied[:]=0;self.d.xfrc_applied[self.car,:3]=fw;self.d.xfrc_applied[self.car,5]=.11*wrench[2]
   self.max_force=max(self.max_force,float(np.linalg.norm(fw)))
   mujoco.mj_step(self.m,self.d);self.steps+=1
   self.contacts+=sum(1 for x in self.d.contact if ('piece_' in (mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,int(x.geom1)) or '') or 'piece_' in (mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,int(x.geom2)) or '')))
   for contact in self.d.contact:
    a,b=int(contact.geom1),int(contact.geom2);ba,bb=int(self.m.geom_bodyid[a]),int(self.m.geom_bodyid[b])
    if (ba in self.pieces and bb>0 and bb not in self.pieces) or (bb in self.pieces and ba>0 and ba not in self.pieces):
     piece_id=self.pieces.index(ba if ba in self.pieces else bb);robot_geom=b if ba in self.pieces else a;name=mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,robot_geom)or'';role='wheel'if name.startswith('support_')else('chassis'if name=='chassis_collision'else('plate'if name.startswith('v5_')else'other_robot'));self.judge_contact_events.append(dict(time_s=self.time,piece_id=piece_id,robot_geom=name,role=role,point=contact.pos.tolist()))
     self.last_control_contacts.add(self.pieces.index(ba if ba in self.pieces else bb));self.robot_piece_contacts+=1
     if self.first_robot_piece_contact is None:self.first_robot_piece_contact=self.time
   if not np.isfinite(self.d.qpos).all() or any(w.number for w in self.d.warning):raise RuntimeError('nonfinite_or_engine_warning')
   ref.update(self.time,self.truth())
   if ref.done:break
 def close(self):
  if self.renderer:self.renderer.close()
