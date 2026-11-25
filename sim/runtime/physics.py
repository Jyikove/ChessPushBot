"""Private MuJoCo runner: finite four-wheel PWM force/slip model; no velocity targets."""
import os,sys,math,xml.etree.ElementTree as E
from copy import deepcopy
from pathlib import Path
import numpy as np,mujoco,cv2
ROOT=Path(__file__).resolve().parents[1]
from sim003_scene import scene_xml
from camera_alignment_adapter import load_profile,apply_scene,warp
from core import WHEEL,validate_action
from room_appearance import apply_room,camera_response
from photo_room import apply_photos

def wheel_force(pwm, speed, deadband, limit, rotation, velocity):
 effective=np.sign(pwm)*np.maximum(0,abs(pwm)-deadband)/(255-deadband)*limit
 speed+=(effective-speed)*(1-math.exp(-1/240/.09))
 body_velocity=rotation.T@velocity[:3]
 slip=speed-WHEEL@np.array([body_velocity[0],body_velocity[1],.11*velocity[5]])
 wrench=WHEEL.T@np.clip(8*slip,-.65,.65)
 return rotation@np.array([wrench[0],wrench[1],0]),.11*wrench[2]

class Physics:
 def __init__(self,c,positions,car=(1.5,1.,.001),yaw=0,render=True,appearance='baseline',opponent=None):
  self.c=c;self.car_start=car;self.positions=positions
  wheel_response=c.get('wheel_response',{})
  self.wheel_deadband=np.asarray(wheel_response.get('deadband_pwm',18),dtype=float)
  self.wheel_max_speed=np.asarray(wheel_response.get('max_speed_m_s',.32),dtype=float)
  xml=E.fromstring(scene_xml(ROOT.parent/'assets/robot',position=car,pieces=positions,yaw=yaw))
  # Static fixed camera, gimbal posture represented explicitly by published extrinsics.
  for name in ['pan','tilt']:
   joint=xml.find('.//joint[@name="'+name+'"]');parent=xml.find('.//body[@name="'+('pan_body' if name=='pan' else 'tilt_body')+'"]');parent.remove(joint)
  xml.remove(xml.find('actuator'))
  profile,self.map,rf=load_profile(ROOT/'resources/camera')
  profile.update(c['camera'])
  tilt=xml.find('.//body[@name="tilt_body"]');E.SubElement(tilt,'camera',name='onboard')
  apply_scene(xml,profile,rf)
  world=xml.find('worldbody')
  if opponent is not None:
   start,heading=opponent
   second=deepcopy(world.find("body[@name='car']"))
   for node in second.iter():
    if 'name' in node.attrib:node.set('name',node.get('name')+'_B')
   second.set('pos',' '.join(map(str,start)))
   second.set('quat',f'{math.cos(heading/2)} 0 0 {math.sin(heading/2)}')
   world.append(second)
  apply_room(xml,appearance);self.camera_gain=camera_response(appearance,self.map.shape[:2])
  E.SubElement(world,'camera',name='overview',pos='1.5 1 3',xyaxes='1 0 0 0 1 0',fovy='60')
  assets={}
  if appearance=='photo-room':
   folder=os.environ.get('SIM_PHOTO_DIR',ROOT.parent.parent/'field-calibration/data/images')
   assets=apply_photos(xml,folder,ROOT/'resources/room/background.png')
  for element in xml.findall('asset/*[@file]'):
   if element.get('file') in assets:continue
   source=Path(element.get('file'));name=element.get('name')+source.suffix
   assets[name]=source.read_bytes();element.set('file',name)
  self.m=mujoco.MjModel.from_xml_string(E.tostring(xml,encoding='unicode'),assets=assets);self.d=mujoco.MjData(self.m)
  self.room_lighting=appearance!='baseline'
  if self.room_lighting:
   light_names=[mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_LIGHT,index) for index in range(self.m.nlight)]
   self.strip_lights=[index for index,name in enumerate(light_names) if name and name.startswith('room_strip_')]
   self.window_light=light_names.index('field_overhead_fill')
   self.ambient_light=light_names.index('field_overhead_key')
   self.light_diffuse=self.m.light_diffuse.copy()
   self.light_specular=self.m.light_specular.copy()
   self.light_ambient=self.m.light_ambient[self.ambient_light].copy()
  self.soft_light=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_LIGHT,'field_overhead_fill') if appearance=='photo-room' else None
  for _ in range(240):mujoco.mj_step(self.m,self.d)
  self.d.time=0;self.d.qvel[:]=0;self.d.qacc_warmstart[:]=0;mujoco.mj_forward(self.m,self.d)
  self.car=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,'car');self.pieces=[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,f'piece_{i:02}') for i in range(len(positions))]
  self.renderer=mujoco.Renderer(self.m,height=960,width=1280) if render else None;self.opt=mujoco.MjvOption();self.opt.geomgroup[:]=[1,1,0,0,0,0]
  self.onboard_opt=mujoco.MjvOption();self.onboard_opt.geomgroup[:]=[1,1,0,1,0,0]
  self.wheel_speed=np.zeros(4);self.delay=[];self.u=np.zeros(4);self.max_force=0;self.steps=0;self.contacts=0;self.rgb=None;self.image_time=0;self.last_control_contacts=set();self.judge_contact_events=[];self.robot_piece_contacts=0;self.first_robot_piece_contact=None
 def set_room_lighting(self,strip_percent,window_percent):
  strip,window=strip_percent/100,window_percent/100
  for channel,original in ((self.m.light_diffuse,self.light_diffuse),(self.m.light_specular,self.light_specular)):
   channel[self.strip_lights]=original[self.strip_lights]*strip
   channel[self.window_light]=original[self.window_light]*window
  self.m.light_ambient[self.ambient_light]=self.light_ambient*(.25+.45*strip+.30*window)
 def pose(self):
  p=self.d.xpos[self.car];r=self.d.xmat[self.car].reshape(3,3)
  return dict(x_m=float(p[0]),y_m=float(p[1]),yaw_rad=math.atan2(r[1,0],r[0,0]),time_s=self.time)
 @property
 def time(self):return self.steps/240
 def truth(self):return [(self.d.xpos[i].copy(),self.d.xmat[i].reshape(3,3).copy()) for i in self.pieces]
 def render(self,overview=False,*,camera='onboard'):
  self.renderer.update_scene(self.d,camera='overview' if overview else camera,scene_option=self.opt if overview else self.onboard_opt)
  a=self.renderer.render().copy()
  if overview:return cv2.resize(a,(640,480))
  if self.soft_light is not None and np.any(self.m.light_diffuse[self.soft_light]):
   light=self.soft_light;direction=self.m.light_dir[light].copy();mix=a.astype(np.float32)
   try:
    offsets=((-0.08,-0.05),(-0.14,0.04),(0.08,-0.05),(0.14,0.04))
    for horizontal,vertical in offsets:
     shifted=direction+np.array([0,horizontal,vertical]);self.m.light_dir[light]=shifted/np.linalg.norm(shifted)
     mujoco.mj_camlight(self.m,self.d)
     self.renderer.update_scene(self.d,camera=camera,scene_option=self.onboard_opt)
     mix+=self.renderer.render()
   finally:
    self.m.light_dir[light]=direction;mujoco.mj_camlight(self.m,self.d)
   a=np.rint(mix/(len(offsets)+1)).astype(np.uint8)
  rgb=warp(a,self.map)
  return rgb if self.camera_gain is None else np.clip(np.rint(rgb*self.camera_gain),0,255).astype(np.uint8)
 def obs(self):
  if self.rgb is None or self.steps%48==0:self.rgb=self.render();self.image_time=self.time
  return dict(time_s=self.time,dt_s=.1,rgb=self.rgb.copy(),image_time_s=self.image_time,pose=self.pose())
 def step(self,a,ref):
  self.last_control_contacts=set();u=np.array(validate_action(a));self.delay.append((self.steps+6,u))
  for _ in range(24):
   while self.delay and self.delay[0][0]<=self.steps:_,self.u=self.delay.pop(0)
   fw,torque=wheel_force(self.u,self.wheel_speed,self.wheel_deadband,self.wheel_max_speed,
                        self.d.xmat[self.car].reshape(3,3),self.d.qvel[:6])
   self.d.xfrc_applied[:]=0;self.d.xfrc_applied[self.car,:3]=fw;self.d.xfrc_applied[self.car,5]=torque
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
