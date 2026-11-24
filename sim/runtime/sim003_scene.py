"""CPU-friendly 3D scene builder and finite-force dynamic equivalent omni drive."""
from pathlib import Path
import json,math,xml.etree.ElementTree as E
import numpy as np
import mujoco
from field_artwork import apply_field_artwork
from field_spec import FIELD_WIDTH_M, FIELD_DEPTH_M, WALL_HEIGHT_M, WALL_THICKNESS_M
from chess_visuals import add_chess_visuals
from piece_spec import RADIUS_M,HALF_HEIGHT_M,MASS_KG,SPAWN_Z_M,inertia as piece_inertia

ROOT=Path(__file__).resolve().parent
def vec(a):return ' '.join(f'{float(x):.12g}' for x in np.asarray(a).ravel())
def add(p,tag,**kw):return E.SubElement(p,tag,{k:vec(v) if isinstance(v,(tuple,list,np.ndarray)) else str(v) for k,v in kw.items()})
def scene_xml(asset_dir,position=(1,1,0.001),pieces=((1.19,1,.18),),yaw=0):
    asset_dir=Path(asset_dir).resolve();config=json.loads((asset_dir/'simulation.json').read_text())
    m=E.Element('mujoco',model='raspbot_sim003');add(m,'compiler',angle='radian',inertiafromgeom='false',autolimits='true',usethread='false')
    add(m,'option',timestep=1/240,gravity=(0,0,-9.81),solver='Newton',iterations=80,tolerance=1e-9,cone='elliptic')
    visual=add(m,'visual');add(visual,'global',offwidth=640,offheight=480)
    default=add(m,'default');add(default,'geom',friction=(.3,.001,.0001),margin=0,gap=0,solref=(.01,1),solimp=(.95,.99,.001),condim=3)
    asset=add(m,'asset')
    for item in config['visuals']:add(asset,'mesh',name=item['name']+'_visual',file=str(asset_dir/item['file']))
    for item in config['convex_parts']:add(asset,'mesh',name=item['name']+'_mesh',vertex=vec(item['vertices']))
    world=add(m,'worldbody');add(world,'light',pos=(1.5,1,3),dir=(0,0,-1),diffuse=(.8,.8,.8))
    x, y = FIELD_WIDTH_M, FIELD_DEPTH_M
    add(world,'geom',name='floor',type='plane',pos=(x/2,y/2,0),size=(x/2,y/2,.1),rgba=(.28,.36,.4,1),group=0)
    half_height = WALL_HEIGHT_M / 2
    half_thickness = WALL_THICKNESS_M / 2
    walls = [
        ('west', (-half_thickness, y/2, half_height), (half_thickness, y/2+WALL_THICKNESS_M, half_height)),
        ('east', (x+half_thickness, y/2, half_height), (half_thickness, y/2+WALL_THICKNESS_M, half_height)),
        ('south', (x/2, -half_thickness, half_height), (x/2, half_thickness, half_height)),
        ('north', (x/2, y+half_thickness, half_height), (x/2, half_thickness, half_height)),
    ]
    for name,pos,size in walls:add(world,'geom',name='wall_'+name,type='box',pos=pos,size=size,rgba=(.7,.7,.65,1),group=0)
    # Ruler markers provide a rendered 100mm reference without influencing contacts.
    for i in range(11):add(world,'geom',name=f'ruler_{i}',type='box',pos=(1+i*.01,.75,.0002),size=(.0003,.004 if i%5 else .008,.0002),rgba=(1,1,1,1),contype=0,conaffinity=0,group=0)
    car=add(world,'body',name='car',pos=position,quat=(math.cos(yaw/2),0,0,math.sin(yaw/2)));add(car,'freejoint',name='car_free')
    add(car,'inertial',pos=(-.01,0,.045),mass=1.2,diaginertia=(.002,.004,.005))
    colors={'body':(.34,.39,.41,1),'wheel':(.15,.18,.19,1),'board':(.35,.54,.42,1),'metal':(.68,.61,.4,1),'push':(.9,.5,.17,1)}
    for item in config['visuals']:add(car,'geom',name=item['name']+'_visual',type='mesh',mesh=item['name']+'_visual',rgba=colors[item['name']],contype=0,conaffinity=0,group=1)
    add(car,'geom',name='chassis_collision',type='box',pos=(-.01,0,.046),size=(.075,.046,.02),rgba=(.1,.8,.8,.28),group=2)
    for i,(x,y) in enumerate([(x,y) for x in [-.069965,.046961] for y in [-.066275,.066275]]):add(car,'geom',name=f'support_{i}',type='ellipsoid',pos=(x,y,.030234),size=(.03014,.015375,.030234),friction=(.03,.001,.0001),priority=1,rgba=(.2,.8,.2,.28),group=2)
    for item in config['convex_parts']:add(car,'geom',name=item['name'],type='mesh',mesh=item['name']+'_mesh',rgba=(.1,1,.4,.5),group=2)
    for item in config['tip_caps']:add(car,'geom',name=item['name'],type='cylinder',pos=item['position'],size=(item['radius'],item['half_height']),rgba=(.1,1,.4,.5),group=2)
    pan=add(car,'body',name='pan_body',pos=(.02,0,.10));add(pan,'joint',name='pan',type='hinge',axis=(0,0,1),range=(-math.pi/2,math.pi/2),damping=.002,armature=.00001)
    add(pan,'inertial',pos=(0,0,0),mass=.004,diaginertia=(.000001,.000001,.000001));add(pan,'geom',name='pan_proxy',type='cylinder',size=(.012,.003),rgba=(.3,.6,.9,1),contype=0,conaffinity=0,group=1)
    tilt=add(pan,'body',name='tilt_body',pos=(.006,0,.008));add(tilt,'joint',name='tilt',type='hinge',axis=(0,1,0),range=(-math.pi/6,math.pi/6),damping=.002,armature=.00001)
    add(tilt,'inertial',pos=(.008,0,0),mass=.006,diaginertia=(.000001,.000001,.000001));add(tilt,'geom',name='tilt_proxy',type='box',pos=(.008,0,0),size=(.008,.012,.006),rgba=(.1,.3,.6,1),contype=0,conaffinity=0,group=1)
    for i,pos in enumerate(pieces):
        piece=add(world,'body',name=f'piece_{i:02}',pos=(*pos[:2], max(pos[2], SPAWN_Z_M)));add(piece,'freejoint',name=f'piece_{i:02}_free')
        add(piece,'inertial',pos=(0,0,0),mass=MASS_KG,diaginertia=piece_inertia())
        add(piece,'geom',name=f'piece_{i:02}_geom',type='cylinder',size=(RADIUS_M,HALF_HEIGHT_M),rgba=(.85,.04,.03,1) if i%2==0 else (.04,.04,.04,1),group=0)
    actuators=add(m,'actuator')
    for name in ['pan','tilt']:add(actuators,'position',name=name+'_servo',joint=name,kp=.05,kv=.002,ctrllimited='true',ctrlrange=(-math.pi/2,math.pi/2) if name=='pan' else (-math.pi/6,math.pi/6),forcelimited='true',forcerange=(-.02,.02))
    add_chess_visuals(m, asset_dir)
    apply_field_artwork(m, asset_dir)
    return E.tostring(m,encoding='unicode')

class DynamicCar:
    def __init__(self,model,data):
        self.m=model;self.d=data;self.body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'car');self.max_force=1.2;self.max_torque=.15
    def apply(self,vx=0,vy=0,wz=0,pan=0,tilt=0):
        target=np.array([vx,vy]);speed=np.linalg.norm(target)
        if speed>.25:target*=.25/speed
        rot=self.d.xmat[self.body].reshape(3,3);world_target=(rot@np.array([*target,0]))[:2]
        # qvel first 3 are translational velocity in world frame, first 6 belong to free base.
        force=1.2*12*(world_target-self.d.qvel[:2]);norm=np.linalg.norm(force)
        if norm>self.max_force:force*=self.max_force/norm
        torque=float(np.clip(.15*(np.clip(wz,-1,1)-self.d.qvel[5]),-self.max_torque,self.max_torque))
        self.d.xfrc_applied[:]=0;self.d.xfrc_applied[self.body,:2]=force;self.d.xfrc_applied[self.body,5]=torque
        self.d.ctrl[:]=[np.clip(pan,-math.pi/2,math.pi/2),np.clip(tilt,-math.pi/6,math.pi/6)]
        return float(np.linalg.norm(force)),abs(torque)
