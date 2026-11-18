from pathlib import Path
import json,math,numpy as np,trimesh
from OCP.gp import gp_Pnt,gp_Vec,gp_Ax2,gp_Dir
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox,BRepPrimAPI_MakeCylinder,BRepPrimAPI_MakePrism
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon,BRepBuilderAPI_MakeFace
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut,BRepAlgoAPI_Fuse,BRepAlgoAPI_Common
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.STEPControl import STEPControl_Writer,STEPControl_AsIs,STEPControl_Reader
from OCP.StlAPI import StlAPI_Writer
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE,TopAbs_SOLID,TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
p=Path(__file__).parent
PILLAR_HEIGHT=14.;PLATE_T=2.;IMU_RISER=10.;IMU_PCB_Z=PLATE_T+IMU_RISER
MOUNT_PITCH_X=49.;MOUNT_PITCH_Y=58.
mount_xy=[(sx*24.5,sy*29) for sx in [-1,1] for sy in [-1,1]]
imu_xy=[(sx*8.5,sy*12.6) for sx in [-1,1] for sy in [-1,1]]
def box(x,y,z,dx,dy,dz):return BRepPrimAPI_MakeBox(gp_Pnt(x,y,z),dx,dy,dz).Shape()
def cyl(x,y,z,r,h):return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,y,z),gp_Dir(0,0,1)),r,h).Shape()
def polygon(points,z,h):
 w=BRepBuilderAPI_MakePolygon()
 for x,y in points:w.Add(gp_Pnt(x,y,z))
 w.Close();return BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(w.Wire()).Face(),gp_Vec(0,0,h)).Shape()
def fuse(items):
 out=items[0]
 for a in items[1:]:out=BRepAlgoAPI_Fuse(out,a).Shape()
 return out

def rounded_rect(cx,cy,dx,dy,r,z,h):return fuse([box(cx-dx/2+r,cy-dy/2,z,dx-2*r,dy,h),box(cx-dx/2,cy-dy/2+r,z,dx,dy-2*r,h)]+[cyl(cx+sx*(dx/2-r),cy+sy*(dy/2-r),z,r,h) for sx in [-1,1] for sy in [-1,1]])
def slot(a,b,r,z,h):
 a=np.array(a);b=np.array(b);v=b-a;v=np.array([-v[1],v[0]])/np.linalg.norm(v)*r
 return fuse([cyl(*a,z,r,h),cyl(*b,z,r,h),polygon([a+v,b+v,b-v,a-v],z,h)])
# Frame side centrelines pass through the IMU pillar axes.
# Four independent rounded arms
# connect the centre to the Raspberry Pi mounting pattern; no outside frame.
base=rounded_rect(0,0,22,30.2,2.5,0,PLATE_T)
base=BRepAlgoAPI_Cut(base,rounded_rect(0,0,12,20.2,2,-1,PLATE_T+2)).Shape()
for sx in [-1,1]:
 for sy in [-1,1]:
  arm=slot((sx*8.5,sy*12.6),(sx*24.5,sy*29),2.5,0,PLATE_T)
  base=BRepAlgoAPI_Fuse(base,arm).Shape()
  base=BRepAlgoAPI_Fuse(base,cyl(sx*24.5,sy*29,0,3.5,PLATE_T)).Shape()
for x,y in imu_xy:base=BRepAlgoAPI_Fuse(base,cyl(x,y,0,3,IMU_PCB_Z)).Shape()
# Align to the four Raspberry Pi mounting columns, rather than the fan body.
mount_cuts=[]
for x,y in mount_xy:
 cut=cyl(x,y,-1,1.45,16);mount_cuts.append(cut);base=BRepAlgoAPI_Cut(base,cut).Shape()
# Prototype direct-fastening version: 1.7 mm pilot, 8 mm deep from boss top.
# M2x8 screw passes through 1.6 mm PCB and nominally engages 6.4 mm of plastic.
for x,y in imu_xy:
 base=BRepAlgoAPI_Cut(base,cyl(x,y,IMU_PCB_Z-8,.85,9)).Shape()
assert BRepCheck_Analyzer(base).IsValid();e=TopExp_Explorer(base,TopAbs_SOLID);n=0
while e.More():n+=1;e.Next()
assert n==1,n

def mesh(sh):
 BRepMesh_IncrementalMesh(sh,.06,False,.12,True);e=TopExp_Explorer(sh,TopAbs_FACE);arr=[]
 while e.More():
  f=TopoDS.Face_s(e.Current());loc=TopLoc_Location();t=BRep_Tool.Triangulation_s(f,loc)
  if t:
   ns=np.array([[v.X(),v.Y(),v.Z()] for v in [t.Node(k).Transformed(loc.Transformation()) for k in range(1,t.NbNodes()+1)]])
   idx=np.array([t.Triangle(k).Get() for k in range(1,t.NbTriangles()+1)])-1
   if f.Orientation()==TopAbs_REVERSED:idx=idx[:,::-1]
   arr.extend(ns[idx])
  e.Next()
 return np.array(arr,dtype=np.float32)
def export(sh,name):
 w=STEPControl_Writer();w.Transfer(sh,STEPControl_AsIs);w.Write(str(p/f'{name}.step'));BRepMesh_IncrementalMesh(sh,.06,False,.12,True);sw=StlAPI_Writer();sw.ASCIIMode=False;sw.Write(sh,str(p/f'{name}.stl'))
export(base,'imu_mount');np.save(p/'adapter_mesh.npy',mesh(base))
