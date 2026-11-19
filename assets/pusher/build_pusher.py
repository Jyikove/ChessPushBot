from pathlib import Path
import json,numpy as np
from OCP.gp import gp_Pnt,gp_Vec,gp_Ax2,gp_Dir
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox,BRepPrimAPI_MakeCylinder,BRepPrimAPI_MakePrism
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon,BRepBuilderAPI_MakeFace,BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire
from OCP.GC import GC_MakeArcOfCircle
import math
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse,BRepAlgoAPI_Cut
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.STEPControl import STEPControl_Writer,STEPControl_AsIs
from OCP.StlAPI import StlAPI_Writer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE,TopAbs_SOLID
from OCP.TopoDS import TopoDS
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
p=Path(__file__).parent;src=p.parent
origin=np.array([-50.34601384544353,55.78782961366948,182.29703418374])
mount_height=18.985724369286885  # mm; established mounting surface above model ground
# All exported geometry uses millimetres, origin at mounting-hole midpoint in XY, ground at Z=0.
def box(x,y,z,dx,dy,dz):return BRepPrimAPI_MakeBox(gp_Pnt(x,y,z),dx,dy,dz).Shape()
def prism(points,vec):
 w=BRepBuilderAPI_MakePolygon()
 for pt in points:w.Add(gp_Pnt(*pt))
 w.Close();return BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(w.Wire()).Face(),gp_Vec(*vec)).Shape()
low=3.;high=27.;wall=3.;mount_thickness=3.
hole_region_thickness=2.;hole_region_radius=3.5
# Three principal circular arcs, tangent at both joins. Width includes the 3 mm wall.
target_width=200.
target_centre_arc_depth=10.
r=26.;half_width=target_width/2

def centre_arc_sagitta(radius):
    """Sagitta of the outside central arc for a candidate centre-line radius."""
    join=(half_width-r-wall/2)/(1-r/radius)
    angle=math.asin(join/radius)
    return (radius+wall/2)*(1-math.cos(angle))

# Solve the radius from the requested outside sagitta so the drawing and metadata
# continue to describe the actual exported solid rather than a rounded estimate.
lo,hi=100.,2000.
for _ in range(80):
    mid=(lo+hi)/2
    if centre_arc_sagitta(mid)>target_centre_arc_depth:lo=mid
    else:hi=mid
R=(lo+hi)/2
join_x=(half_width-r-wall/2)/(1-r/R)
theta=math.asin(join_x/R);phi0=-math.pi/2+theta;phi1=math.radians(35)
cy=17+R
right=(join_x-r*math.sin(theta),cy-(R-r)*math.cos(theta));left=(-right[0],right[1])
wire=BRepBuilderAPI_MakeWire()
def arc(center,radius,start,end):
 def point(t):return gp_Pnt(center[0]+radius*math.cos(t),center[1]+radius*math.sin(t),low)
 curve=GC_MakeArcOfCircle(point(start),point((start+end)/2),point(end)).Value()
 wire.Add(BRepBuilderAPI_MakeEdge(curve).Edge())
# Outside boundary, left tip -> back -> right tip.
arc(left,r+wall/2,math.pi-phi1,math.pi-phi0)
arc((0,cy),R+wall/2,-math.pi/2-theta,phi0)
arc(right,r+wall/2,phi0,phi1)
rt=(right[0]+r*math.cos(phi1),right[1]+r*math.sin(phi1))
lt=(-rt[0],rt[1])
arc(rt,wall/2,phi1,phi1+math.pi)
# Inside boundary, reversed, followed by rounded left tip.
arc(right,r-wall/2,phi1,phi0)
arc((0,cy),R-wall/2,phi0,-math.pi/2-theta)
arc(left,r-wall/2,math.pi-phi0,math.pi-phi1)
arc(lt,wall/2,-phi1,math.pi-phi1)
curved_wall=BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(wire.Wire()).Face(),gp_Vec(0,0,high-low)).Shape()
# Mount sheet is 3 mm thick; underside pockets leave 2 mm around each screw.
parts=[box(-16,-8,mount_height-mount_thickness,32,18,mount_thickness),box(-16,8,mount_height-mount_thickness,32,10,mount_thickness),curved_wall]
# Gradual underside transition: 3 mm to 5 mm over 4 mm of forward travel.
transition=prism([(-16,7,mount_height-mount_thickness),(-16,7,mount_height),(-16,11,mount_height),(-16,11,mount_height-5)],(32,0,0))
parts.append(transition)
# Five millimetre thick fan spreads the load into a 60 mm wide section of the back of the blade.
fan=prism([(-16,11,mount_height-5),(16,11,mount_height-5),(30,18,mount_height-5),(-30,18,mount_height-5)],(0,0,5))
parts.append(fan)
# Three 4 mm ribs support the fan below, without entering screw/standoff clearance zones.
for cx in [-12,0,12]:
 x=cx-2;parts.append(prism([(x,7,mount_height-mount_thickness),(x,18,low+1),(x,18,mount_height-mount_thickness)],(4,0,0)))
reinforcement={'mount_sheet_mm':mount_thickness,'hole_region_thickness_mm':hole_region_thickness,'hole_region_radius_mm':hole_region_radius,'connector_max_thickness_mm':5,'connector_width_at_blade_mm':60,'transition_y_mm':[7,11],'thickness_transition_mm':[mount_thickness,5],'ribs_count':3,'rib_thickness_mm':4,'added_material_bounds_mm':[[-30,7,low+1],[30,18,mount_height]],'installation_surface_z_mm':mount_height,'nominal_ground_clearance_mm':low}
(p/'reinforcement.json').write_text(json.dumps(reinforcement,ensure_ascii=False,indent=2))
# Store analytic plan geometry to allow exact inspection independent of mesh tessellation.
central_arc_depth=round(centre_arc_sagitta(R),6)
curve_params={'centre_arc_radius_mm':R,'side_arc_radius_mm':r,'central_arc_centre':[0,cy],'right_arc_centre':list(right),'left_arc_centre':list(left),'central_join_half_angle_deg':math.degrees(theta),'central_arc_depth_mm':central_arc_depth,'side_tip_angle_deg':35,'side_arc_sweep_deg':math.degrees(phi1-phi0),'side_arc_reduction_deg':20,'tip_round_radius_mm':wall/2,'right_tip_centre':list(rt),'join_tangent_error_deg':0.0,'continuity':'G1 tangent, curvature changes at joins','width_including_wall_mm':target_width,'reference':'shape.png reflected so opening faces +Y'}
(p/'arcs.json').write_text(json.dumps(curve_params,ensure_ascii=False,indent=2))
shape=parts[0]
for part in parts[1:]:shape=BRepAlgoAPI_Fuse(shape,part).Shape()
for x in [-9.5,9.5]:
 pocket=BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,0,mount_height-mount_thickness-1),gp_Dir(0,0,1)),hole_region_radius,mount_thickness-hole_region_thickness+1).Shape()
 shape=BRepAlgoAPI_Cut(shape,pocket).Shape()
 hole=BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,0,mount_height-5),gp_Dir(0,0,1)),1.5,8).Shape();shape=BRepAlgoAPI_Cut(shape,hole).Shape()
assert BRepCheck_Analyzer(shape).IsValid()
exp=TopExp_Explorer(shape,TopAbs_SOLID);n=0
while exp.More():n+=1;exp.Next()
assert n==1,n
w=STEPControl_Writer();w.Transfer(shape,STEPControl_AsIs);w.Write(str(p/'pusher.step'))
BRepMesh_IncrementalMesh(shape,.15,False,.18,True);sw=StlAPI_Writer();sw.ASCIIMode=False;sw.Write(shape,str(p/'pusher.stl'))
arr=[];exp=TopExp_Explorer(shape,TopAbs_FACE)
while exp.More():
 loc=TopLoc_Location();f=TopoDS.Face_s(exp.Current());t=BRep_Tool.Triangulation_s(f,loc)
 if t:
  ns=np.array([[v.X(),v.Y(),v.Z()] for v in [t.Node(k).Transformed(loc.Transformation()) for k in range(1,t.NbNodes()+1)]])
  ts=np.array([t.Triangle(k).Get() for k in range(1,t.NbTriangles()+1)])-1
  if str(f.Orientation()).endswith('REVERSED'):ts=ts[:,::-1]
  arr.extend(ns[ts])
 exp.Next()
np.save(p/'push_mesh.npy',np.array(arr,dtype=np.float32))
props=GProp_GProps();BRepGProp.VolumeProperties_s(shape,props)
params={'status':'concept preview, not validated for printing','unit':'mm','hole_pitch':19,'hole_diameter':3,'mounting_tongue_thickness':mount_thickness,'hole_region_thickness_mm':hole_region_thickness,'hole_region_radius_mm':hole_region_radius,'reinforced_connector_thickness':5,'ribs_count':3,'rib_thickness':4,'nominal_ground_clearance':low,'push_height':high-low,'max_width':target_width,'central_arc_depth_mm':central_arc_depth,'forward_extent_from_hole_line':rt[1]+wall/2,'mounting_surface_height_from_model_ground':mount_height,'volume_mm3':props.Mass(),'retained_standoffs':True,'excluded_original_shells':[417,195,438],'car_source':'../robot/raspbot.step'}
(p/'parameters.json').write_text(json.dumps(params,ensure_ascii=False,indent=2));print(json.dumps(params,ensure_ascii=False),flush=True)
