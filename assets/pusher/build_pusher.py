from pathlib import Path
import json,numpy as np
from OCP.gp import gp_Pnt,gp_Vec,gp_Ax2,gp_Dir
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder,BRepPrimAPI_MakePrism
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon,BRepBuilderAPI_MakeFace,BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire
from OCP.GC import GC_MakeArcOfCircle
import math
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse,BRepAlgoAPI_Cut
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.STEPControl import STEPControl_Writer,STEPControl_AsIs
from OCP.StlAPI import StlAPI_Writer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
p=Path(__file__).parent;src=p.parent
origin=np.array([-50.34601384544353,55.78782961366948,182.29703418374])
mount_height=18.985724369286885  # mm; established mounting surface above model ground
# All exported geometry uses millimetres, origin at mounting-hole midpoint in XY, ground at Z=0.
def prism(points,vec):
 w=BRepBuilderAPI_MakePolygon()
 for pt in points:w.Add(gp_Pnt(*pt))
 w.Close();return BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(w.Wire()).Face(),gp_Vec(*vec)).Shape()
low=3.;high=27.;mount_thickness=3.
hole_region_thickness=1.;hole_region_radius=3.5
# Keep the mounting interface and Z datums fixed as the planform changes.
target_width=290.
plan_scale=target_width/200.
wall=3.*plan_scale
side_inner_diameter=67.
side_tip_angle_deg=35.
side_sweep_deg=110.62997775278676  # Keep the original 290 mm blade's side arc sweep.
phi1=math.radians(side_tip_angle_deg)
phi0=phi1-math.radians(side_sweep_deg)
theta=phi0+math.pi/2
r=(side_inner_diameter+wall)/2
# Each side arc reaches its widest point at its horizontal radius. Solve the
# middle radius from the required width and the common tangent at the join.
right_cx=target_width/2-(r+wall/2)
R=r+right_cx/math.sin(theta)
cy=17*plan_scale+R
right=(right_cx,cy-(R-r)*math.cos(theta));left=(-right[0],right[1])
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
parts=[curved_wall]
reinforcement_primitives=[]
def add_reinforcement(name,points,vector):
    parts.append(prism(points,vector))
    reinforcement_primitives.append(dict(name=name,points_mm=points,extrude_mm=vector))

# Keep the existing rear mounting tongue and both underside screw pockets.
mount_z=mount_height-mount_thickness
add_reinforcement('mount',[(-16,-8,mount_z),(16,-8,mount_z),(16,18,mount_z),(-16,18,mount_z)],
                  (0,0,mount_thickness))
# Gradual underside transition: 3 mm to 5 mm over 4 mm of forward travel.
add_reinforcement('transition',[(-16,7,mount_z),(-16,7,mount_height),
    (-16,11,mount_height),(-16,11,mount_height-5)],(32,0,0))
# Widen the fan where it meets the blade. Extend it forward far enough to
# maintain a solid overlap with the enlarged central arc.
fan_half=30*plan_scale
fan_end=18*plan_scale
add_reinforcement('fan',[(-16,11,mount_height-5),(16,11,mount_height-5),
    (fan_half,fan_end,mount_height-5),(-fan_half,fan_end,mount_height-5)],(0,0,5))
rib_width=4*plan_scale
for index,cx in enumerate([-12,0,12]):
    x=cx-rib_width/2
    add_reinforcement(f'rib{index}',[(x,7,mount_z),(x,fan_end,low+1),(x,fan_end,mount_z)],
                      (rib_width,0,0))
reinforcement={'mount_sheet_mm':mount_thickness,'hole_region_thickness_mm':hole_region_thickness,'hole_region_radius_mm':hole_region_radius,'connector_max_thickness_mm':5,'connector_width_at_blade_mm':2*fan_half,'transition_y_mm':[7,11],'thickness_transition_mm':[mount_thickness,5],'ribs_count':3,'rib_thickness_mm':rib_width,'added_material_bounds_mm':[[-fan_half,7,low+1],[fan_half,fan_end,mount_height]],'installation_surface_z_mm':mount_height,'nominal_ground_clearance_mm':low,'primitives':reinforcement_primitives}
(p/'reinforcement.json').write_text(json.dumps(reinforcement,ensure_ascii=False,indent=2))
# Store analytic plan geometry to allow exact inspection independent of mesh tessellation.
central_arc_depth=round((R+wall/2)*(1-math.cos(theta)),6)
curve_params={'centre_arc_radius_mm':R,'side_arc_radius_mm':r,'side_inner_diameter_mm':side_inner_diameter,'central_arc_centre':[0,cy],'right_arc_centre':list(right),'left_arc_centre':list(left),'central_join_half_angle_deg':math.degrees(theta),'central_arc_depth_mm':central_arc_depth,'side_tip_angle_deg':side_tip_angle_deg,'side_arc_sweep_deg':side_sweep_deg,'side_arc_reduction_deg':20,'tip_round_radius_mm':wall/2,'right_tip_centre':list(rt),'join_tangent_error_deg':0.0,'continuity':'G1 tangent, curvature changes at joins','width_including_wall_mm':target_width,'reference':'shape.png reflected so opening faces +Y'}
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
step=p/'pusher.step'
step.write_bytes(b'\n'.join(line.rstrip() for line in step.read_bytes().splitlines())+b'\n')
BRepMesh_IncrementalMesh(shape,.08,False,.10,True);sw=StlAPI_Writer();sw.ASCIIMode=False;sw.Write(shape,str(p/'pusher.stl'))
props=GProp_GProps();BRepGProp.VolumeProperties_s(shape,props)
params={'status':'concept preview, not validated for printing','unit':'mm',
        'planform_scale':plan_scale,'hole_pitch':19,'hole_diameter':3,
        'mounting_tongue_thickness':mount_thickness,'hole_region_thickness_mm':hole_region_thickness,
        'hole_region_radius_mm':hole_region_radius,'reinforced_connector_thickness':5,'ribs_count':3,
        'rib_thickness':rib_width,'wall_thickness_mm':wall,'nominal_ground_clearance':low,
        'push_height':high-low,'max_width':target_width,'central_arc_depth_mm':central_arc_depth,
        'side_inner_diameter_mm':side_inner_diameter,'side_arc_sweep_deg':side_sweep_deg,
        'forward_extent_from_hole_line':rt[1]+wall/2,'mounting_surface_height_from_model_ground':mount_height,
        'volume_mm3':props.Mass(),'retained_standoffs':True,'excluded_original_shells':[417,195,438],
        'car_source':'../robot/raspbot.step'}
(p/'parameters.json').write_text(json.dumps(params,ensure_ascii=False,indent=2));print(json.dumps(params,ensure_ascii=False),flush=True)
