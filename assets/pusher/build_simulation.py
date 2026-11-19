"""Regenerate the pusher visual and convex collision primitives in robot coordinates."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import trimesh

def build():
    source=Path(__file__).resolve().parent
    out=source.parent/'robot'
    params=json.loads((source/'arcs.json').read_text())
    design=json.loads((source/'parameters.json').read_text())
    data=json.loads((out/'simulation.json').read_text())
    def transform(points):
        p=np.asarray(points,dtype=float)*.001
        return np.column_stack([p[:,1]+.075,-p[:,0],p[:,2]])
    push=trimesh.load_mesh(source/'pusher.stl',process=True)
    assert push.is_watertight
    push.vertices=transform(push.vertices)
    push.export(source/'pusher.obj')
    for item in data['visuals']:
        if item['name']=='push':item['triangles']=len(push.faces)
    # Annular quadrilateral prisms are convex individually. Adjacent endpoints coincide.
    wedges=[];error=0
    phi0=math.radians(-90+params['central_join_half_angle_deg'])
    arcs=[('central',params['central_arc_centre'],params['centre_arc_radius_mm'],math.radians(-90-params['central_join_half_angle_deg']),phi0),('right',params['right_arc_centre'],params['side_arc_radius_mm'],phi0,math.radians(35)),('left',params['left_arc_centre'],params['side_arc_radius_mm'],math.pi-phi0,math.pi-math.radians(35))]
    for name,c,r,a,b in arcs:
        max_angle=2*math.acos(1-.025/(r+1.5));n=math.ceil(abs(b-a)/max_angle)
        angles=np.linspace(a,b,n+1)
        error=max(error,(r+1.5)*(1-math.cos(abs(b-a)/n/2)))
        for i,(aa,bb) in enumerate(zip(angles[:-1],angles[1:])):
            xy=[[c[0]+rr*math.cos(t),c[1]+rr*math.sin(t)] for rr,t in [(r-1.5,aa),(r+1.5,aa),(r+1.5,bb),(r-1.5,bb)]]
            points=[[x,y,z] for z in [3,27] for x,y in xy]
            wedges.append({'name':f'v5_{name}_{i:03}','vertices':transform(points).tolist(),'part':'blade'})
    # Reinforcement is decomposed from documented design primitives; screw bores omitted only in collision.
    def prism(name,poly,vec):
        points=np.asarray(poly);points=np.concatenate([points,points+vec]);wedges.append({'name':name,'vertices':transform(points).tolist(),'part':'reinforcement'})
    def box(name,x,y,z,dx,dy,dz):prism(name,[[x,y,z],[x+dx,y,z],[x+dx,y+dy,z],[x,y+dy,z]],[0,0,dz])
    h=design['mounting_surface_height_from_model_ground'];thickness=design['mounting_tongue_thickness']
    box('v5_mount1',-16,-8,h-thickness,32,18,thickness);box('v5_mount2',-16,8,h-thickness,32,10,thickness)
    prism('v5_transition',[[-16,7,h-thickness],[-16,7,h],[-16,11,h],[-16,11,h-5]],[32,0,0])
    prism('v5_fan',[[-16,11,h-5],[16,11,h-5],[30,18,h-5],[-30,18,h-5]],[0,0,5])
    for i,c in enumerate([-12,0,12]):prism(f'v5_rib{i}',[[c-2,7,h-thickness],[c-2,18,4],[c-2,18,h-thickness]],[4,0,0])
    caps=[{'name':'v5_tip_'+name,'position':transform([[sign*params['right_tip_centre'][0],params['right_tip_centre'][1],15]])[0].tolist(),'radius':.0015,'half_height':.012} for name,sign in [('right',1),('left',-1)]]
    data.update(convex_parts=wedges,tip_caps=caps,max_radial_chord_error_m=error*.001,blade_segment_count=sum(x['part']=='blade' for x in wedges))
    data['omissions'][0]='M3 screw bores and diameter-7mm underside pockets filled in collision only; mounting sheet collision is 3mm thick. Tilted-piece interaction there not validated.'
    data['sources']=[{'file':'../pusher/'+name,'sha256':hashlib.sha256((source/name).read_bytes()).hexdigest()} for name in ['arcs.json','parameters.json','pusher.stl']]
    (out/'simulation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'blade_segments':data['blade_segment_count'],'visual_triangles':len(push.faces),'chord_error_m':data['max_radial_chord_error_m']}))

if __name__=='__main__':build()
