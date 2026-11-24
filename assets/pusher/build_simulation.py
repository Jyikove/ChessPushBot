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
    reinforcement=json.loads((source/'reinforcement.json').read_text())
    data=json.loads((out/'simulation.json').read_text())
    def transform(points):
        p=np.asarray(points,dtype=float)*.001
        return np.column_stack([p[:,1]+.075,-p[:,0],p[:,2]])
    push=trimesh.load_mesh(source/'pusher.stl',process=True)
    assert push.is_watertight
    push.vertices=transform(push.vertices)
    (source/'pusher.obj').write_text(trimesh.exchange.obj.export_obj(push).rstrip()+'\n')
    for item in data['visuals']:
        if item['name']=='push':item['triangles']=len(push.faces)
    # Annular quadrilateral prisms are convex individually. Adjacent endpoints coincide.
    wedges=[];error=0
    phi0=math.radians(-90+params['central_join_half_angle_deg'])
    phi1=math.radians(params['side_tip_angle_deg'])
    half_wall=design['wall_thickness_mm']/2
    low=design['nominal_ground_clearance']
    high=low+design['push_height']
    arcs=[('central',params['central_arc_centre'],params['centre_arc_radius_mm'],math.radians(-90-params['central_join_half_angle_deg']),phi0),('right',params['right_arc_centre'],params['side_arc_radius_mm'],phi0,phi1),('left',params['left_arc_centre'],params['side_arc_radius_mm'],math.pi-phi0,math.pi-phi1)]
    for name,c,r,a,b in arcs:
        max_angle=2*math.acos(1-.025/(r+half_wall));n=math.ceil(abs(b-a)/max_angle)
        angles=np.linspace(a,b,n+1)
        error=max(error,(r+half_wall)*(1-math.cos(abs(b-a)/n/2)))
        for i,(aa,bb) in enumerate(zip(angles[:-1],angles[1:])):
            xy=[[c[0]+rr*math.cos(t),c[1]+rr*math.sin(t)] for rr,t in [(r-half_wall,aa),(r+half_wall,aa),(r+half_wall,bb),(r-half_wall,bb)]]
            points=[[x,y,z] for z in [low,high] for x,y in xy]
            wedges.append({'name':f'v5_{name}_{i:03}','vertices':transform(points).tolist(),'part':'blade'})
    # Reinforcement is decomposed from documented design primitives; screw bores omitted only in collision.
    def prism(name,poly,vec):
        points=np.asarray(poly);points=np.concatenate([points,points+vec]);wedges.append({'name':name,'vertices':transform(points).tolist(),'part':'reinforcement'})
    for item in reinforcement['primitives']:
        prism('v5_'+item['name'],item['points_mm'],item['extrude_mm'])
    caps=[{'name':'v5_tip_'+name,'position':transform([[sign*params['right_tip_centre'][0],
        params['right_tip_centre'][1],(low+high)/2]])[0].tolist(),
        'radius':half_wall*.001,'half_height':(high-low)*.0005}
        for name,sign in [('right',1),('left',-1)]]
    data.update(convex_parts=wedges,tip_caps=caps,max_radial_chord_error_m=error*.001,blade_segment_count=sum(x['part']=='blade' for x in wedges))
    data['omissions'][0]='M3 screw bores and diameter-7mm underside pockets filled in collision only; mounting sheet collision is 3mm thick. Tilted-piece interaction there not validated.'
    data['sources']=[{'file':'../pusher/'+name,'sha256':hashlib.sha256((source/name).read_bytes()).hexdigest()} for name in ['arcs.json','parameters.json','reinforcement.json','pusher.stl']]
    (out/'simulation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'blade_segments':data['blade_segment_count'],'visual_triangles':len(push.faces),'chord_error_m':data['max_radial_chord_error_m']}))

if __name__=='__main__':build()
