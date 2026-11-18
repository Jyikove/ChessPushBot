"""Per-piece private meshes, one collision cylinder/body and three massless visuals."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as E
from piece_spec import RADIUS_M,HALF_HEIGHT_M,MASS_KG,inertia,identity

def add_chess_visuals(model,asset_dir):
    folder=Path(__file__).resolve().parents[1]/'resources/chess';metadata=json.loads((folder/'chess_manifest.json').read_text());asset=model.find('asset')
    bodies=[b for b in model.findall('.//body') if b.get('name','').startswith('piece_')]
    if not 1<=len(bodies)<=32:raise ValueError('one to32 independent chess bodies required')
    used=sorted({mesh for i in range(len(bodies)) for mesh in metadata['pieces'][i]['mesh_parts'].values()})
    for index in used:
        info=metadata['meshes'][str(index)];file=folder/info['file']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('chess visual asset hash mismatch')
        E.SubElement(asset,'mesh',name=f'chess_visual_{index:02}',file=str(file))
    for i,body in enumerate(bodies):
        info=metadata['pieces'][i]
        if any(info[k]!=v for k,v in identity(i).items()):raise ValueError('piece identity/order mismatch')
        collision=body.find('geom');collision.set('size',f'{RADIUS_M} {HALF_HEIGHT_M}');collision.set('rgba','0 0 0 0');collision.set('group','2')
        inertial=body.find('inertial');inertial.set('mass',str(MASS_KG));inertial.set('diaginertia',' '.join(map(str,inertia())))
        for part,index in info['mesh_parts'].items():
            mesh=metadata['meshes'][str(index)]
            E.SubElement(body,'geom',name=f'piece_{i:02}_visual_{part}',type='mesh',mesh=f'chess_visual_{index:02}',rgba=' '.join(map(str,mesh['rgba'])),contype='0',conaffinity='0',mass='0',group='0')
