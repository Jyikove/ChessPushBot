"""Appearance-only local floor artwork; standard library, no engine initialization."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as E

def apply_field_artwork(model, asset_dir):
    folder=Path(asset_dir).parent/'field'
    asset=model.find('asset');world=model.find('worldbody')
    E.SubElement(asset,'texture',name='field_pdf_texture',type='2d',file=str(folder/'field.png'),hflip='false',vflip='false')
    E.SubElement(asset,'material',name='field_pdf_material',texture='field_pdf_texture',texrepeat='1 1',texuniform='false',rgba='1 1 1 1',specular='0',shininess='0',reflectance='0')
    E.SubElement(asset,'mesh',name='field_pdf_surface_mesh',file=str(folder/'field.obj'))
    # Keep the collision floor and original diagnostic target/ruler definitions.
    # Hide their old paint: only one visible surface at z=0, so no layered decals.
    for geom in world.findall('geom'):
        name=geom.get('name','')
        if name in ['floor','goal_mark'] or name.startswith('ruler_'):
            old=geom.get('rgba','1 1 1 1').split();geom.set('rgba',' '.join(old[:3]+['0']));geom.set('group','5')
    E.SubElement(world,'geom',name='field_pdf_visual',type='mesh',mesh='field_pdf_surface_mesh',material='field_pdf_material',rgba='1 1 1 1',pos='0 0 0',contype='0',conaffinity='0',group='0')

    # Explicit overhead key; no camera-facing primary lamp.
    visual=model.find('visual');head=visual.find('headlight')
    if head is None:head=E.SubElement(visual,'headlight')
    head.attrib.update(active='0',ambient='0 0 0',diffuse='0 0 0',specular='0 0 0')
    lights=world.findall('light')
    if len(lights)!=1:raise ValueError('expected one original world key light')
    lights[0].attrib.update(name='field_overhead_key',pos='1.5 1 3',dir='0 0 -1',directional='true',castshadow='false',ambient='.28 .28 .28',diffuse='.62 .62 .62',specular='.02 .02 .02')
    E.SubElement(world,'light',name='field_overhead_fill',pos='1.5 1 3.5',dir='0 0 -1',directional='true',castshadow='true',ambient='0 0 0',diffuse='.06 .06 .06',specular='0 0 0')
