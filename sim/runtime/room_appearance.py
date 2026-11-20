"""Appearance presets inspired by the indoor field photos, not a photometric calibration."""
import xml.etree.ElementTree as ET

import numpy as np


PRESETS = {
    'baseline': None,
    'overcast': dict(ambient='.28 .29 .30', overhead='.43 .44 .46', side='.16 .19 .23',
                     window='.42 .56 .65', gain=(.94, .97, 1.0), vignette=.06),
    'window-sun': dict(ambient='.19 .19 .19', overhead='.28 .28 .29', side='.78 .72 .60',
                       window='.71 .82 .94', gain=(1.03, 1.01, .98), vignette=.08),
    'warm-room': dict(ambient='.19 .18 .17', overhead='.38 .33 .26', side='.07 .09 .14',
                      window='.15 .19 .26', gain=(1.02, .98, .92), vignette=.11),
    'photo-room': dict(ambient='.36 .37 .38', overhead='.40 .42 .44', side='.23 .21 .19',
                       window='.55 .62 .67', gain=(.98, 1., 1.01), vignette=.06),
}


def apply_room(model, preset):
    style = PRESETS[preset]
    if style is None:
        return
    world = model.find('worldbody')
    asset = model.find('asset')
    floor = asset.find("material[@name='field_pdf_material']")
    floor.attrib.update(specular='.52', shininess='.12', reflectance='0')
    ET.SubElement(asset, 'material', name='room_board', emission='.58', specular='.06')
    for wall in world.findall("geom"):
        if wall.get('name', '').startswith('wall_'):
            wall.set('rgba', '.97 .97 .95 1')
            wall.set('material', 'room_board')

    def box(name, pos, size, rgb, material=None):
        attributes = dict(name='room_'+name, type='box', pos=' '.join(map(str, pos)),
                          size=' '.join(map(str, size)), rgba=rgb+' 1',
                          contype='0', conaffinity='0', group='3')
        if material:
            attributes['material'] = material
        ET.SubElement(world, 'geom', **attributes)

    ET.SubElement(asset, 'material', name='room_glazing', emission='.65', specular='0')
    ET.SubElement(asset, 'material', name='room_plaster', emission='.36', specular='.04')
    box('west_wall', (-.65, 1, 1.1), (.05, 1.48, 1.2), '.78 .78 .74', 'room_plaster')
    box('south_wall', (1.45, -.49, 1.1), (2.1, .045, 1.2), '.74 .75 .73', 'room_plaster')
    box('south_baseboard', (1.45, -.43, .09), (2.1, .018, .09), '.55 .57 .55')
    box('socket_plate', (1.96, -.433, .36), (.055, .012, .038), '.84 .82 .73')
    for index, center in enumerate((.61, 1.36, 2.11)):
        box(f'window_{index}', (3.59, center, .89), (.025, .345, .89), style['window'], 'room_glazing')
        box(f'window_base_{index}', (3.54, center, .105), (.038, .345, .085), '.26 .31 .33')
        box(f'window_rail_{index}', (3.53, center, .59), (.048, .35, .020), '.15 .18 .18')
    for index, center in enumerate((.25, .99, 1.74, 2.48)):
        box(f'window_mullion_{index}', (3.51, center, .91), (.063, .027, .92), '.13 .15 .15')
    box('window_pillar', (3.50, -.045, 1.1), (.10, .26, 1.15), '.71 .70 .63', 'room_plaster')
    box('north_wall', (1.46, 2.50, 1.11), (2.10, .06, 1.14), '.65 .65 .61', 'room_plaster')
    box('north_cabinet', (1.10, 2.32, .43), (.70, .08, .43), '.13 .14 .14')
    box('north_cabinet_top', (1.10, 2.30, .88), (.72, .10, .022), '.30 .30 .27')
    for index, center in enumerate((.67, 1.08, 1.48)):
        box(f'cabinet_handle_{index}', (center, 2.225, .69), (.065, .012, .01), '.39 .41 .40')

    key = world.find("light[@name='field_overhead_key']")
    direct_sun = preset in ('window-sun', 'photo-room')
    key.attrib.update(ambient=style['ambient'], diffuse='0 0 0', specular='0 0 0',
                      castshadow='false')
    ET.SubElement(asset, 'material', name='room_strip_diffuser', emission='.72', specular='0')
    channels = tuple(map(float, style['overhead'].split()))
    strip_color = ' '.join(f'{channel/max(channels):.3f}' for channel in channels)
    diffuse = ' '.join(f'{channel/4:.3f}' for channel in channels)
    specular = ' '.join(f'{channel/3:.3f}' for channel in channels)
    for row, row_y in enumerate((.36, 1.54)):
        box(f'strip_diffuser_{row}', (1.55, row_y, 1.62), (1.24, .032, .014),
            strip_color, 'room_strip_diffuser')
        for sample, sample_x in enumerate((.52, 1.55, 2.58)):
            ET.SubElement(world, 'light', name=f'room_strip_{row}_{sample}',
                          pos=f'{sample_x} {row_y} 1.59', dir='0 0 -1', directional='false',
                          diffuse=diffuse, specular=specular, ambient='0 0 0',
                          attenuation='1 0 .08', cutoff='80', exponent='0',
                          castshadow='true' if not direct_sun and (row, sample)==(0, 1) else 'false')
    fill = world.find("light[@name='field_overhead_fill']")
    fill.attrib.update(pos='3.2 .44 1.9', dir='-.76 .21 -.62', diffuse=style['side'],
                       specular=style['side'], castshadow='true' if direct_sun else 'false')
    model.find('visual/quality').set('shadowsize', '2048')


def camera_response(preset, shape):
    style = PRESETS[preset]
    if style is None:
        return None
    height, width = shape
    row = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    col = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    vignette = 1-style['vignette']*(row**2+col**2)*.5
    return vignette[..., None]*np.asarray(style['gain'], dtype=np.float32)
