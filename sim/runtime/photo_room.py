"""Local photo-backed room surfaces; all geometry here is render-only."""
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


PANELS = (
    ('east', ((3.52, 2.52), (3.52, -.52))),
    ('north', ((-.52, 2.52), (3.52, 2.52))),
    ('west', ((-.52, -.52), (-.52, 2.52))),
    ('south', ((3.52, -.52), (-.52, -.52))),
)


def source_image(folder, stamp):
    path = folder / f'image_20261119_{stamp}.jpg'
    if not path.is_file():
        raise FileNotFoundError(f'photo-room needs {path}; set SIM_PHOTO_DIR to the original capture folder')
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (320, 416):
        raise ValueError(f'Expected uncropped 416x320 capture: {path}')
    return image


def encode(image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('Could not encode scene texture')
    return data.tobytes()


def apply_photos(model, folder, atlas_path):
    folder = Path(folder)
    world = model.find('worldbody')
    asset = model.find('asset')
    assets = {}
    for geom in list(world.findall('geom')):
        name = geom.get('name', '')
        if name.startswith('room_') and not name.startswith('room_strip_diffuser'):
            world.remove(geom)
    atlas = cv2.imdecode(np.frombuffer(atlas_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if atlas is None or min(atlas.shape[:2]) < 100:
        raise ValueError(f'Invalid 2x2 room atlas: {atlas_path}')
    height, width = atlas.shape[:2]
    for index, (name, endpoints) in enumerate(PANELS):
        row, col = divmod(index, 2)
        texture = atlas[row*height//2+2:(row+1)*height//2-2,
                        col*width//2+2:(col+1)*width//2-2]
        assets[f'photo_{name}.png'] = encode(texture)
        ET.SubElement(asset, 'texture', name=f'photo_{name}', type='2d', file=f'photo_{name}.png')
        ET.SubElement(asset, 'material', name=f'photo_{name}', texture=f'photo_{name}',
                      emission='.88', specular='0')
        (left_x, left_y), (right_x, right_y) = endpoints
        inset = np.array([right_y-left_y, left_x-right_x])
        inset *= .004 / np.linalg.norm(inset)
        corners = ((left_x, left_y, -.06), (right_x, right_y, -.06),
                   (right_x, right_y, 1.74), (left_x, left_y, 1.74))
        mesh = ''.join(f'v {x+depth*inset[0]} {y+depth*inset[1]} {z}\n'
                       for depth in (0, 1) for x, y, z in corners)
        mesh += ('vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n'
                 'f 1/1 2/2 3/3\nf 1/1 3/3 4/4\nf 5/1 7/3 6/2\nf 5/1 8/4 7/3\n'
                 'f 1/1 5/1 6/2\nf 1/1 6/2 2/2\nf 2/2 6/2 7/3\nf 2/2 7/3 3/3\n'
                 'f 3/3 7/3 8/4\nf 3/3 8/4 4/4\nf 4/4 8/4 5/1\nf 4/4 5/1 1/1\n')
        assets[f'photo_{name}.obj'] = mesh.encode('ascii')
        ET.SubElement(asset, 'mesh', name=f'photo_{name}', file=f'photo_{name}.obj')
        ET.SubElement(world, 'geom', name=f'photo_{name}', type='mesh', mesh=f'photo_{name}',
                      material=f'photo_{name}', group='3', contype='0', conaffinity='0')
    artwork = asset.find("texture[@name='field_pdf_texture']")
    asset.find("material[@name='room_board']").set('emission', '.43')
    printed = cv2.imdecode(np.frombuffer(Path(artwork.get('file')).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    multiplier = .87
    if (folder / 'image_20261119_154929_622.jpg').is_file():
        sample = cv2.cvtColor(source_image(folder, '154929_622')[218:314, 48:368],
                              cv2.COLOR_BGR2GRAY).astype(np.float32)
        grain = sample - cv2.GaussianBlur(sample, (0, 0), 2.4)
        grain = np.block([[grain, grain[:, ::-1]], [grain[::-1], grain[::-1, ::-1]]])
        rows, cols = printed.shape[:2]
        grain = np.tile(grain, ((rows+191)//192, (cols+639)//640))[:rows, :cols]
        multiplier += np.clip(grain, -12, 12)[..., None]*.004
    printed = np.clip(printed.astype(np.float32) * multiplier, 0, 255).astype(np.uint8)
    artwork.set('file', 'photo_field.png')
    assets['photo_field.png'] = encode(printed)
    return assets
