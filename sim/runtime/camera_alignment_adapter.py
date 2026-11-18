"""Opt-in provisional calibrated camera, SI/OpenCV conventions.

No real hardware access. apply_scene() replaces only the onboard camera.
CameraRig integration requires raw RGB/depth/instance to use the SAME warp.
The supplied profile is NOT a multi-view calibration or a training-ready claim.
"""
from pathlib import Path
import json, math
import numpy as np

HERE=Path(__file__).resolve().parent

def load_profile(folder=HERE):
    folder=Path(folder)
    p=json.loads((folder/'camera_profile.json').read_text())
    w=np.load(folder/'warp_map.npz')
    return p,w['map_xy'],float(w['render_f_px'])

def apply_scene(root,profile,render_f):
    """Input camera profile at simulated pan=tilt=0 == real commands77/0.

    At other cloud gimbal positions, kinematics remain uncalibrated.
    Existing parent origins are read, not guessed from an unrelated URDF.
    """
    import xml.etree.ElementTree as E
    pan=root.find('.//body[@name="pan_body"]')
    tilt=root.find('.//body[@name="tilt_body"]')
    camera=tilt.find('camera[@name="onboard"]')
    if camera is None:raise ValueError('onboard camera missing')
    parent=np.fromstring(pan.get('pos'),sep=' ')+np.fromstring(tilt.get('pos'),sep=' ')
    pos=np.array(profile['camera_base_position_m'])-parent
    R=np.array(profile['R_ground_optical'])@np.diag([1,-1,-1])
    camera.attrib.clear()
    camera.attrib.update(name='onboard',pos=' '.join(map(str,pos)),
        xyaxes=' '.join(map(str,np.r_[R[:,0],R[:,1]])),
        fovy=str(math.degrees(2*math.atan(960/(2*render_f)))))
    root.find('visual/global').attrib.update(offwidth='1280',offheight='960')
    quality=root.find('visual/quality')
    if quality is None:quality=E.SubElement(root.find('visual'),'quality')
    quality.set('offsamples','0')  # no blended segmentation IDs at object boundaries
    return root

def warp(array,map_xy,nearest=False):
    """Inverse raw-pixel→ideal-ray map, not an undistortion applied backwards.

    RGB uses bilinear interpolation. Depth and categorical masks use nearest
    sampling to avoid blending foreground/background, IDs or invalid depths.
    Geometry labels/bboxes MUST be recomputed after this operation.
    """
    x,y=map_xy[...,0],map_xy[...,1]
    if x.min()<0 or y.min()<0 or x.max()>array.shape[1]-1 or y.max()>array.shape[0]-1:
        raise ValueError('insufficient overscan: border rays not rendered')
    if nearest:return array[np.rint(y).astype(int),np.rint(x).astype(int)].copy()
    x0=x.astype(int);y0=y.astype(int);x1=np.minimum(x0+1,array.shape[1]-1);y1=np.minimum(y0+1,array.shape[0]-1)
    dx=x-x0;dy=y-y0
    if array.ndim==3:dx=dx[...,None];dy=dy[...,None]
    out=array[y0,x0]*(1-dx)*(1-dy)+array[y0,x1]*dx*(1-dy)+array[y1,x0]*(1-dx)*dy+array[y1,x1]*dx*dy
    return np.clip(np.rint(out),0,255).astype(np.uint8) if array.dtype==np.uint8 else out.astype(array.dtype)

def supervision(rgb,depth,geometry_ids,map_xy,geom_to_instance):
    # Call before any physical step: all inputs must be at exactly the same state.
    rgb=warp(rgb,map_xy)
    depth=warp(depth,map_xy,nearest=True)
    geom=warp(geometry_ids,map_xy,nearest=True)
    instance=np.zeros(geom.shape,np.uint16)
    for gid,iid in geom_to_instance.items():instance[geom==gid]=iid
    labels=[]
    for iid in sorted(set(geom_to_instance.values())):
        y,x=np.where(instance==iid)
        labels.append({'instance_id':iid,'visible_pixels':len(x),
            'bbox_xyxy_exclusive':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None})
    return rgb,depth,instance,labels

def make_rig(model,data,folder,base_rig_class):
    """Use the SAME profiled image path in the normal algorithm environment."""
    import mujoco
    profile,map_xy,rf=load_profile(folder)
    yy,xx=np.indices((480,640))
    diagnostic_map=np.stack([2*xx+.5,2*yy+.5],-1).astype('float32')
    class ProfiledRig(base_rig_class):
        def __init__(self,m,d):
            super().__init__(m,d)
            self.renderer.close()
            self.renderer=mujoco.Renderer(m,height=960,width=1280)
            self._set_intrinsics()
        def _set_intrinsics(self):
            if self.camera_name=='onboard':
                self.K=np.array(profile['K']);self.f=float(self.K[0,0])
        def select(self,name):
            super().select(name);self._set_intrinsics()
        def _raw(self,channel):
            a=super()._raw(channel)
            return warp(a,map_xy if self.camera_name=='onboard' else diagnostic_map,
                        nearest=channel!='rgb')
        def calibration(self):
            c=super().calibration()
            if self.camera_name=='onboard':
                c.update(K=profile['K'],hfov_deg=profile['hfov_pinhole_deg'],vfov_deg=profile['vfov_pinhole_deg'],
                    distortion_model='opencv_radial_k1_k2',distortion_coefficients=profile['opencv_distortion'],
                    effective_raw_hfov_centerline_deg=profile['effective_raw_hfov_centerline_deg'],
                    effective_raw_vfov_centerline_deg=profile['effective_raw_vfov_centerline_deg'],
                    calibration_status=profile['status'],training_ready=False,
                    mount_assumption='height85mm/setback40mm at real servo77/0; provisional tile fit',
                    gimbal_reference=profile['gimbal_reference'],
                    sensor_warp='shared inverse radial map; RGB bilinear, depth/IDs nearest; bbox after warp',
                    output_frame='distorted raw640x480; overscan1280x960 internal')
            return c
    return ProfiledRig(model,data)
