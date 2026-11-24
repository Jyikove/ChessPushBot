import cv2, math
import numpy as np
from piece_spec import HALF_HEIGHT_M
from shape_runtime import Detector, prepare, postprocess
class Camera:
 def __init__(self,c,model_folder):
  self.c=c;self.detector=Detector(model_folder,(416,320),threads=1)
 def project(self,uv,pose):
  c=self.c['camera'];q=cv2.undistortPointsIter(np.asarray(uv,np.float64).reshape(-1,1,2),np.array(c['K']),np.array(c['opencv_distortion']),None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,1e-10)).reshape(-1,2)
  rays=np.column_stack([q,np.ones(len(q))])@np.array(c['R_ground_optical']).T;origin=np.array(c['camera_base_position_m']);out=[]
  R=np.array([[math.cos(pose[2]),-math.sin(pose[2])],[math.sin(pose[2]),math.cos(pose[2])]])
  for ray in rays:
   if ray[2]>-.04:out.append(None);continue
   v=origin+ray*(2*HALF_HEIGHT_M-origin[2])/ray[2]
   out.append(pose[:2]+R@v[:2] if .12<v[0]<2.8 else None)
  return out
