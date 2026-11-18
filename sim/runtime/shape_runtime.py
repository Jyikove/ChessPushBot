"""Fixed-shape FP32 NCNN, one class piece. No hardware/control access."""
from pathlib import Path
import time
import numpy as np,cv2,ncnn
cv2.setNumThreads(1)

def prepare(image,wh):
 w,h=wh;ih,iw=image.shape[:2];r=min(w/iw,h/ih);nw,nh=round(iw*r),round(ih*r);left,top=round((w-nw)/2-.1),round((h-nh)/2-.1)
 resize=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR) if (nw,nh)!=(iw,ih) else image
 padded=cv2.copyMakeBorder(resize,top,h-nh-top,left,w-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114));assert padded.shape==(h,w,3)
 tensor=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1),dtype=np.float32)/255.
 return ncnn.Mat(tensor).clone(),(r,left,top,iw,ih)

class Detector:
 def __init__(self,folder,wh,threads=2):
  self.wh=tuple(wh);self.output_shape=(5,sum((wh[0]//s)*(wh[1]//s) for s in [8,16,32]));self.net=ncnn.Net();opt=self.net.opt;opt.num_threads=threads;opt.use_vulkan_compute=False;opt.use_fp16_storage=False;opt.use_fp16_arithmetic=False;opt.use_fp16_packed=False
  assert self.net.load_param(str(Path(folder)/'model.ncnn.param'))==0 and self.net.load_model(str(Path(folder)/'model.ncnn.bin'))==0
 def forward(self,mat):
  start=time.perf_counter_ns()
  with self.net.create_extractor() as ex:
   assert ex.input('in0',mat)==0;code,blob=ex.extract('out0');assert code==0;end=time.perf_counter_ns();raw=np.array(blob,dtype=np.float32,copy=True)
  assert raw.shape==self.output_shape and np.isfinite(raw).all()
  return raw,(end-start)/1e6

def postprocess(raw,tf,conf=.25,iou=.7):
 scale,left,top,iw,ih=tf;idx=np.flatnonzero(raw[4]>=conf)
 if not len(idx):return []
 p=raw[:4,idx].T;valid=(p[:,2]>0)&(p[:,3]>0);p=p[valid];scores=raw[4,idx][valid];boxes=p.copy();boxes[:,:2]-=p[:,2:]/2;keep=cv2.dnn.NMSBoxes(boxes.tolist(),scores.tolist(),conf,iou);out=[]
 for i in np.asarray(keep).reshape(-1)[:300]:
  x,y,w,h=boxes[i];b=[float(np.clip((x-left)/scale,0,iw)),float(np.clip((y-top)/scale,0,ih)),float(np.clip((x+w-left)/scale,0,iw)),float(np.clip((y+h-top)/scale,0,ih))];out.append({'class':0,'confidence':float(scores[i]),'box':b})
 return out

def iou(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return float(inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter))

def summarize(values):
 a=np.asarray(values,float);return {'samples':len(a),'mean_ms':float(a.mean()),'p50_ms':float(np.percentile(a,50)),'p95_ms':float(np.percentile(a,95)),'FPS':float(1000/a.mean())}
