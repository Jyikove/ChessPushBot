"""Vendor 7e23 checksum protocol. No device writes in the capture path."""
import struct
class Parser:
 def __init__(self):self.buf=bytearray();self.bad=0;self.noise=0
 def feed(self,data):
  self.buf.extend(data);out=[]
  while len(self.buf)>=5:
   i=self.buf.find(b'\x7e\x23')
   if i<0:self.noise+=len(self.buf)-1;self.buf[:]=self.buf[-1:];break
   if i:self.noise+=i;del self.buf[:i]
   n=self.buf[2]
   if not 5<=n<=40:del self.buf[0];self.noise+=1;continue
   if len(self.buf)<n:break
   f=bytes(self.buf[:n]);del self.buf[:n]
   if sum(f[:-1])&255!=f[-1]:self.bad+=1;continue
   k,p=f[3],f[4:-1];r={'function':k}
   if k==4 and len(p)==18:
    a=struct.unpack('<9h',p);r.update(accel_g=[v*16/32767 for v in a[:3]],gyro_rad_s=[v*2000/32767*.017453292519943295 for v in a[3:6]],mag_uT=[v*800/32767 for v in a[6:]],raw_int16=list(a))
   elif k==0x16 and len(p)==16:r['q_wxyz']=list(struct.unpack('<4f',p))
   elif k==0x26 and len(p)==12:r['rpy_rad']=list(struct.unpack('<3f',p))
   out.append(r)
  return out
