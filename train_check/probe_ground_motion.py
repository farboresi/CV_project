"""Probe: is the tracked box moving relative to the ground?

Usage: python probe_ground_motion.py <video.mp4> <clip_pose.npy>

Camera motion per frame = similarity transform (shift+zoom+rotation) fitted with
LK optical flow + RANSAC on background corners outside the tracked box.
Residual = where the box centre actually went minus where a ground-fixed point
would have gone.  Residuals are SUMMED over a 1 s window (60 frames) and
divided by body height: box jitter cancels out, real travel accumulates.

Observed on b002 / women_029 (train_check):
  skier in-run / air / landing : 0.4 - 22 body-heights per second
  Pacifico banner figure       : 0.10 - 0.28
  foreground cameraman         : 0.03 - 0.17
Per-frame values (no accumulation) do NOT separate them: ~0.03 for everyone.
Masks assume the *_pipeline_check.mp4 overlays (top band, top-left inset).
"""
import cv2, numpy as np, sys
vid, posef = sys.argv[1], sys.argv[2]
P = np.load(posef); T=len(P)
def box(k):
    m=k[:,2]>0.2
    if m.sum()<4: return None
    return np.array([k[m,0].min(),k[m,1].min(),k[m,0].max(),k[m,1].max()])
boxes=[box(P[t]) for t in range(T)]
cap=cv2.VideoCapture(vid); fr=[]
while True:
    ok,f=cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY))
H,W=fr[0].shape
base=np.ones((H,W),np.uint8)*255; base[:110]=0; base[:300,:200]=0; base[H-20:]=0
res=np.full((T,3),np.nan)  # dx,dy residual (px), dlogscale residual
bhs=np.full(T,np.nan)
for t in range(T-1):
    b0,b1=boxes[t],boxes[t+1]
    if b0 is None or b1 is None: continue
    bh=max(b0[3]-b0[1],10.); bhs[t]=bh
    m=base.copy(); p=0.5*bh
    x0,y0,x1,y1=[int(v) for v in (b0[0]-p,b0[1]-p,b0[2]+p,b0[3]+p)]
    m[max(y0,0):max(y1,0),max(x0,0):max(x1,0)]=0
    pts=cv2.goodFeaturesToTrack(fr[t],400,0.005,12,mask=m)
    if pts is None or len(pts)<12: continue
    p1,st,_=cv2.calcOpticalFlowPyrLK(fr[t],fr[t+1],pts,None,winSize=(21,21),maxLevel=3)
    ok=st.ravel()==1
    A,_=cv2.estimateAffinePartial2D(pts[ok].reshape(-1,2),p1[ok].reshape(-1,2),method=cv2.RANSAC,ransacReprojThreshold=2.0)
    if A is None: continue
    s=np.sqrt(abs(np.linalg.det(A[:,:2])))
    c0=(b0[:2]+b0[2:])/2; c1=(b1[:2]+b1[2:])/2
    pred=A[:,:2]@c0+A[:,2]
    h1=b1[3]-b1[1]
    res[t]=[*(c1-pred), np.log(max(h1,1)/bh)-np.log(s)]
WIN=60
print(" frm   bh  disp1s/bh  scale1s   verdict")
for w in range(0,T-WIN,30):
    r=res[w:w+WIN]; ok=~np.isnan(r[:,0])
    if ok.sum()<WIN*0.6: print(f"{w:4d}  (insufficient)"); continue
    bh=np.nanmedian(bhs[w:w+WIN])
    disp=np.linalg.norm(np.nansum(r[:,:2],axis=0))/bh
    sc=abs(np.nansum(r[:,2]))
    print(f"{w:4d} {bh:5.0f}   {disp:6.2f}    {sc:5.2f}")
