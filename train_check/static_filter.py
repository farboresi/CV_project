import cv2, numpy as np, sys
vid, posef, outvid = sys.argv[1], sys.argv[2], sys.argv[3]
WIN, THR, MIN_RUN = 60, 0.4, 60
P=np.load(posef); T=len(P)
def box(k):
    m=k[:,2]>0.2
    if m.sum()<4: return None
    return np.array([k[m,0].min(),k[m,1].min(),k[m,0].max(),k[m,1].max()])
boxes=[box(P[t]) for t in range(T)]
cap=cv2.VideoCapture(vid); fps=cap.get(cv2.CAP_PROP_FPS); col=[]
while True:
    ok,f=cap.read()
    if not ok: break
    col.append(f)
gr=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in col]
H,W=gr[0].shape
base=np.ones((H,W),np.uint8)*255; base[:110]=0; base[:300,:200]=0; base[H-20:]=0
res=np.full((T,2),np.nan); bhs=np.full(T,np.nan)
for t in range(T-1):
    b0,b1=boxes[t],boxes[t+1]
    if b0 is None or b1 is None: continue
    bh=max(b0[3]-b0[1],10.); bhs[t]=bh
    m=base.copy(); p=0.5*bh
    x0,y0,x1,y1=[int(v) for v in (b0[0]-p,b0[1]-p,b0[2]+p,b0[3]+p)]
    m[max(y0,0):max(y1,0),max(x0,0):max(x1,0)]=0
    pts=cv2.goodFeaturesToTrack(gr[t],400,0.005,12,mask=m)
    if pts is None or len(pts)<12: continue
    p1,st,_=cv2.calcOpticalFlowPyrLK(gr[t],gr[t+1],pts,None,winSize=(21,21),maxLevel=3)
    ok=st.ravel()==1
    if ok.sum()<6: continue
    A,_=cv2.estimateAffinePartial2D(pts[ok].reshape(-1,2),p1[ok].reshape(-1,2),method=cv2.RANSAC,ransacReprojThreshold=2.0)
    if A is None: continue
    c0=(b0[:2]+b0[2:])/2; c1=(b1[:2]+b1[2:])/2
    res[t]=c1-(A[:,:2]@c0+A[:,2])
# Dwell test: integrate camera-compensated motion into a ground-relative
# trajectory; a stretch >= MIN_RUN frames whose trajectory never leaves a
# radius of R body heights is an object fixed to the ground -> drop it.
R=0.3
g=np.zeros((T,2)); valid=np.array([b is not None for b in boxes])
for t in range(1,T):
    step=res[t-1] if not np.isnan(res[t-1,0]) else 0.0
    g[t]=g[t-1]+step
drop=np.zeros(T,bool); score=np.full(T,np.nan)
t=0
while t<T:
    if not valid[t]: t+=1; continue
    u=t; bh=bhs[t] if not np.isnan(bhs[t]) else 50
    while u+1<T and valid[u+1] and np.linalg.norm(g[u+1]-g[t])<R*bh: u+=1
    if u-t+1>=MIN_RUN:
        drop[t:u+1]=True; score[t:u+1]=np.linalg.norm(g[u]-g[t])/bh/((u-t+1)/fps)
        t=u+1
    else: t+=1
runs=[]; t=0
while t<T:
    if drop[t]:
        u=t
        while u<T and drop[u]: u+=1
        runs.append((t,u-1)); t=u
    else: t+=1
print("dropped runs:",runs, " total dropped",drop.sum(),"/",sum(b is not None for b in boxes))
vw=cv2.VideoWriter(outvid,cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
for t,f in enumerate(col):
    f=f.copy(); b=boxes[t] if t<T else None
    if b is not None:
        c=(0,0,255) if drop[t] else (0,220,0)
        cv2.rectangle(f,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),c,4)
        lab=("DROPPED static" if drop[t] else "KEPT")+(f"  {score[t]:.2f} bh/s" if not np.isnan(score[t]) else "")
        cv2.putText(f,lab,(int(b[0]),max(int(b[1])-10,130)),cv2.FONT_HERSHEY_SIMPLEX,0.9,c,2)
    cv2.putText(f,f"frame {t}",(W-220,H-40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    vw.write(f)
vw.release()
np.save(outvid.replace('.mp4','_drop.npy'),drop)
