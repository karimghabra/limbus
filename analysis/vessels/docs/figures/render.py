"""Pictures of the two things worth looking at."""
import sys, numpy as np, cv2
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
sys.path[:0]=['.',]
from vessels import benchmark as bench
from vessels import image as vimg, network as net, graph as gr, scenarios as scen

GOOD = gr._tangent
def legacy(C, at_start, arc=10.0, standoff=0.0):
    C=np.asarray(C,float)
    if len(C)<2: return np.array([1.0,0.0])
    walk=np.concatenate([[0.0],np.cumsum(np.hypot(*np.diff(C,axis=0).T))])
    if at_start:
        n=int(np.searchsorted(walk,arc)); n=min(max(n,1),len(C)-1); d=C[0]-C[n]
    else:
        back=walk[-1]-walk
        n=int(np.searchsorted(-back,-arc)); n=min(max(n,1),len(C)-1); d=C[-1]-C[len(C)-1-n]
    return d/max(np.hypot(*d),1e-9)

def grey(A, valid):
    lo,hi = np.percentile(A[valid],[1,99.6])
    g=np.clip((A-lo)/(hi-lo)*255,0,255).astype(np.uint8)
    return cv2.cvtColor(255-g, cv2.COLOR_GRAY2BGR)

def draw(img, ves, thick=2):
    out=img.copy()
    Ls=sorted((float(np.hypot(*np.diff(np.asarray(x['centreline'],float),axis=0).T).sum()), i)
              for i,x in enumerate(ves))
    for rank,(L,i) in enumerate(Ls):
        hue=int((i*137.508)%180)
        col=tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue,220,255]]]),cv2.COLOR_HSV2BGR)[0,0])
        C=np.rint(np.asarray(ves[i]['centreline'],float)).astype(np.int32)
        cv2.polylines(out,[C],False,col,thick,cv2.LINE_AA)
    return out

def label(img, text, sub=""):
    out=cv2.copyMakeBorder(img,44,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    cv2.putText(out,text,(10,26),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
    if sub: cv2.putText(out,sub,(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.45,(90,90,90),1,cv2.LINE_AA)
    return out

# ---- 1. the end-tangent fix on a real network -----------------------------
A,v = vimg.prepare(bench.load_crop('1522L'))
img = grey(A,v)
panes=[]
for name, fn in (('before: end tangent over the whole piece', legacy),
                 ('after: end tangent over its last 10 px', GOOD)):
    gr._tangent = fn
    ves,_,_ = net.detect(A,v,net.NetConfig())
    Ls=[float(np.hypot(*np.diff(np.asarray(x['centreline'],float),axis=0).T).sum()) for x in ves]
    panes.append(label(draw(img,ves),
                       name,
                       f"{len(ves)} vessels, longest {max(Ls):.0f} px, median {np.median(Ls):.0f} px"))
gr._tangent = GOOD
cv2.imwrite('fig_tangent.png', np.vstack(panes))
print('wrote fig_tangent.png')

# ---- 2. a crowded junction --------------------------------------------------
A2,v2 = vimg.prepare(bench.load_crop('1522R'))
ref = lambda AA,vv: [np.asarray(x['centreline'],float) for x in net.detect(AA,vv,net.NetConfig())[0]]
free = scen.free_ground(A2,v2,ref)
scenes = scen.build_scenes(A2, free, 'ladder', dict(n_j=3,window=30), seeds=[4], per=2)
Ai, copies = scenes[0]
sub = None
for lines in copies:
    P=np.vstack([l['C'] for l in lines]); x0,y0=P.min(0)-60; x1,y1=P.max(0)+60
    if x0>0 and y0>0 and x1<A2.shape[1] and y1<A2.shape[0]:
        sub=(int(x0),int(y0),int(x1),int(y1), lines); break
x0,y0,x1,y1,lines = sub
panes=[]
for name, cfg in (('shipped', net.NetConfig()), ('complete linkage', net.NetConfig(cluster='complete'))):
    ves,_,_ = net.detect(Ai,v2,cfg)
    base = grey(Ai,v2)
    for l in lines:
        cv2.polylines(base,[np.rint(l['C']).astype(np.int32)],False,(210,210,210),5,cv2.LINE_AA)
    out = draw(base, ves, thick=2)[y0:y1, x0:x1]
    out = cv2.resize(out, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    near=[x for x in ves if np.hypot(*(np.asarray(x['centreline'],float)-[(x0+x1)/2,(y0+y1)/2]).T).min()<90]
    panes.append(label(out, name, f"{len(near)} detected vessels over this trunk and its 3 branches"
                                  "   (truth in grey, detections in colour)"))
h=max(p.shape[0] for p in panes)
panes=[cv2.copyMakeBorder(p,0,h-p.shape[0],0,8,cv2.BORDER_CONSTANT,value=(255,255,255)) for p in panes]
cv2.imwrite('fig_crowded.png', np.hstack(panes))
print('wrote fig_crowded.png')
