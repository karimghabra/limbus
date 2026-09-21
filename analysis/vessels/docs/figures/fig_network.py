"""The network, in the format the earlier figures used: the averaged frame
unannotated, then each variant drawn over it at full width.

usage: python fig_network.py <crop key>
"""
import sys, numpy as np, cv2
sys.path[:0]=['.', r'E:\Conjunctiva Code\limbus\analysis',
              r'C:\Users\PETERC~1\AppData\Local\Temp\claude\E--\e4e3ef18-f4c7-497a-8ee6-40091257ed39\scratchpad\velocity\week']
import bench
from vessels import image as vimg, network as net, graph as gr

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

key = sys.argv[1] if len(sys.argv)>1 else '1522L'
A, valid = vimg.prepare(bench.load_crop(key))
H, W = A.shape
T=np.exp(-A); lo,hi = np.percentile(T[valid],[0.5,99.8])
g=np.clip((T-lo)/(hi-lo)*255,0,255).astype(np.uint8)
plain = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def panel(img, title):
    bar=np.full((30,W,3),25,np.uint8)
    cv2.putText(bar,title,(10,21),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
    return np.vstack([bar,img])

def draw(ves, number=True):
    img=(plain*0.6+50).astype(np.uint8)
    for v in ves:
        hue=int((v['id']*137.508)%180)
        col=tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue,255,255]]]),cv2.COLOR_HSV2BGR)[0,0])
        C=np.rint(np.asarray(v['centreline'],float)).astype(np.int32)
        cv2.polylines(img,[C],False,col,2,cv2.LINE_AA)
        if number and v['length_px']>=120:
            m=C[len(C)//2]
            for th,cc in ((3,(0,0,0)),(1,col)):
                cv2.putText(img,str(v['id']),(int(m[0])+4,int(m[1])-4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,cc,th,cv2.LINE_AA)
    return img

RUNS = [("before this week: end tangent read over the whole piece", legacy, dict()),
        ("shipped now: end tangent read over its last 10 px", GOOD, dict()),
        ("not shipped - complete-linkage junction clustering", GOOD, dict(cluster='complete'))]
panels=[panel(plain, f"{key}: averaged stabilized frame, no annotation")]
for title, fn, kw in RUNS:
    gr._tangent = fn
    ves,_,_ = net.detect(A, valid, net.NetConfig(**kw))
    gr._tangent = GOOD
    L=np.array([v['length_px'] for v in ves])
    panels.append(panel(draw(ves),
        f"{title}  -  {len(ves)} vessels, median {np.median(L):.0f} px, longest {L.max():.0f} px, "
        f"{L.sum():.0f} px total"))
    print(title,'->',len(ves),'vessels, longest',int(L.max()), flush=True)
sep=np.full((6,W,3),255,np.uint8)
out=[panels[0]]
for p in panels[1:]: out += [sep, p]
cv2.imwrite(f'fig_network_{key}.png', np.vstack(out))
print('wrote', f'fig_network_{key}.png')
