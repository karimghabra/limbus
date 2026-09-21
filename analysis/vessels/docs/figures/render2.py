import sys, numpy as np, cv2
sys.path[:0]=['.', r'E:\Conjunctiva Code\limbus\analysis',
              r'C:\Users\PETERC~1\AppData\Local\Temp\claude\E--\e4e3ef18-f4c7-497a-8ee6-40091257ed39\scratchpad\velocity\week']
import bench
from vessels import image as vimg, network as net, graph as gr
from render import legacy, GOOD, grey, label

A,v = vimg.prepare(bench.load_crop('1522R'))
img = grey(A,v)
panes=[]
for name, fn in (('before: end tangent read over the whole piece', legacy),
                 ('after: end tangent read over its last 10 px', GOOD)):
    gr._tangent = fn
    ves,_,_ = net.detect(A,v,net.NetConfig())
    Ls=[float(np.hypot(*np.diff(np.asarray(x['centreline'],float),axis=0).T).sum()) for x in ves]
    k=int(np.argmax(Ls))
    out=img.copy()
    for i,x in enumerate(ves):                      # everything else, pale
        if i==k: continue
        cv2.polylines(out,[np.rint(np.asarray(x['centreline'],float)).astype(np.int32)],
                      False,(185,185,185),2,cv2.LINE_AA)
    C=np.rint(np.asarray(ves[k]['centreline'],float)).astype(np.int32)
    cv2.polylines(out,[C],False,(0,0,0),6,cv2.LINE_AA)
    cv2.polylines(out,[C],False,(40,60,230),3,cv2.LINE_AA)   # the longest vessel
    for p in (C[0], C[-1]):
        cv2.circle(out,tuple(int(q) for q in p),7,(0,0,0),-1)
        cv2.circle(out,tuple(int(q) for q in p),5,(40,60,230),-1)
    panes.append(label(out, name,
        f"longest vessel {max(Ls):.0f} px (red, ends marked) - {len(ves)} vessels, "
        f"{sum(Ls):.0f} px of centreline in total"))
gr._tangent = GOOD
cv2.imwrite('fig_tangent2.png', np.vstack(panes))
print('wrote fig_tangent2.png')
