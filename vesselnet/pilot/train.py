"""Train the pilot junction network on scenes from gen.py (seed % 10 == 0 held
out for validation).

    python vesselnet/pilot/train.py --data data/pilot --iters 3000 --run runs/pilot1 --gain 0.6,2.6

Every --val-every iterations it scores the held-out scenes at several heat
thresholds (vesselscene's matching rule, net.score_scene) and saves
model.pt and val_<it>.json in --run.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import net  # noqa: E402

THRESHOLDS = (0.1, 0.15, 0.2, 0.3, 0.4)


def seed_of(path):
    return int(os.path.basename(path)[1:6])


def validate(model, scenes, dev):
    res = {t: dict(tp=0, fp=0, n=0, type_ok=0) for t in THRESHOLDS}
    for S in scenes:
        P = net.predict(model, S["X"])
        dets = net.peaks(P, thr=min(THRESHOLDS), valid=S["valid"])
        for t in THRESHOLDS:
            r = net.score_scene(dets, S, t)
            for k in r:
                res[t][k] += r[k]
    model.train()
    out = {}
    for t, r in res.items():
        rec, prec = r["tp"] / max(r["n"], 1), r["tp"] / max(r["tp"] + r["fp"], 1)
        out[t] = dict(recall=round(rec, 3), precision=round(prec, 3),
                      f1=round(2 * rec * prec / max(rec + prec, 1e-9), 3),
                      type_acc=round(r["type_ok"] / max(r["tp"], 1), 3), tp=r["tp"], fp=r["fp"], n=r["n"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data/pilot")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--run", default="runs/pilot")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--gain", default="0.6,2.6", help="contrast augmentation range (log-uniform)")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    net.GAIN = tuple(float(g) for g in a.gain.split(","))
    torch.manual_seed(0)
    os.makedirs(a.run, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.data, "s*.npz")))
    tr = [net.load_scene(p) for p in paths if seed_of(p) % 10 != 0]
    va = [net.load_scene(p) for p in paths if seed_of(p) % 10 == 0]
    print("train %d scenes, validation %d, device %s" % (len(tr), len(va), a.device), flush=True)
    dev = torch.device(a.device)
    model = net.UNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters,
                                                pct_start=min(0.3, max(0.05, 3.0 / a.iters)))
    dl = torch.utils.data.DataLoader(net.Crops(tr, size=a.crop, n=a.iters * a.batch), batch_size=a.batch,
                                     num_workers=0)
    model.train()
    t0, hist = time.time(), []
    for it, batch in enumerate(dl, 1):
        X, heat, typ, jcare, lumen, centre, vcare = [b.to(dev) for b in batch]
        out = model(X)
        lj, lt, ll, lc = net.losses(out, heat, typ, jcare, lumen, centre, vcare)
        loss = lj + 0.5 * lt + ll + lc
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append([lj.item(), lt.item(), ll.item(), lc.item()])
        if it % 50 == 0:
            m = np.mean(hist[-50:], 0)
            print("it %d  heat %.3f type %.3f lumen %.3f centre %.3f  %.2f s/it" % (it, *m, (time.time() - t0) / it),
                  flush=True)
        if it % a.val_every == 0 or it == a.iters:
            v = validate(model, va, dev)
            print("it %d validation %s" % (it, json.dumps(v)), flush=True)
            torch.save(model.state_dict(), os.path.join(a.run, "model.pt"))
            json.dump(dict(it=it, val=v, args=vars(a)), open(os.path.join(a.run, "val_%05d.json" % it), "w"))


if __name__ == "__main__":
    main()
