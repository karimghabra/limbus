"""Ridge evidence that keeps the orientations apart instead of collapsing them.

The Hessian gives one answer per pixel: one strength, one direction. Where two
vessels cross there are two directions, and a single-orientation filter has to
describe both with one number. At a right-angle crossing the two curvatures
cancel and the ridge strength falls into a hole - measured on planted
crossings, the evidence along one vessel drops to **0.12** of its value away
from the junction (0.07 for r = 4 px), with a single-vessel null reading
0.99-1.00, so the estimator is not at fault. The pipeline currently survives
that hole because hysteresis bridges it, which works and is luck: it depends
on a faint-ridge threshold being crossed either side of a gap 88 % deep.

At shallow angles the opposite happens. The two vessels superpose and the
evidence nearly doubles (1.89 at 15 degrees), so a shallow crossing looks like
one strong wide vessel - which is the fusion problem, not a hole.

So: do not collapse. For each orientation theta, filter with a kernel LONG
along theta and NARROW across it - the second derivative across the vessel,
smoothed along it - and keep the stack S[theta, y, x]. A vessel running at
theta lives in the plane at theta; two vessels crossing live in two planes and
never compete. Measured on the same crossings, the plane at one vessel's own
angle reads **1.00** at 90 degrees and 1.43 at 15, against the Hessian's 0.12
and 1.89.

In the Fourier domain, for the unit vectors t (along) and n (across):

    H(k) = sigma_t^(2 gamma) (k.n)^2 exp(-[sigma_l^2 (k.t)^2 + sigma_t^2 (k.n)^2] / 2)

the transfer function of -d^2/dn^2 of an anisotropic Gaussian, with
Lindeberg's gamma-normalisation so scales can be compared. One forward FFT of
the image, then one multiply and one inverse FFT per (theta, scale).

WHAT THIS DOES NOT FIX. Two vessels running close together IN PARALLEL are in
the same plane, and the lift does nothing for them: measured against the
Hessian on planted pairs it gives the same answer at every separation. That
limit is the transverse scale, not the filter's shape, and it is dealt with
elsewhere.
"""
import cv2
import numpy as np

GAMMA = 0.75          # ridge normalisation, as in the Hessian stage
ASPECT = 4.0          # how much longer the kernel is along the vessel than across
N_THETA = 16          # orientations over [0, pi)


def stack(A, valid=None, sigmas=(1.0, 1.5, 2.0, 3.0), n_theta=N_THETA,
          aspect=ASPECT, gamma=GAMMA, per_scale_z=True):
    """S[theta, y, x], the winning scale, and the orientations.

    Each scale is put in units of its own robust spread before the scales are
    compared, for the reason the Hessian stage does it: a coarse scale
    responds strongly to the sclera's blotches and would otherwise drown the
    fine scales everywhere.
    """
    A = np.asarray(A, np.float32)
    if valid is not None:
        A = np.where(valid, A - np.median(A[valid]), 0).astype(np.float32)
    H, W = A.shape
    F = np.fft.rfft2(A.astype(np.float64))
    ky = np.fft.fftfreq(H)[:, None] * 2 * np.pi
    kx = np.fft.rfftfreq(W)[None, :] * 2 * np.pi
    thetas = np.arange(n_theta) * np.pi / n_theta
    out = np.zeros((n_theta, H, W), np.float32)
    best_s = np.zeros((n_theta, H, W), np.float32)
    for st in sigmas:
        sl = aspect * st
        resp = np.empty((n_theta, H, W), np.float32)
        for i, th in enumerate(thetas):
            tx, ty = np.cos(th), np.sin(th)
            nx, ny = -ty, tx
            kt = kx * tx + ky * ty
            kn = kx * nx + ky * ny
            Hk = (st ** (2 * gamma)) * (kn ** 2) * np.exp(
                -0.5 * (sl ** 2 * kt ** 2 + st ** 2 * kn ** 2))
            resp[i] = np.fft.irfft2(F * Hk, s=A.shape).astype(np.float32)
        if per_scale_z:
            v = resp if valid is None else resp[:, valid]
            med = float(np.median(v))
            mad = 1.4826 * float(np.median(np.abs(v - med))) + 1e-9
            resp = (resp - med) / mad
        better = resp > out
        best_s = np.where(better, st, best_s)
        out = np.where(better, resp, out)
    return out, best_s, thetas


def collapse(S, thetas):
    """The stack read as one strength and one direction per pixel.

    The maximum over orientations, and which orientation attained it. At a
    crossing this is the stronger of the two vessels rather than a cancelled
    average, which is the whole of the difference from the Hessian: the hole
    is filled because the two vessels were never asked to share a number.
    """
    i = np.argmax(S, 0)
    z = np.take_along_axis(S, i[None], 0)[0]
    ang = thetas[i] + np.pi / 2          # the direction ACROSS the ridge, for suppression
    return z, ang.astype(np.float32), i


def ridge_z(A, valid, sigmas, with_angle=False, with_scale=False, **kw):
    """Drop-in replacement for evidence.ridge_z, reading the stack.

    The angle returned is the one across the winning orientation's ridge, so
    the non-maximum suppression downstream suppresses in the right direction -
    which at a crossing is the direction of the vessel that actually won,
    rather than of the blur between two of them.
    """
    S, sc, thetas = stack(A, valid, sigmas, **kw)
    z, ang, i = collapse(S, thetas)
    z = z.copy()
    z[~valid] = 0
    out = (z,)
    if with_angle:
        out += (ang,)
    if with_scale:
        out += (np.take_along_axis(sc, i[None], 0)[0],)
    return out if len(out) > 1 else z
