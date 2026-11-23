"""The eye in the head, and the slit lamp that records it: a patient from a seed, where the eye points, how it moves
while it fixates and between fixations, the lids that cover part of it, and bursts recorded at any gaze.

    from synth.gaze import patient, SlitLamp
    eye = patient(7)                                    # the same seed is always the same patient
    lamp = SlitLamp(eye, aim=(40.0, 0.0))               # the camera aimed 40° temporal of the primary gaze
    burst = lamp.record(gaze=(-10.0, 5.0), n_frames=16) # the patient looks 10° nasal and 5° up: frames + truth

Head coordinates: µm, the eye's centre at the origin, +z towards the camera in primary gaze, +y up, +x towards the
temporal side (a right eye seen from in front, as synth.eye). Gaze (h, v) in degrees: h positive temporal, v
positive up. The eye turns about its centre (the globe's centre stands for the centre of rotation, about 13 mm
behind the cornea) by Listing's law: from the primary position about the axis perpendicular to both the primary and
the new direction, so a gaze has no torsion of its own. The camera is fixed in the head (the patient rests on a chin
rest); its axis meets the globe at the direction aim = (h, v) in head coordinates. Sources: docs §12.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .camera import Motion
from .eye import Eye, EyeAnatomy, Pose, assemble_view, view_scene
from .simulate import render_network

ARCMIN = math.pi / (180 * 60)


@dataclass
class Oculomotor:
    """Fixational and voluntary eye movements, the lids and blinks (docs §12)."""
    drift_arcmin_per_s: float = 52.0               # ocular drift with the head fixed [Cherici2012]
    microsaccade_rate_hz: float = 1.3              # [Cherici2012]; 1-2 per second [MartinezConde2004]
    microsaccade_arcmin: tuple = (20.0, 7.0)       # amplitude mean, SD [Cherici2012]
    fixation_error_deg: float = 0.25               # SD of where a fixation lands about its target (assumed, docs §12)
    head_motion_um: float = 176.0                  # SD of the image's motion over 5 s on a chin rest [Felder2016]
    head_motion_s: float = 2.0                     # its correlation time (assumed)
    saccade_ms_per_deg: float = 2.7                # duration grows 2.7 ms per degree [Baloh1975]
    saccade_ms_base: float = 22.0                  # duration of the smallest saccades (assumed)
    lid_gain_down: float = 1.0                     # the upper lid follows the eye down one to one [BeckerFuchs1988]
    lid_gain_up: float = 0.8                       # and upward somewhat less [BeckerFuchs1988] (ratio assumed)
    lower_lid_gain: float = 0.3                    # the lower lid moves less (assumed)
    blink_rate_per_min: float = 14.0               # primary gaze 8.0-21.0 per minute [Doughty2001]
    blink_duration_s: float = 0.3                  # closing, closed and reopening (assumed)
    lid_margin_um: float = 600.0                   # the dark lid margin and lashes seen at its edge (assumed)


def listing(h_deg, v_deg):
    """The rotation (3 x 3, eye to head coordinates) that turns the eye from the primary position to gaze (h, v) by
    Listing's law."""
    h, v = math.radians(h_deg), math.radians(v_deg)
    g = np.array([math.sin(h) * math.cos(v), math.sin(v), math.cos(h) * math.cos(v)])
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, g)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3)
    axis /= s
    angle = math.atan2(s, float(z @ g))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def direction_of(h_deg, v_deg):
    h, v = math.radians(h_deg), math.radians(v_deg)
    return np.array([math.sin(h) * math.cos(v), math.sin(v), math.cos(h) * math.cos(v)])


def patient(seed: int, fine_region=None, scene=None) -> Eye:
    """A patient's eye from a seed: the globe, limbus, fissure and fornices drawn about their cited means within the
    cited spread, the heart rate and haematocrit likewise, then the whole vascular network grown from the same seed
    (docs §12). The same seed always gives the same eye."""
    from .anatomy import Scene
    rng = np.random.default_rng([seed, 2026])
    a = EyeAnatomy()
    k = lambda mean, sd, lo, hi: float(np.clip(rng.normal(mean, sd), lo, hi))
    a = replace(a,
                radius_um=k(12200.0, 600.0, 11000.0, 13500.0),          # 11.2-13.1 mm [Jesus2017, Choi2014]
                limbus_radius_um=k(5900.0, 210.0, 5400.0, 6400.0),      # 11.80 ± 0.42 mm white to white [GaneshLin2023]
                fissure_half_width_mm=k(12.8, 0.6, 11.5, 14.0),         # 25.6 mm wide [Rana2023] (spread assumed)
                upper_lid_mm=k(4.1, 0.5, 3.0, 5.2),                      # [Rana2023] (spread assumed)
                lower_lid_mm=k(-5.9, 0.5, -7.0, -4.8))
    scene = scene or Scene()
    ph = scene.physiology
    ph = replace(ph, heart_rate_hz=1000.0 / k(909.0, 86.0, 700.0, 1150.0),       # 909 ± 86 ms [Neriyanuri2023]
                 inflow_discharge_haematocrit=k(ph.inflow_discharge_haematocrit, 0.025, 0.34, 0.46))
    return Eye(seed=seed, anatomy=a, scene=replace(scene, physiology=ph), fine_region=fine_region)


@dataclass
class Burst:
    """One recording at a fixation, with its exact truth."""
    result: object                  # synth.simulate.Result (frames, network in the camera's coordinates, flow, ...)
    pose: Pose                      # where the camera's field lay on the globe in frame 0 (eye coordinates)
    gaze: np.ndarray                # (T, 2) degrees: the eye's actual gaze in every frame
    target: tuple                   # the commanded gaze
    occlusion: np.ndarray           # (T,) share of the field under a lid in every frame
    blink: np.ndarray               # (T,) how far a blink has closed the lids, 0..1
    poses: list                     # every frame's Pose (the field's centre and roll on the globe)


@dataclass
class SlitLamp:
    """The camera, fixed in the head, recording one eye. aim: (h, v) degrees, the head direction its axis meets the
    globe at; scene: the camera settings (magnification, field size, optics) - the eye's own by default."""
    eye: Eye
    aim: tuple = (40.0, 0.0)
    scene: object = None
    motor: Oculomotor = field(default_factory=Oculomotor)
    seed: int = 0

    def __post_init__(self):
        self.scene = view_scene(self.eye, Pose(), self.scene)
        self.rng = np.random.default_rng([self.eye.seed, self.seed, 41])
        self.t = 0.0                                   # session clock, s
        self.head = np.zeros(2)                        # head offset on the chin rest, µm (x, y in head coordinates)
        self.next_blink = self.rng.exponential(60.0 / self.motor.blink_rate_per_min)

    # ---------------------------------------------------------------- geometry
    def field_pose(self, gaze, head=(0.0, 0.0)):
        """The Pose (eye coordinates) of the field's centre for a gaze and head offset, and the roll of the camera's
        image about it."""
        g = self.eye.globe
        R = listing(*gaze)
        a = direction_of(*self.aim)
        # the head offset moves the field over the eye: to first order a shift across the camera axis
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, a)
        right /= np.linalg.norm(right)
        up_c = np.cross(a, right)
        p_head = a * g.R + right * head[0] + up_c * head[1]
        p_head *= g.R / np.linalg.norm(p_head)
        p_eye = R.T @ p_head
        s, clock, _ = g.surface_coordinates(p_eye[None])
        pose = Pose(s_mm=float(s[0]) / 1000.0, clock_deg=float(clock[0]))
        # roll: the camera's image right (head) in the field's local frame
        c, e_circ, e_mer = g.frame(pose.s_mm * 1000.0, pose.clock_deg)
        r_eye = R.T @ right
        roll = math.degrees(math.atan2(float(r_eye @ e_mer), float(r_eye @ e_circ)))
        return pose, roll

    def lid_margins(self, v_deg, closing=0.0):
        """Heights (mm, head coordinates, at the fissure's middle) of the upper and lower lid margins for a vertical
        gaze, the upper lid following the eye [BeckerFuchs1988], a blink closing them (0 open, 1 shut)."""
        a, m = self.eye.anatomy, self.motor
        R = self.eye.globe.R * 1e-3
        gain = m.lid_gain_down if v_deg < 0 else m.lid_gain_up
        upper = R * math.sin(math.asin(np.clip(a.upper_lid_mm / R, -1, 1)) + math.radians(gain * v_deg))
        lower = R * math.sin(math.asin(np.clip(a.lower_lid_mm / R, -1, 1)) + math.radians(m.lower_lid_gain * v_deg))
        meet = lower + 0.15 * (upper - lower)
        return upper + closing * (meet - upper), lower + closing * (meet - lower)

    # ---------------------------------------------------------------- movement
    def _fixational(self, n, dt, target):
        """Gaze (n, 2) degrees and head offsets (n, 2) µm through a fixation."""
        m, rng = self.motor, self.rng
        gaze = np.zeros((n, 2))
        land = np.asarray(target, float) + rng.normal(0, m.fixation_error_deg, 2)
        g = land.copy()
        # drift: a random walk whose mean speed is the cited one (a 2D Gaussian step of SD s has mean length s sqrt(pi/2))
        step_sd = m.drift_arcmin_per_s / 60.0 * dt / math.sqrt(math.pi / 2)
        head = np.zeros((n, 2))
        h = self.head.copy()
        rho = math.exp(-dt / m.head_motion_s)
        head_sd = m.head_motion_um / math.sqrt(2)
        for k in range(n):
            g = g + rng.normal(0, step_sd, 2)
            if rng.random() < m.microsaccade_rate_hz * dt:
                amp = max(rng.normal(*m.microsaccade_arcmin), 3.0) / 60.0
                back = land - g                                # microsaccades tend to correct the drift
                ang = math.atan2(back[1], back[0]) + rng.normal(0, 0.8)
                g = g + amp * np.array([math.cos(ang), math.sin(ang)])
            h = rho * h + math.sqrt(1 - rho ** 2) * rng.normal(0, head_sd, 2)
            gaze[k] = g
            head[k] = h
        self.head = h
        return gaze, head

    def saccade_duration(self, amplitude_deg):
        return (self.motor.saccade_ms_base + self.motor.saccade_ms_per_deg * amplitude_deg) * 1e-3

    def _blinks(self, n, dt):
        """How far the lids are closed in each frame (0..1)."""
        out = np.zeros(n)
        m = self.motor
        for k in range(n):
            tk = self.t + k * dt
            if tk >= self.next_blink + m.blink_duration_s:
                self.next_blink = tk + self.rng.exponential(60.0 / m.blink_rate_per_min)
            phase = (tk - self.next_blink) / m.blink_duration_s
            if 0 <= phase <= 1:
                # fast closing, a short closed spell, slower reopening
                out[k] = min(phase / 0.25, 1.0) if phase < 0.4 else max(0.0, 1 - (phase - 0.4) / 0.6)
        return out

    # ---------------------------------------------------------------- recording
    def record(self, gaze, n_frames=16, previous=None, record_cells=False, record_masks=False, scene=None):
        """Ask the patient to look at gaze (h, v) degrees and record a burst once the eye has landed (the saccade
        from the previous gaze takes its main-sequence duration). Returns a Burst with the frames and the truth."""
        scene = scene or self.scene
        fps = scene.optics.fps
        dt = 1.0 / fps
        if previous is not None:
            self.t += self.saccade_duration(float(np.hypot(*(np.asarray(gaze) - np.asarray(previous))))) + 0.2
        gazes, heads = self._fixational(n_frames, dt, gaze)
        blink = self._blinks(n_frames, dt)
        poses, rolls = zip(*[self.field_pose(gazes[k], heads[k]) for k in range(n_frames)])
        ref = poses[0]
        g = self.eye.globe
        frame = g.frame(ref.s_mm * 1000.0, ref.clock_deg)
        um = scene.anatomy.um_per_px
        dx, dy, rot = np.zeros(n_frames), np.zeros(n_frames), np.zeros(n_frames)
        roll0 = math.radians(rolls[0])
        for k, p in enumerate(poses):
            centre = g.point(p.s_mm * 1000.0, p.clock_deg, 0.0)
            local = g.to_local(centre[None], frame)[0]
            # the tissue at the new centre appears at the image centre: the image moves by minus that offset, in the
            # camera's own (rolled) axes
            x, y = local[0], local[1]
            cr, sr = math.cos(roll0), math.sin(roll0)
            xc, yc = cr * x + sr * y, -sr * x + cr * y
            dx[k], dy[k] = -xc / um, -yc / um
            rot[k] = -(math.radians(rolls[k]) - roll0)
        rng = np.random.default_rng([self.eye.seed, self.seed, int(self.t * 1000) % (2 ** 31)])
        o = scene.optics
        from .camera import _smooth
        focus = o.focus_depth_um + o.focus_wander_um * _smooth(rng, n_frames, 12.0)
        motion = Motion(dx, dy, rot, focus, np.zeros(0, int))
        view = replace(scene, seed=int(rng.integers(2 ** 31)))
        net, flow = assemble_view(self.eye, ref, view, fine=True, roll_deg=rolls[0])
        lids = _LidOccluder(self, net, view, frame, roll0, gazes, blink)
        result = render_network(view, net, flow, n_frames, record_cells=record_cells, record_masks=record_masks,
                                motion=motion, occluder=lids, time_offset=self.t)
        self.t += n_frames * dt
        return Burst(result, ref, gazes, tuple(gaze), lids.share, blink, list(poses))


class _LidOccluder:
    """Which part of each frame the lids cover, and what they look like there (skin, and the dark lid margin with
    its lashes), from the eye's actual gaze in that frame."""

    def __init__(self, lamp, net, scene, frame, roll0, gazes, blink):
        self.lamp, self.scene, self.frame, self.roll0 = lamp, scene, frame, roll0
        self.gazes, self.blink = gazes, blink
        self.share = np.zeros(len(gazes))
        h, w = scene.height_px, scene.width_px
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        self.xx, self.yy = xx, yy
        rng = np.random.default_rng(scene.seed + 5)
        from .camera import texture_field
        self.skin = 1.12 * np.exp(0.6 * texture_field((h, w), scene, rng))

    def __call__(self, k, renderer):
        lamp, g = self.lamp, self.lamp.eye.globe
        scene = self.scene
        h, w = self.xx.shape
        um = scene.anatomy.um_per_px
        dx, dy, rot, _ = renderer.motion.at(k)
        # camera pixel -> frame-0 pixel -> local µm (field frame, rolled) -> eye -> head
        c = np.array([w / 2, h / 2])
        cos, sin = math.cos(-rot), math.sin(-rot)
        qx, qy = self.xx - c[0] - dx, self.yy - c[1] - dy
        px, py = cos * qx - sin * qy, sin * qx + cos * qy
        cr, sr = math.cos(self.roll0), math.sin(self.roll0)
        X, Y = px * um, py * um
        lx, ly = cr * X - sr * Y, sr * X + cr * Y
        local = np.stack([lx.ravel(), ly.ravel(), np.zeros(lx.size)], 1)
        p_eye = g.from_local(local, self.frame)
        R = listing(*self.gazes[k])
        p_head = p_eye @ R.T
        x_mm, y_mm = p_head[:, 0] * 1e-3, p_head[:, 1] * 1e-3
        upper, lower = lamp.lid_margins(float(self.gazes[k][1]), float(self.blink[k]))
        half = lamp.eye.anatomy.fissure_half_width_mm
        shape = np.sqrt(np.clip(1 - (x_mm / half) ** 2, 0, None))
        up_y, lo_y = upper * shape, lower * shape
        # distance into the lid (µm, positive under it), the larger of the two lids
        depth = np.maximum(y_mm - up_y, lo_y - y_mm) * 1000.0
        depth = np.where(np.abs(x_mm) > half, 1e4, depth).reshape(h, w)
        edge = 25.0                                                # soft margin of the occlusion, µm
        mask = np.clip(0.5 + depth / (2 * edge), 0, 1).astype(np.float32)
        margin = lamp.motor.lid_margin_um
        dark = 1.0 - 0.75 * np.exp(-np.clip(depth, 0, None) / (0.35 * margin))
        lid = (self.skin * dark).astype(np.float32)
        self.share[k] = float(mask.mean())
        return mask, lid
