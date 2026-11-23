"""A small 3D view of the matched scan itself: an orthographic volume render (CT:
translucent soft tissue and bone; MRI: the masked brain) with the current slice as a
translucent plane and the crop as a brighter patch on it.

The volume part depends only on the scan, so it is built once per scan as
front-to-back partial sums of colour and transmittance. Each frame then inserts its
plane analytically at the right depth along every ray, which avoids moire from
voxelizing a thin oblique sheet and costs one gather per pixel.

The view assumes the volume axes are RAS, which the loaders guarantee (closest
canonical orientation); the small residual scanner tilt is not drawn."""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .geometry import footprint

YAW, PITCH = -35., -12.          # viewer at the patient's left-anterior, slightly above
STEP_MM = {True: 3., False: 1.5}  # ray step and sampling pitch for CT and MRI
MARGIN_MM = 25.


def rotation():
    """World (RAS millimetres) to view; view axis 1 points at the viewer."""
    a, e = np.deg2rad(YAW), np.deg2rad(PITCH)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def screen_axes():
    """On-screen direction (right, down) of the patient's R, A and S axes."""
    R = rotation()
    return {name: (-R[0, k], -R[2, k]) for k, name in enumerate('RAS')}


def _classify(ct, v):
    """Opacity per step and grey brightness for each voxel."""
    v = v.astype(np.float32)
    if ct:
        hu = v * (2000/255) - 1000            # inverse of the CT preparation window
        tissue = (v > 0) & (hu > -400)
        bone = np.clip((hu - 180) / 250, 0, 1)
        alpha = np.where(tissue, .03, 0) + .6*bone
        bright = np.where(tissue, .55, 0) + .45*bone
    else:
        # Constant opacity: sequences differ in overall brightness (PD is bright
        # everywhere), so signal only sets the grey, not how solid the brain looks.
        tissue = v > 12
        alpha = np.where(tissue, .022, 0)
        bright = np.where(tissue, .45 + .55*np.clip(v/255, 0, 1), 0)
    return alpha.astype(np.float32), np.clip(bright, 0, 1).astype(np.float32)


class View:
    """Rotation and sampling grid around the scan's tissue bounding box. World
    coordinates are millimetres with voxel i of axis k at i * spacing[k]."""
    def __init__(self, b):
        self.step = STEP_MM[b.ct]
        self.f = max(1, round(self.step / float(min(b.spacing))))
        self.data = b.data[::self.f, ::self.f, ::self.f]
        self.sp = b.spacing * self.f
        lo, hi = [], []
        for k in range(3):
            occupied = np.flatnonzero(self.data.any(axis=tuple(i for i in range(3) if i != k)))
            lo.append(occupied[0] * self.sp[k] - MARGIN_MM)
            hi.append(occupied[-1] * self.sp[k] + MARGIN_MM)
        lo, hi = np.array(lo), np.array(hi)
        self.tissue = (lo + MARGIN_MM, hi - MARGIN_MM)
        self.n = int(np.ceil(np.linalg.norm(hi - lo) / self.step))
        self.R = rotation()
        self.M = self.R.T
        self.cw = (lo + hi) / 2
        self.co = np.full(3, self.n * self.step / 2)

    def world(self, o):
        """Output grid coordinates (..., 3) -> world millimetres."""
        return (o * self.step - self.co) @ self.M.T + self.cw

    def screen(self, w):
        """World millimetres (..., 3) -> (column, row) in the uncropped render."""
        o = ((np.asarray(w, float) - self.cw) @ self.R.T + self.co) / self.step
        return np.stack([self.n - 1 - o[..., 0], self.n - 1 - o[..., 2]], axis=-1)


class Volume:
    def __init__(self, b):
        v = self.view = View(b)
        n = v.n
        mat = (v.M * v.step) / v.sp[:, None]
        off = (v.cw - v.M @ v.co) / v.sp
        vals = ndimage.affine_transform(v.data, mat, off, (n, n, n), order=1, cval=0, prefilter=False)
        alpha, bright = _classify(b.ct, vals)
        inside = vals > (0 if b.ct else 12)
        shell = inside & ~ndimage.binary_erosion(inside)     # glassy outline of the surface
        alpha += shell * (.10 if b.ct else .12)
        bright = np.where(shell, np.maximum(bright, .85), bright)
        # Partial sums in compositing order (s = 0 is in front, at the largest depth index).
        self.front_c = np.zeros((n+1, n, n), np.float16)
        self.front_t = np.ones((n+1, n, n), np.float16)
        c = np.zeros((n, n), np.float32)
        T = np.ones((n, n), np.float32)
        for s, k in enumerate(range(n-1, -1, -1)):
            al = alpha[:, k, :]
            c += T * al * bright[:, k, :]
            T *= 1 - al
            self.front_c[s+1], self.front_t[s+1] = c, T
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        self.base = v.world(np.stack([ii, np.zeros_like(ii), jj], -1).astype(np.float32))
        # Tight crop of everything the render can show, fixed per scan so the body does
        # not jump between frames.
        img = np.transpose(self.front_c[-1].astype(np.float32), (1, 0))[::-1, ::-1]
        rows, cols = np.nonzero(img > .025)
        # Include the tissue box corners, which bound every plane's outline.
        lo, hi = v.tissue
        corners = v.screen(np.array([[(lo, hi)[i >> k & 1][k] for k in range(3)] for i in range(8)]))
        q = 8
        r0, r1 = min(rows.min(), corners[:, 1].min()) - q, max(rows.max(), corners[:, 1].max()) + q
        c0, c1 = min(cols.min(), corners[:, 0].min()) - q, max(cols.max(), corners[:, 0].max()) + q
        self.box = (max(0, int(r0)), min(n, int(np.ceil(r1))), max(0, int(c0)), min(n, int(np.ceil(c1))))


@lru_cache(maxsize=3)
def volume(b):
    return Volume(b)


def render(b, p, size, accent=(226, 178, 76), plane_alpha=.22, crop_alpha=.7):
    """RGB uint8 image no larger than size = (width, height)."""
    vol = volume(b)
    v, n, a = vol.view, vol.view.n, p.axis
    rest = [i for i in range(3) if i != a]
    target = p.index * b.spacing[a]
    dk = v.M[a, 1] * v.step                       # world change along the ray per step
    if abs(dk) > 1e-6:
        k = (target - vol.base[..., a]) / dk
    else:
        k = np.full(vol.base.shape[:2], -1.)
    hit = vol.base + k[..., None] * (v.M[:, 1] * v.step)
    s = np.clip(np.round(n - 1 - k), 0, n).astype(np.intp)
    # The plane covers the slice's tissue bounding box plus a margin.
    plane = b.plane(a, p.index)
    nv, nu = plane.shape
    rows, cols = np.nonzero(plane > 0)
    pad = 20 / min(b.pitch(a))
    x0, x1, y0, y1 = cols.min()-pad, cols.max()+pad, rows.min()-pad, rows.max()+pad
    px = hit[..., rest[0]] / b.spacing[rest[0]]
    py = (nv - 1) - hit[..., rest[1]] / b.spacing[rest[1]]
    on = (k >= 0) & (k <= n-1) & (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
    poly = np.array(footprint(b, p), np.float32)
    mask = np.zeros((nv, nu), np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    in_crop = on & (mask[np.clip(np.round(py), 0, nv-1).astype(np.intp), np.clip(np.round(px), 0, nu-1).astype(np.intp)] > 0)
    pa = np.where(in_crop, crop_alpha, np.where(on, plane_alpha, 0)).astype(np.float32)
    c_front = np.take_along_axis(vol.front_c, s[None], 0)[0].astype(np.float32)
    t_front = np.take_along_axis(vol.front_t, s[None], 0)[0].astype(np.float32)
    behind = vol.front_c[-1].astype(np.float32) - c_front     # already attenuated by t_front
    acc = np.array(accent, np.float32) / 255
    col = c_front[..., None] + (t_front*pa)[..., None]*acc + ((1 - pa)*behind)[..., None]
    img = np.transpose(np.clip(col, 0, 1), (1, 0, 2))[::-1, ::-1]
    r0, r1, c0, c1 = vol.box
    img = (img[r0:r1, c0:c1] * 255).astype(np.uint8)
    h, w = img.shape[:2]
    scale = min(size[0]/w, size[1]/h)
    out = Image.fromarray(cv2.resize(img, (max(1, round(w*scale)), max(1, round(h*scale))),
                                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC))

    def project(points):
        world = np.zeros((len(points), 3))
        world[:, a] = target
        world[:, rest[0]] = [x * b.spacing[rest[0]] for x, _ in points]
        world[:, rest[1]] = [(nv - 1 - y) * b.spacing[rest[1]] for _, y in points]
        return [((x - c0) * scale, (y - r0) * scale) for x, y in v.screen(world)]

    d = ImageDraw.Draw(out)
    rect = project([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    d.line(rect + rect[:1], fill=tuple(int(c*.7) for c in accent), width=1)
    crop = project(poly.tolist())
    d.line(crop + crop[:1], fill=accent, width=2)
    return np.asarray(out)
