"""All crops are similarities in the physical plane of an acquired voxel slice."""
from dataclasses import dataclass, asdict
from functools import lru_cache
from json import dumps as json_dumps, loads as json_loads

import cv2
import nibabel as nib
import numpy as np

from .inputs import ASSETS, SOURCES

PLANES = ('sagittal', 'coronal', 'axial')
ASPECT = 4 / 3
PANEL_SIZE = (1040, 780)
SMOOTH_VOXELS = .6


@dataclass(frozen=True)
class Pose:
    scan: int
    axis: int
    index: int
    x_mm: float
    y_mm: float
    width_mm: float
    angle_deg: float
    mirror: bool
    invert: bool

    def record(self):
        return asdict(self)


CT_WINDOW = (-1000., 1000.)   # Hounsfield units mapped to 0-255: air, lung, fat, muscle, bone
HEAD_MARGIN_MM = 25.          # CT is cut this far above the top of the clavicles
SKULL_MARGIN_MM = 15.         # and at least this far below the lowest skull voxel
CT_PREP = 'ct-prep-v4'


class Brain:
    """One masked volume in closest-RAS orientation, as uint8. Brain MRI uses the
    published brain mask; whole-body CT uses a body mask with the head removed."""
    def __init__(self, spec):
        self.key = spec['key']
        self.ct = self.key.startswith('ts_')
        if self.ct:
            self._load_ct(spec)
            # Whole-body silhouettes need far larger crops than brain anatomy.
            self.widths, self.width_clip = (40., 700.), (30., 760.)
        else:
            self._load_mri()
            self.widths, self.width_clip = (32., 166.), (30., 180.)
        self.limits = []
        for axis in range(3):
            count = (self.data > 0).sum(axis=tuple(i for i in range(3) if i != axis))
            valid = np.flatnonzero(count >= 100)
            self.limits.append((int(valid[0]), int(valid[-1])))

    def _load_mri(self):
        folder = ASSETS / self.key
        # Some IXI labels store a 3D volume with a trailing singleton dimension.
        source = nib.squeeze_image(nib.load(folder / 'image.nii.gz'))
        mask = nib.squeeze_image(nib.load(folder / 'mask.nii.gz'))
        if source.ndim != 3 or mask.ndim != 3:
            raise ValueError(f'Expected a single spatial volume for {self.key}')
        if source.shape != mask.shape or not np.allclose(source.affine, mask.affine, atol=1e-4, rtol=0):
            raise ValueError(f'Mask grid differs for {self.key}')
        image = nib.as_closest_canonical(source)
        mask = np.asanyarray(nib.as_closest_canonical(mask).dataobj) != 0
        raw = np.asanyarray(image.dataobj)
        if image.header.get_xyzt_units()[0] != 'mm' or not mask.any():
            raise ValueError(f'Unexpected geometry or empty brain mask for {self.key}')
        if not np.isfinite(raw[mask]).all():
            raise ValueError(f'Non-finite MRI signal in {self.key}')
        self.upper = float(np.percentile(raw[mask], 99.7))
        if self.upper <= 0:
            raise ValueError(f'Invalid intensity window for {self.key}')
        self.data = np.zeros(raw.shape, dtype=np.uint8)
        self.data[mask] = np.rint(np.clip(raw[mask] * (255.0 / self.upper), 0, 255))
        self.spacing = np.asarray(nib.affines.voxel_sizes(image.affine))
        self.meta = dict(key=self.key, source_shape=list(source.shape),
                         source_affine=source.affine.tolist(), ras_shape=list(image.shape),
                         ras_affine=image.affine.tolist(), spacing_mm=self.spacing.tolist(),
                         intensity_upper=self.upper)

    def _load_ct(self, spec):
        folder = ASSETS / self.key
        cache = folder / f'{CT_PREP}.npz'
        stamp = f"{CT_PREP}:{spec['image']['sha256']}:{spec['skull']['sha256']}"
        if cache.exists():
            saved = np.load(cache)
            if str(saved['stamp']) == stamp:
                self.data, self.meta = saved['data'], json_loads(str(saved['meta']))
                self.spacing = np.asarray(self.meta['spacing_mm'])
                return
        self.data, self.meta = prepare_ct(folder, self.key)
        self.spacing = np.asarray(self.meta['spacing_mm'])
        np.savez(cache, data=self.data, meta=json_dumps(self.meta), stamp=stamp)

    @lru_cache(maxsize=768)
    def plane(self, axis, index):
        # Ascending first remaining RAS axis across, descending second axis down.
        return np.ascontiguousarray(np.take(self.data, index, axis=axis).T[::-1])

    @lru_cache(maxsize=768)
    def smooth(self, axis, index):
        """The displayed slice: the native slice with a 0.6-voxel Gaussian, which rounds
        voxel stair-steps on thresholded edges without moving them."""
        return cv2.GaussianBlur(self.plane(axis, index).astype(np.float32), (0, 0), SMOOTH_VOXELS)

    def pitch(self, axis):
        return tuple(float(self.spacing[i]) for i in range(3) if i != axis)


def load_brains():
    return [Brain(spec) for spec in SOURCES['scans'] + SOURCES.get('ct_scans', [])]


def prepare_ct(folder, key):
    """Deterministic CT preprocessing, documented in the README:
    1. Reorient to closest RAS. 2. Remove everything above the clavicles plus
    HEAD_MARGIN_MM, or SKULL_MARGIN_MM below the lowest skull voxel if that is lower (each label
    reduced to its largest connected component), so no face or
    head anatomy remains. 3. Body mask: voxels above -500 HU, opened by two voxels,
    keeping every connected component larger than 0.5% of the largest except flat,
    wide plates (the scanner table); holes such as lungs and bowel gas are filled in
    each axial slice; then only the largest connected piece, the body, is kept, which
    drops detached equipment such as head holders, curved shells and lines (no body
    part is detached in this cohort: the largest other piece is 1.6% of the body).
    4. Map CT_WINDOW linearly to 0-255 inside the mask; zero outside."""
    from scipy import ndimage
    image = nib.as_closest_canonical(nib.squeeze_image(nib.load(folder / 'image.nii.gz')))
    def mask(name):
        m = nib.as_closest_canonical(nib.squeeze_image(nib.load(folder / f'{name}.nii.gz')))
        if m.shape != image.shape or not np.allclose(m.affine, image.affine, atol=1e-3):
            raise ValueError(f'{name} grid differs from the CT for {key}')
        return np.asanyarray(m.dataobj) != 0
    hu = np.asanyarray(image.dataobj, dtype=np.float32) if image.dataobj.slope == 1 and image.dataobj.inter == 0 \
        else image.get_fdata(dtype=np.float32)
    spacing = np.asarray(nib.affines.voxel_sizes(image.affine))
    def main_part(m):
        # Reference labels contain stray voxels (one scan has five "skull" voxels in
        # the legs), so each structure is its largest connected component.
        labels, n = ndimage.label(m)
        if n <= 1:
            return m
        return labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))
    clavicles = main_part(mask('clavicula_left')) | main_part(mask('clavicula_right'))
    if not clavicles.any():
        raise ValueError(f'No clavicles labelled in {key}; cannot place the head cut')
    top = int(np.flatnonzero(clavicles.any(axis=(0, 1))).max())
    cut = top + int(round(HEAD_MARGIN_MM / spacing[2]))
    skull = main_part(mask('skull'))
    if skull.any():
        cut = min(cut, int(np.flatnonzero(skull.any(axis=(0, 1))).min()) - 1 - int(round(SKULL_MARGIN_MM / spacing[2])))
    if cut < 10:
        raise ValueError(f'Head cut at slice {cut} leaves no body in {key}; check its labels')
    keep = np.zeros(hu.shape[2], bool)
    keep[:cut+1] = True
    body = (hu > -500) & keep[None, None, :]
    body = ndimage.binary_opening(body, iterations=2)
    labels, n = ndimage.label(body)
    sizes = np.bincount(labels.ravel())[1:]
    selected = np.zeros(n + 1, bool)
    for i, box in enumerate(ndimage.find_objects(labels)):
        width_x = (box[0].stop - box[0].start) * spacing[0]
        depth_y = (box[1].stop - box[1].start) * spacing[1]
        plate = width_x > 250 and depth_y < 80
        selected[i + 1] = sizes[i] >= .005 * sizes.max() and not plate
    body = selected[labels]
    del labels
    body = ndimage.binary_dilation(body, iterations=2) & (hu > -500) & keep[None, None, :]
    for z in range(body.shape[2]):
        if body[:, :, z].any():
            body[:, :, z] = ndimage.binary_fill_holes(body[:, :, z])
    labels, n = ndimage.label(body)
    if n > 1:
        body = labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))
    del labels
    lo, hi = CT_WINDOW
    data = np.zeros(hu.shape, np.uint8)
    data[body] = np.rint(np.clip((hu[body] - lo) * (255. / (hi - lo)), 1, 255))
    if skull.any() and (data[skull] > 0).any():
        raise AssertionError(f'Skull voxels survived the head cut in {key}')
    meta = dict(key=key, source_shape=list(image.shape), ras_shape=list(image.shape),
                ras_affine=image.affine.tolist(), spacing_mm=spacing.tolist(),
                ct_window_hu=list(CT_WINDOW), head_cut_index=cut, clavicle_top_index=top,
                preprocessing=CT_PREP)
    return data, meta


@lru_cache(maxsize=24)
def unit_grid(w, h):
    x = ((np.arange(w, dtype=np.float32) + .5) / w - .5)[None, None, :]
    y = ((np.arange(h, dtype=np.float32) + .5) / h - .5)[None, :, None] / ASPECT
    return x, y


def maps(params, pitch, w, h, corners=False):
    # Rows are x_mm, y_mm, width_mm, angle_deg, mirror. The mapping runs
    # from output pixel centres to voxel centres in the displayed full slice.
    p = np.asarray(params, dtype=np.float32)
    x, y, width, angle, mirror = (col[:, None, None] for col in p.T)
    if corners:
        gx = np.array([[[-.5, .5, .5, -.5]]], dtype=np.float32)
        gy = np.array([[[-.5, -.5, .5, .5]]], dtype=np.float32) / ASPECT
    else:
        gx, gy = unit_grid(w, h)
    a = np.deg2rad(angle)
    c, s = width * np.cos(a), width * np.sin(a)
    u = gx * (1 - 2 * mirror)
    return (x + c*u - s*gy) / pitch[0], (y + s*u + c*gy) / pitch[1]


def inside(params, pitch, shape):
    x, y = maps(params, pitch, 0, 0, corners=True)
    return ((x >= 0) & (x <= shape[1]-1) & (y >= 0) & (y <= shape[0]-1)).all(axis=(1, 2))


def sample(plane, pitch, params, w, h, interp=cv2.INTER_LINEAR):
    # Stack hundreds of independent maps into one optimized OpenCV call.
    x, y = maps(params, pitch, w, h)
    out = cv2.remap(plane, x.reshape(-1, w), y.reshape(-1, w), interp)
    return out.reshape(-1, h, w)


def values(p):
    return [p.x_mm, p.y_mm, p.width_mm, p.angle_deg, float(p.mirror)]


def crop(brains, p, w, h, interp=cv2.INTER_LINEAR):
    b = brains[p.scan]
    return sample(b.plane(p.axis, p.index), b.pitch(p.axis), [values(p)], w, h, interp)[0]


def smooth_crop(brains, p, w, h):
    b = brains[p.scan]
    return sample(b.smooth(p.axis, p.index), b.pitch(p.axis), [values(p)], w, h)[0]


def silhouette(brains, p, level, w, h, supersample=2):
    """Black and white image of one crop: pixels at or above the level are white
    (black when inverted). Thresholding at twice the resolution and averaging gives
    one-pixel anti-aliased edges. The same level applies to the whole crop."""
    field = smooth_crop(brains, p, w*supersample, h*supersample)
    white = ((field >= level) != p.invert).astype(np.float32) * 255
    return np.rint(cv2.resize(white, (w, h), interpolation=cv2.INTER_AREA)).astype(np.uint8)


GRAY_WINDOWS = (36, 72, 112)


def tone(raw, level, window, invert):
    """Gray display: one monotonic contrast curve for the whole crop, crossing 50% at
    the matched level. A narrow ramp of the given window forms the silhouette and a
    broad ramp keeps the scan's texture visible on both sides of it. The narrow ramp
    shortens near either end so empty background reaches full black or white."""
    delta = raw.astype(np.float32) - level
    half = np.where(delta < 0, min(window/2, max(level, 1)), min(window/2, max(255-level, 1)))
    narrow = np.clip(.5 + .5*delta / half, 0, 1)
    broad = np.clip(.5 + .5*delta / np.where(delta < 0, max(level, 1), max(255-level, 1)), 0, 1)
    result = np.rint(255 * (.62*narrow + .38*broad)).astype(np.uint8)
    return 255 - result if invert else result


def render_crop(brains, p, match, w, h):
    """The displayed image: gray when the match records a contrast window, otherwise
    the black and white silhouette."""
    if match.get('window'):
        return tone(crop(brains, p, w, h, cv2.INTER_CUBIC), match['level'], match['window'], p.invert)
    return silhouette(brains, p, match['level'], w, h)


def panel(brains, p, match):
    return render_crop(brains, p, match, *PANEL_SIZE)


def footprint(brain, p):
    x, y = maps([values(p)], brain.pitch(p.axis), 0, 0, corners=True)
    return np.stack([x.ravel(), y.ravel()], axis=1).tolist()
