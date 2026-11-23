"""Render saved matches as a radiology reading-room screen: the matched crop as the
main viewport with corner annotations, orientation letters and a scale; the source
frame; a 3D view of the matched scan with the slice plane; the cohort strip; and a
localizer with the crop's footprint.

Every annotation is derived from the scan files or the run: orientation letters and
positions from each scan's affine, CT values in Hounsfield units through the inverse
of the preparation window, sizes from the crop geometry."""
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
from collections import deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import model3d
from .geometry import CT_WINDOW, PANEL_SIZE, PLANES, Brain, Pose, footprint, panel
from .inputs import SOURCES, verify
from .media import VIDEO, frames

def layout(name):
    """Panel boxes (x, y, width, height). The viewport keeps the native panel size in
    both layouts, so its pixels are exactly the rendered crop."""
    W, H = PANEL_SIZE
    if name == 'vertical':
        # 9:16 for phone feeds: the viewport at the centre, clear of the apps' top tabs
        # and bottom captions; source, localizer and 3D view above, cohort strip below.
        size = (1080, 1920)
        hero = (20, (size[1] - H)//2, W, H)
        top, lw = 40, 380
        band = hero[1] - 6 - top
        ref = (20, top, lw, round(lw*3/4))
        local = (20, ref[1] + ref[3] + 6, lw, hero[1] - 6 - (ref[1] + ref[3] + 6))
        model = (20 + lw + 6, top, W - lw - 6, band)
        strip = (20, hero[1] + H + 6, W, size[1] - 34 - (hero[1] + H + 6))
        spark = (500, 9, 340, 14)
    else:
        size, hero = (1440, 1080), (6, 40, W, H)
        rx, rw = 1052, 382                   # right column
        ref = (rx, 40, rw, round(rw*3/4))
        model = (rx, ref[1]+ref[3]+6, rw, hero[1]+H - (ref[1]+ref[3]+6))
        strip = (hero[0], hero[1]+H+6, W, size[1] - 34 - (hero[1]+H+6))
        local = (rx, strip[1], rw, strip[3])
        spark = (880, 9, 330, 14)
    return dict(SIZE=size, HERO=hero, REF=ref, MODEL=model, STRIP=strip, LOCAL=local, SPARK=spark)


def use_layout(name):
    """Switch the module's layout; call before render() so forked workers inherit it."""
    globals().update(layout(name))


use_layout('landscape')
INK, MUTE, RULE, ACCENT, BAR, TILE = '#e8e8e8', '#8e959c', '#2e3236', '#e2b24c', '#141618', '#1a1d20'
PLANE_SHORT = ('Sag', 'Cor', 'Ax')
SEQUENCE = {'t1': 'T1-weighted MRI', 't2': 'T2-weighted MRI', 'pd': 'Proton density MRI', 'mra': 'MR angiography'}
STUDY = {'ct polytrauma': 'Polytrauma', 'ct whole body': 'Whole body',
         'ct angiography neck-thx-abd-pelvis-leg': 'Angiography, neck to leg',
         'ct neck-thorax-abdomen-pelvis': 'Neck to pelvis'}
STUDY_OF = {s['key']: s.get('study_type', '') for s in SOURCES.get('ct_scans', [])}
FONTS = {False: ['/System/Library/Fonts/Supplemental/Arial.ttf', '/Library/Fonts/Arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'],
         True: ['/System/Library/Fonts/Supplemental/Arial Bold.ttf', '/Library/Fonts/Arial Bold.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']}
HU_PER_UNIT = (CT_WINDOW[1] - CT_WINDOW[0]) / 255


@lru_cache(maxsize=40)
def font(size, bold=False):
    for name in FONTS[bold]:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


class Painter:
    def __init__(self, canvas):
        self.c, self.d = canvas, ImageDraw.Draw(canvas)

    def text(self, xy, s, size=13, fill=INK, anchor='la', bold=False, overlay=False):
        """Overlay text has a thin outline, and turns dark with a light outline where it
        sits on bright image, as real viewers' inverse annotation does."""
        f = font(size, bold)
        stroke, sfill = 0, None
        if overlay:
            stroke, sfill = 1, '#000000'
            box = [int(v) for v in self.d.textbbox(xy, s, font=f, anchor=anchor)]
            region = np.asarray(self.c.crop(box).convert('L'))
            if region.size and region.mean() > 150:
                fill, sfill = '#121212', '#f2f2f2'
        self.d.text(xy, s, font=f, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=sfill)

    def rows(self, xy, rows, size=15, anchor='la', overlay=True):
        """Corner annotation block; bottom anchors stack upwards."""
        x, y = xy
        gap = round(size * 1.27)
        up = anchor[1] in 'bd'
        for i, s in enumerate(reversed(rows) if up else rows):
            self.text((x, y - i*gap if up else y + i*gap), s, size, anchor=anchor, overlay=overlay)

    def frame(self, box, outline=RULE):
        x, y, w, h = box
        self.d.rectangle((x-1, y-1, x+w, y+h), outline=outline)

    def dashed(self, pts, fill, width=1, dash=9, gap=6):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            n = float(np.hypot(x1-x0, y1-y0))
            s = 0.
            while s < n:
                e = min(n, s + dash)
                self.d.line((x0+(x1-x0)*s/n, y0+(y1-y0)*s/n, x0+(x1-x0)*e/n, y0+(y1-y0)*e/n), fill=fill, width=width)
                s = e + gap


def image_fit(plane, pitch, box, tight=False, corners=()):
    x, y, w, h = box
    ox = oy = 0
    if tight:
        bx, by, bw, bh = cv2.boundingRect((plane > 0).astype(np.uint8))
        if bw and bh:
            x0, x1, y0, y1 = bx, bx+bw, by, by+bh
            for px, py in corners:
                x0, x1 = min(x0, int(np.floor(px))), max(x1, int(np.ceil(px))+1)
                y0, y1 = min(y0, int(np.floor(py))), max(y1, int(np.ceil(py))+1)
            pad = max(3, round(.06*max(x1-x0, y1-y0)))
            ox, oy = max(0, x0-pad), max(0, y0-pad)
            plane = plane[oy:y1+pad, ox:x1+pad]
    # Pixel edges define the image extent. Record the exact post-rounding scale.
    scale = min(w/(plane.shape[1]*pitch[0]), h/(plane.shape[0]*pitch[1]))
    ow = max(1, round(plane.shape[1]*pitch[0]*scale))
    oh = max(1, round(plane.shape[0]*pitch[1]*scale))
    pos = (x+(w-ow)//2, y+(h-oh)//2)
    img = Image.fromarray(plane).resize((ow, oh), Image.Resampling.BILINEAR)

    def project(point):
        px, py = point
        return (pos[0]+(px-ox+.5)*ow/plane.shape[1]-.5, pos[1]+(py-oy+.5)*oh/plane.shape[0]-.5)
    return img, pos, project


def subject(b):
    return ('CT ' + b.key[4:]) if b.ct else ('IXI ' + b.key[-3:])


def thumbnail_plane(b, axis):
    """Central slice of the scanned extent, with its tight physical width and height."""
    lo, hi = b.limits[axis]
    plane = b.plane(axis, (lo+hi)//2)
    _, _, bw, bh = cv2.boundingRect((plane > 0).astype(np.uint8))
    pitch = b.pitch(axis)
    return plane, max(bw, 1)*pitch[0], max(bh, 1)*pitch[1]


def mmss(t):
    return f'{int(t//60):02}:{t % 60:05.2f}'


# ---------------------------------------------------------------- scan geometry
def letters(v):
    """Anatomical letters for a RAS direction, strongest first; obliques get two."""
    names = (('L', 'R'), ('P', 'A'), ('I', 'S'))
    parts = sorted(((abs(v[i]), names[i][int(v[i] > 0)]) for i in range(3)), reverse=True)
    return ''.join(c for m, c in parts if m >= .38*np.linalg.norm(v))


def plane_to_world(b, axis, dx, dy):
    """A step of (dx, dy) displayed-plane pixels as a RAS direction through the scan's
    affine. plane() puts the first remaining axis across and the second reversed down."""
    rest = [i for i in range(3) if i != axis]
    d = np.zeros(3)
    d[rest[0]], d[rest[1]] = dx, -dy
    return np.asarray(b.meta['ras_affine'])[:3, :3] @ d


def crop_edges(b, p):
    """Letters at the main viewport's left, right, top and bottom edges."""
    a = np.deg2rad(p.angle_deg)
    pitch = b.pitch(p.axis)
    right = np.array([np.cos(a), np.sin(a)]) * (1 - 2*p.mirror)      # millimetres, x right, y down
    down = np.array([-np.sin(a), np.cos(a)])
    r = plane_to_world(b, p.axis, right[0]/pitch[0], right[1]/pitch[1])
    dn = plane_to_world(b, p.axis, down[0]/pitch[0], down[1]/pitch[1])
    return dict(l=letters(-r), r=letters(r), t=letters(-dn), b=letters(dn))


def plane_edges(b, axis):
    r, dn = plane_to_world(b, axis, 1, 0), plane_to_world(b, axis, 0, 1)
    return dict(l=letters(-r), r=letters(r), t=letters(-dn), b=letters(dn))


def describe(b, p, match, entry, t, total):
    A = np.asarray(b.meta['ras_affine'])
    centre = np.array([s/2 for s in b.data.shape] + [1.])
    centre[p.axis] = p.index
    level, window = match['level'], match.get('window')
    if b.ct:
        wl = (f'W {window*HU_PER_UNIT:.0f}  L {CT_WINDOW[0] + level*HU_PER_UNIT:.0f}' if window
              else f'Threshold {CT_WINDOW[0] + level*HU_PER_UNIT:.0f}')
        study = STUDY.get(STUDY_OF.get(b.key, ''), 'Body')
    else:
        wl = f'W {window}  L {level:.0f}' if window else f'Threshold {level:.0f}'
        study = 'Brain'
    return dict(
        t=t, total=total, time=entry['time_s'], id=subject(b), score=match['metrics']['score'],
        modality='CT' if b.ct else SEQUENCE[b.key.split('_')[1]], study=study,
        slice=f'{PLANE_SHORT[p.axis]}  Sl {p.index+1}/{b.data.shape[p.axis]}',
        pos=f"{'XYZ'[p.axis]} {(A @ centre)[p.axis]:.1f} mm",
        fov=f'FOV {p.width_mm:.0f} x {p.width_mm*PANEL_SIZE[1]/PANEL_SIZE[0]:.0f} mm',
        rot=f'Rot {p.angle_deg:+.1f}°' + ('  Flip H' if p.mirror else ''),
        zoom=f'Zoom {PANEL_SIZE[0]/p.width_mm*b.pitch(p.axis)[0]:.1f}',
        wl=wl, hu=' HU' if b.ct else '', polarity='Inverted' if p.invert else 'Normal',
        source='TotalSegmentator v2.0.1' if b.ct else 'IXI dataset',
        licence='CC BY 4.0' if b.ct else 'CC BY-SA 4.0')


# ---------------------------------------------------------------- static layout
def cohort(brains, kind):
    if kind == 'ct':
        return [b for b in brains if b.ct], 1
    return [b for b in brains if b.key.startswith('ixi_t1_')], 2


def cell_key(b):
    """CT tiles are scans; MRI tiles are subjects, shared by their four sequences."""
    return b.key if b.ct else b.key[-3:]


def scores(records):
    return np.array([r['silhouette']['metrics']['score'] for r in records])


def spark_points(records):
    s = scores(records)
    lo = max(0., np.floor(s.min()*10)/10)
    x, y, w, h = SPARK
    return [(x + w*j/max(1, len(s)-1), y + h*(1 - (v-lo)/max(1e-6, 1-lo))) for j, v in enumerate(s)]


def toolbar_glyphs(d):
    g = '#b8bec4'
    for k in range(6):
        cx = 140 + k*34
        if k == 2:      # the active tool: window and level
            d.rectangle((cx-13, 4, cx+13, 28), outline='#5a6067', fill='#202326')
        if k == 0:      # layout
            for i in range(2):
                for j in range(2):
                    d.rectangle((cx-8+i*9, 8+j*9, cx-1+i*9, 15+j*9), outline=g)
        elif k == 1:    # stack
            for i in range(3):
                d.rectangle((cx-7+i*3, 7+i*3, cx+3+i*3, 17+i*3), outline=g, fill=BAR)
        elif k == 2:    # window and level
            d.ellipse((cx-8, 8, cx+8, 24), outline=g)
            d.pieslice((cx-8, 8, cx+8, 24), 90, 270, fill=g)
        elif k == 3:    # zoom
            d.ellipse((cx-8, 7, cx+4, 19), outline=g, width=2)
            d.line((cx+3, 18, cx+8, 24), fill=g, width=2)
        elif k == 4:    # measure
            d.line((cx-9, 22, cx+9, 10), fill=g, width=2)
            for s in range(4):
                xx, yy = cx-6+s*4.5, 20-s*3
                d.line((xx, yy, xx-2, yy-3), fill=g)
        else:           # cine
            d.polygon([(cx-5, 8), (cx-5, 24), (cx+7, 16)], outline=g)


def base_canvas(brains, kind, records, fps):
    """Everything that does not change within one cohort: chrome, labels, the dim
    similarity trace and the cohort strip. Returns the canvas and each tile's box."""
    canvas = Image.new('RGB', SIZE, '#000000')
    P = Painter(canvas)
    d = P.d
    d.rectangle((0, 0, SIZE[0], 32), fill=BAR)
    d.line((0, 32, SIZE[0], 32), fill=RULE)
    P.text((12, 16), 'BAD APPLE!!', 14, anchor='lm', bold=True)
    toolbar_glyphs(d)
    d.line(spark_points(records), fill='#4a5056', width=1)
    for box in (REF, MODEL, STRIP, LOCAL):
        P.frame(box)
    P.frame(HERO, ACCENT)
    x, y, w, h = MODEL
    P.text((x+8, y+8), '3D', 13)
    P.text((x+30, y+8), 'Matched scan volume', 13, MUTE)
    atlas, axis = cohort(brains, kind)
    x, y, w, h = STRIP
    P.text((x+8, y+6), f"{len(atlas)} {'bodies' if kind == 'ct' else 'brains'}", 13, MUTE)
    thumbs = [thumbnail_plane(b, axis) for b in atlas]
    aspect = float(np.median([tw/th for _, tw, th in thumbs]))
    gx, gy, gw, gh = x+4, y+24, w-8, h-28

    def fit(cols):
        rows = -(-len(atlas)//cols)
        tw, th = gw//cols - 4, gh//rows - 4
        return min(tw/aspect, th) if tw > 0 and th > 0 else 0
    cols = max(range(1, len(atlas)+1), key=fit)
    rows = -(-len(atlas)//cols)
    tw, th = gw//cols, gh//rows
    ox, oy = gx + (gw - tw*cols)//2, gy + (gh - th*rows)//2
    cells = {}
    for i, (b, (plane, _, _)) in enumerate(zip(atlas, thumbs)):
        cx, cy = ox + (i % cols)*tw, oy + (i//cols)*th
        d.rectangle((cx, cy, cx+tw-2, cy+th-2), fill='#050607', outline=TILE)
        img, pos, _ = image_fit(plane, b.pitch(axis), (cx+2, cy+2, tw-6, th-6), True)
        canvas.paste(img, pos)
        cells[cell_key(b)] = (cx-1, cy-1, cx+tw-1, cy+th-1)
    d.rectangle((0, SIZE[1]-30, SIZE[0], SIZE[1]), fill=BAR)
    d.line((0, SIZE[1]-30, SIZE[0], SIZE[1]-30), fill=RULE)
    P.text((12, SIZE[1]-15), 'IXI MRI  CC BY-SA 4.0      Brain masks  Hoopes et al.      '
           'CT  TotalSegmentator  CC BY 4.0      Shadow PV  Anira', 12, MUTE, 'lm')
    P.text((SIZE[0]-12, SIZE[1]-15), f'{fps} fps', 12, MUTE, 'rm')
    return canvas, cells


# ---------------------------------------------------------------- per frame
def ruler(P, x, y0, h, mm_per_px):
    """Vertical scale in the lower half of the viewport, clear of the edge letters."""
    fits = [L for L in (5, 10, 20, 50, 100, 200, 500) if L/mm_per_px <= h*.5 - 140]
    length = max(fits) if fits else 5
    px = length/mm_per_px
    top = y0 + h - 120 - px
    step = 10 if px/(length/10) >= 6 else length/5
    P.d.line((x, top, x, top+px), fill='#000000', width=3)
    P.d.line((x, top, x, top+px), fill=INK, width=1)
    k = 0
    while k*step <= length + 1e-6:
        yy = top + k*step/mm_per_px
        major = abs((k*step) % (length/2 if length >= 20 else length)) < 1e-6
        P.d.line((x, yy, x - (10 if major else 5), yy), fill=INK, width=1)
        k += 1
    P.text((x-14, top+px), f'{length/10:g} cm', 13, anchor='rd', overlay=True)


def edge_text(P, box, e, size, fill=INK, inset=8):
    x, y, w, h = box
    for anchor, xy, s in (('lm', (x+inset, y+h/2), e['l']), ('rm', (x+w-inset, y+h/2), e['r']),
                          ('mt', (x+w/2, y+inset), e['t']), ('mb', (x+w/2, y+h-inset), e['b'])):
        P.text(xy, s, size, fill, anchor, overlay=True)


def localizer(P, b, p):
    x, y, w, h = LOCAL
    plane = b.plane(p.axis, p.index)
    if not b.ct:
        # Lateral brain slices are small and dim: stretch this view only, for legibility.
        top = np.percentile(plane[plane > 0], 99) if (plane > 0).any() else 255
        plane = np.clip(plane.astype(np.float32) * (255/max(top, 1)), 0, 255).astype(np.uint8)
    corners = footprint(b, p)
    img, pos, project = image_fit(plane, b.pitch(p.axis), (x+4, y+4, w-8, h-8), tight=True, corners=corners)
    P.c.paste(img, pos)
    pts = [project(c) for c in corners]
    P.dashed(pts + pts[:1], '#000000', 4)
    P.dashed(pts + pts[:1], ACCENT, 2)
    P.d.line((pts[0], pts[1]), fill=ACCENT, width=3)       # the crop's top edge
    P.text((x+8, y+8), 'Localizer', 13, overlay=True)
    P.text((x+w-8, y+h-8), f'{PLANE_SHORT[p.axis]} {p.index+1}', 13, anchor='rd', overlay=True)
    edge_text(P, LOCAL, plane_edges(b, p.axis), 13, MUTE, 6)


def draw_frame(bases, brains, originals, records, t):
    r = records[t]
    match = r['silhouette']
    p = Pose(**match['pose'])
    b = brains[p.scan]
    base, cells = bases['ct' if b.ct else 'mri']
    canvas = base.copy()
    P = Painter(canvas)
    d = P.d
    f = describe(b, p, match, r, t, len(records))
    # Toolbar: frame, time, score and the played part of the similarity trace.
    P.text((SIZE[0]-12, 16), f"{t+1:04} / {len(records):04}     {mmss(r['time_s'])}", 14, MUTE, 'rm')
    P.text((SPARK[0]-10, 16), f"Sim {f['score']:.3f}", 13, INK, 'rm')
    pts = spark_points(records)
    if t > 0:
        d.line(pts[:t+1], fill=INK, width=1)
    d.line((pts[t][0], SPARK[1]-2, pts[t][0], SPARK[1]+SPARK[3]+2), fill=ACCENT)
    # Main viewport.
    hx, hy, W, H = HERO
    canvas.paste(Image.fromarray(panel(brains, p, match)), (hx, hy))
    P.rows((hx+10, hy+10), [f['id'], f['modality'], f['study'], f['slice']])
    P.rows((hx+W-10, hy+10), [f['source'], f['licence'], f'Fr {t+1}', mmss(f['time'])], anchor='ra')
    P.rows((hx+10, hy+H-10), [f['pos'], f['fov'], f['rot'], f['zoom']], anchor='ld')
    P.rows((hx+W-34, hy+H-10), [f['wl'], f['polarity'], f"Sim {f['score']:.3f}"], anchor='rd')
    edge_text(P, HERO, crop_edges(b, p), 20)
    ruler(P, hx+W-14, hy, H, p.width_mm/W)
    # Source frame.
    x, y, w, h = REF
    canvas.paste(Image.fromarray(originals[t]).resize((w, h), Image.Resampling.LANCZOS), (x, y))
    P.rows((x+8, y+8), ['Reference', 'Shadow PV'], 13)
    P.text((x+w-8, y+h-8), f'Fr {t+1}', 13, anchor='rd', overlay=True)
    # 3D view of the matched scan with the slice plane.
    x, y, w, h = MODEL
    img = model3d.render(b, p, (w-24, h-60))
    canvas.paste(Image.fromarray(img), (x + (w-img.shape[1])//2, y + 28 + (h-56-img.shape[0])//2))
    P.text((x+w-8, y+8), f'{PLANE_SHORT[p.axis]} {p.index+1}', 13, ACCENT, 'ra')
    # Orientation gizmo, drawn over the render: the patient's R, A and S axes as this
    # fixed view shows them.
    ox, oy = x+56, y+h-40
    for name, (dx, dy) in model3d.screen_axes().items():
        d.line((ox, oy, ox + 32*dx, oy + 32*dy), fill=MUTE, width=1)
        P.text((ox + 42*dx, oy + 42*dy), name, 11, MUTE, 'mm')
    # Cohort strip and localizer.
    d.rectangle(cells[cell_key(b)], outline=ACCENT, width=2)
    P.text((STRIP[0]+STRIP[2]-8, STRIP[1]+6), f['id'], 13, ACCENT, 'ra')
    localizer(P, b, p)
    return canvas


# ---------------------------------------------------------------- video
_JOB = {}
BLOCK = 12


def _render_block(bounds):
    lo, hi = bounds
    j = _JOB
    return [draw_frame(j['bases'], j['brains'], j['originals'], j['records'], t).tobytes() for t in range(lo, hi)]


def render_workers():
    n = os.cpu_count() or 1
    for name in ('SLURM_CPUS_PER_TASK', 'CPU_LIMIT'):
        if os.environ.get(name):
            n = max(1, int(float(os.environ[name])))
    return max(1, min(8, n))   # each worker caches a few 3D volumes of ~120 MB


def render(output, record, brains, originals):
    conf, matches = record['config'], record['frames']
    kinds = ['mri'] + (['ct'] if any(b.ct for b in brains) else [])
    bases = {kind: base_canvas(brains, kind, matches, conf['fps']) for kind in kinds}
    path = output/'bad-apple-mri.mp4'
    # Unique same-directory temporary file keeps interrupted encodes out of replay.
    with tempfile.NamedTemporaryFile(dir=output, suffix='.mp4', delete=False) as f:
        temp = Path(f.name)
    args = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(conf['fps']), '-i', 'pipe:0',
            '-ss', str(conf['start']), '-i', str(VIDEO), '-map', '0:v:0', '-map', '1:a:0?',
            '-t', str(len(matches)/conf['fps']), '-c:v', 'libx264', '-preset', 'fast', '-crf', '16',
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart',
            '-metadata', 'title=Bad Apple!! / Closest Medical Scan',
            '-metadata', 'comment=IXI MRI via SynthStrip v1.5, Hoopes et al. (2022), CC BY-SA 4.0. '
            'CT: TotalSegmentator v2.0.1, Wasserthal et al., CC BY 4.0. Shadow PV: Anira. '
            'Music: Alstroemeria Records, nomico, ZUN. See accompanying credits.', str(temp)]
    blocks = [(lo, min(lo+BLOCK, len(matches))) for lo in range(0, len(matches), BLOCK)]
    workers = render_workers()
    _JOB.update(bases=bases, brains=brains, originals=originals, records=matches)
    done = 0
    try:
        with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
            try:
                def write(chunk):
                    nonlocal done
                    for raw in chunk:
                        if done == 0:
                            Image.frombytes('RGB', SIZE, raw).save(output/'preview.jpg', quality=94, subsampling=0)
                        proc.stdin.write(raw)
                        done += 1
                    if done % 600 < BLOCK or done == len(matches):
                        print(f'Rendered {done}/{len(matches)} frames', flush=True)
                gpu = sys.modules.get('mriapple.gpu')
                if workers > 1 and (sys.platform == 'linux' or not getattr(gpu, 'MPS_USED', False)):
                    # Forked workers share the loaded volumes; a bounded window of
                    # in-flight blocks keeps memory flat if encoding is the bottleneck.
                    with multiprocessing.get_context('fork').Pool(workers) as pool:
                        pending = deque()
                        for blk in blocks:
                            pending.append(pool.apply_async(_render_block, (blk,)))
                            if len(pending) >= 2*workers:
                                write(pending.popleft().get())
                        while pending:
                            write(pending.popleft().get())
                else:
                    for blk in blocks:
                        write(_render_block(blk))
            finally:
                proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError('FFmpeg failed to encode video')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
        _JOB.clear()
    print(f'Rendered {path}', flush=True)


def scans_for(record):
    """Brains for a saved run, in its own scan order. Every scan must still be pinned
    with the same hashes, so runs made before the manifest grew still replay."""
    pinned = {s['key']: s for s in SOURCES['scans'] + SOURCES.get('ct_scans', [])}
    saved = record['dataset']['scans'] + record['dataset'].get('ct_scans', [])
    for spec in saved:
        if pinned.get(spec['key']) != spec:
            raise ValueError(f"{spec['key']} differs from the pinned inputs")
    order = [m['key'] for m in record['scans']]
    return [Brain(pinned[k]) for k in order]


def replay(output, style=None, to=None):
    verify()
    record = json.loads((output/'matches.json').read_text())
    if record['schema'] != 1 or record['dataset']['video'] != SOURCES['video']:
        raise ValueError('Saved match data references a different source or an unsupported schema')
    c = record['config']
    originals, indices, rate = frames(c['start'], c['seconds'], c['fps'])
    if rate != record['source_fps'] or indices != [r['source_frame'] for r in record['frames']]:
        raise ValueError('Source frame mapping differs from the saved run')
    brains = scans_for(record)
    out = to or output
    if style and style != record.get('display', 'bw'):
        from .report import apply_style, report
        record = apply_style(brains, record, originals, style)
    if out != output or style:
        out.mkdir(parents=True, exist_ok=True)
        (out/'matches.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        from .report import report
        report(out, record)
    render(out, record, brains, originals)
