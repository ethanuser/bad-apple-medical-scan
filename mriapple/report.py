"""Report both matchers on the pixels of their full-size MRI panels."""
import shutil
import time
from statistics import median

import numpy as np

from .geometry import Pose, panel
from .inputs import ROOT
from .metrics import down, score_image
from .search import METRIC_W


def measure_panels(brains, matches, originals):
    begin = time.perf_counter()
    for row, frame in zip(matches, originals, strict=True):
        p = Pose(**row['pose'])
        actual = panel(brains, p, row)
        row['metrics'] = score_image(down(actual, METRIC_W), down(frame, METRIC_W))
    return time.perf_counter()-begin


def pitch(scan, axis):
    return [v for i, v in enumerate(scan['spacing_mm']) if i != axis]


def apply_style(brains, record, originals, style):
    """Switch a saved run between the black and white and the gray display, keeping
    every pose and level. Gray picks each frame's contrast window by score, then all
    metrics are re-measured on the full panels as displayed."""
    from .geometry import GRAY_WINDOWS, render_crop
    rows = [r[k] for r in record['frames'] for k in ('coarse_ncc', 'silhouette')]
    targets = [down(f, METRIC_W) for f in originals for _ in range(2)]
    for row, target in zip(rows, targets):
        row.pop('window', None)
        if style == 'gray':
            p = Pose(**row['pose'])
            row['window'] = max(GRAY_WINDOWS, key=lambda w: score_image(
                down(render_crop(brains, p, dict(row, window=w), METRIC_W*2, METRIC_W*3//2), METRIC_W),
                target)['score'])
    for key in ('coarse_ncc', 'silhouette'):
        matches = [r[key] for r in record['frames']]
        measure_panels(brains, matches, originals)
        record['benchmark'][key] = {m: sum(r['metrics'][m] for r in matches)/len(matches) for m in matches[0]['metrics']}
    record['display'] = style
    return record


def report(output, record):
    cfg, bench, frames = record['config'], record['benchmark'], record['frames']
    coarse = bench['coarse_seconds']
    baseline_sec = coarse - bench['shape_retrieval_extra_seconds'] + bench['baseline_finish_seconds']
    refined_sec = coarse + bench['baseline_finish_seconds'] + bench['shape_finish_seconds']
    rows = []
    for title, key, seconds in [('Grayscale NCC', 'coarse_ncc', baseline_sec),
                                ('Contour search + temporal refinement', 'silhouette', refined_sec)]:
        stats = bench[key]
        sources = [row[key]['source_key'] for row in frames]
        changes = sum(a != b for a, b in zip(sources, sources[1:]))
        voxels = median([row[key]['pose']['width_mm'] / np.sqrt(np.prod(pitch(record['scans'][row[key]['pose']['scan']],
                                                                       row[key]['pose']['axis']))) for row in frames])
        rows.append(f"| {title} | {stats['miou']:.3f} | {stats['boundary_f1']:.3f} | {stats['detail']:.3f} | "
                    f"{stats['ncc']:.3f} | {stats['score']:.3f} | {voxels:.0f} | {changes} | {seconds:.1f} s |")
    scans = len(record['scans'])
    out = [
        '# Matching results', '',
        f"{len(frames)} frames from {cfg['start']:.1f} to {cfg['start']+cfg['seconds']:.1f} s, "
        f"sampled at {cfg['fps']} fps. The search covered {scans} volumes, "
        f"{bench['sampled_slices']:,} native sections and {bench['valid_geometries']:,} in-bounds crops. "
        f"Mirroring and intensity polarity yielded {bench['oriented_polarities']:,} views per target frame.", '',
        '| Matcher | Balanced IoU | Boundary F1 | Thin detail | Grayscale NCC | Similarity | Native voxels | Volume changes | Matching time |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|', *rows, '',
        'Metrics are frame means. Both methods use the same sources, geometry proposals and threshold fitting. '
        'An overlap of 1.0 means both silhouette classes match exactly. Boundary F1 allows two pixels '
        'at 96 × 72. Thin detail is the share of target features under five pixels wide (hair, fingers, narrow gaps) '
        'that the match reproduces. The displayed similarity is 28% balanced IoU, 27% boundary F1, 20% thin detail, '
        '12.5% positive global NCC and 12.5% positive regional NCC. '
        'Selection also adds a small preference for crops spanning more native voxels and a small cost for '
        'thresholds that trace the brain-extraction mask. Neither enters the displayed similarity. '
        'Native voxels is the median crop width in source voxels; the panel magnifies it to 1040 pixels.', '',
        ('The baseline time uses the shared search duration less the measured contour-only work. '
         if bench.get('device', 'numpy') == 'numpy' else
         f"Retrieval ran on {bench['device'].upper()} with {bench['coarse_width']} × {bench['coarse_width']*3//4} thumbnails. "
         'Both retrievals share one pass over the bank, so the baseline time includes the contour retrieval. ') +
        'The refined time includes the baseline and the full refinement. Both include final measurement '
        'on 1040 × 780 MRI panels. Loading, input verification, source decoding and video encoding are excluded.', '',
        'The source and transform for each measurement are in [matches.json](matches.json). '
        'Attribution and reuse terms are in [CREDITS.md](CREDITS.md).', ''
    ]
    (output/'comparison.md').write_text('\n'.join(out))
    shutil.copyfile(ROOT/'CREDITS.md', output/'CREDITS.md')
