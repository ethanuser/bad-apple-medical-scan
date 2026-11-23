"""Stream a randomized physical-pose bank through Accelerate BLAS, then refine locally."""
from dataclasses import replace
import gzip
import heapq
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time

import cv2
import numpy as np

from .geometry import Pose, inside, sample, silhouette, smooth_crop, values
from .metrics import (down, unit_rows, fit_levels, regional_features, score_image,
                      resolution_bonus, thin_parts, MASK_EDGE_LEVEL, MASK_EDGE_PENALTY)

COARSE_W = 16
REFINE_W = 48
METRIC_W = 96
KEEP = 48

# Selection-only preferences, never part of the displayed similarity. Grazing slices
# at the edge of a scan and crops centred off the tissue match Bad Apple's simple
# blobs well, but no longer read as anatomy. Slice offset is the distance from the
# middle of the scanned extent as a fraction of its half-length; crop offset is the
# distance from the slice's tissue centroid in units of its equivalent radius.
CENTRE_SLICE_WEIGHT = .035
CENTRE_CROP_WEIGHT = .03
# Heavily magnified crops (few native voxels across) turn into blurry smudges, which
# the centrality costs would otherwise push toward. Full cost at 32 voxels, none at 64.
BLUR_WEIGHT = .05


@lru_cache(maxsize=8192)
def slice_tissue(brain, axis, index):
    """Tissue centroid (mm) and equivalent-circle radius (mm) of one displayed slice."""
    ys, xs = np.nonzero(brain.plane(axis, index) > 0)
    if not len(xs):
        return None
    pitch = brain.pitch(axis)
    return float(xs.mean()*pitch[0]), float(ys.mean()*pitch[1]), float(np.sqrt(len(xs)*pitch[0]*pitch[1]/np.pi))


def centrality(brain, p):
    lo, hi = brain.limits[p.axis]
    s = abs(p.index - (lo + hi)/2) / max(1., (hi - lo)/2)
    tissue = slice_tissue(brain, p.axis, p.index)
    d = np.hypot(p.x_mm - tissue[0], p.y_mm - tissue[1]) / tissue[2] if tissue else 2.
    return float(s), float(d)


def prior(brain, p):
    """Penalty subtracted from the selection objective."""
    s, d = centrality(brain, p)
    voxels = p.width_mm / min(brain.pitch(p.axis))
    return (CENTRE_SLICE_WEIGHT * float(np.clip((s - .5)/.5, 0, 1))**1.5
            + CENTRE_CROP_WEIGHT * float(np.clip(d - .7, 0, 1))
            + BLUR_WEIGHT * float(np.clip((64 - voxels)/32, 0, 1)))


def propose(brain, axis, index, rng, count):
    plane = brain.plane(axis, index)
    dx, dy = brain.pitch(axis)
    yy, xx = np.nonzero(plane[::2, ::2] > 0)
    if not len(xx):
        return np.empty((0, 5), np.float32)
    picks = rng.integers(len(xx), size=count)
    width = np.exp(rng.uniform(*np.log(brain.widths), count))
    x = xx[picks]*2*dx + rng.uniform(-.40, .40, count)*width
    y = yy[picks]*2*dy + rng.uniform(-.40, .40, count)*width
    params = np.column_stack([x, y, width, rng.uniform(-180, 180, count), np.zeros(count)])
    return params[inside(params, (dx, dy), plane.shape)]


def update_pools(corr, pools, params, si, axis, index, serial, pitch):
    n = len(pools)
    direct, reflected = corr[:n], corr[n:]
    flip = np.abs(reflected) > np.abs(direct)
    signed = np.where(flip, reflected, direct)
    quality = np.abs(signed) + resolution_bonus(params[:, 2], pitch)[None, :]
    k = min(3, len(params))
    best = np.argpartition(quality, -k, axis=1)[:, -k:]
    for t, row in enumerate(best):
        pool = pools[t]
        for j in row:
            val = float(quality[t, j])
            if len(pool) == KEEP and val <= pool[0][0]:
                continue
            x, y, w, a, _ = map(float, params[j])
            pose = Pose(si, axis, index, x, y, w, a, bool(flip[t,j]), bool(signed[t,j] < 0))
            serial += 1
            item = (val, serial, pose)
            if len(pool) == KEEP:
                heapq.heapreplace(pool, item)
            else:
                heapq.heappush(pool, item)
    return serial


def dot_rows(a, b):
    # Some Accelerate kernels leave floating-point exception flags set even for
    # finite products of bounded inputs. Validate the returned values directly.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        product = a @ b
    if not np.isfinite(product).all():
        raise FloatingPointError('Non-finite descriptor similarity')
    return product


def gradient_query(images):
    """Discrete integration by parts: dot(image, GᵀG target) equals dot(G image, G target)."""
    imgs = np.asarray(images, dtype=np.float32)
    dx, dy = np.diff(imgs, axis=2), np.diff(imgs, axis=1)
    lap = np.zeros_like(imgs)
    lap[:, :, 1:] += dx
    lap[:, :, :-1] -= dx
    lap[:, 1:, :] += dy
    lap[:, :-1, :] -= dy
    norm = np.sqrt((dx*dx).sum(axis=(1, 2)) + (dy*dy).sum(axis=(1, 2)))
    return np.ascontiguousarray((lap.reshape(len(imgs), -1) / np.maximum(norm[:, None], 1e-6)).T)


def gradient_norm(images):
    dx, dy = np.diff(images, axis=2), np.diff(images, axis=1)
    return np.maximum(np.sqrt((dx*dx).sum(axis=(1, 2)) + (dy*dy).sum(axis=(1, 2))), 1e-3)


def retrieve(brains, targets, seed, step, samples):
    rng = np.random.default_rng(seed)
    images = np.array([down(f, COARSE_W) for f in targets])
    n = len(images)
    bank = np.concatenate([images, images[:, :, ::-1]]).astype(np.float32)
    query = np.ascontiguousarray(unit_rows(bank.reshape(2*n, -1)).T)
    highpass = np.array([v-cv2.GaussianBlur(v, (0, 0), 1.35) for v in bank])
    contour_query = np.ascontiguousarray(.62*query + .38*unit_rows(highpass.reshape(2*n, -1)).T)
    edge_query = gradient_query(bank)
    regional_query = np.ascontiguousarray(regional_features(bank / 255, query=True).T)
    blank = images.max(axis=(1, 2)) == images.min(axis=(1, 2))
    pools = [[] for _ in targets]
    shape_pools = [[] for _ in targets]
    views = slices = serial = 0
    shape_only = 0.
    begin = time.perf_counter()
    for si, brain in enumerate(brains):
        for axis in range(3):
            lo, hi = brain.limits[axis]
            for index in range(lo, hi+1, step):
                plane = brain.plane(axis, index)
                pitch = brain.pitch(axis)
                proposals = propose(brain, axis, index, rng, samples)
                if len(proposals) < 2:
                    continue
                slices += 1
                small = cv2.GaussianBlur(plane, (0, 0), 1.1)
                for start in range(0, len(proposals), 1024):
                    params = proposals[start:start+1024]
                    patches = sample(small, pitch, params, COARSE_W, COARSE_W*3//4)
                    views += len(params)
                    flat = patches.reshape(len(params), -1)
                    corr = dot_rows(unit_rows(flat), query).T
                    if blank.any():
                        brightness = flat.mean(axis=1) / 255
                        for t in np.flatnonzero(blank):
                            white = images[t, 0, 0] > 127
                            signed = np.where(brightness >= .5, brightness, brightness-1)
                            corr[[t, t+n]] = signed if white else -signed
                    serial = update_pools(corr, pools, params, si, axis, index, serial, pitch)
                    extra = time.perf_counter()
                    # Also retrieve internal tissue boundaries. Varying the proposed
                    # level independently of pose avoids preferring only the brain outline.
                    flat_f = flat.astype(np.float32)
                    level = rng.uniform(40, 215, size=(len(params), 1)).astype(np.float32)
                    best = corr * 0.
                    for center in (level, 255-level):
                        soft = np.clip((flat_f - center) / 55, -.5, .5)
                        c = dot_rows(unit_rows(soft), contour_query).T
                        grad = dot_rows(soft, edge_query).T / gradient_norm(soft.reshape(patches.shape))
                        tiled = regional_features(soft.reshape(patches.shape))
                        local = dot_rows(tiled, regional_query).T
                        c = .32*c + .23*grad + .45*local
                        best = np.where(np.abs(c) > np.abs(best), c, best)
                    serial = update_pools(best, shape_pools, params, si, axis, index, serial, pitch)
                    shape_only += time.perf_counter()-extra
        print(f'Coarse search {si+1}/{len(brains)} scans, {views:,} crops', flush=True)
    info = dict(coarse_seconds=time.perf_counter()-begin, shape_retrieval_extra_seconds=shape_only,
                sampled_slices=slices, valid_geometries=views, oriented_polarities=views*4)
    def ranked(groups):
        return [[p for _, _, p in sorted(group, reverse=True)] for group in groups]
    return ranked(pools), ranked(shape_pools), info


def neighbors(p, brain, rng, count, scale):
    # In-plane offsets, log zoom, angle and adjacent slice are jointly perturbed.
    delta = rng.normal(size=(count, 5)) * scale
    lo, hi = brain.limits[p.axis]
    rows = []
    for dx, dy, dw, da, dz in delta:
        rows.append(replace(p, x_mm=p.x_mm+dx*p.width_mm*.09,
                            y_mm=p.y_mm+dy*p.width_mm*.09,
                            width_mm=float(np.clip(p.width_mm*np.exp(dw*.13), *brain.width_clip)),
                            angle_deg=float((p.angle_deg+da*11+180)%360-180),
                            index=int(np.clip(p.index+round(dz*2.3), lo, hi))))
    valid = inside([values(q) for q in rows], brain.pitch(p.axis), brain.plane(p.axis, p.index).shape)
    return [q for q, ok in zip(rows, valid) if ok]


def patches_for(brains, poses, width):
    groups = {}
    for i, p in enumerate(poses):
        groups.setdefault((p.scan, p.axis, p.index), []).append(i)
    h = width*3//4
    output = np.empty((len(poses), h, width), np.uint8)
    for (si, axis, index), ids in groups.items():
        b = brains[si]
        for off in range(0, len(ids), 30000//h):
            batch = ids[off:off+30000//h]
            output[batch] = sample(b.plane(axis, index), b.pitch(axis),
                                   [values(poses[i]) for i in batch], width, h)
    return output


def choose_seeds(poses, quality, n):
    # Avoid spending all refinements in one near-identical patch.
    result = []
    for i in np.argsort(-quality, kind='stable'):
        p = poses[i]
        if any(p.scan == q.scan and p.axis == q.axis and abs(p.index-q.index) <= 2
               and p.mirror == q.mirror and p.invert == q.invert
               and abs(p.x_mm-q.x_mm)+abs(p.y_mm-q.y_mm) < .06*p.width_mm
               and abs(p.width_mm-q.width_mm) < .07*p.width_mm
               and abs((p.angle_deg-q.angle_deg+180)%360-180) < 7 for q in result):
            continue
        result.append(p)
        if len(result) == n:
            break
    return result


def refine(brains, poses, target, rng):
    low = down(target, REFINE_W)
    white, black = thin_parts(low >= 128, 3)
    for scale, n, k in [(1.15, 16, 10), (.60, 24, 7), (.28, 20, 7)]:
        candidates = poses + [q for p in poses for q in neighbors(p, brains[p.scan], rng, n, scale)]
        small = patches_for(brains, candidates, REFINE_W)
        inv = np.array([p.invert for p in candidates])
        levels, iou = fit_levels(small, low, inv)
        # Soft contrast gives sub-pixel feedback in addition to binary overlap.
        soft = np.clip((small.astype(np.float32) - levels[:, None, None])/62 + .5, 0, 1)
        soft = np.where(inv[:, None, None], 1-soft, soft)
        agreement = 1 - np.abs(soft - low.astype(np.float32)/255).mean(axis=(1, 2))
        g = gradient_query(low[None])
        soft = np.ascontiguousarray(soft, dtype=np.float32)
        edge = dot_rows(soft.reshape(len(soft), -1), g).ravel() / gradient_norm(soft)
        regional = regional_features(soft)
        q = regional_features(low[None].astype(np.float32)/255, query=True)
        local = (regional*q).sum(axis=1)
        pred = soft.reshape(len(soft), -1) >= .5
        parts = [pred[:, white.ravel()].mean(axis=1)] if white.any() else []
        parts += [(~pred[:, black.ravel()]).mean(axis=1)] if black.any() else []
        detail = np.mean(parts, axis=0) if parts else iou
        sharp = np.array([resolution_bonus(c.width_mm, brains[c.scan].pitch(c.axis)) - prior(brains[c.scan], c)
                          for c in candidates])
        quality = .35*iou + .08*agreement + .17*edge + .20*local + .20*detail + sharp
        poses = choose_seeds(candidates, quality, k)
    return poses


def evaluate(brains, p, target):
    """Fit the threshold on the smoothed crop, then score the black and white image
    exactly as displayed (thresholded at 2x and area-reduced to the metric size)."""
    raw = np.clip(np.rint(smooth_crop(brains, p, METRIC_W*2, METRIC_W*3//2)), 0, 255).astype(np.uint8)
    level, _ = fit_levels(raw[None], cv2.resize(target, (raw.shape[1], raw.shape[0]),
                           interpolation=cv2.INTER_LINEAR), [p.invert])
    best = None
    # The IoU-optimal level ignores edges and thin detail; try its neighbours on the full score.
    for offset in (-8, -4, 0, 4, 8):
        level_k = float(np.clip(level[0] + offset, .5, 254.5))
        image = silhouette(brains, p, level_k, METRIC_W, METRIC_W*3//4)
        metrics = score_image(image, target)
        if best is None or metrics['score'] > best['metrics']['score']:
            best = dict(pose=p.record(), level=level_k, metrics=metrics)
    best['objective'] = (best['metrics']['score'] + float(resolution_bonus(p.width_mm, brains[p.scan].pitch(p.axis)))
                         - MASK_EDGE_PENALTY*(best['level'] < MASK_EDGE_LEVEL) - prior(brains[p.scan], p))
    return best


def follow(p, previous, current, brain):
    """Translate and uniformly zoom the same acquired patch using silhouette moments."""
    a, b = previous >= 128, current >= 128
    if a.mean() > .5:
        a, b = ~a, ~b
    if not a.any() or not b.any() or a.all() or b.all():
        return p
    ratio = np.sqrt(b.sum()/a.sum())
    if not .64 <= ratio <= 1.56:
        return p
    h, w = a.shape
    def centroid(mask):
        y, x = np.nonzero(mask)
        return np.array([(x.mean()+.5-w/2)/w, (y.mean()+.5-h/2)/w])
    width = float(np.clip(p.width_mm/ratio, *brain.width_clip))
    u, v = p.width_mm*centroid(a) - width*centroid(b)
    if p.mirror:
        u = -u
    ang = np.deg2rad(p.angle_deg)
    q = replace(p, width_mm=width, x_mm=p.x_mm + np.cos(ang)*u - np.sin(ang)*v,
                y_mm=p.y_mm + np.sin(ang)*u + np.cos(ang)*v)
    valid = inside([values(q)], brain.pitch(p.axis), brain.plane(p.axis, p.index).shape)
    return q if valid[0] else p


CUT_COST = .038


def switch_cost(a, b):
    if (a.scan, a.axis, a.mirror, a.invert) != (b.scan, b.axis, b.mirror, b.invert):
        return CUT_COST
    distance = np.hypot(a.x_mm-b.x_mm, a.y_mm-b.y_mm)/a.width_mm
    turn = abs((a.angle_deg-b.angle_deg+180)%360-180)/90
    return min(CUT_COST, .018*(distance + turn + abs(np.log(b.width_mm/a.width_mm)))
               + .001*abs(a.index-b.index))


def sequence(choices):
    """Select a maximum-score path, charging a small cost for discontinuous cuts."""
    prior = np.zeros(1)
    links, old = [], []
    for i, row in enumerate(choices):
        poses = [Pose(**c['pose']) for c in row]
        if i:
            costs = np.array([[switch_cost(a,b) for b in poses] for a in old])
            path = prior[:,None] - costs
            parent = path.argmax(axis=0)
            prior = path[parent, np.arange(len(row))]
            links.append(parent)
        prior = prior + np.array([c.get('objective', c['metrics']['score']) for c in row])
        old = poses
    index = int(prior.argmax())
    result = []
    for i in range(len(choices)-1, -1, -1):
        result.append(choices[i][index])
        if i:
            index = int(links[i-1][index])
    return result[::-1]


def load_retrieval(path, key):
    """Reuse a saved retrieval when every input that shapes it is unchanged."""
    if path is None or not path.exists():
        return None
    with gzip.open(path, 'rt') as f:
        saved = json.load(f)
    if saved['key'] != key:
        print(f'Ignoring {path.name}: it was made with different settings or inputs', flush=True)
        return None
    print(f'Reusing retrieval from {path.name}', flush=True)
    poses = [[[Pose(**p) for p in pool] for pool in group] for group in saved['pools']]
    return poses[0], poses[1], saved['stats']


def save_retrieval(path, key, pools, shape_pools, stats):
    temp = path.with_suffix('.tmp')
    with gzip.open(temp, 'wt') as f:
        json.dump(dict(key=key, stats=stats, pools=[[[p.record() for p in pool] for pool in group]
                                                   for group in (pools, shape_pools)]), f)
    temp.replace(path)


_BLOCK_JOB = {}


def _run_block(bounds):
    return _BLOCK_JOB['fn'](*bounds)


def run_blocks(fn, bounds, workers):
    """Forked processes share the loaded volumes copy-on-write and avoid Python's
    interpreter lock, which caps threads at about 1.6x. Forking is used on Linux, and
    on macOS only when this process has not used the Apple GPU, which is not
    fork-safe. Otherwise threads."""
    import multiprocessing
    import sys
    gpu = sys.modules.get('mriapple.gpu')
    if workers > 1 and (sys.platform == 'linux' or not getattr(gpu, 'MPS_USED', False)):
        _BLOCK_JOB['fn'] = fn
        try:
            with multiprocessing.get_context('fork').Pool(workers) as pool:
                return pool.map(_run_block, bounds, chunksize=1), 'processes'
        finally:
            _BLOCK_JOB.clear()
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda b: fn(*b), bounds)), 'threads'


def default_workers():
    """CPUs this process may use: Slurm's allocation, a notebook server's CPU_LIMIT, or the CPU
    affinity mask, which containers set below the host's core count."""
    for name in ('SLURM_CPUS_PER_TASK', 'CPU_LIMIT'):
        if os.environ.get(name):
            return max(1, int(float(os.environ[name])))
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def pipeline(brains, targets, seed, step, samples, coarse_width=COARSE_W, device='numpy',
             cache=None, cache_key=None, workers=None):
    saved = load_retrieval(cache, cache_key)
    if saved:
        pools, shape_pools, stats = saved
    elif device == 'numpy':
        if coarse_width != COARSE_W:
            raise ValueError(f'The NumPy search uses {COARSE_W}-pixel thumbnails; install torch for others')
        pools, shape_pools, stats = retrieve(brains, targets, seed, step, samples)
    else:
        from .gpu import retrieve as retrieve_gpu
        pools, shape_pools, stats = retrieve_gpu(brains, targets, seed, step, samples, coarse_width, device)
    if cache is not None and not saved:
        save_retrieval(cache, cache_key, pools, shape_pools, stats)
    baseline = []
    begin = time.perf_counter()
    metric_targets = [down(f, METRIC_W) for f in targets]
    for pool, target in zip(pools, metric_targets):
        baseline.append(evaluate(brains, pool[0], target))
    stats['baseline_finish_seconds'] = time.perf_counter()-begin
    begin = time.perf_counter()
    def refine_block(first, last):
        # Frames in a block are refined in order, each seeded with its predecessor's
        # pick. Each frame's random generator depends only on its index.
        picked, block_rows = None, []
        for t in range(first, last):
            target = metric_targets[t]
            seeds = list(dict.fromkeys(pools[t] + shape_pools[t]))
            if picked:
                prev = Pose(**picked['pose'])
                seeds += [prev, follow(prev, metric_targets[t-1], target, brains[prev.scan])]
            finalists = refine(brains, seeds, target, np.random.default_rng(seed+10000+t))
            choices = [baseline[t]] + [evaluate(brains, p, target) for p in finalists]
            if picked:
                choices.append(evaluate(brains, Pose(**picked['pose']), target))
            picked = max(choices, key=lambda c: c['objective'])
            block_rows.append(choices)
        print(f'Refined frames {first+1}-{last} of {len(targets)}', flush=True)
        return block_rows

    # Contiguous blocks are refined in parallel. Only the first frame of each block
    # lacks its predecessor's pose as a seed; the reverse propagation and the joint
    # sequence below still link the blocks.
    workers = max(1, min(workers or default_workers(), len(targets)//8 or 1))
    edges = np.linspace(0, len(targets), workers+1).round().astype(int)
    bounds = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
    blocks, mode = run_blocks(refine_block, bounds, workers)
    rows = [r for block in blocks for r in block]
    stats['refine_workers'], stats['refine_parallelism'] = workers, mode
    # Propagate good poses in reverse as well, then choose the sequence jointly.
    for t in range(len(rows)-2, -1, -1):
        following = max(rows[t+1], key=lambda c: c['objective'])
        p = Pose(**following['pose'])
        propagated = follow(p, metric_targets[t+1], metric_targets[t], brains[p.scan])
        rows[t].append(evaluate(brains, propagated, metric_targets[t]))
    selected = sequence(rows)
    stats['shape_finish_seconds'] = time.perf_counter()-begin
    return baseline, selected, stats
