"""PyTorch coarse retrieval. Runs on Apple-silicon MPS, CUDA (an H100) or CPU.

The random proposals and contrast levels are drawn on the CPU in exactly the order
used by search.retrieve, so both implementations search the same bank. Each group of
up to 1,024 proposals from one slice keeps its best three poses per target, which are
merged into a running top 48 per target, as in the NumPy version.
"""
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .geometry import ASPECT, Pose
from .metrics import RES_WEIGHT, down

CHUNK = 1024
KEEP = 48


def pick_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def unit_rows(a):
    a = a - a.mean(dim=1, keepdim=True)
    return a / a.norm(dim=1, keepdim=True).clamp_min(1e-6)


def regional(images, query=False):
    n, h, w = images.shape
    tiles = images.reshape(n, 3, h//3, 4, w//4).permute(0, 1, 3, 2, 4).reshape(n, 12, -1)
    tiles = tiles - tiles.mean(dim=2, keepdim=True)
    norms = tiles.norm(dim=2, keepdim=True)
    features = tiles / norms.clamp_min(.025*np.sqrt(tiles.shape[2]))
    if query:
        weights = norms.sqrt()
        features = features * weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return features.reshape(n, -1)


def regional_raster(images):
    """Query regional features in raster pixel order (each 4 × 3 tile zero-mean)."""
    n, h, w = images.shape
    f = regional(images, query=True).reshape(n, 3, 4, h//3, w//4).permute(0, 1, 3, 2, 4)
    return f.reshape(n, h*w)


def tile_index(h, w, device):
    rows = torch.arange(h, device=device)[:, None] // (h//3)
    cols = torch.arange(w, device=device)[None, :] // (w//4)
    return (rows*4 + cols).reshape(-1)


class Scorer:
    """Correlation, gradient agreement and regional NCC against zero-mean queries.

    Because every query column has zero mean (globally, and per tile for the regional
    query), centring the crops is unnecessary: only their norms are needed, which
    come from sums and sums of squares instead of materialized centred copies."""
    def __init__(self, h, w, device):
        self.h, self.w = h, w
        self.tile = tile_index(h, w, device)
        self.tiles = F.one_hot(self.tile, 12).float()           # (D, 12)
        self.per_tile = h*w/12

    def norms(self, x, tol):
        # Flat crops have a zero centred norm, while their uncentred product with a
        # zero-mean query is only rounding error. The floor (tol per pixel, far below
        # any visible contrast) keeps that ratio near zero, as centring first would.
        d = x.shape[1]
        s1, s2 = x.sum(1), (x*x).sum(1)
        return (s2 - s1*s1/d).clamp_min(0).sqrt().clamp_min(tol*np.sqrt(d))

    def regional(self, x, rq):
        t1, t2 = x @ self.tiles, (x*x) @ self.tiles
        norm = (t2 - t1*t1/self.per_tile).clamp_min(0).sqrt()
        inv = 1/norm.clamp_min(.025*np.sqrt(self.per_tile))
        return (x * inv[:, self.tile]) @ rq

    def gradient_norm(self, x):
        img = x.reshape(-1, self.h, self.w)
        dx, dy = img.diff(dim=2), img.diff(dim=1)
        return ((dx*dx).sum(dim=(1, 2)) + (dy*dy).sum(dim=(1, 2))).sqrt().clamp_min(1e-3)


def gradient_query(imgs):
    dx, dy = imgs.diff(dim=2), imgs.diff(dim=1)
    lap = torch.zeros_like(imgs)
    lap[:, :, 1:] += dx
    lap[:, :, :-1] -= dx
    lap[:, 1:, :] += dy
    lap[:, :-1, :] -= dy
    norm = ((dx*dx).sum(dim=(1, 2)) + (dy*dy).sum(dim=(1, 2))).sqrt()
    return (lap.reshape(len(imgs), -1) / norm.clamp_min(1e-6)[:, None]).T.contiguous()


def gradient_norm(imgs):
    dx, dy = imgs.diff(dim=2), imgs.diff(dim=1)
    return ((dx*dx).sum(dim=(1, 2)) + (dy*dy).sum(dim=(1, 2))).sqrt().clamp_min(1e-3)


def proposals_for(brain, axis, index, rng, count):
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
    # Same footprint test as geometry.inside.
    a = np.deg2rad(params[:, 3])[:, None]
    gx = np.array([-.5, .5, .5, -.5])[None]
    gy = np.array([-.5, -.5, .5, .5])[None] / ASPECT
    c, s = params[:, 2:3]*np.cos(a), params[:, 2:3]*np.sin(a)
    px = (params[:, :1] + c*gx - s*gy) / dx
    py = (params[:, 1:2] + s*gx + c*gy) / dy
    ok = ((px >= 0) & (px <= plane.shape[1]-1) & (py >= 0) & (py <= plane.shape[0]-1)).all(axis=1)
    return params[ok]


class Pools:
    """Running best poses per target, kept on the device."""
    def __init__(self, targets, device):
        self.value = torch.full((targets, KEEP), -np.inf, device=device)
        self.payload = torch.zeros((targets, KEEP, 10), device=device)

    def update(self, quality, signed, flip, payload):
        # quality, signed, flip: (G, CHUNK, T). payload: (G, CHUNK, 8).
        g, c, t = quality.shape
        top, at = quality.topk(3, dim=1)                                  # (G, 3, T)
        rows = payload[:, :, None, :].expand(g, c, t, 8).gather(1, at[..., None].expand(g, 3, t, 8))
        extra = torch.stack([flip.gather(1, at).float(), (signed.gather(1, at) < 0).float()], dim=-1)
        cand = torch.cat([rows, extra], dim=-1).permute(2, 0, 1, 3).reshape(t, g*3, 10)
        value = torch.cat([self.value, top.permute(2, 0, 1).reshape(t, g*3)], dim=1)
        best, idx = value.topk(KEEP, dim=1)
        self.payload = torch.cat([self.payload, cand], dim=1).gather(1, idx[..., None].expand(t, KEEP, 10))
        self.value = best

    def ranked(self):
        value, payload = self.value.cpu().numpy(), self.payload.cpu().numpy()
        out = []
        for v, rows in zip(value, payload):
            poses = []
            for val, (x, y, w, a, _, scan, axis, index, flip, inv) in zip(v, rows):
                if np.isfinite(val):
                    poses.append(Pose(int(scan), int(axis), int(index), float(x), float(y),
                                      float(w), float(a), bool(flip), bool(inv)))
            out.append(poses)
        return out


def batch_size(dev, targets, pixels):
    """Chunks of 1,024 crops per batch. Scoring keeps about a dozen (crops × 2 targets)
    float tensors alive, so the batch shrinks as the number of target frames grows."""
    per_chunk = CHUNK*4*(12*2*targets + 8*pixels)
    if dev.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(dev)
        budget = free*.5
    elif dev.type == 'mps':
        budget = min(2.5e9, torch.mps.recommended_max_memory()*.15)
    else:
        budget = 2e9
    return int(np.clip(budget // per_chunk, 1, 256))


MPS_USED = False   # forking after the Apple GPU has been used is unsafe


def retrieve(brains, targets, seed, step, samples, width=16, device='auto', batch_chunks=None):
    global MPS_USED
    dev = pick_device(device)
    MPS_USED = MPS_USED or dev.type == 'mps'
    rng = np.random.default_rng(seed)
    h = width*3//4
    images = np.array([down(f, width) for f in targets])
    n = len(images)
    bank = np.concatenate([images, images[:, :, ::-1]]).astype(np.float32)
    bank_t = torch.from_numpy(bank).to(dev)
    query = unit_rows(bank_t.reshape(2*n, -1)).T.contiguous()
    sigma = 1.35*width/16
    highpass = torch.from_numpy(np.array([v-cv2.GaussianBlur(v, (0, 0), sigma) for v in bank])).to(dev)
    contour_query = (.62*query + .38*unit_rows(highpass.reshape(2*n, -1)).T).contiguous()
    edge_query = gradient_query(bank_t)
    regional_query = regional_raster(bank_t / 255).T.contiguous()
    scorer = Scorer(h, width, dev)
    shape_query = torch.cat([contour_query, edge_query], dim=1)
    blank = torch.from_numpy(images.max(axis=(1, 2)) == images.min(axis=(1, 2))).to(dev)
    blank_white = torch.from_numpy(images[:, 0, 0] > 127).to(dev)
    gx = ((torch.arange(width, device=dev, dtype=torch.float32)+.5)/width-.5)[None, None, :]
    gy = ((torch.arange(h, device=dev, dtype=torch.float32)+.5)/h-.5)[None, :, None]/ASPECT
    pools, shape_pools = Pools(n, dev), Pools(n, dev)
    batch_chunks = batch_chunks or batch_size(dev, n, h*width)
    print(f'Retrieval on {dev} with {batch_chunks} × {CHUNK} crops per batch', flush=True)
    blur = max(.6, 1.1*16/width)
    views = slices = 0
    begin = time.perf_counter()

    def run(planes, pitch, chunks):
        # planes: (S, H, W) float on device. chunks: list of (slice slot, params, levels, meta).
        g = len(chunks)
        params = np.zeros((g, CHUNK, 5), np.float32)
        levels = np.zeros((g, CHUNK), np.float32)
        meta = np.zeros((g, CHUNK, 3), np.float32)
        valid = np.zeros((g, CHUNK), bool)
        slot = np.array([c[0] for c in chunks])
        for i, (_, p, lv, m) in enumerate(chunks):
            params[i, :len(p)], levels[i, :len(p)], meta[i, :len(p)], valid[i, :len(p)] = p, lv[:, 0], m, True
            params[i, len(p):] = p[0]  # padding repeats a valid pose and is masked out below
        P = torch.from_numpy(params).to(dev)
        x, y, wd, ang, mir = (P[..., k][..., None, None] for k in range(5))
        a = torch.deg2rad(ang)
        c, s = wd*torch.cos(a), wd*torch.sin(a)
        u = gx*(1-2*mir)
        px = (x + c*u - s*gy) / pitch[0]
        py = (y + s*u + c*gy) / pitch[1]
        H, W = planes.shape[1:]
        grid = torch.stack([(2*px+1)/W-1, (2*py+1)/H-1], dim=-1).reshape(g, CHUNK*h, width, 2)
        src = planes[torch.from_numpy(slot).to(dev)][:, None]
        patches = F.grid_sample(src, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        flat = patches.reshape(g*CHUNK, h*width).round()
        mask = torch.from_numpy(valid).to(dev).reshape(-1)
        corr = (flat @ query) / scorer.norms(flat, .01)[:, None]               # (M, 2n)
        if blank.any():
            bright = flat.mean(dim=1) / 255
            signed = torch.where(bright >= .5, bright, bright-1)[:, None]
            fill = torch.where(blank_white[None], signed, -signed)
            corr = torch.cat([torch.where(blank[None], fill, corr[:, :n]),
                              torch.where(blank[None], fill, corr[:, n:])], dim=1)
        lv = torch.from_numpy(levels).to(dev).reshape(-1, 1)
        best = torch.zeros_like(corr)
        for center in (lv, 255-lv):
            soft = ((flat - center)/55).clamp(-.5, .5)
            both = soft @ shape_query
            cc = both[:, :2*n] / scorer.norms(soft, 1e-3)[:, None]
            grad = both[:, 2*n:] / scorer.gradient_norm(soft)[:, None]
            local = scorer.regional(soft, regional_query)
            cc = .32*cc + .23*grad + .45*local
            best = torch.where(cc.abs() > best.abs(), cc, best)
        widths = P[..., 2].reshape(-1)
        voxels = widths / float(np.sqrt(pitch[0]*pitch[1]))
        bonus = RES_WEIGHT*((voxels/32).log2()/2).clamp(0, 1)
        payload = torch.cat([P, torch.from_numpy(meta).to(dev)], dim=-1)  # (g, CHUNK, 8)
        for scores, pool in ((corr, pools), (best, shape_pools)):
            direct, reflected = scores[:, :n], scores[:, n:]
            flip = reflected.abs() > direct.abs()
            signed = torch.where(flip, reflected, direct)
            quality = torch.where(mask[:, None], signed.abs() + bonus[:, None], -torch.inf)
            pool.update(quality.reshape(g, CHUNK, n), signed.reshape(g, CHUNK, n),
                        flip.reshape(g, CHUNK, n), payload)

    for si, brain in enumerate(brains):
        for axis in range(3):
            lo, hi = brain.limits[axis]
            pitch = brain.pitch(axis)
            planes, chunks = [], []
            for index in range(lo, hi+1, step):
                proposals = proposals_for(brain, axis, index, rng, samples)
                if len(proposals) < 2:
                    continue
                slices += 1
                views += len(proposals)
                k = len(planes)
                planes.append(cv2.GaussianBlur(brain.plane(axis, index), (0, 0), blur))
                for start in range(0, len(proposals), CHUNK):
                    p = proposals[start:start+CHUNK]
                    lv = rng.uniform(40, 215, size=(len(p), 1)).astype(np.float32)
                    m = np.tile(np.array([[si, axis, index]], np.float32), (len(p), 1))
                    chunks.append((k, p, lv, m))
            if not planes:
                continue
            stack = torch.from_numpy(np.stack(planes)).to(dev).float()
            for start in range(0, len(chunks), batch_chunks):
                run(stack, pitch, chunks[start:start+batch_chunks])
        print(f'Coarse search {si+1}/{len(brains)} scans, {views:,} crops', flush=True)
    if dev.type == 'mps':
        torch.mps.synchronize()
    elapsed = time.perf_counter()-begin
    info = dict(coarse_seconds=elapsed, shape_retrieval_extra_seconds=0., sampled_slices=slices,
                valid_geometries=views, oriented_polarities=views*4, device=str(dev), coarse_width=width,
                batch_chunks=batch_chunks)
    return pools.ranked(), shape_pools.ranked(), info
