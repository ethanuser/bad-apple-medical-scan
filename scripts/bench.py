"""Measure search and refinement speed for the full song, to choose the density and
time limit of the real run. Search throughput (crops per second) depends on the
thumbnail width and on the number of target frames, not on slice step or sample
count, which only change how many crops there are. So each width is timed against
every target frame on a subset of volumes, and the full run is extrapolated from the
126.4 million crops of all 250 volumes at slice step 4 and 6,400 samples.

    .venv/bin/python scripts/bench.py                 # defaults: 30 fps, every 10th volume
    .venv/bin/python scripts/bench.py --seconds 5 --every 100 --widths 16 --refine-frames 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mriapple.geometry import load_brains  # noqa: E402
from mriapple.gpu import retrieve  # noqa: E402
from mriapple.media import frames  # noqa: E402
from mriapple.search import pipeline, default_workers  # noqa: E402

FULL_CROPS = 126_370_150      # all 250 volumes, slice step 4, 6,400 samples (measured)
DENSITIES = [(4, 6400), (2, 6400), (2, 12800), (1, 12800)]


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--fps', type=int, default=30)
    cli.add_argument('--seconds', type=float, default=232)
    cli.add_argument('--every', type=int, default=10, help='Use every n-th volume')
    cli.add_argument('--widths', type=int, nargs='+', default=[32, 48, 64])
    cli.add_argument('--refine-frames', type=int, default=200)
    cli.add_argument('--device', default='auto')
    cli.add_argument('--output', type=Path, default=Path('output/bench.json'))
    args = cli.parse_args()
    cv2.setNumThreads(1)
    t = time.perf_counter()
    brains = load_brains()
    load = time.perf_counter() - t
    subset = brains[::args.every]
    targets, _, _ = frames(0., args.seconds, args.fps)
    print(f'{len(brains)} volumes loaded in {load:.0f} s; timing {len(subset)} of them '
          f'against {len(targets)} frames', flush=True)
    results = dict(fps=args.fps, frames=len(targets), volumes=len(subset), load_seconds=load, widths={})
    pools = None
    for width in args.widths:
        t = time.perf_counter()
        p, s, info = retrieve(subset, targets, 1513, 4, 6400, width, args.device)
        seconds = time.perf_counter() - t
        rate = info['valid_geometries'] / seconds
        results['widths'][width] = dict(crops=info['valid_geometries'], seconds=seconds, crops_per_second=rate,
                                        device=info['device'], batch_chunks=info['batch_chunks'])
        print(f'width {width}: {rate:,.0f} crops/s on {info["device"]} '
              f'({info["valid_geometries"]:,} crops in {seconds:.0f} s)', flush=True)
        if pools is None:
            pools = (p, s)
    # Refinement: CPU-only, timed on the first frames with every core.
    n = min(args.refine_frames, len(targets))
    workers = default_workers()
    t = time.perf_counter()
    import mriapple.search as search
    saved = search.load_retrieval
    search.load_retrieval = lambda path, key: (pools[0][:n], pools[1][:n], {})
    try:
        pipeline(subset, targets[:n], 1513, 4, 6400, args.widths[0], args.device, Path('unused'), {}, workers)
    finally:
        search.load_retrieval = saved
    per_frame = (time.perf_counter() - t) / n
    results['refine'] = dict(frames=n, workers=workers, seconds_per_frame=per_frame)
    print(f'refinement: {per_frame*1000:.0f} ms per frame with {workers} workers', flush=True)
    print('\nEstimated full-song matching time (search + refinement), all 250 volumes:', flush=True)
    rows = []
    for width, r in results['widths'].items():
        for step, samples in DENSITIES:
            crops = FULL_CROPS * (4/step) * (samples/6400)
            minutes = (crops / r['crops_per_second'] + per_frame*len(targets) + load) / 60
            rows.append(dict(width=width, slice_step=step, samples=samples, crops=crops, minutes=minutes))
            print(f'  --coarse-width {width} --slice-step {step} --samples {samples}: '
                  f'{crops/1e6:,.0f}M crops, about {minutes:,.0f} min', flush=True)
    results['estimates'] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))
    print(f'\nSaved {args.output}', flush=True)


if __name__ == '__main__':
    main()
