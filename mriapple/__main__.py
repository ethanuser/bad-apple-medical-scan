import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import PLANES, Pose, footprint, load_brains
from .inputs import ROOT, SOURCES, fetch, verify
from .media import frames
from .search import pipeline
from .report import measure_panels, report


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def ffmpeg_version():
    try:
        return subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    except FileNotFoundError:
        return 'not installed; frames decoded by OpenCV ' + cv2.__version__


def run(args):
    verify()
    originals, indices, rate = frames(args.start, args.seconds, args.fps)
    begin = time.perf_counter()
    brains = load_brains()
    load_time = time.perf_counter() - begin
    print(f'Loaded {len(brains)} masked volumes in {load_time:.1f} s', flush=True)
    # The GPU retrieval is the expensive step. Save it so an interrupted refinement
    # can resume with the same command instead of searching the bank again.
    args.output.mkdir(exist_ok=True, parents=True)
    key = dict(seed=args.seed, slice_step=args.slice_step, samples=args.samples,
               coarse_width=args.coarse_width, source_frames=indices,
               scans=hashlib.sha256(json.dumps(SOURCES['scans']).encode()).hexdigest())
    baseline, selected, stats = pipeline(brains, originals, args.seed, args.slice_step, args.samples,
                                         args.coarse_width, args.device,
                                         args.output/'retrieval.json.gz', key, args.workers)
    stats['baseline_finish_seconds'] += measure_panels(brains, baseline, originals)
    stats['shape_finish_seconds'] += measure_panels(brains, selected, originals)
    records = []
    for t, (a, b) in enumerate(zip(baseline, selected)):
        for match in (a, b):
            p = Pose(**match['pose'])
            match.update(source_key=brains[p.scan].key, plane=PLANES[p.axis],
                         corners_slice_xy_vox=footprint(brains[p.scan], p))
        records.append(dict(frame=t, time_s=args.start+t/args.fps, source_frame=indices[t],
                            coarse_ncc=a, silhouette=b))
    stats['load_seconds'] = load_time
    for name, matches in [('coarse_ncc', baseline), ('silhouette', selected)]:
        stats[name] = {metric: float(np.mean([m['metrics'][metric] for m in matches]))
                       for metric in matches[0]['metrics']}
    args.output.mkdir(exist_ok=True, parents=True)
    record = dict(schema=1, config={k:v for k,v in vars(args).items() if k not in ('cmd','output')},
                  dataset=SOURCES, scans=[b.meta for b in brains], source_fps=rate,
                  environment=dict(machine=platform.machine(), os=platform.platform(),
                      python=platform.python_version(), ffmpeg=ffmpeg_version(),
                      packages={k:importlib.metadata.version(k) for k in (
                          'numpy','opencv-python-headless','nibabel','pillow', 'torch')
                          if k != 'torch' or args.device != 'numpy'}),
                  benchmark=stats, frames=records)
    write_json(args.output/'matches.json', record)
    report(args.output, record)
    print(json.dumps(stats, indent=2), flush=True)
    if args.cmd == 'run':
        from .render import render
        render(args.output, record, brains, originals)


def main():
    cli = argparse.ArgumentParser(description='Match Bad Apple!! to native IXI MRI slices.')
    commands = cli.add_subparsers(dest='cmd', required=True)
    commands.add_parser('fetch', help='Download and hash-check the pinned media')
    for name in ['run','match','render']:
        parser = commands.add_parser(name)
        parser.add_argument('--output', type=Path, default=ROOT/'output')
        if name == 'render':
            parser.add_argument('--style', choices=['bw', 'gray'],
                                help='Display: black and white silhouette, or gray scan with texture')
            parser.add_argument('--to', type=Path, help='Write the restyled run and video to this directory instead')
            parser.add_argument('--layout', choices=['landscape', 'vertical'], default='landscape',
                                help='1440x1080 reading room, or 1080x1920 for phone feeds')
        if name != 'render':
            parser.add_argument('--start', type=float, default=8.)
            parser.add_argument('--seconds', type=float, default=7.)
            parser.add_argument('--fps', type=int, default=10)
            parser.add_argument('--seed', type=int, default=1513)
            parser.add_argument('--slice-step', type=int, default=4)
            parser.add_argument('--samples', type=int, default=6400, help='Proposed geometries per coarse slice')
            parser.add_argument('--coarse-width', type=int, default=32,
                                help='Retrieval thumbnail width (multiple of 4; 16 for the NumPy search)')
            parser.add_argument('--workers', type=int, default=None,
                                help='Refinement threads (default: Slurm CPUs or all cores)')
            parser.add_argument('--device', default='auto',
                                help='auto, mps, cuda, cpu (PyTorch) or numpy (CPU reference search)')
    args = cli.parse_args()
    cv2.setNumThreads(1)
    if args.cmd == 'fetch':
        fetch()
    elif args.cmd == 'render':
        from .render import replay, use_layout
        use_layout(args.layout)
        replay(args.output, args.style, args.to)
    else:
        if args.fps <= 0 or args.slice_step < 1 or args.samples < 4 or args.seed < 0:
            cli.error('fps and slice step must be positive, samples >= 4, seed >= 0')
        if args.coarse_width % 4 or args.coarse_width < 8:
            cli.error('coarse width must be a multiple of 4 and at least 8')
        if args.device != 'numpy':
            try:
                import torch  # noqa: F401
            except ImportError:
                if args.device != 'auto':
                    cli.error('PyTorch is required for --device ' + args.device)
                print('PyTorch not installed; using the NumPy search at 16 pixels', flush=True)
                args.device, args.coarse_width = 'numpy', 16
        run(args)


if __name__ == '__main__':
    main()
