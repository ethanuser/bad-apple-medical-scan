"""Pinned local inputs, fetched individually and checked before use."""
import hashlib
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
SOURCES = json.loads((ROOT / 'sources.json').read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


CT_ROLES = ('image', 'clavicula_left', 'clavicula_right', 'skull')


def inputs():
    yield ASSETS / 'badapple.mp4', SOURCES['video']
    for scan in SOURCES['scans']:
        for role in ('image', 'mask'):
            yield ASSETS / scan['key'] / f'{role}.nii.gz', scan[role]
    for scan in SOURCES.get('ct_scans', []):
        for role in CT_ROLES:
            yield ASSETS / scan['key'] / f'{role}.nii.gz', dict(scan[role], url=SOURCES['ct_archive'])


def fetch_one(item):
    path, spec = item
    if path.exists() and digest(path) == spec['sha256']:
        return
    path.parent.mkdir(exist_ok=True, parents=True)
    if 'member' in spec:
        # One member of the pinned TotalSegmentator archive, fetched by byte range.
        from .remotezip import fetch_member
        data = fetch_member(spec['url'], spec)
        if hashlib.sha256(data).hexdigest() != spec['sha256']:
            raise ValueError(f'Checksum mismatch for {path.name} in {path.parent.name}')
        temp = path.with_suffix('.part')
        temp.write_bytes(data)
        temp.replace(path)
        print(f'Verified {path.relative_to(ASSETS)}', flush=True)
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        partial = Path(temp.name)
    try:
        subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '180',
                        '-o', str(partial), spec['url']], check=True)
        if digest(partial) != spec['sha256']:
            raise ValueError(f'Checksum mismatch for {path.name} in {path.parent.name}')
        partial.replace(path)
        print(f'Verified {path.relative_to(ASSETS)}', flush=True)
    finally:
        partial.unlink(missing_ok=True)


def fetch():
    items = list(inputs())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch_one, [i for i in items if 'member' not in i[1]]))
    for item in items:
        if 'member' in item[1]:
            fetch_one(item)


def verify():
    for path, spec in inputs():
        if not path.is_file() or digest(path) != spec['sha256']:
            raise ValueError(f'Missing or altered input {path}. Run python -m mriapple fetch.')
