"""Choose whole-body CT scans from the TotalSegmentator v2.0.1 archive and write them
to ct-candidates.json. Reads only the archive's directory, meta.csv and the first
bytes of each candidate (its NIfTI header) through HTTP range requests.

    .venv/bin/python scripts/select_ct.py [count]
"""
import csv
import io
import zlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mriapple.remotezip import member_prefix, open_archive  # noqa: E402

URL = 'https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1'
TYPES = {'ct angiography neck-thx-abd-pelvis-leg', 'ct polytrauma', 'ct whole body',
         'ct neck-thorax-abdomen-pelvis'}


def header(info):
    gz = member_prefix(URL, info, 1 << 16)
    raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(gz, 352)
    h = nib.Nifti1Header.from_fileobj(io.BytesIO(raw))
    return h.get_data_shape()[:3], [float(v) for v in h.get_zooms()[:3]]


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    archive = open_archive(URL)
    with archive.open('meta.csv') as f:
        meta = {r['image_id']: r for r in csv.DictReader(io.TextIOWrapper(f, 'utf-8-sig'), delimiter=';')}
    infos = {i.filename: i for i in archive.infolist()}
    cases = sorted(k for k, r in meta.items() if r['study_type'].strip() in TYPES)
    def probe(case):
        shape, zooms = header(infos[f'{case}/ct.nii.gz'])
        sizes = {name: infos[f'{case}/segmentations/{name}.nii.gz'].compress_size
                 for name in ('clavicula_left', 'femur_left', 'humerus_left', 'skull', 'brain')}
        extent = [s*z for s, z in zip(shape, zooms)]
        return dict(case=case, study_type=meta[case]['study_type'].strip(), shape=list(shape),
                    zooms=zooms, extent_mm=extent, seg_bytes=sizes)
    # One request at a time: Zenodo rate-limits bursts.
    rows = []
    for i, case in enumerate(cases):
        rows.append(probe(case))
        if (i+1) % 50 == 0:
            print(f'{i+1}/{len(cases)} headers', flush=True)
    # Empty masks compress to a few kilobytes; present structures are larger.
    empty = min(min(r['seg_bytes'].values()) for r in rows) * 3
    for r in rows:
        r['has'] = {k: v > empty for k, v in r['seg_bytes'].items()}
    whole = [r for r in rows if r['has']['clavicula_left'] and r['has']['femur_left']]
    whole.sort(key=lambda r: -max(r['extent_mm']))
    out = Path(__file__).resolve().parents[1] / 'ct-candidates.json'
    out.write_text(json.dumps(dict(url=URL, probed=len(rows), whole_body=len(whole), rows=whole), indent=1))
    lengths = np.array([max(r['extent_mm']) for r in whole])
    print(f'{len(rows)} probed, {len(whole)} span clavicles to femurs; longest {lengths[:count].min():.0f}-{lengths.max():.0f} mm '
          f'for the top {count}; {sum(r["has"]["humerus_left"] for r in whole[:count])} include arms')


def pin(count):
    """Download the top candidates' CT and the masks used for head removal, and record
    offsets and SHA-256 digests in sources.json under ct_scans."""
    import hashlib
    from mriapple.inputs import ASSETS, CT_ROLES, ROOT
    from mriapple.remotezip import fetch_member
    candidates = json.loads((ROOT/'ct-candidates.json').read_text())['rows'][:count]
    archive = open_archive(URL)
    infos = {i.filename: i for i in archive.infolist()}
    members = dict(image='ct.nii.gz', clavicula_left='segmentations/clavicula_left.nii.gz',
                   clavicula_right='segmentations/clavicula_right.nii.gz', skull='segmentations/skull.nii.gz')
    scans = []
    for n, row in enumerate(candidates):
        scan = dict(key=f"ts_{row['case']}", study_type=row['study_type'])
        for role in CT_ROLES:
            info = infos[f"{row['case']}/{members[role]}"]
            spec = dict(member=info.filename, offset=info.header_offset,
                        compressed_size=info.compress_size, compress_type=info.compress_type)
            path = ASSETS/scan['key']/f'{role}.nii.gz'
            path.parent.mkdir(parents=True, exist_ok=True)
            data = fetch_member(URL, spec)
            path.write_bytes(data)
            spec['sha256'] = hashlib.sha256(data).hexdigest()
            scan[role] = spec
        scans.append(scan)
        print(f'Pinned {n+1}/{count} {scan["key"]}', flush=True)
    sources = json.loads((ROOT/'sources.json').read_text())
    sources['ct_archive'] = URL
    sources['ct_edition'] = 'TotalSegmentator dataset v2.0.1 (Wasserthal et al.), CC BY 4.0'
    sources['ct_scans'] = scans
    (ROOT/'sources.json').write_text(json.dumps(sources, indent=2) + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'pin':
        pin(int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    else:
        main()
