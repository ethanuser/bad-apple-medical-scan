"""Integration check against actual NIfTI voxels and the encoded movie, after a run."""
import json
import unittest

import cv2
import nibabel as nib
import numpy as np

from mriapple.geometry import prepare_ct
from mriapple.inputs import ASSETS, ROOT
from mriapple.render import HERO


@unittest.skipUnless((ROOT/'output/bad-apple-mri.mp4').exists(), 'requires a completed default render')
class RenderedProvenanceTests(unittest.TestCase):
    def test_decoded_hero_samples_agree_with_native_masked_voxels(self):
        results = json.loads((ROOT/'output/matches.json').read_text())
        all_scans = results['scans']
        images = {}
        cap = cv2.VideoCapture(str(ROOT/'output/bad-apple-mri.mp4'))
        left, top, width, height = HERO
        # Sample inside the margins that hold the viewer's corner annotations, edge
        # letters and scale (at most about 90 px deep and 80 px from the right edge).
        cols, rows = np.meshgrid(np.arange(55, width-80, 41), np.arange(110, height-110, 39))
        errors = []
        try:
            for row in results['frames']:
                ok, video_frame = cap.read()
                self.assertTrue(ok)
                match = row['silhouette']
                key, pose = match['source_key'], match['pose']
                meta = all_scans[pose['scan']]
                if key not in images:
                    folder = ASSETS/key
                    if key.startswith('ts_'):
                        # CT: the prepared volume (head cut, body mask, HU window), whose
                        # preparation is checked separately in test_ct_privacy.
                        data, _ = prepare_ct(folder, key)
                        images[key] = (data, data > 0)
                    else:
                        def load(name):
                            return np.asanyarray(nib.as_closest_canonical(nib.squeeze_image(nib.load(folder/name))).dataobj)
                        images[key] = (load('image.nii.gz'), load('mask.nii.gz'))
                raw, mask = images[key]
                self.assertEqual(key, meta['key'])
                axis = pose['axis']
                a, b = [i for i in range(3) if i != axis]
                corners = np.asarray(match['corners_slice_xy_vox'])
                self.assertTrue(np.all(corners >= 0))
                self.assertTrue(np.all(corners <= [raw.shape[a]-1, raw.shape[b]-1]))
                # Independent reconstruction using only the four logged footprint corners.
                xy = (corners[0] + (cols[...,None]+.5)/width*(corners[1]-corners[0])
                      + (rows[...,None]+.5)/height*(corners[3]-corners[0]))
                xy = np.round(xy*32)/32
                base = np.floor(xy).astype(int)
                frac = xy-base
                # Independent reconstruction: mask and scale the native voxels, apply the
                # documented 0.6-voxel Gaussian, then interpolate bilinearly at each pixel.
                idx = [slice(None)]*3
                idx[axis] = pose['index']
                scale = 1. if key.startswith('ts_') else 255./meta['intensity_upper']
                values = np.rint(np.clip(raw[tuple(idx)]*scale, 0, 255)) * (mask[tuple(idx)]!=0)
                plane = cv2.GaussianBlur(values.T[::-1].astype(np.float32), (0, 0), .6)
                px, py = xy[..., 0], xy[..., 1]
                x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
                fx, fy = px-x0, py-y0
                x1, y1 = np.minimum(x0+1, plane.shape[1]-1), np.minimum(y0+1, plane.shape[0]-1)
                signal = (plane[y0, x0]*(1-fx)*(1-fy) + plane[y0, x1]*fx*(1-fy)
                          + plane[y1, x0]*(1-fx)*fy + plane[y1, x1]*fx*fy)
                white = (signal >= match['level']) != pose['invert']
                # Edge pixels are anti-aliased, so compare only clearly classified points.
                clear = np.abs(signal - match['level']) > 3
                expected = np.where(white, 255., 0.)
                decoded = video_frame[top+rows, left+cols, 1].astype(float)
                errors.append(np.abs(decoded-expected)[clear].mean())
            self.assertFalse(cap.read()[0])
        finally:
            cap.release()
        # H.264 is lossy and the video encoder converts RGB through a limited-range YUV space.
        self.assertLess(max(errors), 4.0, 'Decoded scan pixels disagree with the saved source')
