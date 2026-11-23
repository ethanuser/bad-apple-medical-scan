"""The viewport annotations describe the real scan: orientation letters agree with the
crop mapping pushed through each scan's affine, the scale matches the crop, and CT
levels convert back to Hounsfield units through the preparation window."""
import unittest

import numpy as np

from mriapple.geometry import CT_WINDOW, PANEL_SIZE, Brain, Pose, maps, values
from mriapple.inputs import ASSETS, SOURCES
from mriapple.render import HU_PER_UNIT, crop_edges, letters, plane_to_world


def spec(prefix):
    for s in SOURCES['scans'] + SOURCES.get('ct_scans', []):
        if s['key'].startswith(prefix) and (ASSETS/s['key']/'image.nii.gz').exists():
            return s


SCANS = [s for s in (spec('ixi_t2_'), spec('ts_')) if s]


def truth(b, p):
    """Letters from finite steps through maps() itself, independent of crop_edges."""
    W, H = PANEL_SIZE
    x, y = maps([values(p)], b.pitch(p.axis), W, H)
    x, y = x.reshape(H, W), y.reshape(H, W)
    cy, cx = H//2, W//2
    right = plane_to_world(b, p.axis, x[cy, cx+40]-x[cy, cx], y[cy, cx+40]-y[cy, cx])
    down = plane_to_world(b, p.axis, x[cy+40, cx]-x[cy, cx], y[cy+40, cx]-y[cy, cx])
    return dict(l=letters(-right), r=letters(right), t=letters(-down), b=letters(down)), x, y


@unittest.skipUnless(SCANS, 'requires fetched scans')
class AnnotationTests(unittest.TestCase):
    def test_orientation_letters_and_scale(self):
        for s in SCANS:
            b = Brain(s)
            for axis in range(3):
                index = sum(b.limits[axis]) // 2
                for angle in range(-180, 180, 15):
                    for mirror in (False, True):
                        p = Pose(0, axis, index, 60., 60., 50., float(angle), mirror, False)
                        with self.subTest(scan=b.key, axis=axis, angle=angle, mirror=mirror):
                            expected, x, y = truth(b, p)
                            self.assertEqual(crop_edges(b, p), expected)
                            pitch = b.pitch(axis)
                            step = np.hypot((x[0, 1]-x[0, 0])*pitch[0], (y[0, 1]-y[0, 0])*pitch[1])
                            self.assertAlmostEqual(step, p.width_mm/PANEL_SIZE[0], places=4)

    def test_ras_letters(self):
        self.assertEqual(letters(np.array([1., 0, 0])), 'R')
        self.assertEqual(letters(np.array([0, -1., 0])), 'P')
        self.assertEqual(letters(np.array([0, .6, .8])), 'SA')

    def test_ct_units(self):
        # Preparation maps CT_WINDOW linearly onto 0-255.
        for hu in (-1000., -533., 0., 400., 1000.):
            stored = (hu - CT_WINDOW[0]) * 255 / (CT_WINDOW[1] - CT_WINDOW[0])
            self.assertAlmostEqual(CT_WINDOW[0] + stored*HU_PER_UNIT, hu, places=6)


if __name__ == '__main__':
    unittest.main()
