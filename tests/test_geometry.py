"""Regression checks for anisotropic source slices, mirrored footprints and polarity."""
import unittest

import numpy as np

from mriapple.geometry import inside, maps, sample
from mriapple.metrics import fit_levels, score_image


class GeometryTests(unittest.TestCase):
    def test_rotated_footprint_preserves_lengths_in_mm(self):
        pitch = (.7, 2.2)
        for mirror in (0, 1):
            params = [[40., 60., 32., 37., mirror]]
            x, y = maps(params, pitch, 0, 0, corners=True)
            points = np.column_stack((x.ravel(), y.ravel())) * pitch
            a, b = points[1]-points[0], points[3]-points[0]
            self.assertAlmostEqual(np.linalg.norm(a), 32., places=4)
            self.assertAlmostEqual(np.linalg.norm(b), 24., places=4)
            self.assertAlmostEqual(np.dot(a, b), 0., places=3)
            np.testing.assert_allclose(points.mean(axis=0), [40, 60], atol=1e-4)

    def test_identity_and_mirror_sample_native_voxel_centres(self):
        plane = np.arange(12*16, dtype=np.uint8).reshape(12, 16)
        params = [[7.5, 5.5, 8., 0., 0], [7.5, 5.5, 8., 0., 1]]
        actual = sample(plane, (1., 1.), params, 8, 6)
        expected = plane[3:9, 4:12]
        np.testing.assert_array_equal(actual[0], expected)
        np.testing.assert_array_equal(actual[1], expected[:, ::-1])
        self.assertTrue(inside(params, (1., 1.), plane.shape).all())
        self.assertFalse(inside([[0,0,8,30,1]], (1., 1.), plane.shape)[0])

    def test_fitted_threshold_reproduces_target_in_both_polarities(self):
        patch = np.tile(np.arange(0, 256, 16, dtype=np.uint8), (12, 1))
        for reverse in (False, True):
            target = np.uint8((patch >= 128) ^ reverse)*255
            levels, iou = fit_levels(patch[None], target, [reverse])
            white = (patch >= levels[0]) != reverse
            np.testing.assert_array_equal(white, target >= 128)
            self.assertEqual(iou[0], 1.)

    def test_identical_flat_and_nonuniform_silhouettes_score_one(self):
        frame = np.zeros((72, 96), dtype=np.uint8)
        for w in (0, 40, 96):
            frame[:, :w] = 255
            self.assertAlmostEqual(score_image(frame, frame)['score'], 1., places=5)


if __name__ == '__main__':
    unittest.main()
