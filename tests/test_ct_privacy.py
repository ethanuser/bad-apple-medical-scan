"""Every prepared CT must have its head removed: nothing at or above the logged cut,
no skull voxel anywhere, and the cut below the skull by the documented margin."""
import unittest

import nibabel as nib
import numpy as np
from scipy import ndimage

from mriapple.geometry import SKULL_MARGIN_MM, prepare_ct
from mriapple.inputs import ASSETS, SOURCES


def available():
    return [s['key'] for s in SOURCES.get('ct_scans', [])
            if all((ASSETS/s['key']/f'{r}.nii.gz').exists() for r in ('image', 'skull', 'clavicula_left', 'clavicula_right'))]


@unittest.skipUnless(available(), 'requires fetched CT scans')
class CTPrivacyTests(unittest.TestCase):
    def test_no_head_anatomy_survives_preparation(self):
        for key in available():
            with self.subTest(key=key):
                data, meta = prepare_ct(ASSETS/key, key)
                cut = meta['head_cut_index']
                self.assertFalse(data[:, :, cut+1:].any(), 'voxels above the head cut')
                skull = np.asanyarray(nib.as_closest_canonical(nib.squeeze_image(
                    nib.load(ASSETS/key/'skull.nii.gz'))).dataobj) != 0
                # Stray label voxels far from the head (seen in the legs) are ignored;
                # the head is the skull label's largest connected component.
                labels, n = ndimage.label(skull)
                if n > 1:
                    skull = labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))
                self.assertFalse(data[skull].any(), 'skull voxels in the prepared CT')
                if skull.any():
                    lowest = np.flatnonzero(skull.any(axis=(0, 1))).min()
                    self.assertLessEqual((cut - lowest) * meta['spacing_mm'][2], -SKULL_MARGIN_MM)


if __name__ == '__main__':
    unittest.main()
