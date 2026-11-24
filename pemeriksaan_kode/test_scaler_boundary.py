"""Check that future extremes cannot affect train-only preprocessing."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fungsi_bersama'))
import common


class ScalerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'temperature': [10., 20., 100., -50.],
                                   'humidity': [30., 50., 99., 1.]},
                                  index=pd.date_range('2026-01-01', periods=4, freq='1D', tz='Asia/Jakarta'))
        self.pre = {'scaler_fit': 'train_only', 'split': {'train': ['2026-01-01 00:00+07:00', '2026-01-03 00:00+07:00']}}

    def test_future_extremes_do_not_change_train_extrema_or_clip(self):
        lo, hi = common.fit_sensor_scaler(self.frame, list(self.frame), self.pre)
        np.testing.assert_array_equal(lo, [10., 30.])
        np.testing.assert_array_equal(hi, [20., 50.])
        transformed = (self.frame - lo) / (hi - lo)
        self.assertEqual(transformed.temperature.iloc[2], 9.)
        self.assertEqual(transformed.temperature.iloc[3], -6.)
        changed = self.frame.copy(); changed.iloc[2:] *= 100
        other_lo, other_hi = common.fit_sensor_scaler(changed, list(changed), self.pre)
        np.testing.assert_array_equal(other_lo, lo)
        np.testing.assert_array_equal(other_hi, hi)

    def test_invalid_mode_rejected(self):
        self.pre['scaler_fit'] = 'typo'
        with self.assertRaises(ValueError):
            common.fit_sensor_scaler(self.frame, list(self.frame), self.pre)

    def test_legacy_config_remains_global(self):
        self.pre.pop('scaler_fit')
        lo, hi = common.fit_sensor_scaler(self.frame, list(self.frame), self.pre)
        np.testing.assert_array_equal(lo, [-50., 1.])
        np.testing.assert_array_equal(hi, [100., 99.])

    def test_empty_train_rejected(self):
        self.pre['split']['train'] = ['2025-01-01 00:00+07:00', '2025-01-02 00:00+07:00']
        with self.assertRaises(ValueError):
            common.fit_sensor_scaler(self.frame, list(self.frame), self.pre)


if __name__ == '__main__':
    unittest.main()
