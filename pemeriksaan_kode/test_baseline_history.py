"""Synthetic regression checks for forecast origin; not thesis performance scores."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('baseline_history', ROOT / '05_baseline/03_baselines.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class RecordedFit:
    def __init__(self):
        self.histories, self.appends = [], []

    def apply(self, history, refit):
        assert refit is False
        self.histories.append(history.copy())
        self.last = history[-1]
        return self

    def append(self, values, refit):
        assert refit is False
        self.appends.extend(values)
        self.last = values[-1]
        return self

    def forecast(self, steps):
        return np.repeat(self.last, steps)


def history():
    index = pd.date_range('2026-09-01', periods=210, freq='5min', tz='UTC')
    values = np.arange(210, dtype=float)
    values[140:150] = np.nan  # Old segment is longer than current segment.
    return pd.Series(values, index=index)


class BaselineHistoryTests(unittest.TestCase):
    def test_latest_segment_ends_exactly_at_issue_without_future(self):
        series = history()
        selected = baseline.latest_history(series, series.index[199], pd.Timedelta(minutes=5))
        self.assertEqual(selected.index[0], series.index[150])
        self.assertEqual(selected.index[-1], series.index[199])
        self.assertEqual(len(selected), 50)

    def test_reset_forecast_uses_latest_segment_and_append_next_observation(self):
        series, fit = history(), RecordedFit()
        issues = series.index[[199, 200]].to_numpy()
        prediction = baseline.arima_predictions(fit, series, issues, 12, 'UTC')
        np.testing.assert_array_equal(fit.histories[0], np.arange(150., 200.))
        self.assertEqual(fit.appends, [200.])
        np.testing.assert_array_equal(prediction, [199., 200.])

    def test_real_random_walk_forecast_origin_is_latest_observation(self):
        series = history()
        fitted = ARIMA(series.iloc[:100].to_numpy(), order=(0, 1, 0)).fit()
        prediction = baseline.arima_predictions(fitted, series, series.index[[199, 200]].to_numpy(), 12, 'UTC')
        np.testing.assert_allclose(prediction, [199., 200.], atol=1e-6)

    def test_short_latest_segment_does_not_fall_back_to_stale_long_segment(self):
        series, fit = history(), RecordedFit()
        prediction = baseline.arima_predictions(fit, series, series.index[[190]].to_numpy(), 12, 'UTC')
        self.assertTrue(np.isnan(prediction[0]))
        self.assertEqual(fit.histories, [])

    def test_missing_issue_or_nonfinite_issue_never_uses_older_origin(self):
        series = history()
        for issue in (series.index[145], series.index[199] + pd.Timedelta(minutes=1)):
            self.assertTrue(baseline.latest_history(series, issue, pd.Timedelta(minutes=5)).empty)

    def test_omitted_calendar_rows_are_a_segment_break(self):
        series = history().drop(history().index[180])
        selected = baseline.latest_history(series, series.index[-1], pd.Timedelta(minutes=5))
        self.assertEqual(selected.index[0], pd.Timestamp('2026-09-01 15:05', tz='UTC'))

    def test_duplicate_or_out_of_order_issues_rejected(self):
        series = history()
        for positions in ([200, 200], [200, 199]):
            with self.assertRaisesRegex(ValueError, 'sorted and unique'):
                baseline.arima_predictions(RecordedFit(), series, series.index[positions].to_numpy(), 12, 'UTC')


if __name__ == '__main__':
    unittest.main()
