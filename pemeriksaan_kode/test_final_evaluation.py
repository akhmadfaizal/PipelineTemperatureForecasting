"""Synthetic checks only: alignment, no seed pooling, and protocol gates."""
import importlib.util
import sys
import json
import tempfile
import pandas as pd
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fungsi_bersama'))
sys.path.insert(0, str(ROOT / '04_training'))
spec = importlib.util.spec_from_file_location('final_evaluation', ROOT / '06_evaluasi/05_evaluate.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def packages():
    target = np.array(['2026-09-22T01:00', '2026-09-22T01:05', '2026-09-22T01:10'], dtype='datetime64[m]')
    issue = target - np.timedelta64(60, 'm')
    y = np.array([26., 27., 28.])
    baseline = {'t_test': target, 'issue_test': issue, 'y_true_test': y}
    baseline.update({f'{name}_test': y + 1 for name in ev.BASELINES})
    neural = {seed: {'t': target[::-1], 'issue': issue[::-1], 'y_true': y[::-1],
                     'lstm_test': y[::-1] + seed, 'gru_test': y[::-1] - seed} for seed in ev.SEEDS}
    return baseline, neural


class EvaluationTests(unittest.TestCase):
    def test_reorders_timestamp_and_uses_all_seed_intersection(self):
        baseline, neural = packages()
        neural[3]['gru_test'][1] = np.nan
        frames, dropped = ev.align_packages(baseline, neural, 12)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(frames['baseline']), 2)
        np.testing.assert_allclose(frames[1]['LSTM'], frames['baseline']['y'] + 1)
        self.assertEqual(str(frames['baseline'].index[0]), '2026-09-21 18:00:00+00:00')

    def test_missing_seed_rejected(self):
        baseline, neural = packages()
        del neural[2]
        with self.assertRaisesRegex(ValueError, 'three seeds'):
            ev.align_packages(baseline, neural, 12)

    def test_wrong_target_issue_rejected(self):
        baseline, neural = packages()
        neural[1]['issue'] = neural[1]['issue'] - np.timedelta64(5, 'm')
        with self.assertRaisesRegex(ValueError, 'Issue timestamps'):
            ev.align_packages(baseline, neural, 12)

    def test_ground_truth_mismatch_rejected(self):
        baseline, neural = packages()
        neural[1]['y_true'] = neural[1]['y_true'] + .1
        with self.assertRaisesRegex(ValueError, 'Ground truth'):
            ev.align_packages(baseline, neural, 12)

    def test_summary_counts_seeds_not_pooled_observations(self):
        rows = [{'model': model, 'horizon_minutes': 60, 'seed': seed, 'n': 100,
                 'mae': seed, 'rmse': seed, 'mape': seed} for model in ('LSTM', 'GRU') for seed in ev.SEEDS]
        summary = ev.seed_summary(rows)
        self.assertTrue((summary.n_test_per_seed == 100).all())
        self.assertTrue((summary.n_seeds == 3).all())
        self.assertTrue((summary.mae_sd == 1).all())
        with self.assertRaises(ValueError):
            ev.seed_summary(rows[:-1])

    def test_complete_evaluation_writes_counts_and_explicit_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            for directory in ('results', 'state', 'models'):
                (run / directory).mkdir()
            cfg = {'preprocessing': {'horizons': [12]}, 'evaluation': {'alpha': .05}}
            (run / 'state' / 'training_identity.json').write_text(json.dumps({'fingerprint': 'synthetic', 'config': cfg}))
            baseline, neural = packages()
            np.savez(run / 'results' / 'baseline_predictions_h12.npz', **baseline)
            progress, models, predictions = [], [], []
            for seed, package in neural.items():
                relative = f'results/neural_predictions_h12_seed{seed}.npz'
                np.savez(run / relative, **package, run_fingerprint='synthetic', seed=seed)
                predictions.append({'path': relative, 'sha256': ev.common.sha256_file(run / relative), 'horizon_slots': 12, 'seed': seed})
                for architecture in ('lstm', 'gru'):
                    row = {'status': 'success', 'phase': 'final', 'fingerprint': 'synthetic',
                           'horizon_slots': 12, 'architecture': architecture, 'seed': seed}
                    for prefix in ('model', 'history', 'checkpoint'):
                        artifact = f'models/{architecture}_{seed}_{prefix}.txt'
                        (run / artifact).write_text('synthetic artifact, never real model')
                        row[f'{prefix}_path'] = artifact
                        row[f'{prefix}_sha256'] = ev.common.sha256_file(run / artifact)
                    progress.append(row)
                    models.append({key: row[key] for key in ('architecture', 'horizon_slots', 'seed', 'model_path', 'model_sha256')})
            (run / 'results' / 'final_models_manifest.json').write_text(json.dumps({'run_fingerprint': 'synthetic', 'models': models, 'predictions': predictions}))
            progress_path = run / 'results' / 'final_training_progress.csv'
            pd.DataFrame(progress).to_csv(progress_path, index=False)
            status = ev.evaluate(cfg, run)
            self.assertEqual(status['dm'], 'blocked_seed_family_decision_required')
            self.assertEqual(status['early_warning'], 'blocked_protocol_decision_required')
            metrics = pd.read_csv(run / 'results' / 'metrics_per_seed.csv')
            self.assertEqual(len(metrics), 10)
            self.assertTrue((metrics.n == 3).all())
            summary = pd.read_csv(run / 'results' / 'metrics_seed_summary.csv')
            self.assertEqual(len(summary), 2)
            self.assertTrue((summary.n_test_per_seed == 3).all())
            altered = dict(cfg, evaluation={'alpha': .01})
            with self.assertRaisesRegex(ValueError, 'locked training config'):
                ev.evaluate(altered, run)
            changed_progress = [dict(row, fingerprint='old-run') for row in progress]
            pd.DataFrame(changed_progress).to_csv(progress_path, index=False)
            with self.assertRaisesRegex(ValueError, 'Missing or duplicate successful final fit'):
                ev.evaluate(cfg, run)
            pd.DataFrame(progress).to_csv(progress_path, index=False)
            (run / models[0]['model_path']).write_text('tampered')
            with self.assertRaisesRegex(RuntimeError, 'Checksum'):
                ev.evaluate(cfg, run)

    def test_holm_keeps_nine_comparisons_when_undefined(self):
        adjusted = ev.holm_nine([.001] + [np.nan] * 8)
        self.assertAlmostEqual(adjusted[0], .009)
        self.assertTrue(np.isnan(adjusted[1:]).all())

    def test_dm_rejects_gapped_calendar_instead_of_compressing_time(self):
        times = pd.date_range('2026-09-22', periods=30, freq='5min', tz='UTC').delete(10)
        y = np.arange(len(times), dtype=float)
        statistic, pvalue, status = ev.calendar_dm_hln(times, y, y + 1, y + 2, 12)
        self.assertEqual(status, 'gapped_intersection')
        self.assertTrue(np.isnan(statistic) and np.isnan(pvalue))

    def test_dm_on_regular_calendar_preserves_existing_statistic(self):
        times = pd.date_range('2026-09-22', periods=50, freq='5min', tz='UTC')
        y = np.zeros(50)
        first = np.tile([0., 1.], 25)
        second = np.ones(50)
        expected = ev.dm_hln(y, first, second, 1)
        for unit in ['ns', 'us']:
            self.assertEqual(ev.calendar_dm_hln(times.as_unit(unit), y, first, second, 1), expected)

    def test_dm_length_mismatch_rejected(self):
        times = pd.date_range('2026-09-22', periods=3, freq='5min', tz='UTC')
        with self.assertRaisesRegex(ValueError, 'equal length'):
            ev.calendar_dm_hln(times, np.zeros(2), np.zeros(2), np.zeros(2), 1)

    def test_mape_zero_is_explicit_not_epsilon_substituted(self):
        result = ev.regression_metrics(np.array([0., 1.]), np.array([1., 1.]))
        self.assertTrue(np.isnan(result['mape']))
        self.assertEqual(result['mape_status'], 'undefined_actual_zero_or_near_zero')

if __name__ == '__main__':
    unittest.main()
