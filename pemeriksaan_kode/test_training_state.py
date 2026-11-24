import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fungsi_bersama'))
sys.path.insert(0, str(ROOT / '04_training'))
import common
import pc_training_state as state
spec = importlib.util.spec_from_file_location("pc_train", ROOT / "04_training/04_train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
import pandas as pd


class TrainingStateTests(unittest.TestCase):
    def test_identity_rejects_environment_source_or_prepared_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'data').mkdir()
            fixture = run / 'data' / 'scaler.json'
            fixture.write_text('{}')
            cfg = {'model': {'final_seeds': [1, 2, 3]}, '_run_dir': tmp}
            metadata = {'sha256': 'dataset', 'fingerprint': 'preprocessing'}
            env = {'packages': {'tensorflow': '2.18.1'}, 'hostname': 'pc', 'captured_at_utc': 't1'}
            identity = state.bind_run_identity(run, cfg, metadata, env)
            self.assertEqual(identity, state.bind_run_identity(run, cfg, metadata, {**env, 'captured_at_utc': 't2'}))
            for changed in [{**env, 'hostname': 'other'}, {**env, 'packages': {'tensorflow': '2.20.0'}}]:
                with self.assertRaises(RuntimeError):
                    state.bind_run_identity(run, cfg, metadata, changed)
            fixture.write_text('{"changed":true}')
            with self.assertRaises(RuntimeError):
                state.bind_run_identity(run, cfg, metadata, env)

    def test_completed_fit_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            row = {}
            for prefix in ['model', 'history', 'checkpoint']:
                path = run / f'{prefix}.bin'
                path.write_bytes(b'original')
                row[f'{prefix}_path'] = path.name
                row[f'{prefix}_sha256'] = common.sha256_file(path)
            state.verify_artifacts(run, row)
            (run / 'history.bin').write_bytes(b'tampered')
            with self.assertRaises(RuntimeError):
                state.verify_artifacts(run, row)

    def test_selection_ignores_other_run_and_final_seed(self):
        hp = {'units': 32, 'layers': 1, 'dropout': .1, 'learning_rate': .001, 'batch_size': 32}
        rows = []
        for config_id, value in [('a', 2.0), ('b', 1.0)]:
            for architecture in ['lstm', 'gru']:
                rows.append(dict(hp, config_id=config_id, architecture=architecture, horizon_slots=12,
                                 status='success', seed=1, phase='tuning', fingerprint='current', val_mae_celsius=value))
        rows.append(dict(rows[0], fingerprint='old', val_mae_celsius=.001))
        rows.append(dict(rows[0], seed=3, phase='final', val_mae_celsius=.001))
        cfg = {'model': {'seed_tuning': 1}, 'preprocessing': {'horizons': [12]}}
        selected = train.shared_selection(pd.DataFrame(rows), cfg, 'current')
        self.assertEqual(selected['12']['shared_hyperparameters']['config_id'], 'b')
        self.assertEqual(train.shared_selection(pd.DataFrame(rows), cfg, 'unknown'), {})

    def test_interrupted_fit_restarts_seed_preserving_attempts(self):
        class Model:
            def save(self, path):
                Path(path).write_bytes(b'complete-model')
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'results').mkdir()
            cfg = {'model': {'seed_tuning': 1}}
            hp = {'config_id': 'test'}
            progress = run / 'results' / 'progress.csv'
            with mock.patch.object(common, 'load_sequence', return_value={}):
                with mock.patch.object(train, 'train_once', side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        train.run_trial(cfg, run, 'fp', 'gru', 12, hp, progress, seed=3, phase='final')
                first = train.load_progress(progress).iloc[0]
                self.assertEqual(first['status'], 'interrupted')
                def success(*args, **kwargs):
                    self.assertEqual(args[4], 3)
                    directory = kwargs['attempt_dir']
                    (directory / 'history.csv').write_text('epoch,loss,val_loss\n0,1,1\n')
                    (directory / 'best.keras').write_bytes(b'checkpoint')
                    return Model(), {'val_mae_celsius': 1, 'duration_seconds': .2, 'epochs_ran': 1}
                with mock.patch.object(train, 'train_once', side_effect=success):
                    _, row = train.run_trial(cfg, run, 'fp', 'gru', 12, hp, progress, seed=3, phase='final')
                self.assertNotEqual(first['attempt_dir'], row['attempt_dir'])
                self.assertTrue((run / first['attempt_dir'] / 'status.json').exists())
                with mock.patch.object(train, 'train_once', side_effect=AssertionError('must skip')):
                    model, resumed = train.run_trial(cfg, run, 'fp', 'gru', 12, hp, progress, seed=3, phase='final')
                self.assertIsNone(model)
                self.assertEqual(resumed['seed'], 3)

    def test_final_writes_all_eighteen_fits_and_seed_predictions(self):
        import numpy as np
        from types import SimpleNamespace
        class Model:
            def save(self, path):
                Path(path).write_bytes(b'fixture-not-real-keras')
            def predict(self, x, **kwargs):
                return np.zeros((len(x), 1))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for name in ['results', 'models', 'data', 'state']:
                (run / name).mkdir()
            common.atomic_json(run / 'data' / 'scaler.json', {'min': [20], 'max': [40]})
            cfg = {'model': {'seed_tuning': 1, 'final_seeds': [1, 2, 3],
                             'architectures': ['lstm', 'gru']},
                   'preprocessing': {'horizons': [12, 24, 36]}}
            selection = {'run_fingerprint': 'fp', 'selections': {
                str(h): {'shared_hyperparameters': {'config_id': 'fixture'}} for h in [12, 24, 36]}}
            common.atomic_json(run / 'results' / 'selected_shared_configs.json', selection)
            sequence = {'X_val': np.zeros((2, 3, 1)), 'X_test': np.zeros((2, 3, 1)),
                        'ycel_val': np.array([20., 20.]), 'ycel_test': np.array([20., 20.])}
            for split in ['val', 'test']:
                sequence['t_' + split] = np.array(['2026-09-22T01:00', '2026-09-22T01:05'], dtype='datetime64[m]')
                sequence['issue_' + split] = np.array(['2026-09-22T00:00', '2026-09-22T00:05'], dtype='datetime64[m]')
                sequence['actual_at_issue_' + split] = np.array([19., 19.])
            def success(*args, **kwargs):
                directory = kwargs['attempt_dir']
                (directory / 'history.csv').write_text('epoch,loss,val_loss\n0,1,1\n')
                (directory / 'best.keras').write_bytes(b'checkpoint')
                return Model(), {'val_mae_celsius': 0, 'duration_seconds': .2, 'epochs_ran': 1}
            fake_tf = SimpleNamespace(keras=SimpleNamespace(backend=SimpleNamespace(clear_session=lambda: None)))
            with mock.patch.dict(sys.modules, {'tensorflow': fake_tf}):
                with mock.patch.object(common, 'load_sequence', return_value=sequence):
                    with mock.patch.object(train, 'train_once', side_effect=success) as fit:
                        train.final_training(cfg, run, 'fp')
                        self.assertEqual(fit.call_count, 18)
            manifest = json.loads((run / 'results' / 'final_models_manifest.json').read_text())
            self.assertEqual(len(manifest['models']), 18)
            self.assertEqual(len(manifest['predictions']), 9)
            frame = train.load_progress(run / 'results' / 'final_training_progress.csv')
            self.assertEqual(set(frame.seed), {1, 2, 3})
            self.assertEqual(set(frame.status), {'success'})
            for item in manifest['predictions']:
                path = run / item['path']
                self.assertEqual(common.sha256_file(path), item['sha256'])
                with np.load(path, allow_pickle=False) as payload:
                    self.assertEqual(int(payload['seed']), item['seed'])
                    self.assertIn('actual_at_issue_test', payload.files)
                    self.assertEqual(str(payload['run_fingerprint']), 'fp')


if __name__ == '__main__':
    unittest.main()
