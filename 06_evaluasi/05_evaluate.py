"""Final seed-aware evaluation. No seed pooling and no implicit protocol decisions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / 'fungsi_bersama'))
sys.path.insert(0, str(PIPELINE_ROOT / '04_training'))
import common
import pc_training_state as state

BASELINES = ('Naive', 'SeasonalNaive', 'MovingAverage', 'ARIMA')
SEEDS = (1, 2, 3)


def prediction_frame(package, baseline=False):
    def timestamp(values):
        parsed = pd.DatetimeIndex(pd.to_datetime(values))
        return (parsed.tz_localize('Asia/Jakarta') if parsed.tz is None else parsed).tz_convert('UTC')
    t_key, y_key, issue_key = ('t_test', 'y_true_test', 'issue_test') if baseline else ('t', 'y_true', 'issue')
    fields = {'target': timestamp(package[t_key]),
              'issue': timestamp(package[issue_key]),
              'y': np.asarray(package[y_key]).reshape(-1)}
    names = BASELINES if baseline else ('lstm', 'gru')
    for name in names:
        fields[name if baseline else name.upper()] = np.asarray(package[f'{name}_test']).reshape(-1)
    frame = pd.DataFrame(fields).set_index('target').sort_index()
    if frame.index.has_duplicates or frame['issue'].duplicated().any():
        raise ValueError('Duplicate target/issue timestamp; evaluation would double-count observations.')
    return frame


def align_packages(baseline, neural, horizon_slots):
    """One intersection for all architectures, baselines and seeds at each horizon."""
    if set(neural) != set(SEEDS):
        raise ValueError('Final evaluation requires all three seeds 1, 2, 3.')
    frames = {'baseline': prediction_frame(baseline, True)}
    frames.update({seed: prediction_frame(package) for seed, package in neural.items()})
    targets = frames['baseline'].index
    for frame in frames.values():
        targets = targets.intersection(frame.index)
    targets = targets.sort_values()
    if targets.empty:
        raise ValueError('No shared target timestamps.')
    reference = frames['baseline'].loc[targets]
    mask = np.isfinite(reference['y'].to_numpy())
    for frame in frames.values():
        frame = frame.loc[targets]
        if not np.allclose(frame['y'], reference['y'], rtol=0, atol=1e-8, equal_nan=True):
            raise ValueError('Ground truth differs at identical target timestamps.')
        if not frame['issue'].equals(reference['issue']):
            raise ValueError('Issue timestamps differ at identical targets.')
        if not ((frame.index - pd.DatetimeIndex(frame['issue'])) == pd.Timedelta(minutes=horizon_slots * 5)).all():
            raise ValueError('Target minus issue does not equal the configured horizon.')
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        mask &= np.isfinite(numeric).all(axis=1)
    selected = targets[mask]
    if selected.empty:
        raise ValueError('No finite common intersection across all three seeds and baselines.')
    return {key: frame.loc[selected] for key, frame in frames.items()}, len(targets) - len(selected)


def regression_metrics(y, prediction):
    error = np.asarray(y) - np.asarray(prediction)
    nonzero = np.abs(y) > 1e-8
    return {'n': len(error), 'mae': float(np.abs(error).mean()),
            'rmse': float(np.sqrt(np.square(error).mean())),
            'mape': float((np.abs(error[nonzero] / y[nonzero])).mean() * 100) if nonzero.all() else np.nan,
            'mape_status': 'valid' if nonzero.all() else 'undefined_actual_zero_or_near_zero'}


def seed_summary(rows):
    frame = pd.DataFrame(rows)
    neural = frame[frame['model'].isin(['LSTM', 'GRU'])]
    result = []
    for (model, horizon), group in neural.groupby(['model', 'horizon_minutes']):
        if set(group['seed']) != set(SEEDS) or len(group) != 3 or group['n'].nunique() != 1:
            raise ValueError('Summary requires exactly three seeds on the same intersection.')
        item = {'model': model, 'horizon_minutes': horizon, 'seeds': '1,2,3', 'n_seeds': 3,
                'n_test_per_seed': int(group['n'].iloc[0]), 'sd_ddof': 1}
        for metric in ('mae', 'rmse', 'mape'):
            values = group[metric].to_numpy()
            item[f'{metric}_mean'] = float(values.mean())
            item[f'{metric}_sd'] = float(values.std(ddof=1))
        result.append(item)
    return pd.DataFrame(result)


def dm_hln(y, first, second, horizon):
    differential = np.abs(y - first) - np.abs(y - second)
    n = len(differential)
    if n <= horizon:
        return np.nan, np.nan, 'insufficient_samples'
    centered = differential - differential.mean()
    variance = np.dot(centered, centered) / n
    for lag in range(1, horizon):
        variance += 2 * np.dot(centered[lag:], centered[:-lag]) / n
    if variance <= 0:
        return np.nan, np.nan, 'nonpositive_long_run_variance'
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = differential.mean() / np.sqrt(variance / n) * correction
    return float(statistic), float(2 * student_t.sf(abs(statistic), n - 1)), 'computed'


def calendar_dm_hln(times, y, first, second, horizon):
    """Do not compress missing calendar slots into adjacent DM observations."""
    if len(times) != len(y):
        raise ValueError('DM timestamps and observations must have equal length.')
    if not ((times[1:] - times[:-1]) == pd.Timedelta(minutes=5)).all():
        return np.nan, np.nan, 'gapped_intersection'
    return dm_hln(y, first, second, horizon)


def holm_nine(values):
    """Keep the locked nine-member family even if some tests are undefined."""
    if len(values) != 9:
        raise ValueError('Holm family must contain nine comparisons.')
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.where(np.isfinite(values), values, 1.0))
    output = np.full(9, np.nan)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (9 - rank) * (values[index] if np.isfinite(values[index]) else 1.0))
        if np.isfinite(values[index]):
            output[index] = min(running, 1.0)
    return output


def verify_manifest_artifact(run_dir, relative, digest):
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()) or not path.is_file() or common.sha256_file(path) != digest:
        raise ValueError(f'Artifact path/hash invalid: {relative}')


def evaluate(cfg, run_dir, *, dm_family=None):
    """Evaluate frozen training artifacts with a separately recorded analysis protocol."""
    if dm_family not in (None, "per_seed_per_horizon"):
        raise ValueError("Unsupported analysis DM family.")
    results = run_dir / 'results'
    progress = pd.read_csv(results / 'final_training_progress.csv')
    identity = json.loads((run_dir / 'state' / 'training_identity.json').read_text())
    if common.stable_hash(identity['config']) != common.stable_hash({k: v for k, v in cfg.items() if not k.startswith('_')}):
        raise ValueError('Evaluation config differs from locked training config.')
    fingerprint = identity['fingerprint']
    manifest = json.loads((results / 'final_models_manifest.json').read_text())
    if manifest['run_fingerprint'] != fingerprint:
        raise ValueError('Final manifest fingerprint differs from training identity.')
    predictions_manifest = {item['path']: item for item in manifest['predictions']}
    successful = progress[progress['status'].eq('success') & progress['fingerprint'].eq(fingerprint) & progress['phase'].eq('final')]
    rows, dm_rows, intersection_rows = [], [], []
    dm_enabled = (dm_family or cfg['evaluation'].get('dm_final_seed_family')) == 'per_seed_per_horizon'
    for horizon in cfg['preprocessing']['horizons']:
        for seed in SEEDS:
            for architecture in ('lstm', 'gru'):
                matched = successful[successful['seed'].eq(seed) & successful['horizon_slots'].eq(horizon)
                                     & successful['architecture'].str.lower().eq(architecture)]
                if len(matched) != 1:
                    raise ValueError(f'Missing or duplicate successful final fit: {architecture}, h={horizon}, seed={seed}.')
                row = matched.iloc[0].to_dict()
                state.verify_artifacts(run_dir, row)
                models = [item for item in manifest['models'] if item['architecture'] == architecture and item['horizon_slots'] == horizon and item['seed'] == seed]
                if len(models) != 1 or models[0]['model_sha256'] != row['model_sha256']:
                    raise ValueError('Final model manifest differs from verified fit progress.')
                verify_manifest_artifact(run_dir, models[0]['model_path'], models[0]['model_sha256'])
        with np.load(results / f'baseline_predictions_h{horizon}.npz', allow_pickle=False) as package:
            baseline = dict(package)
        neural = {}
        for seed in SEEDS:
            relative = f'results/neural_predictions_h{horizon}_seed{seed}.npz'
            entry = predictions_manifest.get(relative)
            if entry is None or entry['horizon_slots'] != horizon or entry['seed'] != seed:
                raise ValueError('Prediction artifact missing or mismatched in final manifest.')
            verify_manifest_artifact(run_dir, relative, entry['sha256'])
            with np.load(run_dir / relative, allow_pickle=False) as package:
                neural[seed] = dict(package)
            if str(neural[seed]['run_fingerprint'].item()) != fingerprint or int(neural[seed]['seed'].item()) != seed:
                raise ValueError('Prediction package seed/fingerprint does not match current run.')
        aligned, dropped = align_packages(baseline, neural, horizon)
        reference, times = aligned['baseline'], aligned['baseline'].index
        intersection_rows.append({'horizon_minutes': horizon * 5, 'n': len(times), 'dropped_nonfinite': dropped,
                                  'calendar_gap_count': int(((times[1:] - times[:-1]) != pd.Timedelta(minutes=5)).sum()),
                                  'first_target': times[0], 'last_target': times[-1]})
        y = reference['y'].to_numpy()
        for name in BASELINES:
            rows.append({'model': name, 'horizon_minutes': horizon * 5, 'seed': 0,
                         **regression_metrics(y, reference[name].to_numpy())})
        for seed in SEEDS:
            predictions = {name: reference[name].to_numpy() for name in BASELINES}
            predictions.update({name: aligned[seed][name].to_numpy() for name in ('LSTM', 'GRU')})
            for name in ('LSTM', 'GRU'):
                rows.append({'model': name, 'horizon_minutes': horizon * 5, 'seed': seed,
                             **regression_metrics(y, predictions[name])})
            if dm_enabled:
                family = []
                for first, second in [(a, b) for a in ('LSTM', 'GRU') for b in BASELINES] + [('LSTM', 'GRU')]:
                    # Removing missing timestamps breaks equal-spaced lag assumptions.
                    statistic, p, status = calendar_dm_hln(times, y, predictions[first], predictions[second], horizon)
                    family.append({'seed': seed, 'horizon_minutes': horizon * 5, 'model_1': first, 'model_2': second,
                                   'n': len(y), 'dm_statistic': statistic, 'p_value': p, 'status': status,
                                   'mae_difference_1_minus_2': float((np.abs(y-predictions[first])-np.abs(y-predictions[second])).mean())})
                for item, adjusted in zip(family, holm_nine([item['p_value'] for item in family])):
                    item.update(p_holm=adjusted, reject_h0=bool(np.isfinite(adjusted) and adjusted < cfg['evaluation']['alpha']), family='nine_pairs_per_seed_per_horizon')
                dm_rows.extend(family)
    common.atomic_csv(results / 'metrics_per_seed.csv', pd.DataFrame(rows))
    common.atomic_csv(results / 'metrics_seed_summary.csv', seed_summary(rows))
    common.atomic_csv(results / 'evaluation_intersections.csv', pd.DataFrame(intersection_rows))
    status = {'schema_version': 1, 'seeds': list(SEEDS), 'seed_pooling': False,
              'dm': 'computed_per_seed_per_horizon' if dm_enabled else 'blocked_seed_family_decision_required',
              'dm_limitation': 'Holm controls each nine-pair seed/horizon family, not all seeds jointly. Undefined tests retain family size nine.',
              'early_warning': 'blocked_protocol_decision_required',
              'early_warning_reason': 'Confirm prediction >27 versus >=27; cooldown scope and multiple warnings/event matching. Issue time and actual at issue must be used, never target time as trigger.',
              'timestamp_contract': {
                  'interval': '[bin_start, bin_start + 5 minutes)',
                  'issue': 'start label of the last input bin; not actual issuance time',
                  'input_available_at_nominal': 'issue + 5 minutes; ignores acquisition/processing delay',
                  'target_start': 't = issue + horizon_slots * 5 minutes',
                  'target_end': 't + 5 minutes',
                  'nominal_lead_to_target_end_minutes': 'horizon_slots * 5',
                  'nominal_lead_to_target_start_minutes': '(horizon_slots - 1) * 5',
                  'issued_at_actual': 'not recorded in offline prediction packages; do not infer live lead time'},
              'mape_limitation': 'Celsius has an arbitrary zero; MAPE is retained for historical compatibility, not model selection. Undefined for actual zero/near zero.'}
    if dm_enabled:
        common.atomic_csv(results / 'dm_holm_per_seed.csv', pd.DataFrame(dm_rows))
    common.atomic_json(results / 'evaluation_status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(common.load_config(args.config), Path(args.run_dir).resolve()), indent=2))

if __name__ == '__main__':
    main()
