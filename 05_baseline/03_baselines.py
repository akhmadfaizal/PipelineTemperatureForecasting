from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / 'fungsi_bersama'))
sys.path.insert(0, str(PIPELINE_ROOT / '04_training'))
import common


def _series(run_dir: Path, target: str) -> pd.Series:
    frame = pd.read_csv(run_dir / "data" / "prepared_series.csv")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")[f"{target}_celsius"].sort_index()


def _sequence_index(times: np.ndarray, timezone: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(times))
    if index.tz is None:
        index = index.tz_localize(timezone)
    return index.tz_convert("UTC")


def _at(series: pd.Series, times: np.ndarray, timezone: str) -> np.ndarray:
    return series.reindex(_sequence_index(times, timezone)).to_numpy(dtype=float)


def simple_predictions(series: pd.Series, issue: np.ndarray, target_times: np.ndarray,
                       seasonal_lag: int, windows: list[int], timezone: str) -> dict[str, np.ndarray]:
    step = series.index.to_series().diff().dropna().mode().iloc[0]
    issue_index = _sequence_index(issue, timezone)
    target_index = _sequence_index(target_times, timezone)
    result = {"Naive": _at(series, issue, timezone),
              "SeasonalNaive": series.reindex(target_index - seasonal_lag * step).to_numpy(dtype=float)}
    for window in windows:
        moving = series.rolling(window, min_periods=window).mean()
        result[f"MovingAverage_w{window}"] = moving.reindex(issue_index).to_numpy(dtype=float)
    return result


def longest_segment(series: pd.Series) -> pd.Series:
    valid = series.notna()
    groups = (valid != valid.shift()).cumsum()
    candidates = [part.dropna() for _, part in series.groupby(groups) if valid.loc[part.index[0]]]
    if not candidates:
        raise RuntimeError("Tidak ada segmen suhu lengkap untuk ARIMA.")
    return max(candidates, key=len)


def latest_history(series: pd.Series, issue: pd.Timestamp, step: pd.Timedelta) -> pd.Series:
    """Contiguous observed history ending at issue, never a stale earlier segment."""
    history = series.loc[:issue]
    if history.empty or history.index[-1] != issue or not np.isfinite(history.iloc[-1]):
        return history.iloc[:0]
    finite = np.isfinite(history.to_numpy(dtype=float))
    breaks = np.flatnonzero(~finite)
    start = int(breaks[-1] + 1) if len(breaks) else 0
    # Also preserve calendar gaps if an input omitted missing rows entirely.
    calendar_breaks = np.flatnonzero((history.index[1:] - history.index[:-1]) != step)
    if len(calendar_breaks):
        start = max(start, int(calendar_breaks[-1] + 1))
    return history.iloc[start:]


def select_arima(train: pd.Series, cfg: dict) -> tuple[object, tuple[int, int, int], dict]:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    segment = longest_segment(train)
    d, transformed, adf_rows = 0, segment.to_numpy(dtype=float), []
    for candidate_d in range(int(cfg["max_d"]) + 1):
        if len(transformed) < 30:
            break
        statistic, pvalue, *_ = adfuller(transformed, autolag="AIC")
        adf_rows.append({"d": candidate_d, "statistic": float(statistic), "p_value": float(pvalue)})
        d = candidate_d
        if pvalue < 0.05:
            break
        transformed = np.diff(transformed)

    best, best_order, best_aic = None, None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(int(cfg["max_p"]) + 1):
            for q in range(int(cfg["max_q"]) + 1):
                try:
                    fitted = ARIMA(segment.to_numpy(), order=(p, d, q)).fit()
                    if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                        best, best_order, best_aic = fitted, (p, d, q), float(fitted.aic)
                except Exception:
                    continue
    if best is None:
        raise RuntimeError("Tidak ada kandidat ARIMA yang berhasil diestimasi.")
    return best, best_order, {"adf": adf_rows, "order": list(best_order), "aic": best_aic,
                              "training_segment_start": segment.index[0].isoformat(),
                              "training_segment_end": segment.index[-1].isoformat(),
                              "training_segment_n": len(segment)}


def arima_predictions(fitted: object, full_series: pd.Series, issues: np.ndarray,
                      horizon: int, timezone: str) -> np.ndarray:
    if not full_series.index.is_monotonic_increasing or full_series.index.has_duplicates:
        raise ValueError("ARIMA requires sorted, unique history timestamps.")
    if horizon < 1:
        raise ValueError("ARIMA horizon must be positive.")
    output = np.full(len(issues), np.nan)
    issue_index = _sequence_index(issues, timezone)
    if issue_index.has_duplicates or not issue_index.is_monotonic_increasing:
        raise ValueError("ARIMA issues must be sorted and unique.")
    expected_step = full_series.index.to_series().diff().dropna().mode().iloc[0]
    state, previous_issue = None, None
    for i, issue in enumerate(issue_index):
        try:
            if state is None or previous_issue is None or issue - previous_issue != expected_step:
                history = latest_history(full_series, issue, expected_step)
                if len(history) < 50:
                    continue
                state = fitted.apply(history.to_numpy(dtype=float), refit=False)
            else:
                observation = full_series.reindex([issue]).iloc[0]
                if not np.isfinite(observation):
                    state = None
                    continue
                state = state.append([float(observation)], refit=False)
            output[i] = float(state.forecast(steps=horizon)[-1])
            previous_issue = issue
        except Exception:
            state, previous_issue = None, None
    return output


def choose_moving_average(validation: dict[str, np.ndarray], y_true: np.ndarray) -> tuple[str, list[dict]]:
    rows = []
    for name, prediction in validation.items():
        if name.startswith("MovingAverage_"):
            ok = np.isfinite(prediction) & np.isfinite(y_true)
            score = float(np.mean(np.abs(y_true[ok] - prediction[ok]))) if ok.any() else np.inf
            rows.append({"model": name, "validation_mae": score, "n": int(ok.sum())})
    if not rows:
        raise RuntimeError("Tidak ada Moving Average yang bisa dinilai.")
    rows.sort(key=lambda row: (row["validation_mae"], row["model"]))
    return rows[0]["model"], rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    cfg, run_dir = common.load_config(args.config), Path(args.run_dir).resolve()
    timezone = cfg["project"]["timezone"]
    series = _series(run_dir, cfg["data"]["kolom_target"])
    train_end = pd.Timestamp(cfg["preprocessing"]["split"]["train"][1]).tz_convert("UTC")
    fitted, _, arima_meta = select_arima(series.loc[series.index < train_end], cfg["baseline"]["arima"])
    metadata = {"arima": {**arima_meta,
                           "forecast_history_policy": "latest_contiguous_segment_ending_at_issue",
                           "forecast_minimum_history_slots": 50,
                           "refit_at_issue": False},
                "baseline_code_sha256": common.sha256_file(Path(__file__)), "horizons": {}}

    for horizon in cfg["preprocessing"]["horizons"]:
        seq = common.load_sequence(run_dir, horizon)
        val = simple_predictions(series, seq["issue_val"], seq["t_val"],
                                 cfg["baseline"]["seasonal_naive_lag"], cfg["baseline"]["moving_average_windows"], timezone)
        selected_ma, candidates = choose_moving_average(val, seq["ycel_val"])
        test = simple_predictions(series, seq["issue_test"], seq["t_test"],
                                  cfg["baseline"]["seasonal_naive_lag"], cfg["baseline"]["moving_average_windows"], timezone)
        val["ARIMA"] = arima_predictions(fitted, series, seq["issue_val"], horizon, timezone)
        test["ARIMA"] = arima_predictions(fitted, series, seq["issue_test"], horizon, timezone)
        payload = {"y_true_val": seq["ycel_val"], "issue_val": seq["issue_val"], "t_val": seq["t_val"],
                   "y_true_test": seq["ycel_test"], "issue_test": seq["issue_test"], "t_test": seq["t_test"]}
        for name in ["Naive", "SeasonalNaive", selected_ma, "ARIMA"]:
            public = "MovingAverage" if name == selected_ma else name
            payload[f"{public}_val"] = val[name]
            payload[f"{public}_test"] = test[name]
        np.savez_compressed(run_dir / "results" / f"baseline_predictions_h{horizon}.npz", **payload)
        metadata["horizons"][str(horizon)] = {"minutes": horizon * 5,
                                                "moving_average_selected": selected_ma,
                                                "moving_average_validation": candidates}
    common.atomic_json(run_dir / "results" / "baseline_selection.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
