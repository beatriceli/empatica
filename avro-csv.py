"""
avro-csv.py
-----------
Read Empatica EMBRACE+ Avro files, unpack each signal into a time-series
and write one CSV per signal inside a folder named after the participant.

Output layout:
  OUTPUT_DIR/
    <participant_id>/
      accelerometer.csv   (columns: timestamp_utc, timestamp_local, x, y, z)
      eda.csv             (columns: timestamp_utc, timestamp_local, value)
      temperature.csv
      bvp.csv
      steps.csv
      systolic_peaks.csv
      tags.csv
      hr.csv      (columns: timestamp_utc, timestamp_local, ibi_s, bpm)
      hrv.csv             (columns: timestamp_utc, timestamp_local, rmssd_ms, sdnn_ms, pnn50)

Expected directory layout:
  DATA_ROOT/
    <date>/
      <participant_folder>/
        raw_data/v6/
          *.avro

Usage:
  python avro-csv.py                          # uses DATA_ROOT below
  python avro-csv.py --data path/to/data      # override data root
  python avro-csv.py --out  path/to/output    # override output dir
"""

import argparse
from pathlib import Path

import fastavro
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_ROOT = Path.home() / "Downloads" / "empatica_raw"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _us_to_timestamps(start_us: int, fs: float, n: int, tz_offset_s: int):
    """
    Return (utc_series, local_series) for n evenly-spaced samples.

    start_us      – epoch time of first sample in microseconds
    fs            – sampling frequency in Hz
    n             – number of samples
    tz_offset_s   – local = UTC + offset  (e.g. -14400 for UTC-4)
    """
    if fs > 0:
        step_us = 1_000_000.0 / fs
    else:
        step_us = 0.0
    offsets = np.arange(n, dtype=np.float64) * step_us
    us_vals = (start_us + offsets).astype("int64")
    utc = pd.to_datetime(us_vals, unit="us", utc=True)
    local = utc + pd.Timedelta(seconds=tz_offset_s)
    return utc, local


def unpack_record(record: dict) -> dict[str, list[dict]]:
    """
    Unpack a single Avro record into a dict mapping signal name -> list of row dicts.

    Signals with a single value column produce rows: {timestamp_utc, timestamp_local, value}
    Accelerometer produces rows:                     {timestamp_utc, timestamp_local, x, y, z}
    Event signals (peaks, tags) produce rows:        {timestamp_utc, timestamp_local}
    """
    rd = record.get("rawData", {})
    tz = record.get("timezone", 0)   # seconds offset (e.g. -14400)
    signal_data: dict[str, list[dict]] = {}

    def get_nested(rec, dotted_key):
        val = rec
        for p in dotted_key.split("."):
            if not isinstance(val, dict):
                return None
            val = val.get(p)
        return val

    # ------------------------------------------------------------------ #
    # 1. Single-channel uniformly-sampled signals
    # ------------------------------------------------------------------ #
    scalar_signals = [
        ("eda",         "rawData.eda.timestampStart",
                        "rawData.eda.samplingFrequency",
                        "rawData.eda.values"),
        ("temperature", "rawData.temperature.timestampStart",
                        "rawData.temperature.samplingFrequency",
                        "rawData.temperature.values"),
        ("bvp",         "rawData.bvp.timestampStart",
                        "rawData.bvp.samplingFrequency",
                        "rawData.bvp.values"),
        ("steps",       "rawData.steps.timestampStart",
                        "rawData.steps.samplingFrequency",
                        "rawData.steps.values"),
    ]

    for sig_name, start_key, fs_key, val_key in scalar_signals:
        values = get_nested(record, val_key)
        if not values:
            continue
        start_us = get_nested(record, start_key) or 0
        fs       = get_nested(record, fs_key)    or 0.0
        utc_ts, local_ts = _us_to_timestamps(start_us, fs, len(values), tz)
        rows = [
            {"timestamp_utc": utc_ts[i], "timestamp_local": local_ts[i], "value": float(v)}
            for i, v in enumerate(values)
        ]
        signal_data.setdefault(sig_name, []).extend(rows)

    # ------------------------------------------------------------------ #
    # 2. Accelerometer (x, y, z share one timestamp axis)
    # ------------------------------------------------------------------ #
    acc = rd.get("accelerometer", {})
    x_vals = acc.get("x", [])
    y_vals = acc.get("y", [])
    z_vals = acc.get("z", [])
    if x_vals:
        start_us = acc.get("timestampStart", 0) or 0
        fs       = acc.get("samplingFrequency", 0.0) or 0.0
        n        = len(x_vals)
        utc_ts, local_ts = _us_to_timestamps(start_us, fs, n, tz)
        rows = [
            {
                "timestamp_utc":   utc_ts[i],
                "timestamp_local": local_ts[i],
                "x": float(x_vals[i]),
                "y": float(y_vals[i]) if i < len(y_vals) else float("nan"),
                "z": float(z_vals[i]) if i < len(z_vals) else float("nan"),
            }
            for i in range(n)
        ]
        signal_data.setdefault("accelerometer", []).extend(rows)

    # ------------------------------------------------------------------ #
    # 3. Systolic peaks (timestamps in nanoseconds, no value)
    # ------------------------------------------------------------------ #
    peaks_ns = rd.get("systolicPeaks", {}).get("peaksTimeNanos", [])
    if peaks_ns:
        rows = []
        for ns in peaks_ns:
            utc = pd.Timestamp(ns, unit="ns", tz="UTC")
            rows.append({"timestamp_utc": utc, "timestamp_local": utc + pd.Timedelta(seconds=tz)})
        signal_data.setdefault("systolic_peaks", []).extend(rows)

    # ------------------------------------------------------------------ #
    # 4. Tags (timestamps in microseconds, no value)
    # ------------------------------------------------------------------ #
    tags_us = rd.get("tags", {}).get("tagsTimeMicros", [])
    if tags_us:
        rows = []
        for us in tags_us:
            utc = pd.Timestamp(us * 1000, unit="ns", tz="UTC")
            rows.append({"timestamp_utc": utc, "timestamp_local": utc + pd.Timedelta(seconds=tz)})
        signal_data.setdefault("tags", []).extend(rows)

    return signal_data


# --------------------------------------------------------------------------- #
# Heart rate & HRV
# --------------------------------------------------------------------------- #

# Rolling window size (number of beats) for HRV metrics.
HRV_WINDOW_BEATS = 30

# Physiologically valid IBI range (ms). Beats outside this are treated as
# artefacts and excluded before computing HRV.
IBI_MIN_MS = 300   # 200 BPM max
IBI_MAX_MS = 2000  # 30 BPM min


def compute_hr_hrv(peaks_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a list of systolic-peak row dicts (keys: timestamp_utc, timestamp_local),
    return (heart_rate_df, hrv_df).

    heart_rate_df columns: timestamp_utc, timestamp_local, ibi_s, bpm
      – one row per beat (starting from the 2nd peak)

    hrv_df columns: timestamp_utc, timestamp_local, rmssd_ms, sdnn_ms, pnn50
      – one row per beat computed over a rolling window of HRV_WINDOW_BEATS
    """
    if len(peaks_rows) < 2:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(peaks_rows).sort_values("timestamp_utc").reset_index(drop=True)

    # IBI in seconds and milliseconds
    df["ibi_s"]  = df["timestamp_utc"].diff().dt.total_seconds()
    df["ibi_ms"] = df["ibi_s"] * 1000.0

    # Drop the first row (no preceding peak) and artefacts
    df = df.dropna(subset=["ibi_ms"])
    df = df[(df["ibi_ms"] >= IBI_MIN_MS) & (df["ibi_ms"] <= IBI_MAX_MS)].copy()
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ---- Heart rate --------------------------------------------------------
    df["bpm"] = 60.0 / df["ibi_s"]
    hr_df = df[["timestamp_utc", "timestamp_local", "ibi_s", "bpm"]].copy()

    # ---- HRV (rolling window) ----------------------------------------------
    ibi = df["ibi_ms"].to_numpy()

    rmssd_vals = np.full(len(ibi), np.nan)
    sdnn_vals  = np.full(len(ibi), np.nan)
    pnn50_vals = np.full(len(ibi), np.nan)

    for i in range(HRV_WINDOW_BEATS - 1, len(ibi)):
        window = ibi[i - HRV_WINDOW_BEATS + 1 : i + 1]
        successive_diffs = np.diff(window)

        rmssd_vals[i] = np.sqrt(np.mean(successive_diffs ** 2))
        sdnn_vals[i]  = np.std(window, ddof=1)
        pnn50_vals[i] = np.mean(np.abs(successive_diffs) > 50.0) * 100.0

    hrv_df = df[["timestamp_utc", "timestamp_local"]].copy()
    hrv_df["rmssd_ms"] = rmssd_vals
    hrv_df["sdnn_ms"]  = sdnn_vals
    hrv_df["pnn50"]    = pnn50_vals
    hrv_df = hrv_df.dropna(subset=["rmssd_ms"]).reset_index(drop=True)

    return hr_df, hrv_df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(data_root: Path, output_dir: Path | None) -> None:
    # Accumulate rows keyed by (participant_id, signal_name)
    data: dict[tuple[str, str], list[dict]] = {}

    avro_files = sorted(data_root.glob("**/participant_data/*/*/raw_data/*/*.avro"))
    if not avro_files:
        print(f"No .avro files found under {data_root}")
        return

    if output_dir is None:
        participant_data_dir = next(p for p in avro_files[0].parents if p.name == "participant_data")
        output_dir = participant_data_dir.parent / "participant_csv"

    print(f"Found {len(avro_files)} .avro file(s).")

    for avro_path in avro_files:
        participant_folder = avro_path.parts[-4]   # e.g. 0-3YK3K1526F

        with open(avro_path, "rb") as f:
            for record in fastavro.reader(f):
                pid = (record.get("enrollment", {}) or {}).get("participantID")
                if not pid and pid != 0:
                    pid = participant_folder
                pid = str(pid)

                for signal, rows in unpack_record(record).items():
                    data.setdefault((pid, signal), []).extend(rows)

    # Write output: output_dir/<participant_id>/<signal>.csv
    participants_written = set()
    for (pid, signal) in sorted(data.keys()):
        rows = data[(pid, signal)]
        participant_dir = output_dir / pid
        participant_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(rows)
        df.sort_values("timestamp_utc", inplace=True)
        df.reset_index(drop=True, inplace=True)

        out_path = participant_dir / f"{signal}.csv"
        df.to_csv(out_path, index=False)
        print(f"  → {out_path}  ({len(df):,} rows)")
        participants_written.add(pid)

    # Compute heart rate and HRV from systolic peaks
    all_pids = {pid for (pid, _) in data.keys()}
    for pid in sorted(all_pids):
        peaks_rows = data.get((pid, "systolic_peaks"), [])
        if not peaks_rows:
            continue

        hr_df, hrv_df = compute_hr_hrv(peaks_rows)
        participant_dir = output_dir / pid

        if not hr_df.empty:
            out_path = participant_dir / "hr.csv"
            hr_df.to_csv(out_path, index=False)
            print(f"  → {out_path}  ({len(hr_df):,} rows)")

        if not hrv_df.empty:
            out_path = participant_dir / "hrv.csv"
            hrv_df.to_csv(out_path, index=False)
            print(f"  → {out_path}  ({len(hrv_df):,} rows)")

    print(f"\nDone. {len(participants_written)} participant folder(s) written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Empatica Avro files to per-participant CSVs.")
    parser.add_argument("--data", type=Path, default=DATA_ROOT,  help="Root directory of Avro files")
    parser.add_argument("--out",  type=Path, default=None, help="Output directory for CSVs (default: participant_csv/ alongside participant_data/)")
    args = parser.parse_args()
    main(args.data, args.out)
