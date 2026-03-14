"""
hr.py
-----
Extract heart rate and HRV from per-participant, per-session CSVs using NeuroKit2.

Reads systolic_peaks.csv (on-device peak timestamps from the EMBRACE+),
applies nk.signal_fixpeaks() for artifact correction, then computes
beat-by-beat HR and rolling HRV (RMSSD, SDNN, pNN50) over a 30-beat window.

Input (participant_csv/<pid>/<YYYY-MM-DD_HHMM>/):
  systolic_peaks.csv   columns: timestamp_utc, timestamp_local

Output (participant_csv/<pid>/<YYYY-MM-DD_HHMM>/):
  hr.csv               columns: timestamp_utc, timestamp_local, ibi_s, bpm
  hrv.csv              columns: timestamp_utc, timestamp_local, rmssd_ms, sdnn_ms, pnn50

Usage:
  python hr.py                       # auto-discovers participant_csv/ under ~/Downloads/empatica_raw
  python hr.py --data path/to/participant_csv
"""

import argparse
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd

HRV_WINDOW_BEATS = 30

DATA_ROOT = Path.home() / "Downloads" / "empatica_raw"


def compute_hr_hrv(peaks_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute beat-by-beat HR and rolling HRV from systolic_peaks.csv.

    Converts peak timestamps to a 1 ms virtual sample grid, applies
    nk.signal_fixpeaks() for artifact correction, then computes IBIs,
    HR, and rolling HRV metrics.
    """
    df = pd.read_csv(peaks_path)
    df["timestamp_utc"]   = pd.to_datetime(df["timestamp_utc"],   utc=True)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], utc=True)
    df.sort_values("timestamp_utc", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 2:
        return pd.DataFrame(), pd.DataFrame()

    t_ns  = df["timestamp_utc"].astype("int64").to_numpy()
    t0_ns = t_ns[0]
    tz_ns = int((df["timestamp_local"].iloc[0] - df["timestamp_utc"].iloc[0]).total_seconds() * 1e9)

    # Convert to sample indices on a 1000 Hz (1 ms) virtual grid
    peak_indices = np.round((t_ns - t0_ns) / 1e6).astype(int)

    _, peaks_clean = nk.signal_fixpeaks(
        peak_indices, sampling_rate=1000, iterative=True, show=False
    )

    # Reconstruct timestamps from corrected indices
    corrected_ns = (peaks_clean / 1000.0 * 1e9 + t0_ns).astype("int64")
    utc_clean   = pd.to_datetime(corrected_ns, unit="ns", utc=True)
    local_clean = utc_clean + pd.Timedelta(nanoseconds=tz_ns)

    # At 1000 Hz, sample difference == milliseconds
    ibi_ms = np.diff(peaks_clean).astype(float)
    ibi_s  = ibi_ms / 1000.0

    # --- Heart rate ---------------------------------------------------------
    hr_df = pd.DataFrame({
        "timestamp_utc":   utc_clean[1:],
        "timestamp_local": local_clean[1:],
        "ibi_s":           ibi_s,
        "bpm":             60.0 / ibi_s,
    })

    # --- HRV (rolling window) -----------------------------------------------
    n = len(ibi_ms)
    rmssd_vals = np.full(n, np.nan)
    sdnn_vals  = np.full(n, np.nan)
    pnn50_vals = np.full(n, np.nan)

    for i in range(HRV_WINDOW_BEATS - 1, n):
        w     = ibi_ms[i - HRV_WINDOW_BEATS + 1 : i + 1]
        diffs = np.diff(w)
        rmssd_vals[i] = np.sqrt(np.mean(diffs ** 2))
        sdnn_vals[i]  = np.std(w, ddof=1)
        pnn50_vals[i] = np.mean(np.abs(diffs) > 50.0) * 100.0

    hrv_df = pd.DataFrame({
        "timestamp_utc":   utc_clean[1:],
        "timestamp_local": local_clean[1:],
        "rmssd_ms":        rmssd_vals,
        "sdnn_ms":         sdnn_vals,
        "pnn50":           pnn50_vals,
    })
    hrv_df = hrv_df.dropna(subset=["rmssd_ms"]).reset_index(drop=True)

    return hr_df, hrv_df


def process_session(session_dir: Path) -> None:
    peaks_path = session_dir / "systolic_peaks.csv"

    if not peaks_path.exists():
        print(f"    [skip] {session_dir.name}: no systolic_peaks.csv")
        return

    hr_df, hrv_df = compute_hr_hrv(peaks_path)

    if not hr_df.empty:
        hr_df.to_csv(session_dir / "hr.csv", index=False)
        print(f"    → {session_dir / 'hr.csv'}  ({len(hr_df):,} rows)")

    if not hrv_df.empty:
        hrv_df.to_csv(session_dir / "hrv.csv", index=False)
        print(f"    → {session_dir / 'hrv.csv'}  ({len(hrv_df):,} rows)")


def process_participant(pid_dir: Path) -> None:
    session_dirs = sorted(p for p in pid_dir.iterdir() if p.is_dir())
    if not session_dirs:
        print(f"  [skip] {pid_dir.name}: no session folders")
        return

    for session_dir in session_dirs:
        print(f"  {session_dir.name}")
        process_session(session_dir)


def main(csv_root: Path) -> None:
    pid_dirs = sorted(p for p in csv_root.iterdir() if p.is_dir())
    if not pid_dirs:
        print(f"No participant folders found under {csv_root}")
        return

    print(f"Processing {len(pid_dirs)} participant(s) from {csv_root}\n")
    for pid_dir in pid_dirs:
        print(f"{pid_dir.name}")
        process_participant(pid_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract HR/HRV from participant CSVs using NeuroKit2."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to participant_csv/ directory (default: auto-discovered under ~/Downloads/empatica_raw)",
    )
    args = parser.parse_args()

    if args.data is not None:
        csv_root = args.data
    else:
        candidate = next(
            (p.parent / "participant_csv"
             for p in DATA_ROOT.glob("**/participant_data")
             if p.is_dir() and (p.parent / "participant_csv").exists()),
            None,
        )
        if candidate is None:
            print(f"Could not find participant_csv/ under {DATA_ROOT}. Use --data to specify the path.")
            raise SystemExit(1)
        csv_root = candidate

    main(csv_root)
