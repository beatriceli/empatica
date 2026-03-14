"""
avro-csv.py
-----------
Read Empatica EMBRACE+ Avro files, unpack each signal into a time-series
and write one CSV per signal inside a session folder named after the participant.

Output layout:
  OUTPUT_DIR/
    <participant_id>/
      <YYYY-MM-DD_HHMM>/            ← session folder (local time of first file)
        accelerometer.csv   (columns: timestamp_utc, timestamp_local, x, y, z)
        eda.csv             (columns: timestamp_utc, timestamp_local, value)
        temperature.csv
        bvp.csv
        steps.csv
        systolic_peaks.csv
        tags.csv

Sessions are detected by gaps in the actual signal data between consecutive Avro
files. A gap > SESSION_GAP_SECONDS between the last sample of one file and the
first sample of the next is treated as a session boundary.

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
from dataclasses import dataclass, field
from pathlib import Path

import fastavro
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_ROOT = Path.home() / "Downloads" / "empatica_raw"
SESSION_GAP_SECONDS = 30 * 60   # 30-minute data gap → new session

# --------------------------------------------------------------------------- #
# Signal helpers
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
# Session detection
# --------------------------------------------------------------------------- #

@dataclass
class FileInfo:
    path: Path
    t_start: pd.Timestamp          # earliest data timestamp in this file
    t_end: pd.Timestamp            # latest data timestamp in this file
    tz_offset: int                 # seconds (e.g. -14400 for UTC-4)
    data: dict = field(default_factory=dict)   # (pid, signal) -> list[dict]


def read_avro_file(avro_path: Path, participant_folder: str) -> FileInfo | None:
    """Read one Avro file, returning its data and actual data time range."""
    data: dict[tuple[str, str], list[dict]] = {}
    tz_offset = 0

    with open(avro_path, "rb") as f:
        for record in fastavro.reader(f):
            tz_offset = record.get("timezone", 0)
            pid = (record.get("enrollment", {}) or {}).get("participantID")
            if not pid and pid != 0:
                pid = participant_folder
            pid = str(pid)

            for signal, rows in unpack_record(record).items():
                data.setdefault((pid, signal), []).extend(rows)

    if not data:
        return None

    # Derive time range from actual data (first/last of each signal)
    all_ts = []
    for rows in data.values():
        if rows:
            all_ts.append(rows[0]["timestamp_utc"])
            all_ts.append(rows[-1]["timestamp_utc"])

    if not all_ts:
        return None

    return FileInfo(
        path=avro_path,
        t_start=min(all_ts),
        t_end=max(all_ts),
        tz_offset=tz_offset,
        data=data,
    )


def group_into_sessions(file_infos: list[FileInfo]) -> list[list[FileInfo]]:
    """
    Group FileInfos (sorted by t_start) into sessions.
    A new session starts when the gap between the end of the previous file
    and the start of the next exceeds SESSION_GAP_SECONDS.
    """
    gap = pd.Timedelta(seconds=SESSION_GAP_SECONDS)
    sessions: list[list[FileInfo]] = []
    current: list[FileInfo] = []

    for fi in sorted(file_infos, key=lambda x: x.t_start):
        if current and (fi.t_start - current[-1].t_end) > gap:
            sessions.append(current)
            current = []
        current.append(fi)

    if current:
        sessions.append(current)

    return sessions


def format_session_name(t_start: pd.Timestamp, tz_offset_s: int) -> str:
    """Return a folder name like '2026-03-11_2018' from a UTC timestamp and tz offset."""
    local = t_start + pd.Timedelta(seconds=tz_offset_s)
    return local.strftime("%Y-%m-%d_%H%M")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(data_root: Path, output_dir: Path | None) -> None:
    avro_files = sorted(data_root.glob("**/participant_data/*/*/raw_data/*/*.avro"))
    if not avro_files:
        print(f"No .avro files found under {data_root}")
        return

    if output_dir is None:
        participant_data_dir = next(p for p in avro_files[0].parents if p.name == "participant_data")
        output_dir = participant_data_dir.parent / "participant_csv"

    # Group avro files by participant folder
    by_participant: dict[str, list[Path]] = {}
    for f in avro_files:
        participant_folder = f.parts[-4]   # e.g. 0-3YK3K1526F
        by_participant.setdefault(participant_folder, []).append(f)

    print(f"Found {len(avro_files)} .avro file(s) across {len(by_participant)} participant(s).\n")

    participants_written = set()
    for participant_folder, files in sorted(by_participant.items()):
        # Read all files for this participant (single pass)
        file_infos = []
        for avro_path in files:
            fi = read_avro_file(avro_path, participant_folder)
            if fi is not None:
                file_infos.append(fi)

        if not file_infos:
            print(f"{participant_folder}: no data found, skipping")
            continue

        sessions = group_into_sessions(file_infos)
        print(f"{participant_folder}: {len(sessions)} session(s)")

        for session_file_infos in sessions:
            first = session_file_infos[0]
            session_name = format_session_name(first.t_start, first.tz_offset)

            # Merge data across all files in this session
            data: dict[tuple[str, str], list[dict]] = {}
            for fi in session_file_infos:
                for key, rows in fi.data.items():
                    data.setdefault(key, []).extend(rows)

            # Write output: output_dir/<pid>/<session_name>/<signal>.csv
            for (pid, signal) in sorted(data.keys()):
                rows = data[(pid, signal)]
                session_dir = output_dir / pid / session_name
                session_dir.mkdir(parents=True, exist_ok=True)

                df = pd.DataFrame(rows)
                df.sort_values("timestamp_utc", inplace=True)
                df.reset_index(drop=True, inplace=True)

                out_path = session_dir / f"{signal}.csv"
                df.to_csv(out_path, index=False)
                print(f"  → {out_path}  ({len(df):,} rows)")
                participants_written.add(pid)

    print(f"\nDone. {len(participants_written)} participant folder(s) written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Empatica Avro files to per-participant CSVs.")
    parser.add_argument("--data", type=Path, default=DATA_ROOT,  help="Root directory of Avro files")
    parser.add_argument("--out",  type=Path, default=None, help="Output directory for CSVs (default: participant_csv/ alongside participant_data/)")
    args = parser.parse_args()
    main(args.data, args.out)
