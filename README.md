# empatica
Retrieve and parse raw data files from Empatica Cloud.

## Overview

Two scripts cover the full pipeline:

1. **`empatica_aws.py`** — downloads raw Avro files from S3
2. **`avro-csv.py`** — converts those Avro files into per-participant CSVs

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials

Create a `.env` file in the project root:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

S3_URL=s3://your-bucket/your-prefix/
```

---

## Scripts

### `empatica_aws.py`

Downloads `.avro` files from S3 into `~/Downloads/empatica_raw/`, mirroring the S3 key structure. Skips files that already exist locally.

**Usage**
```bash
# Download all available data
python empatica_aws.py

# Download only files on or after a given date (inclusive)
python empatica_aws.py --since 2026-03-01
```

**Arguments**

| Argument | Type | Description |
|----------|------|-------------|
| `--since` | `YYYY-MM-DD` | Optional. Only download files from this date onwards. |

**Output layout**

Mirrors the S3 key structure:
```
~/Downloads/empatica_raw/
  [study_id]/[site_id]/
    metadata/
      [metadata].csv
    participant_data/
      [YYYY-MM-DD]/
        [participant_id]-[EmbracePlus_sn]/
          digital_biomarkers/
            aggregated_per_minute/
              [aggregated].csv
          raw_data/
            [schema_version]/
              [filename].avro
```

---

### `avro-csv.py`

Reads all `.avro` files from `~/Downloads/empatica_raw/` and writes one CSV per signal per participant into `participant_csv/` alongside `participant_data/`.

**Usage**
```bash
# Use default input/output directories
python avro-csv.py

# Override directories
python avro-csv.py --data path/to/avro/root --out path/to/csv/output
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `~/Downloads/empatica_raw` | Root directory of Avro files |
| `--out` | `participant_csv/` alongside `participant_data/` | Output directory for CSVs |

**Output layout**
```
[same level as participant_data]/
  participant_csv/
    <participant_id>/
    accelerometer.csv
    eda.csv
    temperature.csv
    bvp.csv
    steps.csv
    systolic_peaks.csv
    tags.csv
    hr.csv
    hrv.csv
```

**Signal columns**

| File | Columns |
|------|---------|
| `accelerometer.csv` | `timestamp_utc`, `timestamp_local`, `x`, `y`, `z` |
| `eda.csv` | `timestamp_utc`, `timestamp_local`, `value` |
| `temperature.csv` | `timestamp_utc`, `timestamp_local`, `value` |
| `bvp.csv` | `timestamp_utc`, `timestamp_local`, `value` |
| `steps.csv` | `timestamp_utc`, `timestamp_local`, `value` |
| `systolic_peaks.csv` | `timestamp_utc`, `timestamp_local` |
| `tags.csv` | `timestamp_utc`, `timestamp_local` |
| `hr.csv` | `timestamp_utc`, `timestamp_local`, `ibi_s`, `bpm` |
| `hrv.csv` | `timestamp_utc`, `timestamp_local`, `rmssd_ms`, `sdnn_ms`, `pnn50` |

---

## Heart rate & HRV processing

Heart rate and HRV are derived from the systolic peak timestamps detected by the EMBRACE+ PPG sensor. Each inter-beat interval (IBI) is the elapsed time between two consecutive peaks.

### IBI artefact correction

Raw PPG IBIs contain artefacts from motion, poor skin contact, and missed or false peaks. Two correction stages are applied before any HR or HRV metric is computed:

**Stage 1 — Hard physiological bounds**
IBIs outside [300 ms, 2000 ms] (30–200 BPM) are physiologically impossible and are replaced with NaN.

**Stage 2 — Local median threshold**
Each IBI is compared to the median of the 11 surrounding beats (centred window, ±5 beats). If it deviates by more than 20% from that local median it is flagged as an artefact and replaced with NaN. This catches ectopic beats, missed detections, and motion spikes that fall within the hard bounds but are inconsistent with the local rhythm.

Both stages are repaired by linear interpolation to preserve the beat timeline.

### HRV metrics

All metrics are computed over a rolling 30-beat window (~30 s at resting HR).

| Metric | Description |
|--------|-------------|
| `rmssd_ms` | Root mean square of successive IBI differences. Reflects short-term parasympathetic (vagal) activity. Most robust metric for short recordings and wearable data. |
| `sdnn_ms` | Standard deviation of IBIs over the window. Reflects overall HRV from both sympathetic and parasympathetic activity. |
| `pnn50` | Percentage of successive IBI pairs differing by > 50 ms. Parasympathetic index correlated with RMSSD. |

---

## Full pipeline

```bash
python empatica_aws.py --since 2026-03-01
python avro-csv.py
```
