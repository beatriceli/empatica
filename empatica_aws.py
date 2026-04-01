import argparse
import boto3
import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_s3_url = os.getenv("S3_URL", "").lstrip("s3://")
BUCKET, _, PREFIX = _s3_url.partition("/")
_SCRIPT_DIR = Path(__file__).parent

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)


def list_keys(bucket, prefix, since: date | None = None, skip_metadata: bool = False):
    """Yield all .avro and .csv object keys under prefix, handling pagination.

    If since is provided, keys that contain a YYYY-MM-DD path segment are only
    yielded when that date >= since. Keys without a date segment (e.g. metadata)
    are always yielded unless skip_metadata is True.
    """
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not (key.endswith(".avro") or key.endswith(".csv")):
                continue
            has_date = False
            if since is not None or skip_metadata:
                for part in key.split("/"):
                    try:
                        key_date = date.fromisoformat(part)
                    except ValueError:
                        continue
                    has_date = True
                    # Found a date segment — apply the since filter
                    if since is not None and key_date < since:
                        key = None
                    break
                if key is None:
                    continue
                if skip_metadata and not has_date:
                    continue
            yield key


def download_file(bucket, key, local_root):
    """Download a single S3 object, stripping the S3 prefix and leading numeric path segments."""
    parts = key[len(PREFIX):].lstrip("/").split("/")
    while parts and parts[0].isdigit():
        parts.pop(0)
    local_path = local_root / Path(*parts)
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_path))


def main():
    parser = argparse.ArgumentParser(description="Download Empatica data files from S3.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR,
        metavar="DIR",
        help="Directory to save downloaded files (default: script directory).",
    )
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="Only download participant files on or after this date (e.g. 2026-03-01). Metadata is always downloaded unless --skip-metadata is set.",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip files without a date path segment (e.g. metadata folders).",
    )
    args = parser.parse_args()

    local_root = args.output_dir.expanduser() / "empatica_raw"

    since_label = f" (since {args.since})" if args.since else ""
    print(f"Listing s3://{BUCKET}/{PREFIX}{since_label} ...")
    keys = list(list_keys(BUCKET, PREFIX, since=args.since, skip_metadata=args.skip_metadata))
    print(f"Found {len(keys)} files\n")

    for key in keys:
        download_file(BUCKET, key, local_root)

    print(f"\nDone. Files saved under: {local_root.resolve()}")


if __name__ == "__main__":
    main()
