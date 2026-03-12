import argparse
import boto3
import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_s3_url = os.getenv("S3_URL", "").lstrip("s3://")
BUCKET, _, PREFIX = _s3_url.partition("/")
LOCAL_ROOT = Path.home() / "Downloads" / "empatica_raw"

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)


def list_keys(bucket, prefix, since: date | None = None):
    """Yield all .avro and .csv object keys under prefix, handling pagination.

    If since is provided, keys that contain a YYYY-MM-DD path segment are only
    yielded when that date >= since. Keys without a date segment (e.g. metadata)
    are always yielded.
    """
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not (key.endswith(".avro") or key.endswith(".csv")):
                continue
            if since is not None:
                for part in key.split("/"):
                    try:
                        key_date = date.fromisoformat(part)
                    except ValueError:
                        continue
                    # Found a date segment — apply the filter
                    if key_date < since:
                        key = None
                    break
                if key is None:
                    continue
            yield key


def download_file(bucket, key, local_root):
    """Download a single S3 object, stripping the S3 prefix from the local path."""
    relative_key = key[len(PREFIX):]
    local_path = local_root / relative_key
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_path))


def main():
    parser = argparse.ArgumentParser(description="Download Empatica data files from S3.")
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="Only download participant files on or after this date (e.g. 2026-03-01). Metadata is always downloaded.",
    )
    args = parser.parse_args()

    since_label = f" (since {args.since})" if args.since else ""
    print(f"Listing s3://{BUCKET}/{PREFIX}{since_label} ...")
    keys = list(list_keys(BUCKET, PREFIX, since=args.since))
    print(f"Found {len(keys)} files\n")

    for key in keys:
        download_file(BUCKET, key, LOCAL_ROOT)

    print(f"\nDone. Files saved under: {LOCAL_ROOT.resolve()}")


if __name__ == "__main__":
    main()
