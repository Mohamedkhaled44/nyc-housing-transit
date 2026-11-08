"""
Download the two NYC Open Data datasets used in this project.

Housing & Development : hg8x-zxpr  Affordable Housing Production by Building
Transportation        : h9gi-nx95  Motor Vehicle Collisions - Crashes

The crash dataset holds ~2.2M rows in total, which will not fit comfortably in a
notebook, so we pull a justified, reproducible slice: calendar years 2018-2025.
The Socrata API caps a single response at 1,000,000 rows, and our slice is
1,043,903 rows, so the download is paged with $offset.  A stable $order=:id is
used so that the paging cannot skip or duplicate rows between requests.

Raw files land in data/raw/ (git-ignored).  Re-running this script is cheap:
files that already exist are skipped.
"""

import os
import sys
import time
from urllib.parse import urlencode

import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

HOUSING_ID = "hg8x-zxpr"
CRASH_ID = "h9gi-nx95"

# Analysis window: 8 complete calendar years.
# 2026 is excluded from both datasets because it is a partial year and would
# fabricate a downward trend in any time-series chart.
START = "2018-01-01T00:00:00"
END = "2026-01-01T00:00:00"

PAGE_SIZE = 250_000
RESOURCE = "https://data.cityofnewyork.us/resource/{}.csv"


def fetch_csv(dataset_id, params, dest, description):
    """Download one Socrata CSV page to `dest` unless it is already present."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [skip] {description} already downloaded -> {os.path.basename(dest)}")
        return
    url = RESOURCE.format(dataset_id) + "?" + urlencode(params)
    print(f"  [get ] {description} ...", flush=True)
    for attempt in range(1, 4):
        try:
            with requests.get(url, stream=True, timeout=600) as response:
                response.raise_for_status()
                tmp = dest + ".part"
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
                os.replace(tmp, dest)
            size_mb = os.path.getsize(dest) / 1e6
            print(f"  [ok  ] {description} -> {os.path.basename(dest)} ({size_mb:.1f} MB)")
            return
        except Exception as exc:  # network hiccups are common on large pulls
            print(f"  [warn] attempt {attempt} failed: {exc}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not download {description} after 3 attempts")


def download_housing():
    print("Housing & Development - Affordable Housing Production by Building")
    dest = os.path.join(RAW_DIR, "housing_raw.csv")
    # ~9.5k rows in total, small enough to take whole; no filter applied so the
    # notebook can show the full picture and justify its own date trimming.
    fetch_csv(HOUSING_ID, {"$limit": 50_000, "$order": ":id"}, dest, "housing (all rows)")
    return dest


def download_crashes():
    print("Transportation - Motor Vehicle Collisions: Crashes")
    where = f"crash_date >= '{START}' AND crash_date < '{END}'"
    pages = []
    for page_index in range(10):  # generous upper bound; loop breaks when short
        offset = page_index * PAGE_SIZE
        dest = os.path.join(RAW_DIR, f"crashes_raw_p{page_index}.csv")
        params = {
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$where": where,
            "$order": ":id",  # stable order makes the paging reproducible
        }
        fetch_csv(CRASH_ID, params, dest, f"crashes page {page_index} (offset {offset:,})")
        # Count rows with the CSV parser, not by counting lines: several text
        # columns contain embedded newlines, so a line count badly overstates it.
        rows = len(pd.read_csv(dest, usecols=["collision_id"]))
        pages.append(dest)
        print(f"         page {page_index}: {rows:,} rows")
        if rows < PAGE_SIZE:
            break
    return pages


def main():
    housing_path = download_housing()
    crash_pages = download_crashes()

    combined = os.path.join(RAW_DIR, "crashes_raw.csv")
    if not os.path.exists(combined):
        print("Concatenating crash pages ...")
        frames = [pd.read_csv(p, low_memory=False) for p in crash_pages]
        crashes = pd.concat(frames, ignore_index=True)
        # Paging with a stable $order should not duplicate, but verify.
        before = len(crashes)
        crashes = crashes.drop_duplicates(subset="collision_id")
        print(f"  rows {before:,} -> {len(crashes):,} after de-duplicating collision_id")
        crashes.to_csv(combined, index=False)
    print(f"\nDone.\n  housing : {housing_path}\n  crashes : {combined}")


if __name__ == "__main__":
    sys.exit(main())
