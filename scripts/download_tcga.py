#!/usr/bin/env python3
"""
download_tcga.py
----------------
Automated downloader for TCGA Masked Copy Number Segment files from the NCI GDC API.
Downloads into references/tcga_cnv/ relative to the repository root.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "references" / "tcga_cnv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Session with automatic retries for resilient connection
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "cases.project.program.name", "value": "TCGA"}},
        {"op": "=", "content": {"field": "data_category", "value": "Copy Number Variation"}},
        {"op": "=", "content": {"field": "data_type", "value": "Masked Copy Number Segment"}},
        {"op": "=", "content": {"field": "cases.samples.tissue_type", "value": "tumor"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ],
}

params = {
    "filters": json.dumps(filters),
    "fields": "file_id,file_name,cases.project.project_id",
    "format": "JSON",
    "size": "4000",
}


def download_one(item: dict) -> str:
    fid = item["file_id"]
    proj = item["cases"][0]["project"]["project_id"]
    fname = f"{proj}_{item['file_name']}"
    out_path = OUT_DIR / fname
    if out_path.exists():
        return fname
    r = session.get(f"https://api.gdc.cancer.gov/data/{fid}", timeout=30)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return fname


def main():
    resp = session.get("https://api.gdc.cancer.gov/files", params=params, timeout=30)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]
    print(f"Found {len(hits)} available files. Downloading up to 500 profiles...")

    target = hits[:500]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download_one, item) for item in target]
        for idx, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if idx % 100 == 0 or idx == len(target):
                print(f"Progress: {idx}/{len(target)} completed.")

    print(f"TCGA CNV files stored in {OUT_DIR}")


if __name__ == "__main__":
    main()
