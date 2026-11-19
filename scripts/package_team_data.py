#!/usr/bin/env python3
"""
package_team_data.py
--------------------
Bundles all processed features, reference TCGA CNV files, metadata manifests,
and benchmark results into a single clean ZIP archive (mced_project_data_bundle.zip)
so team members can get started immediately without downloading external files.
"""

import os
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_OUT = REPO_ROOT / "mced_project_data_bundle.zip"

FILES_AND_DIRS = [
    "data/metadata/sample_sheet.csv",
    "data/metadata/SraRunTable.csv",
    "data/processed/merged_features.parquet",
    "references/tcga_cnv",
    "results/metrics_by_tf.csv",
    "results/predictions.csv",
    "results/oof_xgboost.csv",
    "results/oof_lightgbm.csv",
    "results/oof_mlp.csv",
    "results/eda",
]

print(f"Packaging project data bundle to: {ZIP_OUT.name} ...")
with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zipf:
    for item in FILES_AND_DIRS:
        p = REPO_ROOT / item
        if not p.exists():
            print(f"Warning: {item} does not exist, skipping.")
            continue
        if p.is_file():
            rel_path = p.relative_to(REPO_ROOT).as_posix()
            zipf.write(p, rel_path)
            print(f"Added file: {rel_path}")
        elif p.is_dir():
            count = 0
            for root, _, files in os.walk(p):
                for file in files:
                    full_p = Path(root) / file
                    rel_path = full_p.relative_to(REPO_ROOT).as_posix()
                    zipf.write(full_p, rel_path)
                    count += 1
            print(f"Added directory: {item} ({count} files)")

zip_size_mb = os.path.getsize(ZIP_OUT) / (1024 * 1024)
print(f"\nSuccessfully created {ZIP_OUT.name}! Total compressed size: {zip_size_mb:.2f} MB")
print("Team members can extract this zip directly into their cloned repo root folder.")
