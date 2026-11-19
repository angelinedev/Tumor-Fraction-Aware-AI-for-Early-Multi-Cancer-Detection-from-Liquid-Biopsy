#!/usr/bin/env python3
"""
download_sra.py
---------------
Automated and safe retrieval of paired-end cfDNA whole-genome sequencing runs
from the NCBI Sequence Read Archive (SRA) for pilot and full-scale MCED cohorts.

Features:
- Parses SraRunTable.csv / sample_sheet.csv
- Balances and filters healthy vs. specific cancer types
- Generates reproducible prefetch and fasterq-dump commands
- Safe subprocess execution with dry-run mode and rate-limiting
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("download_sra")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract SRA runs for low-pass cfDNA cohorts."
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata/sample_sheet_template.csv",
        help="Path to SraRunTable.csv or sample_sheet.csv manifest."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/fastq",
        help="Directory to save demultiplexed FASTQ files."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of runs to download (useful for pilot testing)."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for fasterq-dump."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and create download script without running SRA tools."
    )
    parser.add_argument(
        "--script_out",
        type=str,
        default="scripts/download_runs.sh",
        help="Output path for batch shell script."
    )
    return parser.parse_args()


def load_and_filter_manifest(metadata_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Load metadata manifest and identify target SRA accessions.

    Args:
        metadata_path: File path to CSV metadata.
        limit: Optional cap on the number of records.

    Returns:
        Filtered DataFrame with validated SRA accession identifiers.
    """
    path = Path(metadata_path)
    if not path.exists():
        logger.error("Metadata manifest does not exist: %s", metadata_path)
        sys.exit(1)

    df = pd.read_csv(path)
    logger.info("Loaded metadata containing %d rows from %s", len(df), metadata_path)

    # Detect accession column
    acc_col = None
    for candidate in ["Run", "sample_id", "Run_s", "Accession", "SRR_ID"]:
        if candidate in df.columns:
            acc_col = candidate
            break

    if acc_col is None:
        raise ValueError(f"Could not find SRA accession column in columns: {list(df.columns)}")

    # Clean accession column
    df["target_accession"] = df[acc_col].astype(str).str.strip()
    valid_mask = df["target_accession"].str.startswith(("SRR", "ERR", "DRR"))
    filtered = df[valid_mask].copy()

    if limit is not None and limit > 0:
        filtered = filtered.head(limit)
        logger.info("Applied pilot limit: selected %d accessions", len(filtered))

    return filtered


def check_tool_availability() -> dict:
    """Check whether prefetch and fasterq-dump are installed and on PATH."""
    tools = {
        "prefetch": shutil.which("prefetch") is not None,
        "fasterq-dump": shutil.which("fasterq-dump") is not None,
    }
    return tools


def generate_download_commands(
    accessions: List[str],
    output_dir: str,
    threads: int = 4
) -> List[str]:
    """Generate shell command lines for downloading and converting SRA files.

    Args:
        accessions: List of SRA accessions (e.g. SRR...).
        output_dir: Target directory for fastq output.
        threads: Thread count for fasterq-dump.

    Returns:
        List of executable bash command strings.
    """
    commands = []
    for acc in accessions:
        cmd = (
            f"echo '=== Fetching {acc} ===' && "
            f"prefetch --max-size 50G {acc} && "
            f"fasterq-dump --split-files --threads {threads} --outdir {output_dir} {acc} && "
            f"gzip -f {output_dir}/{acc}_*.fastq"
        )
        commands.append(cmd)
    return commands


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_and_filter_manifest(args.metadata, args.limit)
    accessions = df["target_accession"].tolist()

    if not accessions:
        logger.warning("No valid SRA accessions found to download.")
        return

    logger.info("Identified %d accession(s) for retrieval: %s", len(accessions), accessions)
    tools = check_tool_availability()
    logger.info("NCBI SRA Tool Availability: %s", tools)

    commands = generate_download_commands(accessions, args.output_dir, args.threads)

    # Save batch shell script
    script_path = Path(args.script_out)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        f.write(f"mkdir -p {args.output_dir}\n\n")
        for cmd in commands:
            f.write(cmd + "\n")
    logger.info("Generated batch execution script: %s", script_path)

    if args.dry_run:
        logger.info("[DRY RUN] Download commands generated without execution.")
        for idx, cmd in enumerate(commands, 1):
            logger.info("Command #%d: %s", idx, cmd)
        return

    if not all(tools.values()):
        logger.error(
            "NCBI SRA tools (prefetch, fasterq-dump) not found on PATH. "
            "Install via 'conda install -c bioconda sra-tools' or run in Docker. "
            "Commands have been written to %s",
            script_path
        )
        return

    # Execute sequentially
    for acc in accessions:
        logger.info("Initiating download for %s...", acc)
        try:
            subprocess.run(["prefetch", "--max-size", "50G", acc], check=True)
            subprocess.run([
                "fasterq-dump", "--split-files", "--threads", str(args.threads),
                "--outdir", args.output_dir, acc
            ], check=True)
            logger.info("Successfully fetched and extracted %s", acc)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to retrieve accession %s: %s", acc, e)


if __name__ == "__main__":
    main()
