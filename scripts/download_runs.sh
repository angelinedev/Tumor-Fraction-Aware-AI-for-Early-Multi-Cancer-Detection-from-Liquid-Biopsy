#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/fastq

echo '=== Fetching SRR12340001 ===' && prefetch --max-size 50G SRR12340001 && fasterq-dump --split-files --threads 4 --outdir data/raw/fastq SRR12340001 && gzip -f data/raw/fastq/SRR12340001_*.fastq
echo '=== Fetching SRR12340002 ===' && prefetch --max-size 50G SRR12340002 && fasterq-dump --split-files --threads 4 --outdir data/raw/fastq SRR12340002 && gzip -f data/raw/fastq/SRR12340002_*.fastq
