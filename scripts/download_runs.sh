#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/fastq

echo '=== Fetching SRR17478151 ===' && prefetch --max-size 50G SRR17478151 && fasterq-dump --split-files --threads 4 --outdir data/raw/fastq SRR17478151 && gzip -f data/raw/fastq/SRR17478151_*.fastq
echo '=== Fetching SRR17478152 ===' && prefetch --max-size 50G SRR17478152 && fasterq-dump --split-files --threads 4 --outdir data/raw/fastq SRR17478152 && gzip -f data/raw/fastq/SRR17478152_*.fastq
echo '=== Fetching SRR17478153 ===' && prefetch --max-size 50G SRR17478153 && fasterq-dump --split-files --threads 4 --outdir data/raw/fastq SRR17478153 && gzip -f data/raw/fastq/SRR17478153_*.fastq
echo '=== Fetching SRR17478154 ===' && prefetch --max-size 50G SRR17478154 && fasterq-dump --split-files --threads 4 --outdir data/raw/fastq SRR17478154 && gzip -f data/raw/fastq/SRR17478154_*.fastq
