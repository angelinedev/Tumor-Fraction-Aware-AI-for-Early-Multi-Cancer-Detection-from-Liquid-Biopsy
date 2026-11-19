# Data Directory Organization & Access Guidelines

This directory houses sequencing metadata, raw sequence/alignment files, and processed fragmentomic feature matrices for the **Tumor-Fraction-Aware AI for Early Multi-Cancer Detection from Liquid Biopsy** project.

---

## Directory Hierarchy

```
data/
├── metadata/              # Run tables, clinical annotations, and sample mapping manifests
│   ├── SraRunTable.csv    # SRA metadata downloaded from NCBI Run Selector
│   ├── sample_sheet.csv   # Unified study sample sheet (SampleID, PatientID, CancerType, Stage, BaselineTF)
│   └── .gitkeep
├── raw/                   # Raw sequencing archives and BAM alignments (NOT tracked in git)
│   ├── fastq/             # Demultiplexed FASTQ reads (.fastq.gz)
│   ├── bam/               # Quality-filtered coordinate-sorted BAM files (.bam, .bam.bai)
│   └── .gitkeep
└── processed/             # Extracted features, dilution tables, and training splits
    ├── dilutions/         # In silico titrated BAM files across tumor fractions
    ├── features/          # ichorCNA segment tables & FinaleToolkit fragmentomic TSVs
    ├── merged_matrix.parquet # Unified multimodal feature matrix with patient labels
    └── .gitkeep
```

---

## Cohort Descriptions & Access Modalities

### 1. DELFI Multi-Cancer cfDNA Cohort (`PRJNA795275` / `GSE126044`)
- **Biological Description:** Cell-free DNA whole-genome sequencing (low-pass ~1–2x depth) from healthy individuals and patients across 7 cancer types (breast, colorectal, lung, ovarian, pancreatic, gastric, and bile duct).
- **Source:** NCBI SRA (BioProject `PRJNA795275`).
- **Access Protocol:** Publicly accessible. Download using `scripts/download_sra.py` or NCBI SRA Toolkit (`prefetch` and `fasterq-dump`).

### 2. Nucleosome Footprinting & cfDNA sWGS (`GSE71378`)
- **Biological Description:** Snyder et al. (Cell 2016) low-pass sequencing of healthy donors and late-stage cancer patients with established copy-number alterations and nucleosome protection metrics.
- **Source:** NCBI GEO (`GSE71378`).
- **Access Protocol:** Publicly available SRA runs.

### 3. EGA Validation Cohort (`EGAD00001006869`)
- **Biological Description:** Controlled-access independent European cohort of low-pass WGS cfDNA.
- **Source:** European Genome-phenome Archive (EGA).
- **Access Protocol:** Requires institutional Data Access Committee (DAC) approval via EGA credentials and `pyega3` downloader.

### 4. TCGA Tissue-of-Origin References
- **Biological Description:** The Cancer Genome Atlas segment copy-number calls (`GDC` harmonized hg38) and recurrent arm-level gain/loss frequencies used as tissue-of-origin prior distributions.
- **Source:** NCI Genomic Data Commons (GDC) Portal.

---

## Metadata Specification: `data/metadata/sample_sheet.csv`

Every sample processed by this pipeline must have a matching row in the sample sheet with the following standard columns:

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `sample_id` | `str` | Unique sequencing run or aliquot identifier | `SRR12345678` |
| `patient_id` | `str` | Anonymized patient/donor ID (critical for leakage prevention) | `PATIENT_042` |
| `cohort` | `str` | Dataset source identifier | `PRJNA795275` |
| `condition` | `str` | Primary diagnostic classification (`Healthy` vs `Cancer`) | `Cancer` |
| `cancer_type` | `str` | Specific cancer lineage (or `Healthy`) | `Colorectal`, `Lung`, `Breast` |
| `stage` | `str` | Clinical stage (`Stage I`, `Stage II`, `Stage III`, `Stage IV`, `NA`) | `Stage I` |
| `measured_tf` | `float` | Estimated baseline tumor fraction (via orthogonal assay or ichorCNA) | `0.082` |
| `bam_path` | `str` | Relative path to sorted, indexed BAM file | `data/raw/bam/SRR12345678.sorted.bam` |

---

## Storage & Computational Warnings

> [!WARNING]
> Raw sequencing reads (`.fastq.gz`) and aligned BAM files (`.bam`) require substantial disk storage:
> - Low-pass WGS (~1x to 2x): ~1.5 to 3.0 GB per sample.
> - A pilot cohort of 20 samples requires ~60 GB storage; a full cohort of 500 samples requires ~1.2 TB.
> - Ensure your workspace disk or mount has adequate scratch capacity before triggering mass SRA downloads.
