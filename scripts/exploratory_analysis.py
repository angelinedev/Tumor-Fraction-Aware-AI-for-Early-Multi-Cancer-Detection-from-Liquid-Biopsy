#!/usr/bin/env python3
"""
exploratory_analysis.py
-----------------------
Exploratory Data Analysis (EDA) of acquired liquid biopsy and reference cohorts:
1. PRJNA795275 cfDNA cohort (69 sWGS plasma runs across healthy donors & lung cancer patients)
2. TCGA Masked Copy Number Segment reference cohort (500 tumor tissue profiles: LUAD, KIRC, GBM)

Outputs generated:
- High-resolution figures in results/eda/
- Comprehensive statistical summary report
"""

import glob
import logging
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("eda")

# Set visualization aesthetic
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def run_cfdna_eda(sample_sheet_path: str, output_dir: Path) -> dict:
    """Analyze PRJNA795275 cfDNA cohort metadata."""
    logger.info("Analyzing cfDNA metadata: %s", sample_sheet_path)
    df = pd.read_csv(sample_sheet_path)

    stats = {
        "total_runs": len(df),
        "unique_patients": df["patient_id"].nunique(),
        "healthy_count": int((df["condition"] == "Healthy").sum()),
        "cancer_count": int((df["condition"] == "Cancer").sum()),
        "subtypes": df["cancer_type"].value_counts().to_dict(),
        "sex_dist": df["sex"].value_counts().to_dict(),
        "median_bytes_gb": float(df["bytes"].median() / 1e9),
        "mean_bytes_gb": float(df["bytes"].mean() / 1e9),
    }

    # Plot 1: Cohort Composition & Demographics
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Condition & subtype
    order = df["cancer_type"].value_counts().index
    sns.countplot(data=df, x="cancer_type", order=order, ax=axes[0], hue="cancer_type", palette="Set2", legend=False)
    axes[0].set_title("cfDNA Plasma Samples by Subtype", fontweight="bold")
    axes[0].set_xlabel("Sample Category")
    axes[0].set_ylabel("Number of Samples")
    axes[0].tick_params(axis="x", rotation=25)
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    # Sex distribution
    sns.countplot(data=df, x="sex", hue="condition", ax=axes[1], palette="tab10")
    axes[1].set_title("Demographics by Biological Sex", fontweight="bold")
    axes[1].set_xlabel("Sex")
    axes[1].set_ylabel("Count")

    # Patient replicate distribution (to demonstrate GroupKFold requirement)
    reps_per_pt = df["patient_id"].value_counts()
    multi_pts = reps_per_pt[reps_per_pt > 1]
    sns.barplot(x=multi_pts.index, y=multi_pts.values, ax=axes[2], color="#4C72B0")
    axes[2].set_title(f"Patients with Multi-Timepoint Replicates (N={len(multi_pts)})", fontweight="bold")
    axes[2].set_xlabel("Patient Donor ID")
    axes[2].set_ylabel("Sequencing Aliquots")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig_path1 = output_dir / "01_cfdna_cohort_composition.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    logger.info("Saved %s", fig_path1)

    # Plot 2: Sequencing Library Size Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df["gigabytes"] = df["bytes"] / 1e9
    sns.boxplot(data=df, x="cancer_type", y="gigabytes", ax=axes[0], hue="cancer_type", palette="Set2", legend=False)
    sns.stripplot(data=df, x="cancer_type", y="gigabytes", color="black", alpha=0.5, jitter=0.2, ax=axes[0])
    axes[0].set_title("Sequencing Data Volume per Sample (GB)", fontweight="bold")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Archive Size (GB)")
    axes[0].tick_params(axis="x", rotation=25)

    # Spot length (read length profile)
    sns.histplot(data=df, x="avg_spot_len", bins=15, kde=True, color="teal", ax=axes[1])
    axes[1].set_title("Sequencing Read Length Distribution (bp)", fontweight="bold")
    axes[1].set_xlabel("Average Spot Length (bp)")
    axes[1].set_ylabel("Sample Count")

    plt.tight_layout()
    fig_path2 = output_dir / "02_cfdna_library_depth.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    logger.info("Saved %s", fig_path2)

    return stats


def run_tcga_eda(tcga_dir: Path, output_dir: Path) -> dict:
    """Analyze TCGA Copy Number Segment reference files."""
    logger.info("Analyzing TCGA CNV files in: %s", tcga_dir)
    seg_files = list(tcga_dir.glob("*.seg.v2.txt"))

    project_counts = {}
    sample_dfs = []

    # Sample proportionally from each project
    project_files = {}
    for f in seg_files:
        proj = f.name.split("_")[0]
        project_files.setdefault(proj, []).append(f)

    for proj, pfiles in project_files.items():
        for f in pfiles[:35]:  # sample 35 files per project
            try:
                sdf = pd.read_csv(f, sep="\t")
                sdf["Project"] = proj
                sample_dfs.append(sdf)
            except Exception as e:
                logger.debug("Failed reading %s: %s", f.name, e)

    # Count all files across projects
    all_projects = pd.Series([f.name.split("_")[0] for f in seg_files]).value_counts().to_dict()

    merged_segs = pd.concat(sample_dfs, ignore_index=True) if sample_dfs else pd.DataFrame()
    logger.info("Aggregated %d segment rows from 100 sample profiles.", len(merged_segs))

    stats = {
        "total_tcga_files": len(seg_files),
        "projects_distribution": all_projects,
        "total_segments_inspected": len(merged_segs),
    }

    if not merged_segs.empty:
        # Standardize Chromosome identifiers (1-22)
        merged_segs["Chromosome"] = merged_segs["Chromosome"].astype(str).str.replace("chr", "", case=False)
        autosomes = [str(i) for i in range(1, 23)]
        merged_segs = merged_segs[merged_segs["Chromosome"].isin(autosomes)].copy()
        merged_segs["Chromosome"] = pd.Categorical(merged_segs["Chromosome"], categories=autosomes, ordered=True)

        stats["mean_segment_mean"] = float(merged_segs["Segment_Mean"].mean())
        stats["median_segment_mean"] = float(merged_segs["Segment_Mean"].median())
        stats["amplifications_count"] = int((merged_segs["Segment_Mean"] > 0.2).sum())
        stats["deletions_count"] = int((merged_segs["Segment_Mean"] < -0.2).sum())

        # Plot 3: TCGA Projects & Segment Amplitude
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Project distribution
        proj_series = pd.Series(all_projects)
        sns.barplot(x=proj_series.index, y=proj_series.values, ax=axes[0], hue=proj_series.index, palette="mako", legend=False)
        axes[0].set_title("TCGA Reference Profiles Downloaded (N=500)", fontweight="bold")
        axes[0].set_xlabel("TCGA Project")
        axes[0].set_ylabel("Tumor Tissue Profiles")
        for p in axes[0].patches:
            axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha='center', va='center', color='white', fontweight='bold')

        # Segment Mean distribution (Copy Number Amplitude)
        sns.kdeplot(data=merged_segs, x="Segment_Mean", hue="Project", common_norm=False,
                    clip=(-1.5, 1.5), ax=axes[1], palette="Set1")
        axes[1].axvline(0.0, color="gray", linestyle="--", alpha=0.7, label="Diploid Baseline")
        axes[1].axvline(0.2, color="red", linestyle=":", label="Copy Gain (> +0.2)")
        axes[1].axvline(-0.2, color="blue", linestyle=":", label="Copy Loss (< -0.2)")
        axes[1].set_title("Segment Copy-Ratio Distribution (log2 scale)", fontweight="bold")
        axes[1].set_xlabel("Segment Mean log2(Tumor / Normal)")
        axes[1].legend()

        plt.tight_layout()
        fig_path3 = output_dir / "03_tcga_cnv_distribution.png"
        plt.savefig(fig_path3, dpi=300)
        plt.close()
        logger.info("Saved %s", fig_path3)

        # Plot 4: Chromosomal Alteration Landscape (Focusing on Lung Adenocarcinoma TCGA-LUAD)
        luad_segs = merged_segs[merged_segs["Project"] == "TCGA-LUAD"].copy()
        if not luad_segs.empty:
            plt.figure(figsize=(15, 6))
            chr_summary = luad_segs.groupby("Chromosome", observed=True)["Segment_Mean"].agg(
                ["mean", "std", "count"]
            ).reset_index()

            # Plot mean deviation across chromosomes
            colors = ["#d95f02" if m > 0.02 else "#7570b3" if m < -0.02 else "#1b9e77" for m in chr_summary["mean"]]
            bars = plt.bar(chr_summary["Chromosome"], chr_summary["mean"], color=colors, edgecolor="black", alpha=0.85)
            plt.axhline(0, color="black", linewidth=1)
            plt.title("TCGA-LUAD (Lung Adenocarcinoma) Chromosome-Level Copy Number Trends (Chr 1-22)", fontweight="bold")
            plt.xlabel("Chromosome")
            plt.ylabel("Mean Segment Mean log2(Ratio)")
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            plt.tight_layout()
            fig_path4 = output_dir / "04_tcga_luad_chromosomal_profile.png"
            plt.savefig(fig_path4, dpi=300)
            plt.close()
            logger.info("Saved %s", fig_path4)

    return stats


def main():
    sample_sheet = "data/metadata/sample_sheet.csv"
    tcga_dir = Path("references/tcga_cnv")
    output_dir = Path("results/eda")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA): LIQUID BIOPSY & TCGA REFERENCE COHORTS")
    print("=" * 80)

    cfdna_stats = run_cfdna_eda(sample_sheet, output_dir)
    tcga_stats = run_tcga_eda(tcga_dir, output_dir)

    print("\n1. PRJNA795275 cfDNA Sequencing Cohort:")
    print(f"   - Total Sequencing Runs:     {cfdna_stats['total_runs']}")
    print(f"   - Unique Patient Donors:     {cfdna_stats['unique_patients']}")
    print(f"   - Healthy Control Plasma:    {cfdna_stats['healthy_count']}")
    print(f"   - Cancer Plasma:             {cfdna_stats['cancer_count']}")
    print(f"   - Subtype Breakdown:         {cfdna_stats['subtypes']}")
    print(f"   - Biological Sex Breakdown:  {cfdna_stats['sex_dist']}")
    print(f"   - Median Archive Size (GB):  {cfdna_stats['median_bytes_gb']:.2f} GB")

    print("\n2. TCGA Masked Copy Number Reference Cohort:")
    print(f"   - Total Segment Profiles:    {tcga_stats['total_tcga_files']}")
    print(f"   - Lineage Breakdown:         {tcga_stats['projects_distribution']}")
    if "total_segments_inspected" in tcga_stats:
        print(f"   - Inspected Segments:        {tcga_stats['total_segments_inspected']}")
        print(f"   - Segment Gains (> +0.2):    {tcga_stats['amplifications_count']}")
        print(f"   - Segment Losses (< -0.2):   {tcga_stats['deletions_count']}")

    print("\n3. Generated Diagnostic Figures:")
    for fig_file in sorted(output_dir.glob("*.png")):
        print(f"   - {fig_file}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
